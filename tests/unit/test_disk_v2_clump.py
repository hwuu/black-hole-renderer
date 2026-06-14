#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 团块调制 `F_clump` 单元测试（v2.1 新增）。

覆盖：

- 团块中心可复现性、数量、分布偏内圈。
- `clump_modulation` 在盘外返回中性值 1。
- 团块边界锐利（0.1 σ_r 内幅度跌至 50% 以下）。
- 高倾角穿盘路径上 `F_clump` 出现多次极值（"亮 → 暗 → 亮"振荡）。
- 三维 vs 二维退化（z=None 等同 z=0）。
"""

import unittest

import numpy as np

from disk_v2.params import DiskV2Params, DiskV2StructureParams
from disk_v2.structure_modulations import (
    _ClumpCenters,
    _sample_clump_centers,
    clump_modulation,
    structure_modulation_density,
    structure_modulation_emission,
)


class DiskV2ClumpTest(unittest.TestCase):
    def setUp(self):
        # 使用 v2.1 默认半径范围。
        self.params = DiskV2Params(r_in=3.0, r_out=50.0)
        self.structure_params = DiskV2StructureParams(
            clump_count=400,
            clump_strength=0.6,
            clump_radial_sigma_scale=0.2,
            clump_vertical_sigma_scale=0.5,
            clump_phi_sigma=0.15,
        )

    # --- 团块中心 ---

    def test_clump_centers_count_matches_param(self):
        centers = _sample_clump_centers(self.params, self.structure_params, seed=42)
        self.assertEqual(len(centers.r), self.structure_params.clump_count)
        self.assertEqual(len(centers.phi), self.structure_params.clump_count)
        self.assertEqual(len(centers.z), self.structure_params.clump_count)
        self.assertEqual(len(centers.amplitude), self.structure_params.clump_count)

    def test_clump_centers_are_reproducible_for_same_seed(self):
        a = _sample_clump_centers(self.params, self.structure_params, seed=123)
        b = _sample_clump_centers(self.params, self.structure_params, seed=123)
        c = _sample_clump_centers(self.params, self.structure_params, seed=124)

        self.assertTrue(np.allclose(a.r, b.r))
        self.assertTrue(np.allclose(a.phi, b.phi))
        self.assertTrue(np.allclose(a.z, b.z))
        self.assertFalse(np.allclose(a.r, c.r))

    def test_clump_centers_are_inside_disk_radial_bounds(self):
        centers = _sample_clump_centers(self.params, self.structure_params, seed=7)
        self.assertTrue(np.all(centers.r >= self.params.r_in))
        self.assertTrue(np.all(centers.r <= self.params.r_out))

    def test_clump_centers_log_uniform_biases_toward_inner_disk(self):
        """log-均匀采样应让一半团块落在内 sqrt(r_in*r_out) 之内。"""
        centers = _sample_clump_centers(self.params, self.structure_params, seed=7)
        boundary = np.sqrt(self.params.r_in * self.params.r_out)
        inner_fraction = float(np.mean(centers.r <= boundary))
        # 理论值 0.5，允许 ±0.05 的随机波动（400 个样本下 σ ≈ 0.025）。
        self.assertAlmostEqual(inner_fraction, 0.5, delta=0.05)

    # --- F_clump 行为 ---

    def test_clump_modulation_neutral_outside_disk(self):
        outside_r = self.params.r_out * 1.1
        self.assertEqual(
            clump_modulation(outside_r, 0.0, z=0.0, params=self.params, structure_params=self.structure_params, seed=42),
            1.0,
        )

    def test_clump_modulation_is_positive_and_finite_inside_disk(self):
        r_values = np.linspace(self.params.r_in + 1e-3, self.params.r_out - 1e-3, 32)
        phi_values = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        r_grid, phi_grid = np.meshgrid(r_values, phi_values, indexing="ij")
        z_grid = np.zeros_like(r_grid)

        field = clump_modulation(
            r_grid, phi_grid, z=z_grid,
            params=self.params, structure_params=self.structure_params, seed=42,
        )
        self.assertEqual(field.shape, r_grid.shape)
        self.assertTrue(np.all(field > 0.0))
        self.assertTrue(np.all(np.isfinite(field)))

    def test_clump_modulation_z_none_equals_z_zero(self):
        r_values = np.linspace(self.params.r_in + 1e-3, self.params.r_out - 1e-3, 16)
        phi_values = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        r_grid, phi_grid = np.meshgrid(r_values, phi_values, indexing="ij")

        a = clump_modulation(
            r_grid, phi_grid, z=None,
            params=self.params, structure_params=self.structure_params, seed=42,
        )
        b = clump_modulation(
            r_grid, phi_grid, z=np.zeros_like(r_grid),
            params=self.params, structure_params=self.structure_params, seed=42,
        )
        self.assertTrue(np.allclose(a, b))

    def test_clump_modulation_amplitude_bounded(self):
        """`F_clump` 围绕 1 波动，最大偏移由 `clump_strength` 决定。"""
        r_values = np.linspace(self.params.r_in + 1e-3, self.params.r_out - 1e-3, 64)
        phi_values = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        r_grid, phi_grid = np.meshgrid(r_values, phi_values, indexing="ij")
        z_grid = np.zeros_like(r_grid)

        field = clump_modulation(
            r_grid, phi_grid, z=z_grid,
            params=self.params, structure_params=self.structure_params, seed=42,
        )
        # signed 累积逐点 clamp 到 [-1, 1]，因此 field ∈ [1 - clump_strength, 1 + clump_strength]。
        margin = 1e-9
        self.assertGreaterEqual(float(np.min(field)), 1.0 - self.structure_params.clump_strength - margin)
        self.assertLessEqual(float(np.max(field)), 1.0 + self.structure_params.clump_strength + margin)

    def test_clump_modulation_centers_override(self):
        """如果传入预生成 centers，函数应使用它，等同于该 seed 下的 _sample_clump_centers。"""
        centers = _sample_clump_centers(self.params, self.structure_params, seed=11)
        r_value = float(centers.r[0])
        phi_value = float(centers.phi[0])
        z_value = float(centers.z[0])

        # 用 centers 显式传入。
        a = clump_modulation(
            r_value, phi_value, z=z_value,
            params=self.params, structure_params=self.structure_params,
            centers=centers,
        )
        # 用 seed 自动生成 centers，应得到同一结果。
        b = clump_modulation(
            r_value, phi_value, z=z_value,
            params=self.params, structure_params=self.structure_params, seed=11,
        )
        self.assertAlmostEqual(float(a), float(b), places=12)

    # --- 边界锐度与体积振荡 ---

    def test_clump_modulation_boundary_sharpness(self):
        """单个团块从核到 0.1·σ_r 之外，幅度应快速衰减；用紧支撑核保证有限范围。"""
        # 构造只含一个团块（在盘内某处）的 centers。
        r_k = 0.5 * (self.params.r_in + self.params.r_out)
        phi_k = 0.0
        z_k = 0.0
        amp_k = 1.0
        centers = _ClumpCenters(
            r=np.array([r_k]),
            phi=np.array([phi_k]),
            z=np.array([z_k]),
            amplitude=np.array([amp_k]),
        )

        sigma_r = self.structure_params.clump_radial_sigma_scale * self.params.r_in

        # 测沿径向远离核心的衰减。
        r_samples = np.array([r_k, r_k + 0.5 * sigma_r, r_k + 0.9 * sigma_r, r_k + 1.0 * sigma_r, r_k + 1.5 * sigma_r])
        phi_samples = np.full_like(r_samples, phi_k)
        z_samples = np.zeros_like(r_samples)

        field = clump_modulation(
            r_samples, phi_samples, z=z_samples,
            params=self.params, structure_params=self.structure_params,
            centers=centers,
        )
        # 减去 1 得到 signed 调制。clump 使用逐点 clamp，单团块情况下可直接测
        # 核 vs 远端的相对衰减。
        signed = field - 1.0
        core_amp = float(np.abs(signed[0]))
        out_of_kernel_amp = float(np.abs(signed[-1]))

        self.assertGreater(core_amp, 0.0, msg="核心位置应有非零团块贡献")
        # 紧支撑核保证 d > 1 时核值为 0，团块贡献严格为 0。
        # 逐点 clamp 后仍是 0。
        self.assertAlmostEqual(out_of_kernel_amp, 0.0, places=9, msg="超出紧支撑边界应严格为 0")

        # 半径方向 0.9 σ_r 处应已显著衰减（核值为 0.1，平方再 smoothstep 后 ≈ 0.028）。
        mid_amp = float(np.abs(signed[2]))
        # 核内 r_k 处 d=0、kernel=1 -> smoothstep(1)=1；r_k+0.9σ_r 处 d=0.9、kernel=0.1
        # -> smoothstep(0.1) ≈ 0.028。相对核心 ≈ 2.8%。
        self.assertLess(mid_amp, 0.5 * core_amp, msg="0.9 σ_r 处应明显衰减")

    def test_clump_modulation_volumetric_oscillation(self):
        """沿穿盘垂向路径，`F_clump` 应在团块所在的 z 位置处出现局部极值，
        反映"光线穿盘时反复进入团块/离开团块"的振荡。

        这是 docs/design_ad_v2.md §6 视觉验收的硬指标之一。
        为了让单测确定性强、不依赖 400 团块的随机分布是否覆盖采样路径，
        这里用 `_ClumpCenters` 显式构造 3 个紧密相邻的团块。
        团块在 z 方向间隔约 1.5 倍 σ_z，足以让每个团块产生独立的局部极值。
        """
        from disk_v2.geometry import disk_half_thickness
        r_fixed = 0.5 * (self.params.r_in + self.params.r_out)
        phi_fixed = 0.0
        h_mid = float(disk_half_thickness(r_fixed, self.params))
        sigma_z = self.structure_params.clump_vertical_sigma_scale * h_mid

        # 把 3 个团块沿 z 间隔 1.5 σ_z 放置（这样恰好不重叠，但仍在盘内）。
        # amplitude 全部为正，让它们都产生"亮于中性"的局部极值。
        spacing = 1.5 * sigma_z
        # 确保 ±spacing < H(r)。
        self.assertLess(spacing, h_mid)
        centers = _ClumpCenters(
            r=np.array([r_fixed, r_fixed, r_fixed]),
            phi=np.array([phi_fixed, phi_fixed, phi_fixed]),
            z=np.array([-spacing, 0.0, +spacing]),
            amplitude=np.array([1.0, 1.0, 1.0]),
        )

        z_samples = np.linspace(-0.999 * h_mid, 0.999 * h_mid, 513)
        r_samples = np.full_like(z_samples, r_fixed)
        phi_samples = np.full_like(z_samples, phi_fixed)

        field = clump_modulation(
            r_samples, phi_samples, z=z_samples,
            params=self.params, structure_params=self.structure_params,
            centers=centers,
        )
        # 找局部极大值的数量：先验证 field 不是常数。
        peak_count = int(np.sum(np.diff(np.sign(np.diff(field))) < 0))
        self.assertGreaterEqual(
            peak_count, 3,
            msg=f"沿 z 路径预期看到至少 3 个局部极大值（3 个团块各产生 1 个），实际 {peak_count}。",
        )

    # --- F_struct_density / F_struct_emission 入口 ---

    def test_structure_modulation_density_is_positive(self):
        r = np.linspace(self.params.r_in + 1e-3, self.params.r_out - 1e-3, 16)
        phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        z = np.zeros_like(r)
        r_g, phi_g = np.meshgrid(r, phi, indexing="ij")
        z_g = np.zeros_like(r_g)
        field = structure_modulation_density(
            r_g, phi_g, z_g, self.params, self.structure_params, seed=42,
        )
        self.assertEqual(field.shape, r_g.shape)
        self.assertTrue(np.all(field > 0.0))
        self.assertTrue(np.all(np.isfinite(field)))

    def test_structure_modulation_density_neutral_outside_disk(self):
        outside_r = self.params.r_out * 1.1
        field = structure_modulation_density(
            outside_r, 0.0, 0.0, self.params, self.structure_params, seed=42,
        )
        self.assertEqual(field, 1.0)

    def test_structure_modulation_emission_is_positive(self):
        r = np.linspace(self.params.r_in + 1e-3, self.params.r_out - 1e-3, 16)
        phi = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        r_g, phi_g = np.meshgrid(r, phi, indexing="ij")
        z_g = np.zeros_like(r_g)
        field = structure_modulation_emission(
            r_g, phi_g, z_g, self.params, self.structure_params, seed=42,
        )
        self.assertEqual(field.shape, r_g.shape)
        self.assertTrue(np.all(field > 0.0))
        self.assertTrue(np.all(np.isfinite(field)))

    def test_structure_modulation_emission_neutral_outside_disk(self):
        outside_r = self.params.r_out * 1.1
        field = structure_modulation_emission(
            outside_r, 0.0, 0.0, self.params, self.structure_params, seed=42,
        )
        self.assertEqual(field, 1.0)


if __name__ == "__main__":
    unittest.main()
