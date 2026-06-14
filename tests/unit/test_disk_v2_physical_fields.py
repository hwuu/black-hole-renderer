#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 基础物理场层的单元测试（v2.1 同步）。

v2.1 改动后的关注点：

- `DiskV2Params` 默认值变为 `r_in=3, r_out=50, T_peak_K=1e7, rho_power=1.5`。
- `temp_scale` 参数被移除，替换为 `T_peak_K`。
- 密度场在 `r = r_in` 通过 `[1 - sqrt(r_in/r)]^(1/2)` 项自然取 0。
- `r_in < 3` 会触发 warning + 钳制，而不是 raise。

旧版测试 `setUp` 使用 `r_in=2, r_out=10, temp_scale=1.0` 等参数。
这里既改 setUp，又新增 `test_v2_temperature_range_default` 等 v2.1 专属测试。
"""

import math
import unittest
import warnings

import numpy as np

from disk_v2.physical_fields import (
    _thin_disk_temperature_raw,
    angular_velocity_field,
    density_field,
    midplane_density_field,
    midplane_temperature_field,
    temperature_field,
)
from disk_v2.geometry import (
    disk_half_thickness,
    disk_radial_mask,
    disk_radial_weight,
    disk_vertical_weight,
    disk_volume_mask,
)
from disk_v2.params import DiskV2Params, SCHWARZSCHILD_ISCO_R_S


class DiskV2PhysicalFieldsTest(unittest.TestCase):
    def setUp(self):
        # v2.1 默认参数：r_in=3, r_out=50, T_peak_K=1e7。
        # 为了让既有几何测试（半径采样、容差等）继续可读，这里显式构造一组
        # 较小半径的参数，避免 r_out=50 让一些"接近内边界"的相对距离测试
        # 难以编排。
        self.params = DiskV2Params(
            r_in=3.0,
            r_out=10.0,
            h0=0.05,
            beta_h=0.05,
            rho_power=1.5,
            T_peak_K=1.0e7,
            omega_scale=1.0,
            edge_softness=0.1,
        )

    # --- ISCO 钳制 ---

    def test_r_in_below_isco_is_clamped_with_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            params = DiskV2Params(r_in=2.0, r_out=10.0)

        self.assertEqual(params.r_in, SCHWARZSCHILD_ISCO_R_S)
        # 仅校验"产生了至少一条与 ISCO 钳制相关的 warning"。
        self.assertTrue(any("ISCO" in str(w.message) for w in caught))

    def test_invalid_params_raise_value_error(self):
        # r_in 钳制为 3.0 后，r_out=2.5 仍然 ≤ r_in，应 raise。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(ValueError):
                DiskV2Params(r_in=2.0, r_out=2.5)

        with self.assertRaises(ValueError):
            DiskV2Params(h0=0.0)

        with self.assertRaises(ValueError):
            DiskV2Params(edge_softness=0.5)

        with self.assertRaises(ValueError):
            DiskV2Params(T_peak_K=0.0)

        with self.assertRaises(ValueError):
            DiskV2Params(alpha_density=-0.1)

        with self.assertRaises(ValueError):
            DiskV2Params(beta_temperature=-0.1)

    # --- 几何（保持 v1.0 行为） ---

    def test_disk_half_thickness_is_positive_and_smooth(self):
        radii = np.linspace(self.params.r_in, self.params.r_out, 64)
        thickness = disk_half_thickness(radii, self.params)

        self.assertTrue(np.all(thickness > 0.0))
        self.assertTrue(np.all(np.diff(thickness) > 0.0))
        self.assertFalse(np.any(np.isnan(thickness)))

    def test_disk_radial_mask_respects_inner_and_outer_bounds(self):
        radii = np.array(
            [
                self.params.r_in * 0.9,
                self.params.r_in,
                0.5 * (self.params.r_in + self.params.r_out),
                self.params.r_out,
                self.params.r_out * 1.1,
            ],
            dtype=np.float64,
        )
        mask = disk_radial_mask(radii, self.params)

        self.assertFalse(mask[0])
        self.assertTrue(mask[1])
        self.assertTrue(mask[2])
        self.assertTrue(mask[3])
        self.assertFalse(mask[4])

    def test_disk_volume_mask_respects_radial_and_vertical_bounds(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)

        self.assertTrue(disk_volume_mask(r_mid, 0.0, self.params))
        self.assertFalse(disk_volume_mask(self.params.r_in * 0.9, 0.0, self.params))
        self.assertFalse(disk_volume_mask(self.params.r_out * 1.1, 0.0, self.params))
        self.assertFalse(disk_volume_mask(r_mid, h_mid * 1.01, self.params))

    def test_geometry_masks_use_closed_boundary_convention(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)

        self.assertTrue(disk_radial_mask(self.params.r_in, self.params))
        self.assertTrue(disk_radial_mask(self.params.r_out, self.params))
        self.assertTrue(disk_volume_mask(r_mid, h_mid, self.params))
        self.assertTrue(disk_volume_mask(r_mid, -h_mid, self.params))

    def test_weights_and_fields_vanish_on_exact_boundaries(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)

        self.assertEqual(disk_radial_weight(self.params.r_in, self.params), 0.0)
        self.assertEqual(disk_radial_weight(self.params.r_out, self.params), 0.0)
        self.assertEqual(disk_vertical_weight(r_mid, h_mid, self.params), 0.0)
        self.assertEqual(disk_vertical_weight(r_mid, -h_mid, self.params), 0.0)
        self.assertEqual(midplane_density_field(self.params.r_in, self.params), 0.0)
        self.assertEqual(midplane_temperature_field(self.params.r_in, self.params), 0.0)
        self.assertEqual(density_field(r_mid, h_mid, self.params), 0.0)
        self.assertEqual(temperature_field(r_mid, h_mid, self.params), 0.0)

    def test_disk_vertical_weight_is_symmetric_and_closes_on_surface(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)
        z_samples = np.array([0.0, 0.25 * h_mid, -0.25 * h_mid, h_mid, 1.1 * h_mid], dtype=np.float64)

        weight = disk_vertical_weight(r_mid, z_samples, self.params)

        self.assertAlmostEqual(weight[0], 1.0, places=8)
        self.assertAlmostEqual(weight[1], weight[2], places=8)
        self.assertGreater(weight[1], 0.0)
        self.assertEqual(weight[3], 0.0)
        self.assertEqual(weight[4], 0.0)

    def test_disk_radial_weight_is_flat_in_middle_and_zero_outside(self):
        radii = np.array(
            [
                self.params.r_in - 0.1,
                self.params.r_in + 0.05,
                0.5 * (self.params.r_in + self.params.r_out),
                self.params.r_out - 0.05,
                self.params.r_out + 0.1,
            ],
            dtype=np.float64,
        )
        weight = disk_radial_weight(radii, self.params)

        self.assertEqual(weight[0], 0.0)
        self.assertGreater(weight[1], 0.0)
        self.assertAlmostEqual(weight[2], 1.0, places=6)
        self.assertGreater(weight[3], 0.0)
        self.assertEqual(weight[4], 0.0)

    def test_small_disk_outer_edge_uses_wider_softening_than_inner_edge(self):
        """v2.2：小盘外缘使用更宽软化带，内缘仍保持窄软化以保护 SS 温度峰。"""
        params = DiskV2Params(r_in=3.0, r_out=15.0, edge_softness=0.02)

        # 内边界软化仍很窄：r_in + 0.30 已应接近平台。
        self.assertGreater(disk_radial_weight(params.r_in + 0.30, params), 0.95)

        # 外边界软化有 0.6 r_s 下限：r_out - 0.30 位于 soft edge 中间。
        mid_outer_weight = disk_radial_weight(params.r_out - 0.30, params)
        self.assertGreater(mid_outer_weight, 0.35)
        self.assertLess(mid_outer_weight, 0.65)

    def test_default_temperature_peak_survives_inner_soft_edge(self):
        """v2.2：内边界 soft edge 不应削掉默认 SS 温度峰。"""
        params = DiskV2Params(T_peak_K=1.0e7)
        radii = np.linspace(params.r_in + 1e-3, params.r_out - 1e-3, 4096)
        peak = float(np.max(midplane_temperature_field(radii, params)))

        self.assertGreaterEqual(peak, 0.95 * params.T_peak_K)

    # --- 基础场 ---

    def test_angular_velocity_field_monotonically_decreases_with_radius(self):
        radii = np.linspace(self.params.r_in, self.params.r_out, 128)
        omega = angular_velocity_field(radii, self.params)

        self.assertTrue(np.all(omega > 0.0))
        self.assertTrue(np.all(np.diff(omega) < 0.0))
        self.assertGreater(omega[0], omega[-1])

    def test_density_is_symmetric_and_midplane_dominant(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)

        rho_mid = density_field(r_mid, 0.0, self.params)
        rho_up = density_field(r_mid, 0.5 * h_mid, self.params)
        rho_down = density_field(r_mid, -0.5 * h_mid, self.params)

        self.assertGreater(rho_mid, rho_up)
        self.assertAlmostEqual(rho_up, rho_down, places=8)
        self.assertEqual(density_field(r_mid, 1.1 * h_mid, self.params), 0.0)
        self.assertEqual(density_field(self.params.r_out * 1.1, 0.0, self.params), 0.0)
        self.assertGreater(midplane_density_field(r_mid, self.params), 0.0)

    def test_density_field_vanishes_on_geometric_surface(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)

        self.assertGreater(density_field(r_mid, 0.9 * h_mid, self.params), 0.0)
        self.assertEqual(density_field(r_mid, h_mid, self.params), 0.0)

    def test_density_field_uses_inner_boundary_suppression(self):
        """v2.1：密度场加入 SS 启发式内边界压制项后，应在 r=r_in 处取 0、
        在 r≈1.5 r_in 附近出现峰值。
        """
        radii = np.linspace(self.params.r_in, self.params.r_out, 2048)
        rho = midplane_density_field(radii, self.params)

        # r=r_in 处密度为 0（被内边界压制或 W_r 共同贡献）。
        self.assertEqual(rho[0], 0.0)

        # 峰值位置：内边界压制 sqrt(1 - sqrt(r_in/r)) 与径向幂律 (r/r_in)^(-rho_power)
        # 共同决定。对 rho_power=1.5、内边界因子 ∝ [1-sqrt(r_in/r)]^(1/2)，
        # 解 dρ/dr=0 给出 r_peak / r_in 在 1.5 附近（弱依赖 rho_power）。
        peak_idx = int(np.argmax(rho))
        peak_radius = radii[peak_idx]
        self.assertGreater(peak_radius, self.params.r_in)
        self.assertLess(peak_radius, 2.0 * self.params.r_in)

    def test_temperature_peaks_outside_inner_edge_and_decays_outward(self):
        radii = np.linspace(self.params.r_in, self.params.r_out, 2048)
        temp_mid = midplane_temperature_field(radii, self.params)
        peak_idx = int(np.argmax(temp_mid))
        peak_radius = radii[peak_idx]

        self.assertEqual(temp_mid[0], 0.0)
        # 标准薄盘峰值在 r ≈ 1.36 r_in；但乘上 W_r 后峰值会受外边界软化轻微影响。
        # 这里只检查"峰值在 r_in 之后、在盘内偏内侧"。
        self.assertGreater(peak_radius, self.params.r_in)
        self.assertLess(peak_radius, self.params.r_in + 0.5 * (self.params.r_out - self.params.r_in))
        self.assertTrue(np.all(np.diff(temp_mid[peak_idx:]) <= 1e-8))

    def test_temperature_is_symmetric_and_midplane_dominant(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)

        temp_mid = temperature_field(r_mid, 0.0, self.params)
        temp_up = temperature_field(r_mid, 0.5 * h_mid, self.params)
        temp_down = temperature_field(r_mid, -0.5 * h_mid, self.params)

        self.assertGreater(temp_mid, temp_up)
        self.assertAlmostEqual(temp_up, temp_down, places=8)
        self.assertEqual(temperature_field(r_mid, 1.1 * h_mid, self.params), 0.0)

    def test_temperature_field_vanishes_on_geometric_surface(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)

        self.assertGreater(temperature_field(r_mid, 0.9 * h_mid, self.params), 0.0)
        self.assertEqual(temperature_field(r_mid, h_mid, self.params), 0.0)

    # --- v2.1 新增：温度量纲 + raw profile 跨度 ---

    def test_temperature_carries_kelvin_unit(self):
        """v2.1：中面温度应携带物理 K 量纲，峰值接近 T_peak_K。

        默认 `edge_softness=0.02` 下，内边界软化区跨度约 0.94 r_s，
        SS 温度峰值（r ≈ 1.36 · r_in ≈ 4.08）远离软化区，W_r 在峰值附近 ≈ 1。
        因此完整 `T_mid` 的峰值与 `T_peak_K` 误差应 < 5%。
        """
        params = DiskV2Params(T_peak_K=1.0e7)  # 用全默认半径 + edge_softness
        radii = np.linspace(params.r_in + 1e-3, params.r_out - 1e-3, 4096)
        temp_mid = midplane_temperature_field(radii, params)
        peak = float(np.max(temp_mid))

        self.assertAlmostEqual(peak, params.T_peak_K, delta=0.05 * params.T_peak_K)

    def test_v2_temperature_range_default(self):
        """v2.1 验收硬指标：未乘 W_r 的 raw thin-disk 剖面在默认半径下
        peak-to-outer 比值约 4.3 倍。

        参考 docs/design_ad_v2.md §6 测试方案。
        """
        # 用 v2.1 默认参数 r_in=3, r_out=50。
        params = DiskV2Params()
        self.assertEqual(params.r_in, 3.0)
        self.assertEqual(params.r_out, 50.0)

        radii = np.linspace(params.r_in + 1e-6, params.r_out, 16384)
        raw = _thin_disk_temperature_raw(radii, params.r_in)
        peak = float(np.max(raw))
        outer = float(raw[-1])
        ratio = peak / outer

        # 实算 ≈ 4.32；docs/design_ad_v2.md §6 给出验收区间 [4.0, 4.6]。
        self.assertGreaterEqual(ratio, 4.0)
        self.assertLessEqual(ratio, 4.6)

    def test_thin_disk_raw_peak_location(self):
        """v2.1：raw thin-disk 剖面峰值位置在 r ≈ 1.36 · r_in 附近。"""
        params = DiskV2Params(r_in=3.0, r_out=50.0)
        radii = np.linspace(params.r_in, params.r_out, 16384)
        raw = _thin_disk_temperature_raw(radii, params.r_in)
        peak_idx = int(np.argmax(raw))
        peak_over_r_in = radii[peak_idx] / params.r_in

        expected = 49.0 / 36.0  # ≈ 1.361
        self.assertAlmostEqual(peak_over_r_in, expected, delta=0.01)


if __name__ == "__main__":
    unittest.main()
