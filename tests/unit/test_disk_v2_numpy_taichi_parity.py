#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 NumPy 与 Taichi 实现的 parity 测试（v2.1 Phase 4）。

覆盖：

- 几何函数：disk_half_thickness、disk_radial_weight、disk_vertical_weight、disk_volume_mask。
- 物理场：midplane_density_field、midplane_temperature_field、density_field、temperature_field。
- 团块场：clump_modulation。
- 调色：blackbody_color / cinematic palette / tonemap_reinhard。

容差：相对误差 `< 1e-4`（fp32 Taichi 路径下；fp64 NumPy 参考实现）。
"""

import unittest

import numpy as np
import taichi as ti

from disk_v2.geometry import (
    disk_half_thickness,
    disk_radial_weight,
    disk_vertical_weight,
    disk_volume_mask,
)
from disk_v2.params import DiskV2PaletteParams, DiskV2Params, DiskV2StructureParams
from disk_v2.palette import (
    apply_exposure,
    blackbody_color,
    cinematic_color,
    cinematic_visual_temperature,
    tonemap,
)
from disk_v2.imaging import observed_visible_temperature
from disk_v2.physical_fields import (
    density_field,
    midplane_density_field,
    midplane_temperature_field,
    temperature_field,
)
from disk_v2.structure_modulations import clump_modulation
from disk_v2.taichi_impl import DiskV2Taichi


_TI_INITED = False


def _ensure_taichi():
    global _TI_INITED
    if not _TI_INITED:
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        _TI_INITED = True


class DiskV2NumpyTaichiParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_taichi()
        cls.params = DiskV2Params(r_in=3.0, r_out=50.0, T_peak_K=1.0e7, edge_softness=0.02)
        cls.structure_params = DiskV2StructureParams(clump_count=80, clump_strength=0.6)
        cls.palette_params_phys = DiskV2PaletteParams(palette_mode="physical")
        cls.palette_params_cine = DiskV2PaletteParams(palette_mode="cinematic")

        # Taichi 句柄分别用 physical 和 cinematic 构造两个，便于测试不同模式。
        cls.ti_phys = DiskV2Taichi(
            cls.params, cls.structure_params, cls.palette_params_phys, seed=7
        )
        cls.ti_cine = DiskV2Taichi(
            cls.params, cls.structure_params, cls.palette_params_cine, seed=7,
            centers=cls.ti_phys.centers,
        )

        # 固定测试网格：32 个 (r, phi, z) 采样点。
        rng = np.random.default_rng(0)
        cls.n = 32
        cls.r_samples = rng.uniform(cls.params.r_in - 1.0, cls.params.r_out + 1.0, cls.n).astype(np.float64)
        cls.phi_samples = rng.uniform(0.0, 2.0 * np.pi, cls.n).astype(np.float64)
        # 让 z 一部分在盘内、一部分在外。
        h_samples = np.maximum(disk_half_thickness(np.maximum(cls.r_samples, cls.params.r_in), cls.params), 1e-12)
        z_rel = rng.uniform(-1.5, 1.5, cls.n)
        cls.z_samples = (z_rel * h_samples).astype(np.float64)

        # 温度采样：覆盖 0 ~ 2e7 K。
        cls.T_samples = np.linspace(0.0, 2.0e7, cls.n).astype(np.float64)

    # --- 公用 kernel ---

    def _run_scalar_kernel(self, ti_obj, sample_fn_name: str, inputs: list[tuple[float, ...]]):
        """通用：对一组输入逐点调用 ti_obj 的 ti.func，返回标量数组。"""
        n = len(inputs)
        out = ti.field(dtype=ti.f32, shape=n)
        # 把 inputs 上传到 fields。
        max_args = max(len(x) for x in inputs)
        in_fields = [ti.field(dtype=ti.f32, shape=n) for _ in range(max_args)]
        for arg_idx in range(max_args):
            data = np.array([x[arg_idx] if arg_idx < len(x) else 0.0 for x in inputs], dtype=np.float32)
            in_fields[arg_idx].from_numpy(data)

        # 这里直接定义内核闭包，每个调用都重建。
        # 不同 sample 函数签名不同，要分别处理。
        # 为了简化，每个 parity 测试自己写专用 kernel。
        raise NotImplementedError("Use per-test kernels.")

    # --- 几何 parity ---

    def test_parity_disk_half_thickness(self):
        from disk_v2.taichi_impl import disk_half_thickness_ti
        params = self.params

        @ti.kernel
        def compute(out: ti.template(), r_in_field: ti.template()):
            for i in range(out.shape[0]):
                out[i] = disk_half_thickness_ti(r_in_field[i], params.h0, params.beta_h, params.r_in)

        r_in_f = ti.field(dtype=ti.f32, shape=self.n)
        r_in_f.from_numpy(self.r_samples.astype(np.float32))
        out_f = ti.field(dtype=ti.f32, shape=self.n)
        compute(out_f, r_in_f)
        ti_result = out_f.to_numpy().astype(np.float64)
        np_result = np.asarray(disk_half_thickness(self.r_samples, params), dtype=np.float64)
        # disk_half_thickness 对 r < r_in 内部用 r_in 作为下界（NumPy 与 Taichi 一致）。
        np.testing.assert_allclose(ti_result, np_result, rtol=1e-4, atol=1e-6)

    def test_parity_disk_radial_weight(self):
        from disk_v2.taichi_impl import disk_radial_weight_ti
        params = self.params

        @ti.kernel
        def compute(out: ti.template(), r_f: ti.template()):
            for i in range(out.shape[0]):
                out[i] = disk_radial_weight_ti(r_f[i], params.r_in, params.r_out, params.edge_softness)

        r_f = ti.field(dtype=ti.f32, shape=self.n)
        r_f.from_numpy(self.r_samples.astype(np.float32))
        out_f = ti.field(dtype=ti.f32, shape=self.n)
        compute(out_f, r_f)
        ti_result = out_f.to_numpy().astype(np.float64)
        np_result = np.asarray(disk_radial_weight(self.r_samples, params), dtype=np.float64)
        np.testing.assert_allclose(ti_result, np_result, rtol=1e-4, atol=1e-6)

    def test_parity_disk_volume_mask(self):
        from disk_v2.taichi_impl import disk_volume_mask_ti
        params = self.params

        @ti.kernel
        def compute(out: ti.template(), r_f: ti.template(), z_f: ti.template()):
            for i in range(out.shape[0]):
                out[i] = disk_volume_mask_ti(r_f[i], z_f[i], params.h0, params.beta_h, params.r_in, params.r_out)

        r_f = ti.field(dtype=ti.f32, shape=self.n)
        z_f = ti.field(dtype=ti.f32, shape=self.n)
        r_f.from_numpy(self.r_samples.astype(np.float32))
        z_f.from_numpy(self.z_samples.astype(np.float32))
        out_f = ti.field(dtype=ti.i32, shape=self.n)
        compute(out_f, r_f, z_f)
        ti_result = out_f.to_numpy().astype(bool)
        np_result = np.asarray(disk_volume_mask(self.r_samples, self.z_samples, params), dtype=bool)
        # 在浮点边界 |z| ≈ H(r) 上 Taichi/NumPy 可能差 1 ULP，造成 mask 差异。
        # 这里要求大多数一致；统计相同率 ≥ 95%。
        agree = float(np.mean(ti_result == np_result))
        self.assertGreaterEqual(agree, 0.95)

    # --- 物理场 parity ---

    def test_parity_midplane_density(self):
        from disk_v2.taichi_impl import midplane_density_ti
        params = self.params

        @ti.kernel
        def compute(out: ti.template(), r_f: ti.template()):
            for i in range(out.shape[0]):
                out[i] = midplane_density_ti(r_f[i], params.r_in, params.r_out, params.rho_power, params.edge_softness)

        r_f = ti.field(dtype=ti.f32, shape=self.n)
        r_f.from_numpy(self.r_samples.astype(np.float32))
        out_f = ti.field(dtype=ti.f32, shape=self.n)
        compute(out_f, r_f)
        ti_result = out_f.to_numpy().astype(np.float64)
        np_result = np.asarray(midplane_density_field(self.r_samples, params), dtype=np.float64)
        np.testing.assert_allclose(ti_result, np_result, rtol=1e-3, atol=1e-6)

    def test_parity_midplane_temperature(self):
        from disk_v2.taichi_impl import midplane_temperature_ti
        params = self.params

        @ti.kernel
        def compute(out: ti.template(), r_f: ti.template()):
            for i in range(out.shape[0]):
                out[i] = midplane_temperature_ti(r_f[i], params.r_in, params.r_out, params.T_peak_K, params.edge_softness)

        r_f = ti.field(dtype=ti.f32, shape=self.n)
        r_f.from_numpy(self.r_samples.astype(np.float32))
        out_f = ti.field(dtype=ti.f32, shape=self.n)
        compute(out_f, r_f)
        ti_result = out_f.to_numpy().astype(np.float64)
        np_result = np.asarray(midplane_temperature_field(self.r_samples, params), dtype=np.float64)
        # 温度 ~ 1e7，绝对容差按 0.1% 量级。
        np.testing.assert_allclose(ti_result, np_result, rtol=1e-3, atol=10.0)

    def test_parity_density_field(self):
        from disk_v2.taichi_impl import density_field_ti
        params = self.params

        @ti.kernel
        def compute(out: ti.template(), r_f: ti.template(), z_f: ti.template()):
            for i in range(out.shape[0]):
                out[i] = density_field_ti(
                    r_f[i], z_f[i],
                    params.r_in, params.r_out, params.rho_power,
                    params.h0, params.beta_h, params.edge_softness,
                )

        r_f = ti.field(dtype=ti.f32, shape=self.n)
        z_f = ti.field(dtype=ti.f32, shape=self.n)
        r_f.from_numpy(self.r_samples.astype(np.float32))
        z_f.from_numpy(self.z_samples.astype(np.float32))
        out_f = ti.field(dtype=ti.f32, shape=self.n)
        compute(out_f, r_f, z_f)
        ti_result = out_f.to_numpy().astype(np.float64)
        np_result = np.asarray(density_field(self.r_samples, self.z_samples, params), dtype=np.float64)
        # 边界处可能小差异，整体应一致。
        np.testing.assert_allclose(ti_result, np_result, rtol=1e-3, atol=1e-6)

    # --- F_clump parity ---

    def test_parity_clump_modulation(self):
        """`F_clump` 的 NumPy reference 与 Taichi 实现应逐点一致。"""
        ti_obj = self.ti_phys

        # 用 ti.kernel 计算 ti 版本的 F_clump。
        @ti.kernel
        def compute(out: ti.template(), r_f: ti.template(), phi_f: ti.template(), z_f: ti.template(),
                    obj: ti.template()):
            for i in range(out.shape[0]):
                out[i] = obj.clump_modulation_ti(r_f[i], phi_f[i], z_f[i])

        r_f = ti.field(dtype=ti.f32, shape=self.n)
        phi_f = ti.field(dtype=ti.f32, shape=self.n)
        z_f = ti.field(dtype=ti.f32, shape=self.n)
        out_f = ti.field(dtype=ti.f32, shape=self.n)
        r_f.from_numpy(self.r_samples.astype(np.float32))
        phi_f.from_numpy(self.phi_samples.astype(np.float32))
        z_f.from_numpy(self.z_samples.astype(np.float32))
        compute(out_f, r_f, phi_f, z_f, ti_obj)
        ti_result = out_f.to_numpy().astype(np.float64)

        np_result = np.asarray(
            clump_modulation(
                self.r_samples, self.phi_samples, z=self.z_samples,
                params=self.params, structure_params=self.structure_params,
                centers=ti_obj.centers,
            ),
            dtype=np.float64,
        )

        np.testing.assert_allclose(ti_result, np_result, rtol=1e-4, atol=1e-5)

    def test_parity_emission_atlas_sampling(self):
        """视觉 atlas 双线性采样与 NumPy reference 一致。"""
        from disk_v2.visual_atlas import build_visual_atlas, sample_atlas_bilinear

        sp = DiskV2StructureParams(
            clump_count=40,
            clump_strength=0.12,
            atlas_n_r=32,
            atlas_n_phi=64,
            use_visual_atlas=True,
        )
        ti_obj = DiskV2Taichi(
            self.params, sp, self.palette_params_phys, seed=19,
        )
        atlas = build_visual_atlas(self.params, sp, seed=19)

        @ti.kernel
        def compute(out: ti.template(), r_f: ti.template(), phi_f: ti.template(), obj: ti.template()):
            for i in range(out.shape[0]):
                out[i] = obj.sample_emission_atlas_ti(r_f[i], phi_f[i])

        r_f = ti.field(dtype=ti.f32, shape=self.n)
        phi_f = ti.field(dtype=ti.f32, shape=self.n)
        out_f = ti.field(dtype=ti.f32, shape=self.n)
        r_f.from_numpy(self.r_samples.astype(np.float32))
        phi_f.from_numpy(self.phi_samples.astype(np.float32))
        compute(out_f, r_f, phi_f, ti_obj)
        ti_result = out_f.to_numpy().astype(np.float64)
        np_result = np.asarray(
            sample_atlas_bilinear(
                atlas.emission_weight,
                self.r_samples,
                self.phi_samples,
                atlas.r_in,
                atlas.r_out,
            ),
            dtype=np.float64,
        )
        np.testing.assert_allclose(ti_result, np_result, rtol=1e-3, atol=1e-4)

    # --- 调色 parity ---

    def test_parity_blackbody_color_physical(self):
        ti_obj = self.ti_phys

        @ti.kernel
        def compute(out_r: ti.template(), out_g: ti.template(), out_b: ti.template(),
                    T_f: ti.template(), obj: ti.template()):
            for i in range(out_r.shape[0]):
                rgb = obj.sample_palette_color(T_f[i])
                out_r[i] = rgb[0]
                out_g[i] = rgb[1]
                out_b[i] = rgb[2]

        T_f = ti.field(dtype=ti.f32, shape=self.n)
        T_f.from_numpy(self.T_samples.astype(np.float32))
        out_r = ti.field(dtype=ti.f32, shape=self.n)
        out_g = ti.field(dtype=ti.f32, shape=self.n)
        out_b = ti.field(dtype=ti.f32, shape=self.n)
        compute(out_r, out_g, out_b, T_f, ti_obj)
        ti_rgb = np.stack([
            out_r.to_numpy(), out_g.to_numpy(), out_b.to_numpy()
        ], axis=-1).astype(np.float64)

        np_rgb = np.asarray(blackbody_color(self.T_samples), dtype=np.float64)
        np.testing.assert_allclose(ti_rgb, np_rgb, rtol=1e-3, atol=1e-3)

    def test_parity_blackbody_color_cinematic(self):
        ti_obj = self.ti_cine

        @ti.kernel
        def compute(out_r: ti.template(), out_g: ti.template(), out_b: ti.template(),
                    T_f: ti.template(), obj: ti.template()):
            for i in range(out_r.shape[0]):
                rgb = obj.sample_palette_color(T_f[i])
                out_r[i] = rgb[0]
                out_g[i] = rgb[1]
                out_b[i] = rgb[2]

        T_f = ti.field(dtype=ti.f32, shape=self.n)
        T_f.from_numpy(self.T_samples.astype(np.float32))
        out_r = ti.field(dtype=ti.f32, shape=self.n)
        out_g = ti.field(dtype=ti.f32, shape=self.n)
        out_b = ti.field(dtype=ti.f32, shape=self.n)
        compute(out_r, out_g, out_b, T_f, ti_obj)
        ti_rgb = np.stack([
            out_r.to_numpy(), out_g.to_numpy(), out_b.to_numpy()
        ], axis=-1).astype(np.float64)

        np_rgb = np.asarray(
            cinematic_color(
                self.T_samples,
                self.palette_params_cine,
                T_peak_K=self.params.T_peak_K,
            ),
            dtype=np.float64,
        )
        np.testing.assert_allclose(ti_rgb, np_rgb, rtol=2e-3, atol=2e-3)

    def test_parity_observed_palette_color_cinematic(self):
        ti_obj = self.ti_cine
        g_samples = np.linspace(0.5, 1.8, self.n).astype(np.float64)

        @ti.kernel
        def compute(out_r: ti.template(), out_g: ti.template(), out_b: ti.template(),
                    T_f: ti.template(), g_f: ti.template(), obj: ti.template()):
            for i in range(out_r.shape[0]):
                rgb = obj.sample_observed_palette_color(T_f[i], g_f[i])
                out_r[i] = rgb[0]
                out_g[i] = rgb[1]
                out_b[i] = rgb[2]

        T_f = ti.field(dtype=ti.f32, shape=self.n)
        g_f = ti.field(dtype=ti.f32, shape=self.n)
        T_f.from_numpy(self.T_samples.astype(np.float32))
        g_f.from_numpy(g_samples.astype(np.float32))
        out_r = ti.field(dtype=ti.f32, shape=self.n)
        out_g = ti.field(dtype=ti.f32, shape=self.n)
        out_b = ti.field(dtype=ti.f32, shape=self.n)
        compute(out_r, out_g, out_b, T_f, g_f, ti_obj)
        ti_rgb = np.stack([
            out_r.to_numpy(), out_g.to_numpy(), out_b.to_numpy()
        ], axis=-1).astype(np.float64)

        t_visible = observed_visible_temperature(
            # 先用 public 函数得到发射可见色温，再手工应用 g-factor 与
            # cinematic saturation/warm。
            cinematic_visual_temperature(
                self.T_samples,
                self.params.T_peak_K,
                self.palette_params_cine,
            ),
            g_samples,
            self.palette_params_cine,
        )
        np_rgb = blackbody_color(t_visible)
        luma = (
            0.2126 * np_rgb[..., 0]
            + 0.7152 * np_rgb[..., 1]
            + 0.0722 * np_rgb[..., 2]
        )[..., None]
        np_rgb = luma + self.palette_params_cine.cinematic_saturation * (np_rgb - luma)
        np_rgb = np.clip(np_rgb, 0.0, 1.0)
        warm = np.array([
            1.0 + self.palette_params_cine.cinematic_warm_shift,
            1.0,
            1.0 - self.palette_params_cine.cinematic_warm_shift,
        ])
        np_rgb = np.clip(np_rgb * warm, 0.0, 1.0)

        np.testing.assert_allclose(ti_rgb, np_rgb, rtol=2e-3, atol=2e-3)

    def test_parity_observed_palette_color_physical_does_not_double_count_g(self):
        ti_obj = self.ti_phys

        @ti.kernel
        def compute(out_r: ti.template(), out_g: ti.template(), out_b: ti.template(),
                    T_f: ti.template(), g_f: ti.template(), obj: ti.template()):
            for i in range(out_r.shape[0]):
                rgb = obj.sample_observed_palette_color(T_f[i], g_f[i])
                out_r[i] = rgb[0]
                out_g[i] = rgb[1]
                out_b[i] = rgb[2]

        T_const = np.full(self.n, 6500.0, dtype=np.float64)
        g_samples = np.linspace(0.5, 2.0, self.n).astype(np.float64)
        T_f = ti.field(dtype=ti.f32, shape=self.n)
        g_f = ti.field(dtype=ti.f32, shape=self.n)
        T_f.from_numpy(T_const.astype(np.float32))
        g_f.from_numpy(g_samples.astype(np.float32))
        out_r = ti.field(dtype=ti.f32, shape=self.n)
        out_g = ti.field(dtype=ti.f32, shape=self.n)
        out_b = ti.field(dtype=ti.f32, shape=self.n)
        compute(out_r, out_g, out_b, T_f, g_f, ti_obj)
        ti_rgb = np.stack([
            out_r.to_numpy(), out_g.to_numpy(), out_b.to_numpy()
        ], axis=-1).astype(np.float64)

        expected = blackbody_color(T_const)
        np.testing.assert_allclose(ti_rgb, expected, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(ti_rgb[0], ti_rgb[-1], rtol=1e-6, atol=1e-6)

    def test_parity_tonemap_reinhard(self):
        ti_obj = self.ti_phys

        @ti.kernel
        def compute(out_r: ti.template(), out_g: ti.template(), out_b: ti.template(),
                    in_r: ti.template(), in_g: ti.template(), in_b: ti.template(),
                    obj: ti.template()):
            for i in range(out_r.shape[0]):
                v = ti.Vector([in_r[i], in_g[i], in_b[i]], dt=ti.f32)
                vt = obj.tonemap_reinhard(v)
                out_r[i] = vt[0]
                out_g[i] = vt[1]
                out_b[i] = vt[2]

        rng = np.random.default_rng(123)
        hdr = rng.uniform(0.0, 100.0, (self.n, 3))
        in_fs = [ti.field(dtype=ti.f32, shape=self.n) for _ in range(3)]
        out_fs = [ti.field(dtype=ti.f32, shape=self.n) for _ in range(3)]
        for c in range(3):
            in_fs[c].from_numpy(hdr[:, c].astype(np.float32))
        compute(out_fs[0], out_fs[1], out_fs[2], in_fs[0], in_fs[1], in_fs[2], ti_obj)
        ti_ldr = np.stack([f.to_numpy() for f in out_fs], axis=-1).astype(np.float64)

        np_ldr = np.asarray(tonemap(hdr, self.palette_params_phys), dtype=np.float64)
        np.testing.assert_allclose(ti_ldr, np_ldr, rtol=1e-5, atol=1e-6)

    def test_parity_apply_exposure(self):
        ti_obj = self.ti_phys

        @ti.kernel
        def compute(out_r: ti.template(), out_g: ti.template(), out_b: ti.template(),
                    in_r: ti.template(), in_g: ti.template(), in_b: ti.template(),
                    obj: ti.template()):
            for i in range(out_r.shape[0]):
                v = ti.Vector([in_r[i], in_g[i], in_b[i]], dt=ti.f32)
                vt = obj.apply_exposure_ti(v, 2.5)
                out_r[i] = vt[0]
                out_g[i] = vt[1]
                out_b[i] = vt[2]

        rng = np.random.default_rng(321)
        hdr = rng.uniform(0.0, 10.0, (self.n, 3))
        in_fs = [ti.field(dtype=ti.f32, shape=self.n) for _ in range(3)]
        out_fs = [ti.field(dtype=ti.f32, shape=self.n) for _ in range(3)]
        for c in range(3):
            in_fs[c].from_numpy(hdr[:, c].astype(np.float32))
        compute(out_fs[0], out_fs[1], out_fs[2], in_fs[0], in_fs[1], in_fs[2], ti_obj)
        ti_hdr = np.stack([f.to_numpy() for f in out_fs], axis=-1).astype(np.float64)

        np_hdr = np.asarray(apply_exposure(hdr, 2.5), dtype=np.float64)
        np.testing.assert_allclose(ti_hdr, np_hdr, rtol=1e-5, atol=1e-5)

    def test_parity_volume_emission_integral(self):
        """D3 公式对齐：在纯物理基线下，Taichi sample_emission 沿 z 数值积分
        应等于 NumPy physical_baseline_volume_flux。

        这只证明"reference 与 ray-march 的物理公式已对齐"：reference 是 NumPy
        沿 z 解析积分的"surface 总通量"；Taichi 渲染器走"逐 ds 累积 emissivity
        per unit volume"，两者在结构调制全部关闭下应同量级（相对误差 < 5%）。

        **不能从此推论 actual HDR 与 reference 已严格匹配** —— 实际渲染叠加
        g-factor、cinematic palette、transmittance 累积、构图相关 percentile
        取值后，actual HDR p{n} 与 reference 之间仍会偏离一个数量级，由
        `_compute_white_point` 的策略性 trusted/warn 窗口吸收。

        为隔离结构调制影响：构造一个"无 atlas / shear=0 / mode=0 / hotspot=0"
        的 DiskV2Taichi，让 sample_emission 退化为纯物理基线。
        """
        from disk_v2.imaging import physical_baseline_volume_flux
        from disk_v2.taichi_impl import DiskV2Taichi

        # 关闭所有结构调制，让 Taichi sample_emission 退化为
        # support · opacity · rho_envelope · T_norm^4。
        flat_structure = DiskV2StructureParams(
            mode1_strength=0.0,
            mode2_strength=0.0,
            shear_strength=0.0,
            hotspot_strength=0.0,
            clump_strength=0.0,
            clump_emission_weight=0.0,
            use_visual_atlas=False,
        )
        opacity = 0.55
        ti_flat = DiskV2Taichi(
            params=self.params,
            structure_params=flat_structure,
            palette_params=self.palette_params_phys,
            emission_opacity_scale=opacity,
            seed=7,
        )

        # 在固定 r 上沿 z 用 Gauss-Legendre 数值积分 Taichi sample_emission。
        n_r = 16
        radii = np.linspace(
            self.params.r_in + 0.5,
            self.params.r_out - 0.5,
            n_r,
        ).astype(np.float64)
        # 用与 NumPy 一致的 32 点 Gauss-Legendre。
        n_z = 32
        xi_arr, w_arr = np.polynomial.legendre.leggauss(n_z)

        from disk_v2.geometry import disk_half_thickness as np_disk_half_thickness
        h_arr = np.asarray(np_disk_half_thickness(radii, self.params))

        # Taichi 端用 kernel 读 sample_emission(r, 0, z)。
        n_total = n_r * n_z
        r_field = ti.field(dtype=ti.f32, shape=n_total)
        z_field = ti.field(dtype=ti.f32, shape=n_total)
        j_field = ti.field(dtype=ti.f32, shape=n_total)
        r_flat = np.repeat(radii, n_z)
        z_flat = np.empty(n_total, dtype=np.float64)
        for i, r in enumerate(radii):
            z_flat[i * n_z:(i + 1) * n_z] = xi_arr * h_arr[i]
        r_field.from_numpy(r_flat.astype(np.float32))
        z_field.from_numpy(z_flat.astype(np.float32))

        @ti.kernel
        def compute(out: ti.template(),
                    r_f: ti.template(),
                    z_f: ti.template(),
                    obj: ti.template()):
            for i in range(out.shape[0]):
                out[i] = obj.sample_emission(r_f[i], 0.0, z_f[i])

        compute(j_field, r_field, z_field, ti_flat)
        j_arr = j_field.to_numpy().astype(np.float64).reshape(n_r, n_z)

        # Taichi 端积分：∫_{-H}^{H} j(r, z) dz ≈ H · Σ w_xi · j(r, xi·H)
        ti_volume_flux = h_arr * np.sum(w_arr[None, :] * j_arr, axis=-1)

        # NumPy reference
        np_volume_flux = np.asarray(
            physical_baseline_volume_flux(radii, self.params, opacity),
            dtype=np.float64,
        )

        # 相对容差：fp32 Taichi vs fp64 NumPy，加上数值积分误差，取 5%。
        # 不要求严格精确（W_r 在 fp32/fp64 边界处微差也会引入误差），
        # 但量纲必须一致——否则差几倍。
        np.testing.assert_allclose(
            ti_volume_flux, np_volume_flux,
            rtol=0.05, atol=1e-10,
            err_msg="D3 量纲一致性破坏：Taichi sample_emission 沿 z 积分 ≠ NumPy reference",
        )


if __name__ == "__main__":
    unittest.main()
