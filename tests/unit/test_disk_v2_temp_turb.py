#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小尺度温度湍流（`src/v2/temperature_turbulence.py`、`DiskV2Taichi._temp_turb_*`）单元测试。

覆盖（docs/plans/v2_temperature_turbulence_plan.md §7）：
- 几何：延伸八度频率、长宽比、径向格宽含 `u_scale`（第 1 项）。
- 钳制与透镜权重的端点、单调性与 L = 0（第 2 项）。
- 温度倍率：σ_T = 0 或 V = 0 时为 1；高斯样本下 ⟨f_T⁴⟩ ≈ 1（第 3 项）。
- 参数校验（第 8 项）。
- `*_ti` 一致性：单图案与透镜权重的 Taichi 实现与 NumPy 参考一致（优化级别 0 / 1，第 4 项）；
  `density_I` 级别 0 与 1 一致（第 5 项）；归一化后方差约为 1 / V、真实噪声下 ⟨f_T⁴⟩ ≈ 1（第 6 项）；
  L = 0 时 `density_I` 与关闭时逐位相同（第 7 项）。

渲染核中的步长约束、像素足迹（λ 取段起点 / 中点、θ = ss·像素高度）与关闭时的编译期移除由 §7 第 10–12 项
的端到端验收覆盖（4090，结果见方案 §8）。
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import taichi as ti

from src.v2.params import DiskV2Params, DiskV2VolumeParams
from src.v2.temperature_turbulence import (
    LN_R_ANCHOR,
    smoothstep01,
    temp_turb_clamp_weights,
    temp_turb_gains,
    temp_turb_geometry,
    temp_turb_lens_weight,
    temp_turb_pattern_np,
    temp_turb_temperature,
    temp_turb_variance,
)
from tests.unit.test_disk_v2_noise_ti import _vnoise

_vnoise_vec = np.vectorize(_vnoise, otypes=[np.float64])

# 与默认 DiskV2VolumeParams 一致的主云剪切级联参数
_K0, _NOCT, _AR_SMALL, _TILT = 6.67, 4, 2.5, 0.01


class GeometryTest(unittest.TestCase):
    """延伸八度接在主云最细八度之后，长宽比统一取 `shear_ar_small`。"""

    def test_frequencies_follow_main_cascade(self):
        g = temp_turb_geometry(_K0, _NOCT, _AR_SMALL, _TILT, 2, -1.0)
        np.testing.assert_allclose(g.kr, [_K0 * 3 ** 4, _K0 * 3 ** 5])
        np.testing.assert_allclose(g.aspect, [_AR_SMALL, _AR_SMALL])

    def test_octave_count_bounds(self):
        for n in (0, 3):
            with self.assertRaises(ValueError):
                temp_turb_geometry(_K0, _NOCT, _AR_SMALL, _TILT, n, 1.0)

    def test_gains(self):
        self.assertEqual(temp_turb_gains(2, 0.69), (1.0, 0.69))
        for n, g in ((0, 0.69), (3, 0.69), (2, 0.0), (2, 1.5)):
            with self.assertRaises(ValueError):
                temp_turb_gains(n, g)


class WeightTest(unittest.TestCase):
    """钳制权重与透镜权重的端点、单调性。"""

    def setUp(self):
        self.g = temp_turb_geometry(_K0, _NOCT, _AR_SMALL, _TILT, 2, 1.0)

    def test_clamp_endpoints_use_u_scale(self):
        k = 3.0
        for e in range(2):
            cell_per_r = 1.0 / (self.g.u_scale[e] * self.g.kr[e])
            r = 10.0
            cell = r * cell_per_r
            w_lo = temp_turb_clamp_weights(r, cell / k, 1.0, self.g, k)[..., e]
            w_hi = temp_turb_clamp_weights(r, cell / (2.0 * k), 1.0, self.g, k)[..., e]
            self.assertAlmostEqual(float(w_lo), 0.0, places=12)
            self.assertAlmostEqual(float(w_hi), 1.0, places=12)

    def test_clamp_monotone_in_footprint(self):
        foot = np.geomspace(1e-5, 1e-1, 200)
        w = temp_turb_clamp_weights(10.0, foot, 1.0, self.g, 3.0)
        self.assertEqual(w.shape, (200, 2))
        self.assertTrue(np.all(np.diff(w, axis=0) <= 1e-15))
        self.assertTrue(np.all((w >= 0.0) & (w <= 1.0)))
        # 细八度先淡出
        self.assertTrue(np.all(w[:, 1] <= w[:, 0] + 1e-15))

    def test_lens_weight_zero_kills_all_octaves(self):
        w = temp_turb_clamp_weights(10.0, 1e-9, 0.0, self.g, 3.0)
        np.testing.assert_array_equal(w, np.zeros(2))

    def test_lens_weight_endpoints_and_monotone(self):
        d0 = math.radians(10.0)
        self.assertEqual(float(temp_turb_lens_weight(0.0, d0)), 1.0)
        self.assertEqual(float(temp_turb_lens_weight(0.5 * d0, d0)), 1.0)
        self.assertEqual(float(temp_turb_lens_weight(d0, d0)), 0.0)
        self.assertEqual(float(temp_turb_lens_weight(math.pi, d0)), 0.0)
        d = np.linspace(0.0, 2.0 * d0, 300)
        self.assertTrue(np.all(np.diff(temp_turb_lens_weight(d, d0)) <= 0.0))

    def test_smoothstep(self):
        np.testing.assert_allclose(smoothstep01(np.array([-1.0, 0.0, 0.5, 1.0, 2.0])), [0.0, 0.0, 0.5, 1.0, 1.0])


class TemperatureFactorTest(unittest.TestCase):
    """温度倍率 f_T = exp(σn − 2σ²V)。"""

    def test_identity_cases(self):
        n = np.linspace(-3.0, 3.0, 7)
        np.testing.assert_array_equal(temp_turb_temperature(0.0, n, 1.0), np.ones(7))
        np.testing.assert_array_equal(temp_turb_temperature(0.1, 0.0, 0.0), 1.0)

    def test_mean_flux_conserved_for_gaussian(self):
        rng = np.random.default_rng(3)
        for var in (1.0, 0.4):
            n = rng.normal(0.0, math.sqrt(var), 400_000)
            f4 = temp_turb_temperature(0.1, n, var) ** 4
            self.assertAlmostEqual(float(f4.mean()), 1.0, delta=2e-3)

    def test_variance(self):
        a = temp_turb_gains(2, 0.69)
        self.assertAlmostEqual(float(temp_turb_variance(np.ones(2), a)), 1.0)
        self.assertEqual(float(temp_turb_variance(np.zeros(2), a)), 0.0)
        self.assertAlmostEqual(float(temp_turb_variance(np.array([1.0, 0.0]), a)), 1.0 / (1.0 + 0.69 ** 2))


class ParamsValidationTest(unittest.TestCase):
    """`DiskV2VolumeParams` 温度湍流字段越界报错。"""

    def test_defaults_off(self):
        self.assertEqual(DiskV2VolumeParams().temp_turb_sigma, 0.0)

    def test_invalid(self):
        for kw in ({"temp_turb_sigma": -0.01}, {"temp_turb_sigma": 0.51}, {"temp_turb_octaves": 0},
                   {"temp_turb_octaves": 3}, {"temp_turb_gain": 0.0}, {"temp_turb_gain": 1.01},
                   {"temp_turb_clamp_px": 0.0}, {"temp_turb_lens_deg": 0.0}, {"temp_turb_lens_deg": 91.0}):
            with self.assertRaises(ValueError, msg=str(kw)):
                DiskV2VolumeParams(**kw)


class TaichiParityTest(unittest.TestCase):
    """Taichi 实现与 NumPy 参考一致；`density_I` 级别 0 / 1 一致、L = 0 时与关闭逐位相同。

    构造 3 个 `DiskV2Taichi`（开启 × 级别 0 / 1、关闭 × 级别 1），每个含 κ 标定（CPU 上约 40 s）。
    """

    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        from src.v2.taichi_impl import DiskV2Taichi
        cls.params = DiskV2Params(3.0, 30.0, disk_spin=-1.0)
        cls.vp_on = DiskV2VolumeParams(temp_turb_sigma=0.1)
        cls.on = {opt: DiskV2Taichi(cls.params, cls.vp_on, opt_level=opt) for opt in (0, 1)}
        cls.off = DiskV2Taichi(cls.params, DiskV2VolumeParams(), opt_level=1)
        cls.geom = temp_turb_geometry(_K0, _NOCT, _AR_SMALL, _TILT, 2, -1.0)
        cls.gains = temp_turb_gains(2, 0.69)
        rng = np.random.default_rng(17)
        cls.n = 128
        cls.r = rng.uniform(5.0, 28.0, cls.n)
        cls.phi = rng.uniform(-math.pi, math.pi, cls.n)
        cls.zr = rng.uniform(-0.01, 0.01, cls.n)
        # 足迹跨越完全可见 → 完全淡出；透镜权重含 0、1 与中间值
        cls.foot = 10.0 ** rng.uniform(-6.0, -1.5, cls.n)
        cls.lens = rng.choice([0.0, 0.3, 1.0], cls.n)

    @staticmethod
    def _fields(*arrays):
        out = []
        for a in arrays:
            f = ti.field(ti.f32, shape=len(a))
            f.from_numpy(np.asarray(a, dtype=np.float32))
            out.append(f)
        return out

    @staticmethod
    def _f32(a):
        return np.asarray(a, dtype=np.float32).astype(np.float64)

    def test_pattern_matches_numpy(self):
        f_r, f_p, f_z, f_f, f_l = self._fields(self.r, self.phi, self.zr, self.foot, self.lens)
        r, phi, zr, foot, lens = (self._f32(a) for a in (self.r, self.phi, self.zr, self.foot, self.lens))
        w = temp_turb_clamp_weights(r, foot, lens, self.geom, 3.0)
        ref = temp_turb_pattern_np(np.log(r), phi, zr, 3.7, 11.2, self.geom, self.gains, w, -1.0, _vnoise_vec)
        ref_v = temp_turb_variance(w, self.gains)
        for opt, disk in self.on.items():
            out = ti.field(ti.f32, shape=self.n)
            out_v = ti.field(ti.f32, shape=self.n)

            @ti.kernel
            def k():
                for i in range(self.n):
                    out[i] = disk._temp_turb_pattern(ti.log(f_r[i]), f_r[i], f_p[i], f_z[i], 3.7, 11.2,
                                                     f_f[i], f_l[i])
                    out_v[i] = disk._temp_turb_variance(f_r[i], f_f[i], f_l[i])

            k()
            # 噪声坐标量级 10³，f32 坐标舍入使值噪声相差约 1e-3
            np.testing.assert_allclose(out.to_numpy(), ref, rtol=0, atol=5e-3, err_msg=f"opt_level={opt}")
            np.testing.assert_allclose(out_v.to_numpy(), ref_v, rtol=1e-5, atol=1e-6, err_msg=f"opt_level={opt}")

    def test_lens_weight_matches_numpy(self):
        disk = self.on[1]
        d = np.linspace(0.0, math.radians(20.0), 64)
        (f_c,) = self._fields(np.cos(d))
        out = ti.field(ti.f32, shape=64)

        @ti.kernel
        def k():
            for i in range(64):
                out[i] = disk._temp_turb_lens_weight(f_c[i])

        k()
        ref = temp_turb_lens_weight(np.arccos(self._f32(np.cos(d))), math.radians(10.0))
        # acos 在 δ → 0 处对 cos 的 f32 舍入敏感；L 在该处恒为 1，误差只出现在过渡区
        np.testing.assert_allclose(out.to_numpy(), ref, rtol=0, atol=2e-3)

    def _density(self, disk, lens):
        f_r, f_p, f_f, f_l = self._fields(self.r, self.phi, self.foot, lens)
        (f_z,) = self._fields(self.zr * self.r * 0.2)
        out = ti.Vector.field(6, ti.f32, shape=self.n)

        @ti.kernel
        def k():
            for i in range(self.n):
                a0, a1, a2, a3, a4, a5 = disk.density_I(f_r[i], f_z[i], f_p[i], 0.0, f_f[i], f_l[i])
                out[i] = ti.Vector([a0, a1, a2, a3, a4, a5])

        k()
        return out.to_numpy()

    def test_density_opt_levels_agree(self):
        a = self._density(self.on[0], self.lens)
        b = self._density(self.on[1], self.lens)
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=1e-7)

    def test_density_lens_zero_equals_off(self):
        zero = np.zeros(self.n)
        np.testing.assert_array_equal(self._density(self.on[1], zero), self._density(self.off, zero))

    def test_temperature_actually_modulated(self):
        """开启且八度可见时 tf_c 偏离关闭值（防止实现被意外静态移除）。"""
        ones = np.ones(self.n)
        on = self._density(self.on[1], ones)
        off = self._density(self.off, ones)
        visible = self.foot < 1e-4
        self.assertGreater(np.abs(on[visible, 1] / off[visible, 1] - 1.0).max(), 0.02)
        # 只改温度：其余五个量逐位相同
        np.testing.assert_array_equal(np.delete(on, 1, axis=1), np.delete(off, 1, axis=1))

    def test_normalized_variance_and_flux(self):
        """混合后 n_T 的方差：全部 w_e = 1 时约为 1，只保留最粗八度时约为 V；真实噪声下 ⟨f_T⁴⟩ ≈ 1。"""
        disk = self.on[1]
        m = 20000
        rng = np.random.default_rng(5)
        r = rng.uniform(6.0, 20.0, m)
        phi = rng.uniform(-math.pi, math.pi, m)
        zr = rng.uniform(-0.01, 0.01, m)
        # 只保留最粗八度的足迹：c_0/F = 2K·1.5（可见），c_1/F = c_0/(3F) < K（淡出）
        cell0 = r / (self.geom.u_scale[0] * self.geom.kr[0])
        foot_coarse = cell0 / 9.0
        f_r, f_p, f_z, f_c = self._fields(r, phi, zr, foot_coarse)
        out_full = ti.field(ti.f32, shape=m)
        out_coarse = ti.field(ti.f32, shape=m)
        out_v = ti.field(ti.f32, shape=m)

        @ti.kernel
        def k():
            for i in range(m):
                lnr = ti.log(f_r[i])
                w, ph, ox, oz, nb = disk._band_info(lnr, f_p[i], 0.0)
                out_full[i] = disk._temp_turb_shared(lnr, f_r[i], f_z[i], w, ph, ox, oz, 1e-9, 1.0)
                out_coarse[i] = disk._temp_turb_shared(lnr, f_r[i], f_z[i], w, ph, ox, oz, f_c[i], 1.0)
                out_v[i] = disk._temp_turb_variance(f_r[i], f_c[i], 1.0)

        k()
        full = out_full.to_numpy().astype(np.float64)
        coarse = out_coarse.to_numpy().astype(np.float64)
        v = out_v.to_numpy().astype(np.float64)
        np.testing.assert_allclose(v, 1.0 / (1.0 + 0.69 ** 2), rtol=1e-5)
        self.assertAlmostEqual(float(full.mean()), 0.0, delta=0.05)
        self.assertAlmostEqual(float(full.var()), 1.0, delta=0.1)
        self.assertAlmostEqual(float(coarse.var()), float(v.mean()), delta=0.1)
        f4 = temp_turb_temperature(0.1, full, 1.0) ** 4
        self.assertAlmostEqual(float(f4.mean()), 1.0, delta=0.02)
        # 部分八度可见时按实际 V 补偿，平均通量同样守恒
        f4c = temp_turb_temperature(0.1, coarse, v) ** 4
        self.assertAlmostEqual(float(f4c.mean()), 1.0, delta=0.02)

    def test_anchor_constant(self):
        self.assertAlmostEqual(LN_R_ANCHOR, math.log(20.0))


if __name__ == "__main__":
    unittest.main()
