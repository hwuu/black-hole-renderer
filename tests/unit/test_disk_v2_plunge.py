#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 内边界力矩 β 与坠落区（ISCO 以内）单元测试。

覆盖：
- 参数：`isco_stress` ∈ [0, 1)、`plunge_width` ∈ (0, 0.3]，默认值不改变现有输出。
- Page–Thorne 通量：显式 β = 0 与默认值逐位一致；β > 0 时内缘通量非零、处处不减。
- 内边界因子 f_β = 1 − (1 − β)·√(r_in/r)：β = 0 为零力矩，r = r_in 处等于 β。
- 结构场：β = 0 时 SS 标高 / 柱密度与改动前的公式逐位一致；β > 0 时柱密度在 r_in 处连续、坠落区向内下降
  （ISCO 内侧 < 0.002 r_s 内最多高出 0.02%，其余单调下降），`r = r_in − Δr` 处满足 u_pl = v_I
  （Σ/Σ_I = r_in/(2r)）；坠落区标高 = H(r_in)。
- 坠落区温度倍率 = (Σ/Σ_I)^{2/3}（绝热，标高不变）。
- 温度表：β = 0 与改动前逐位一致；β > 0 时按 β = 0 的峰值归一（额外加热）。
- CLI：`--v2_isco_stress` / `--v2_plunge_width` 只在显式传入时覆盖。
- Taichi 实现（f32）与 NumPy 参考（f64）一致，相对误差 ≤ 2e-5；β > 0 且 r_in ≠ ISCO 时构造报错。
- 渲染输出与改动前逐位一致（β = 0）不在单测里验证，由多视角 HDR 对比渲染验收（见 `docs/design_ad_v2.md` §3.2）。
"""

import math
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

from src.v2.params import DiskV2Params, DiskV2VolumeParams
from src.v2 import physical_fields as pf

R_IN, R_OUT, R_REF, HR = 3.0, 30.0, 10.0, 0.027


class _FakeLut:
    """模拟 `ti.field.from_numpy`：记录写入的数组。"""

    def __init__(self):
        self.arr = None

    def from_numpy(self, a):
        self.arr = np.array(a)


class ParamsTest(unittest.TestCase):
    def test_defaults_keep_zero_torque(self):
        """默认 β = 0（零力矩、无坠落区气体），Δr = 0.1。"""
        vp = DiskV2VolumeParams()
        self.assertEqual((vp.isco_stress, vp.plunge_width), (0.0, 0.1))

    def test_invalid_values_rejected(self):
        for kw in ({"isco_stress": -0.01}, {"isco_stress": 1.0}, {"plunge_width": 0.0}, {"plunge_width": 0.31},
                   {"isco_stress": math.nan}, {"plunge_width": math.nan}, {"temp_turb_lens_deg": math.nan}):
            with self.assertRaises(ValueError, msg=str(kw)):
                DiskV2VolumeParams(**kw)


class FluxTest(unittest.TestCase):
    rs = np.linspace(R_IN, R_OUT, 4001)

    def test_zero_stress_identical(self):
        np.testing.assert_array_equal(pf.page_thorne_flux(self.rs), pf.page_thorne_flux(self.rs, isco_stress=0.0))

    def test_positive_stress_heats_inner_edge(self):
        f0 = pf.page_thorne_flux(self.rs)
        f1 = pf.page_thorne_flux(self.rs, isco_stress=0.005)
        self.assertEqual(f0[0], 0.0)
        self.assertGreater(f1[0], 0.0)
        self.assertTrue(np.all(f1 >= f0))
        self.assertLess(f1[-1] / f0[-1] - 1.0, 0.01)  # 外盘几乎不变

    def test_inner_temperature_matches_design(self):
        """β = 0.005 时内缘温度 ≈ 0.80·T_峰（β = 0），β = 0.01 时 ≈ 0.95·T_峰（方案小样标注值）。"""
        t0 = pf.page_thorne_flux(self.rs) ** 0.25
        for beta, want in ((0.005, 0.80), (0.01, 0.95)):
            t = pf.page_thorne_flux(self.rs, isco_stress=beta) ** 0.25
            self.assertAlmostEqual(t[0] / t0.max(), want, delta=0.01)


class BoundaryFactorTest(unittest.TestCase):
    def test_factor(self):
        r = np.array([3.0, 4.0, 10.0])
        np.testing.assert_allclose(pf.inner_boundary_factor(r, R_IN, 0.0),
                                   np.maximum(1.0 - np.sqrt(R_IN / r), 1e-6))  # r_in 处截断到 1e-6
        self.assertAlmostEqual(float(pf.inner_boundary_factor(R_IN, R_IN, 0.2)), 0.2)


class StructureTest(unittest.TestCase):
    r_disk = np.linspace(R_IN, R_OUT, 501)

    def test_zero_stress_matches_previous_formula(self):
        """β = 0：SS 标高 / 柱密度与改动前的公式逐位一致（含 r < r_in 的外推）。"""
        r = np.linspace(2.0, R_OUT, 701)
        fr = np.maximum(1.0 - np.sqrt(R_IN / np.maximum(r, R_IN)), 1e-6)
        f_ref = 1.0 - math.sqrt(R_IN / R_REF)
        outer = 1.0 - np.clip((r - 0.72 * R_OUT) / (R_OUT - 0.72 * R_OUT), 0, 1)
        outer = outer * outer * (3 - 2 * outer)
        np.testing.assert_array_equal(pf.ss_half_thickness(r, R_IN, HR, R_REF),
                                      HR * r * (r / R_REF) ** 0.125 * (fr / f_ref) ** 0.15)
        np.testing.assert_array_equal(pf.ss_surface_density(r, R_IN, R_OUT, R_REF),
                                      (r / R_REF) ** (-0.75) * (fr / f_ref) ** 0.7 * outer)

    def test_plunge_region(self):
        beta, dr = 0.005, 0.1
        sig = lambda r: pf.ss_surface_density(r, R_IN, R_OUT, R_REF, isco_stress=beta, plunge_width=dr)
        s_in = float(sig(R_IN))
        self.assertGreater(s_in, 0.0)
        self.assertAlmostEqual(float(sig(R_IN - 1e-6)) / s_in, 1.0, places=4)  # r_in 处连续
        r_pl = np.linspace(1.6, R_IN - 0.01, 200)
        self.assertTrue(np.all(np.diff(sig(r_pl)) > 0.0))  # 向内单调下降
        r_h = R_IN - dr
        self.assertAlmostEqual(float(sig(r_h)) / s_in, R_IN / (2.0 * r_h), places=6)
        h = lambda r: pf.ss_half_thickness(r, R_IN, HR, R_REF, isco_stress=beta)
        self.assertEqual(float(h(2.5)), float(h(R_IN)))

    def test_plunge_ratio_and_temperature(self):
        r = np.array([3.0, 2.95, 2.8, 2.0])
        p = pf.plunge_surface_density_ratio(r, R_IN, 0.1)
        self.assertEqual(float(p[0]), 1.0)
        self.assertTrue(np.all(np.diff(p) < 0.0))
        np.testing.assert_allclose(pf.plunge_temperature(r, R_IN, 0.1), p ** (2.0 / 3.0))
        np.testing.assert_array_equal(pf.plunge_surface_density_ratio(np.array([3.5, 10.0]), R_IN, 0.1), 1.0)

    def test_plunge_ratio_near_isco_bounded(self):
        """Δr 取边界值时：ISCO 内侧微小增密 ≤ 0.03%，0.01 r_s 以内即单调下降；光子球处 ≤ 7.2%。"""
        for dr in (0.05, 0.1, 0.3):
            r = np.linspace(1.5, R_IN, 300001)
            p = pf.plunge_surface_density_ratio(r, R_IN, dr)
            self.assertLessEqual(float(p.max()), 1.0003, msg=str(dr))
            inner = r < R_IN - 0.01
            self.assertTrue(np.all(np.diff(p[inner]) > 0.0), msg=str(dr))
            self.assertLessEqual(float(p[0]), 0.072, msg=str(dr))

    def test_isco_velocity(self):
        """v_I = √(1/(3 r_in))·(r_in/(r_in − Δr) − 1)^{3/2}：Δr = 0.1 → ≈ 0.002 c，0.3 → ≈ 0.012 c。"""
        self.assertAlmostEqual(pf.plunge_isco_velocity(R_IN, 0.1), 0.002134, places=5)
        self.assertAlmostEqual(pf.plunge_isco_velocity(R_IN, 0.3), 0.012346, places=5)


class LutTest(unittest.TestCase):
    def test_zero_stress_bitwise(self):
        a, b = _FakeLut(), _FakeLut()
        pf.build_page_thorne_lut(a, R_IN, R_OUT, 512)
        pf.build_page_thorne_lut(b, R_IN, R_OUT, 512, isco_stress=0.0)
        rs = np.linspace(R_IN, R_OUT, 512)
        tt = pf.page_thorne_flux(rs) ** 0.25
        np.testing.assert_array_equal(a.arr, (tt / tt.max()).astype(np.float32))
        np.testing.assert_array_equal(a.arr, b.arr)

    def test_positive_stress_normalized_to_zero_stress_peak(self):
        a = _FakeLut()
        pf.build_page_thorne_lut(a, R_IN, R_OUT, 512, isco_stress=0.005)
        self.assertAlmostEqual(float(a.arr[0]), 0.80, delta=0.01)
        self.assertGreaterEqual(float(a.arr.max()), 1.0)


class CliTest(unittest.TestCase):
    def _args(self, *argv):
        from src import cli
        with patch.object(sys, "argv", ["render.py", "--disk_model", "v2", *argv]):
            return cli, cli.parse_args()

    def test_flags(self):
        cli, a = self._args()
        self.assertEqual(cli.v2_volume_overrides(a), {})
        cli, a = self._args("--v2_isco_stress", "0.005", "--v2_plunge_width", "0.2")
        vp = DiskV2VolumeParams(**cli.v2_volume_overrides(a))
        self.assertEqual((vp.isco_stress, vp.plunge_width), (0.005, 0.2))


class TaichiParityTest(unittest.TestCase):
    """`DiskV2Taichi` 的结构场 / 坠落区与 NumPy 参考一致（β > 0，f32 对 f64，相对误差 ≤ 2e-5）。"""

    @classmethod
    def setUpClass(cls):
        import taichi as ti
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        from src.v2.taichi_impl import DiskV2Taichi
        cls.ti = ti
        cls.vp = DiskV2VolumeParams(isco_stress=0.005, plunge_width=0.1)
        cls.disk = DiskV2Taichi(DiskV2Params(R_IN, R_OUT), cls.vp, opt_level=1)

    def test_structure_and_plunge(self):
        ti, disk, vp = self.ti, self.disk, self.vp
        rr = np.concatenate([np.linspace(1.6, 2.99, 40), np.linspace(3.0, 29.0, 60)])
        n = len(rr)
        f_r = ti.field(ti.f32, shape=n)
        out = ti.Vector.field(4, ti.f32, shape=n)
        f_r.from_numpy(rr.astype(np.float32))

        @ti.kernel
        def k():
            for i in range(n):
                r = f_r[i]
                out[i] = ti.Vector([disk._ss_surface_density(r), disk._ss_half_thickness(r),
                                    disk._plunge_ratio(r), disk._page_thorne_temperature(r) / disk._t_peak_vol])

        k()
        got = out.to_numpy().astype(np.float64)
        r32 = rr.astype(np.float32).astype(np.float64)
        hr = vp.hr_ref * vp.thickness_scale
        np.testing.assert_allclose(got[:, 0], pf.ss_surface_density(r32, R_IN, R_OUT, vp.r_ref, 0.005, 0.1), rtol=2e-5)
        np.testing.assert_allclose(got[:, 1], pf.ss_half_thickness(r32, R_IN, hr, vp.r_ref, 0.005), rtol=2e-5)
        np.testing.assert_allclose(got[:, 2], pf.plunge_surface_density_ratio(r32, R_IN, 0.1), rtol=2e-5)
        self.assertAlmostEqual(float(got[0, 3]), float(got[39, 3]))  # 坠落区温度表取 T(r_in)
        self.assertAlmostEqual(float(got[40, 3]), 0.80, delta=0.01)

    def test_requires_isco_inner_radius(self):
        """β > 0 时 r_in 必须等于 ISCO（3 r_s），否则构造报错。"""
        from src.v2.taichi_impl import DiskV2Taichi
        with self.assertRaises(ValueError):
            DiskV2Taichi(DiskV2Params(4.0, R_OUT), DiskV2VolumeParams(isco_stress=0.005), opt_level=1)

    def test_density_inside_isco(self):
        """β > 0 时 ISCO 以内的采样点有气体（β = 0 时 density_I 在 r ≤ r_in 处恒为 0）。"""
        ti, disk = self.ti, self.disk
        out = ti.Vector.field(2, ti.f32, shape=1)

        @ti.kernel
        def k():
            em_c, tf_c, ab_c, ab_a, em_a, sc_a = disk.density_I(2.95, 0.0, 0.3, 0.0, 1.0, 0.0)
            out[0] = ti.Vector([ab_c, tf_c])

        k()
        ab, tf = out.to_numpy()[0]
        self.assertGreater(float(ab), 0.0)
        self.assertGreater(float(tf), 0.0)


if __name__ == "__main__":
    unittest.main()
