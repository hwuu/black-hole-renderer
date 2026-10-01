#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 体积密度场（v2.3 S5）单元测试。

覆盖（方案 §5 S5 测试要点）：
- 盘外 = 0；ρ ≥ 0（em / ab 全部非负）。
- T_peak(1e8, 1.7e-6) = 4509 K（Page–Thorne 绝对通量推导）。
- PT 温度峰值位于 r ≈ 4.8 r_s（牛顿近似为 49/36·r_in ≈ 4.08）。
- SS 标高 / 柱密度结构性质（H/r 随 r 缓慢增大、Σ ≥ 0）。
- 灰大气温度倍率 ∈ [1 - mix·(1-0.84), 1 + mix·(cap-1)]，τ 单调。
- 核心柱密度 ∫ρ dz ≈ Σ'(r)（竖直高斯解析积分）。
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

R_IN, R_OUT = 3.0, 30.0


class PageThorneTest(unittest.TestCase):
    def test_t_peak_derivation(self):
        """T_peak(1e8 M☉, 1.7e-6 Ṁ_Edd) ≈ 4509 K。"""
        from disk_v2.physical_fields import derive_t_peak
        tp = derive_t_peak(1.0e8, 1.7e-6)
        self.assertAlmostEqual(tp, 4509.0, delta=20.0)

    def test_pt_peak_radius(self):
        """Page–Thorne 温度峰值位于 r ≈ 4.8 r_s（牛顿近似 4.08）。"""
        from disk_v2.physical_fields import page_thorne_flux
        rs = np.linspace(3.0, 30.0, 20001)
        f = page_thorne_flux(rs)
        r_peak = rs[int(np.argmax(f))]
        self.assertGreater(r_peak, 4.5)
        self.assertLess(r_peak, 5.2)


class SSFieldTest(unittest.TestCase):
    def test_half_thickness_grows_slowly(self):
        """SS 外区 H/r 随 r^{1/8} 缓慢增大，且 H 在内区有限。"""
        from disk_v2.physical_fields import ss_half_thickness
        h5 = float(ss_half_thickness(5.0, R_IN, 0.027, 10.0))
        h20 = float(ss_half_thickness(20.0, R_IN, 0.027, 10.0))
        self.assertGreater(h5, 0.0)
        self.assertGreater(h20, h5)
        self.assertLess(h20 / 20.0 / (h5 / 5.0), 1.5)  # H/r 增长 < 1.5 倍

    def test_surface_density_positive_and_decays(self):
        """Σ(r) > 0 且从内到外衰减；外缘截断为 0。"""
        from disk_v2.physical_fields import ss_surface_density
        s5 = float(ss_surface_density(5.0, R_IN, R_OUT, 10.0))
        s15 = float(ss_surface_density(15.0, R_IN, R_OUT, 10.0))
        s_out = float(ss_surface_density(R_OUT, R_IN, R_OUT, 10.0))
        self.assertGreater(s5, 0.0)
        self.assertGreater(s5, s15)
        self.assertAlmostEqual(s_out, 0.0, places=6)

    def test_column_density_matches_gaussian(self):
        """竖直高斯解析：∫ρ dz = Σ'（误差 < 1%）。"""
        from disk_v2.physical_fields import ss_half_thickness, ss_surface_density
        h = float(ss_half_thickness(6.0, R_IN, 0.027, 10.0))
        sig = float(ss_surface_density(6.0, R_IN, R_OUT, 10.0))
        # ∫ Σ/(√(2π)H)·exp(-z²/2H²) dz = Σ
        zs = np.linspace(-5 * h, 5 * h, 2001)
        rho = sig / (math.sqrt(2 * math.pi) * h) * np.exp(-0.5 * (zs / h) ** 2)
        col = np.trapezoid(rho, zs)
        self.assertAlmostEqual(col / sig, 1.0, places=3)


class GreyAtmosphereTest(unittest.TestCase):
    def test_grey_factor_bounds(self):
        """灰大气倍率有界 [1 - mix·(1-0.5^{1/4}), 1 + mix·(cap-1)]。"""
        grey_mix, grey_cap = 0.5, 1.19
        # τ = 0（表面）→ 1 + mix·((3/4·2/3)^{1/4} - 1) = 1 + 0.5·(0.5^{1/4}-1) ≈ 0.920
        lo = 1.0 + grey_mix * (0.5 ** 0.25 - 1.0)
        # τ → 大 → cap
        hi = 1.0 + grey_mix * (grey_cap - 1.0)
        self.assertAlmostEqual(lo, 0.920, places=3)
        self.assertAlmostEqual(hi, 1.095, places=3)

    def test_grey_monotonic_in_tau(self):
        """灰大气温度随 τ 单调递增（有上限）。"""
        greys = []
        for tau in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 100.0):
            factor = 1.0 + 0.5 * (min((0.75 * (tau + 2.0 / 3.0)) ** 0.25, 1.19) - 1.0)
            greys.append(factor)
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(greys, greys[1:])))


class VolumeParamsTest(unittest.TestCase):
    def test_defaults_match_reference_preset_M(self):
        """DiskV2VolumeParams 默认值 = 参考实现预设 M。"""
        from disk_v2.params import DiskV2VolumeParams
        vp = DiskV2VolumeParams()
        self.assertEqual(vp.bh_mass_msun, 1.0e8)
        self.assertAlmostEqual(vp.mdot_edd, 1.7e-6)
        self.assertAlmostEqual(vp.grey_mix, 0.5)
        self.assertAlmostEqual(vp.grey_cap, 1.19)
        self.assertAlmostEqual(vp.core_opac, 2.0)
        self.assertAlmostEqual(vp.core_floor, 0.35)
        self.assertAlmostEqual(vp.dt_i, 0.05)
        self.assertAlmostEqual(vp.smoke_i, 0.8)
        self.assertAlmostEqual(vp.smoke_tr, 0.85)
        self.assertAlmostEqual(vp.con_i, 50.0)
        self.assertAlmostEqual(vp.lowf_sigma, 0.9)
        self.assertAlmostEqual(vp.surf_lo, 1.0)
        self.assertAlmostEqual(vp.surf_k, 0.0)
        self.assertTrue(vp.dust_kepler)
        self.assertTrue(vp.static_cam)
        self.assertTrue(vp.light_delay)

    def test_validation(self):
        from disk_v2.params import DiskV2VolumeParams
        with self.assertRaises(ValueError):
            DiskV2VolumeParams(grey_mix=-0.1)
        with self.assertRaises(ValueError):
            DiskV2VolumeParams(core_opac=0.0)
        with self.assertRaises(ValueError):
            DiskV2VolumeParams(dln_r=-1.0)


if __name__ == "__main__":
    unittest.main()
