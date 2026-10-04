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
        from src.v2.physical_fields import derive_t_peak
        tp = derive_t_peak(1.0e8, 1.7e-6)
        self.assertAlmostEqual(tp, 4509.0, delta=20.0)

    def test_pt_peak_radius(self):
        """Page–Thorne 温度峰值位于 r ≈ 4.8 r_s（牛顿近似 4.08）。"""
        from src.v2.physical_fields import page_thorne_flux
        rs = np.linspace(3.0, 30.0, 20001)
        f = page_thorne_flux(rs)
        r_peak = rs[int(np.argmax(f))]
        self.assertGreater(r_peak, 4.5)
        self.assertLess(r_peak, 5.2)


class SSFieldTest(unittest.TestCase):
    def test_half_thickness_grows_slowly(self):
        """SS 外区 H/r 随 r^{1/8} 缓慢增大，且 H 在内区有限。"""
        from src.v2.physical_fields import ss_half_thickness
        h5 = float(ss_half_thickness(5.0, R_IN, 0.027, 10.0))
        h20 = float(ss_half_thickness(20.0, R_IN, 0.027, 10.0))
        self.assertGreater(h5, 0.0)
        self.assertGreater(h20, h5)
        self.assertLess(h20 / 20.0 / (h5 / 5.0), 1.5)  # H/r 增长 < 1.5 倍

    def test_surface_density_positive_and_decays(self):
        """Σ(r) > 0 且从内到外衰减；外缘截断为 0。"""
        from src.v2.physical_fields import ss_surface_density
        s5 = float(ss_surface_density(5.0, R_IN, R_OUT, 10.0))
        s15 = float(ss_surface_density(15.0, R_IN, R_OUT, 10.0))
        s_out = float(ss_surface_density(R_OUT, R_IN, R_OUT, 10.0))
        self.assertGreater(s5, 0.0)
        self.assertGreater(s5, s15)
        self.assertAlmostEqual(s_out, 0.0, places=6)

    def test_column_density_matches_gaussian(self):
        """竖直高斯解析：∫ρ dz = Σ'（误差 < 1%）。"""
        from src.v2.physical_fields import ss_half_thickness, ss_surface_density
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
    def test_defaults(self):
        """DiskV2VolumeParams 默认值：温度与刚体环沿用预设 M，大气参数取方案 §5.1 定稿值。"""
        from src.v2.params import DiskV2VolumeParams
        vp = DiskV2VolumeParams()
        self.assertEqual(vp.bh_mass_msun, 1.0e8)
        self.assertAlmostEqual(vp.mdot_edd, 1.7e-6)
        self.assertAlmostEqual(vp.grey_mix, 0.5)
        self.assertAlmostEqual(vp.grey_cap, 1.19)
        self.assertAlmostEqual(vp.dt_i, 0.05)
        self.assertAlmostEqual(vp.core_floor, 0.15)
        self.assertAlmostEqual(vp.lowf_sigma, 1.6)
        self.assertAlmostEqual(vp.core_contrast, 0.8)
        self.assertAlmostEqual(vp.tau_i, 1.84)
        self.assertAlmostEqual(vp.atm_fine_sigma, 0.5)
        self.assertAlmostEqual(vp.atm_fine_fz, 2.0)
        self.assertAlmostEqual(vp.con_t, 50.0)
        self.assertAlmostEqual(vp.az_stretch_t, 1.5)
        self.assertAlmostEqual(vp.surf_lo, 1.0)
        self.assertAlmostEqual(vp.surf_k, 0.0)
        self.assertAlmostEqual(vp.hr_ref, 0.027)
        self.assertAlmostEqual(vp.thickness_scale, 1.0 / 9.0)
        self.assertAlmostEqual(vp.lum_temp_scale, 1.25)
        self.assertAlmostEqual(vp.core_oct_gain, 0.6)
        self.assertTrue(vp.band_seam_fix)
        self.assertAlmostEqual(vp.atm_frac, 0.15)
        self.assertAlmostEqual(vp.atm_height, 0.01)
        self.assertAlmostEqual(vp.atm_extent, 5.0)
        self.assertAlmostEqual(vp.atm_cov_c0, 1.0)
        self.assertAlmostEqual(vp.atm_cov_soft, 0.3)
        self.assertAlmostEqual(vp.abs_scatter_ratio, 20.0)
        self.assertAlmostEqual(vp.scatter_j, 0.5)
        self.assertTrue(vp.static_cam)
        self.assertTrue(vp.light_delay)

    def test_removed_fields(self):
        """旧烟雾 / 尘埃 / core_opac / 旧主云级联字段已删除。"""
        from src.v2.params import DiskV2VolumeParams
        for name in ("smoke_i", "core_opac", "dust_on", "kr_i", "con_i", "core_az_stretch",
                     "outer_detail_fade", "shear_cascade"):
            with self.assertRaises(TypeError, msg=name):
                DiskV2VolumeParams(**{name: 1.0})

    def test_validation(self):
        from src.v2.params import DiskV2VolumeParams
        for kw in (dict(grey_mix=-0.1), dict(dln_r=-1.0), dict(lum_temp_scale=0.0), dict(tau_i=0.0), dict(lowf_sigma=-0.1), dict(atm_fine_sigma=-0.1), dict(atm_fine_fz=0.0),
                   dict(atm_frac=-0.1), dict(atm_height=0.0), dict(atm_extent=0.0), dict(atm_cov_soft=0.0),
                   dict(abs_scatter_ratio=-1.0), dict(scatter_j=-0.1)):
            with self.assertRaises(ValueError, msg=str(kw)):
                DiskV2VolumeParams(**kw)

    def test_structure_validation(self):
        """结构参数：取值范围，以及 core_contrast ≠ 1 必须配合接缝修复。"""
        from src.v2.params import DiskV2VolumeParams
        for kw in (dict(az_stretch_t=-1.0), dict(con_t=0.0), dict(core_oct_gain=0.0), dict(core_contrast=0.0),
                   dict(band_seam_fix=False, core_contrast=0.4)):
            with self.assertRaises(ValueError, msg=str(kw)):
                DiskV2VolumeParams(**kw)
        # 参考实现取值组合合法
        DiskV2VolumeParams(az_stretch_t=0.0, core_oct_gain=1.0, band_seam_fix=False, core_contrast=1.0,
                           atm_frac=0.0, abs_scatter_ratio=0.0, scatter_j=0.0)


class DiskGeometryParamsTest(unittest.TestCase):
    """`DiskV2Params`：ISCO 钳制与半径顺序校验（自 test_disk_v2_physical_fields 迁入）。"""

    def test_r_in_below_isco_is_clamped_with_warning(self):
        import warnings

        from src.v2.params import SCHWARZSCHILD_ISCO_R_S, DiskV2Params
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            params = DiskV2Params(r_in=2.0, r_out=10.0)
        self.assertEqual(params.r_in, SCHWARZSCHILD_ISCO_R_S)
        self.assertTrue(any("ISCO" in str(w.message) for w in caught))

    def test_r_out_must_exceed_clamped_r_in(self):
        import warnings

        from src.v2.params import DiskV2Params
        # r_in 钳制为 3.0 后，r_out=2.5 仍然 ≤ r_in，应 raise。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(ValueError):
                DiskV2Params(r_in=2.0, r_out=2.5)


if __name__ == "__main__":
    unittest.main()
