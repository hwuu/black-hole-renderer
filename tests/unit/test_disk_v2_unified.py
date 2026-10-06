#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一气体模型（`src/v2/gas_unified.py`、`DiskV2Taichi.density_I`）单元测试。

覆盖（docs/plans/v2_unified_gas_plan.md §8）：
- 竖直剖面归一化：数值积分 ∫(ab_c + ab_a) dz 与解析柱密度一致。
- ω(ρ)：值域 (0, 1]，随 ρ 单调减，ρ → 0 时 → 1。
- 散射入射光：散射权重 = ω·ab_a·(1 − e^{−τ_c})，随核心光学深度单调增，τ_c → ∞ 时回到 ω·ab_a。
- 温度连续：从中面到大气顶，温度倍率无跳变。
- `*_ti` 一致性：Taichi `density_I` 与 NumPy 参考 `unified_gas_profile` 逐点一致。
- 小尺度起伏：零均值、单位方差，保均值 ⟨exp(σ_a·n − σ_a²/2)⟩ = 1 ± 0.02。
- κ 标定：r ∈ [5.5, 6.5] 的平均 τ⊥（细网格独立积分）= `tau_i` ± 2%。
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import taichi as ti

from src.v2.gas_unified import UnifiedGasLocal, scatter_albedo, unified_column, unified_gas_profile
from src.v2.noise_ti import hashf as _hashf_t
from src.v2.params import DiskV2Params, DiskV2VolumeParams

_PROFILE_KW = dict(core_floor=0.15, atm_frac=0.15, atm_c0=1.0, atm_soft=0.3, q=20.0, kappa=2.8,
                   grey_mix=0.5, grey_cap=1.19, dt_i=0.05)


class ProfileReferenceTest(unittest.TestCase):
    """NumPy 参考实现自身的物理性质。"""

    def setUp(self):
        r = 22.0
        self.loc = UnifiedGasLocal(sig=0.4, h_geo=0.003 * r, h_s=0.0025 * r, cn=1.2, h_a=0.01 * r)

    def test_column_normalization(self):
        z_top = 5.0 * self.loc.h_a
        z = np.linspace(-z_top, z_top, 400001)
        em_c, tf, ab_c, ab_a, em_a, sc_a = unified_gas_profile(z, self.loc, **_PROFILE_KW)
        num = np.trapezoid(ab_c + ab_a, z)
        ref = unified_column(self.loc, core_floor=0.15, atm_frac=0.15, atm_c0=1.0, atm_soft=0.3, atm_extent=5.0)
        self.assertAlmostEqual(num / ref, 1.0, delta=1e-3)

    def test_scatter_albedo_monotone(self):
        rho = np.concatenate([[0.0], np.logspace(-6, 2, 200)])
        w = scatter_albedo(rho, 1.0, 20.0)
        self.assertAlmostEqual(float(w[0]), 1.0)
        self.assertTrue(np.all(w > 0.0) and np.all(w <= 1.0))
        self.assertTrue(np.all(np.diff(w) < 0.0))

    def test_thermal_weight_within_extinction(self):
        z = np.linspace(-0.2, 0.2, 2001)
        _, _, _, ab_a, em_a, sc_a = unified_gas_profile(z, self.loc, **_PROFILE_KW)
        self.assertTrue(np.all(em_a >= 0.0) and np.all(em_a <= ab_a))
        self.assertTrue(np.all(sc_a >= 0.0) and np.all(sc_a <= ab_a - em_a + 1e-15))

    def test_scatter_weight_follows_core_emissivity(self):
        """散射权重 / (ω·ab_a) = 1 − e^{−κ·cfac·Σ}：核心越厚入射光越强，厚极限回到 1。"""
        z = np.array([0.05])
        prev = 0.0
        cfac = 0.15 + 0.85 * self.loc.cn
        for kappa in (0.01, 0.3, 3.0, 300.0):
            kw = dict(_PROFILE_KW, kappa=kappa)
            _, _, _, ab_a, em_a, sc_a = unified_gas_profile(z, self.loc, **kw)
            ratio = float(sc_a[0] / (ab_a[0] - em_a[0]))
            self.assertAlmostEqual(ratio, 1.0 - math.exp(-kappa * cfac * self.loc.sig), places=12)
            self.assertGreater(ratio, prev)
            prev = ratio
        self.assertAlmostEqual(prev, 1.0, places=9)

    def test_temperature_continuous(self):
        z = np.linspace(0.0, 5.0 * self.loc.h_a, 20001)
        _, tf, _, _, _, _ = unified_gas_profile(z, self.loc, **_PROFILE_KW)
        self.assertLess(float(np.max(np.abs(np.diff(tf)) / tf[:-1])), 0.01)
        # 温度随高度单调不增（上方光学深度单调减）
        self.assertTrue(np.all(np.diff(tf) <= 1e-12))


class DensityTaichiTest(unittest.TestCase):
    """Taichi `density_I` 与 NumPy 参考一致；κ 标定。只构造一次（构造含 κ 标定，CPU 上约 40 s）。"""

    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        from src.v2.taichi_impl import DiskV2Taichi
        cls.vp = DiskV2VolumeParams(core_floor=0.15, lowf_sigma=1.6, core_contrast=0.8, tau_i=1.84)
        cls.disk = DiskV2Taichi(DiskV2Params(3.0, 30.0, disk_spin=-1.0), cls.vp, opt_level=1)

    def test_matches_numpy_reference(self):
        disk, vp = self.disk, self.vp
        rng = np.random.default_rng(5)
        n_col = 48
        zr = np.array([0.0, 0.0005, 0.0015, 0.003, 0.008, 0.02, 0.04, -0.012])
        rr = np.repeat(rng.uniform(4.0, 28.0, n_col), len(zr))
        pp = np.repeat(rng.uniform(0.0, 2.0 * math.pi, n_col), len(zr))
        zz = rr * np.tile(zr, n_col)
        n = len(rr)
        f_r, f_p, f_z = (ti.field(ti.f32, shape=n) for _ in range(3))
        loc_f = ti.Vector.field(5, ti.f32, shape=n)
        out_f = ti.Vector.field(6, ti.f32, shape=n)
        f_r.from_numpy(rr.astype(np.float32))
        f_p.from_numpy(pp.astype(np.float32))
        f_z.from_numpy(zz.astype(np.float32))

        @ti.kernel
        def k():
            for i in range(n):
                r, phi, z = f_r[i], f_p[i], f_z[i]
                h_geo = disk._ss_half_thickness(r)
                nl = disk._turb_low(r, phi, 0.0)
                sig = disk._ss_surface_density(r) * ti.exp(disk._lowf_sigma * nl - 0.5 * disk._lowf_sigma ** 2)
                c, tn = disk._flow_I(r, phi, z, 0.0)
                softsat = 1.0 - 1.0 / (ti.max(tn, 0.0) + 1.0)
                h_s = ti.max(h_geo * (1.0 - disk._surf_noise + disk._surf_noise * softsat), 1e-6)
                na = disk._atm_fine_I(r, phi, disk._atm_ffz * z / (disk._atm_h * r), 0.0)
                loc_f[i] = ti.Vector([sig, h_geo, h_s, c / disk._c_mean, na])
                a0, a1, a2, a3, a4, a5 = disk.density_I(r, z, phi, 0.0, 1.0, 0.0)
                out_f[i] = ti.Vector([a0, a1, a2, a3, a4, a5])

        k()
        locs, outs = loc_f.to_numpy().astype(np.float64), out_f.to_numpy().astype(np.float64)
        kw = dict(core_floor=vp.core_floor, atm_frac=vp.atm_frac, atm_c0=vp.atm_cov_c0, atm_soft=vp.atm_cov_soft,
                  q=vp.abs_scatter_ratio, kappa=disk._kappa_vol, grey_mix=vp.grey_mix, grey_cap=vp.grey_cap,
                  dt_i=vp.dt_i, fine_sigma=vp.atm_fine_sigma)
        self.assertGreater(vp.atm_fine_sigma, 0.0)  # 默认开启小尺度起伏，一并校验
        ref = np.array([np.array(unified_gas_profile(zz[i], UnifiedGasLocal(*locs[i, :4], h_a=vp.atm_height * rr[i]),
                                                     fine_n=locs[i, 4], **kw)).ravel() for i in range(n)])
        self.assertGreater(int(np.sum(outs[:, 3] > 0.0)), n // 2)  # 大部分采样点落在大气内
        np.testing.assert_allclose(outs, ref, rtol=2e-4, atol=1e-6)

    def test_fine_structure_preserves_mean(self):
        """小尺度起伏 n：零均值、单位方差；lognormal 调制保均值（柱密度期望不变）。"""
        disk, vp = self.disk, self.vp
        n = 8192
        out = ti.field(ti.f32, shape=n)

        @ti.kernel
        def k():
            for i in out:
                r = 4.0 + _hashf_t(i, 3, 1) * 24.0
                phi = _hashf_t(i, 5, 2) * 2.0 * math.pi
                zeta = (_hashf_t(i, 7, 3) * 2.0 - 1.0) * 5.0 * disk._atm_ffz
                out[i] = disk._atm_fine_I(r, phi, zeta, 0.0)

        k()
        na = out.to_numpy().astype(np.float64)
        s = vp.atm_fine_sigma
        self.assertAlmostEqual(float(na.mean()), 0.0, delta=0.06)
        self.assertAlmostEqual(float(na.std()), 1.0, delta=0.1)
        self.assertAlmostEqual(float(np.exp(s * na - 0.5 * s * s).mean()), 1.0, delta=0.02)

    def test_kappa_calibration(self):
        """细网格（3200 个中点）独立积分的平均 τ⊥ 与 tau_i 一致（验证标定用的 800 点已收敛）。"""
        disk = self.disk
        out = ti.field(ti.f32, shape=256)

        @ti.kernel
        def k():
            for i in out:
                r = 5.5 + ti.cast(i % 16, ti.f32) / 16.0
                phi = ti.cast(i, ti.f32) * 0.61803 * 2.0 * math.pi
                zmax = ti.max(3.0 * disk._ss_half_thickness(r) + 0.01, disk._atm_ext * disk._atm_h * r)
                col = 0.0
                for j in range(3200):
                    z = -zmax + (ti.cast(j, ti.f32) + 0.5) / 3200.0 * 2.0 * zmax
                    a0, a1, ab_c, ab_a, a4, a5 = disk.density_I(r, z, phi, 0.0, 1.0, 0.0)
                    col += (ab_c + ab_a) * 2.0 * zmax / 3200.0
                out[i] = col

        k()
        tau = disk._kappa_vol * float(out.to_numpy().mean())
        self.assertAlmostEqual(tau / self.vp.tau_i, 1.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()
