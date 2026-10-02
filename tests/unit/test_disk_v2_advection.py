#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 刚体环平流（v2.3 S4）单元测试。

覆盖（方案 §5 S4 测试要点）：
- 带权重和 = 1；种子切换（frac = 0）时权重 = 0。
- 带坐标关于 ln r 严格单调（扰动不翻转带序）。
- 纹理 Δφ 与 Ω_b·Δt 一致（误差 < 1°），且 30 个本地轨道后不衰减
  （刚体 → 图样平移不变 ⇒ 螺旋倾角恒定）。
- t = 1e6 时查表相位精度优于直接 f32 计算（与 f64 参考对比）。
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import taichi as ti

from src.v2 import advection as ADV
from src.v2 import noise_ti as N

R_IN, R_OUT, DLN, K_RIGID = 3.0, 30.0, math.log(1.22), 4.0


class WeightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        cls.fbfs = np.linspace(0.0, 1.0, 101, dtype=np.float32)
        cls.fracs = np.linspace(0.0, 1.0, 101, dtype=np.float32)

    def test_band_weights_sum_to_one(self):
        f = ti.field(ti.f32, shape=len(self.fbfs))
        f.from_numpy(self.fbfs)
        w0 = ti.field(ti.f32, shape=len(self.fbfs))
        w1 = ti.field(ti.f32, shape=len(self.fbfs))

        @ti.kernel
        def k():
            for i in range(len(self.fbfs)):
                a, b = ADV.band_mix_weights(f[i])
                w0[i] = a
                w1[i] = b

        k()
        total = w0.to_numpy() + w1.to_numpy()
        np.testing.assert_allclose(total, 1.0, rtol=0, atol=1e-6)
        self.assertAlmostEqual(float(w0.to_numpy()[0]), 1.0, places=6)
        self.assertAlmostEqual(float(w1.to_numpy()[-1]), 1.0, places=6)

    def test_phase_weight_zero_at_seed_switch(self):
        f = ti.field(ti.f32, shape=len(self.fracs))
        f.from_numpy(self.fracs)
        out = ti.field(ti.f32, shape=len(self.fracs))

        @ti.kernel
        def k():
            for i in range(len(self.fracs)):
                out[i] = ADV.phase_weight(f[i])

        k()
        w = out.to_numpy()
        self.assertAlmostEqual(float(w[0]), 0.0, places=7)      # frac=0：切换瞬间
        self.assertAlmostEqual(float(w[-1]), 0.0, places=6)    # frac→1
        self.assertAlmostEqual(float(w[50]), 1.0, places=6)    # frac=0.5
        self.assertTrue(np.all(w >= 0.0) and np.all(w <= 1.0))


class MonotonicityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    def test_band_coordinate_strictly_monotonic(self):
        """扰动斜率 < 基础斜率 ⇒ fb 关于 ln r 严格递增（带序不翻转）。"""
        bands = ADV.RigidRingBands(R_IN, R_OUT, DLN, K_RIGID)
        lnrs = np.linspace(math.log(R_IN) - 2.5 * DLN, math.log(R_OUT) + 1.5 * DLN, 8192)
        f = ti.field(ti.f32, shape=len(lnrs))
        f.from_numpy(lnrs.astype(np.float32))
        out = ti.field(ti.f32, shape=len(lnrs))

        @ti.kernel
        def k():
            for i in range(len(lnrs)):
                out[i] = ADV.band_coordinate(f[i], bands.lnr0, bands.dln)

        k()
        fb = out.to_numpy().astype(np.float64)
        self.assertTrue(np.all(np.diff(fb) > 0.0),
                        msg=f"min diff = {np.diff(fb).min():.3e}")


class PatternShiftTest(unittest.TestCase):
    """图样 Δφ ≈ Ω_b·Δt（< 1°），且 30 个本地轨道后不衰减。"""

    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        cls.bands = ADV.RigidRingBands(R_IN, R_OUT, DLN, K_RIGID)
        cls.fields = ADV.make_fields(cls.bands)
        # 目标带：取覆盖 r = 8 的带，求其带中心的精确半径
        b = cls.bands.band_of_radius(8.0)
        cls.bi = b
        cls.r = float(np.exp(cls.bands.lnr0 + b * cls.bands.dln))
        cls.om = float(np.sqrt(0.5 / cls.r ** 3))
        cls.t_life = float(cls.bands.t_life[b - cls.bands.b_lo])
        # 选 frac_0 = 0.5 的时刻（此时 phase1 frac = 0、权重 0 → 纯 phase0 种子）
        ph0 = float(cls.bands.ph0[b - cls.bands.b_lo])
        cls.t0 = ((12 - ph0) % 1.0 + 11.5) * cls.t_life  # frac0 = 0.5
        cls.n_phi = 2048

    def _profile(self, t: float) -> np.ndarray:
        """在目标带中心 r 处采样方位剖面（经完整查表 + 流坐标 + cascade）。"""
        table = self.bands.phase_table(t)
        ADV.upload(self.fields, table)
        phis = np.linspace(0.0, 2.0 * math.pi, self.n_phi, endpoint=False)
        pf = ti.field(ti.f32, shape=self.n_phi)
        pf.from_numpy(phis.astype(np.float32))
        out = ti.field(ti.f32, shape=self.n_phi)
        flds = self.fields

        @ti.kernel
        def k():
            for i in range(self.n_phi):
                # 该 r 恰为带中心：fbf=0 → 纯带 bi（另一带权重 0）
                idx = self.bi - flds.b_lo
                rot = flds.rot[idx]
                phi_b = flds.phi_b[idx]
                om = flds.om_b[idx]
                t_life = flds.t_life[idx]
                # 种子相位进度（相位 0）：frac = 0.5 + 微小时间修正项由表给出
                fr = flds.frac[idx][0]
                cy = flds.cyc[idx][0]
                w0 = ADV.phase_weight(fr)
                # 带混合（fbf = 0 → w_lo = 1）
                wb_lo, _wb_hi = ADV.band_mix_weights(0.0)
                phi0 = pf[i] - rot - phi_b
                ox = N.hashf(self.bi, cy, 0) * 97.0
                oz = N.hashf(self.bi, cy, 1) * 97.0
                c = N.cascade(0.2 * self.r + ox,
                              phi0 / (2.0 * math.pi) * 2.0,
                              0.0 + oz, 2, 3.0, 5.0, 50.0)
                out[i] = w0 * wb_lo * c
                _ = om
                _ = t_life

        k()
        return out.to_numpy().astype(np.float64)

    def _shift_by_correlation(self, t: float, delta: float) -> float:
        """互相关求 [t, t+delta] 间图样的角位移（度）。"""
        a = self._profile(t)
        b_ = self._profile(t + delta)
        a = a - a.mean()
        b_ = b_ - b_.mean()
        corr = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b_)), n=self.n_phi)
        k = int(np.argmax(corr))
        # IFFT(FFT(a)·conj(FFT(b))) 的峰在 k = −s（s 为图样右移格数）：
        # b[x] = a[x−s] ⇒ 峰 k = −s mod n。图样沿 +φ 移动 s 格。
        return (-360.0 * k / self.n_phi) % 360.0

    def test_shift_matches_kepler_within_1deg(self):
        delta = math.radians(20.0) / self.om   # 20° 的开普勒位移对应的时间
        for base, label in ((self.t0, "t0"), (self.t0 + 30.0 * self.t_life, "30 orbits later")):
            measured = self._shift_by_correlation(base, delta)
            self.assertLess(abs(measured - 20.0), 1.0,
                            msg=f"{label}: measured {measured:.2f}° vs expected 20°")

    def test_pitch_constant_over_30_orbits(self):
        """30 个本地轨道前后，同样 Δt 的图样位移一致（倾角不衰减）。"""
        delta = math.radians(10.0) / self.om
        s1 = self._shift_by_correlation(self.t0, delta)
        s2 = self._shift_by_correlation(self.t0 + 30.0 * self.t_life, delta)
        self.assertLess(abs(s1 - s2), 0.5, msg=f"{s1:.3f} vs {s2:.3f}")


class PhasePrecisionTest(unittest.TestCase):
    """t = 1e6 时查表相位 vs f64 参考，远优于直接 f32 计算。"""

    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        cls.bands = ADV.RigidRingBands(R_IN, R_OUT, DLN, K_RIGID)
        cls.fields = ADV.make_fields(cls.bands)
        cls.t = 1.0e6

    def test_table_phase_precision(self):
        table = self.bands.phase_table(self.t)
        ADV.upload(self.fields, table)
        n = self.bands.n_bands
        out_table = ti.field(ti.f32, shape=n)
        out_naive = ti.field(ti.f32, shape=n)
        flds = self.fields

        @ti.kernel
        def k():
            for i in range(n):
                # 查表路径：rot 已在 CPU 用 f64 取模
                out_table[i] = flds.rot[i]
                # 朴素路径：kernel 内 f32 直接乘
                out_naive[i] = flds.om_b[i] * 1.0e6

        k()
        rot64 = table["rot"]
        err_table = np.abs(out_table.to_numpy().astype(np.float64) - rot64).max()
        # 朴素 f32 的相位误差：取 (om*t mod 2π) 与 f64 之差
        naive = out_naive.to_numpy().astype(np.float64) % (2.0 * math.pi)
        err_naive = np.abs(((naive - rot64 + math.pi) % (2.0 * math.pi)) - math.pi).max()
        self.assertLess(err_table, 1e-5, msg=f"table err {err_table:.2e}")
        self.assertGreater(err_naive, 1e-3,
                           msg="朴素 f32 在 t=1e6 的相位误差应显著大于查表误差")

    def test_cycle_index_bounded(self):
        """种子周期索引被 mod 4096（防 f32 大整数哈希退化），且 frac ∈ [0,1)。"""
        table = self.bands.phase_table(1.0e6)
        self.assertTrue(np.all(table["cyc"] >= 0) and np.all(table["cyc"] < 4096))
        self.assertTrue(np.all(table["frac"] >= 0.0) and np.all(table["frac"] < 1.0))
        # 相位 0/1 恒差半周期
        np.testing.assert_allclose(
            (table["frac"][:, 1] - table["frac"][:, 0]) % 1.0, 0.5, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
