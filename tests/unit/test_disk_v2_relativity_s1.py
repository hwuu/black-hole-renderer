#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 相对论修正（v2.3 S1）单元测试。

覆盖：
- 圆轨道局部速度 β = sqrt(M / (r − 2M))
- 光子坐标方向 → 本地静止观者方向变换
- 盘面 g-factor 与严格 GR 解析解对照（冲击参数由守恒量独立求得）
- Taichi helper 与 NumPy reference parity
"""

import math
import unittest

import numpy as np

from src.v2.relativity import (
    disk_g_factor,
    exact_equatorial_g_factor,
    local_photon_direction,
    orbital_beta_local,
)


def _impact_parameter_from_invariants(pos, d_hat):
    """由光线方程守恒量求冲击参数（与本地方向变换无关的独立推导）。

    光线方程 x'' = −1.5 L² x / r⁵ 对应势 V = −L² / (2 r³)，
    故 |v|² − L²/r³ = |v_∞|²，b = L / |v_∞| = |x × d̂| / sqrt(1 − |x × d̂|² / r³)。
    """
    r = np.linalg.norm(pos)
    lvec = np.cross(pos, d_hat)
    l2 = float(lvec @ lvec)
    return math.sqrt(l2) / math.sqrt(1.0 - l2 / r ** 3), lvec


class OrbitalBetaTest(unittest.TestCase):
    def test_beta_at_isco_is_one_half(self):
        """ISCO（r = 3 r_s = 6M）处静止观者测得 β = 1/2。"""
        self.assertAlmostEqual(orbital_beta_local(3.0), 0.5, places=6)

    def test_beta_formula_matches_schwarzschild(self):
        radii = np.array([3.0, 4.5, 8.0, 30.0])
        expected = np.sqrt(0.5 / (radii - 1.0))
        np.testing.assert_allclose(orbital_beta_local(radii), expected, rtol=1e-12)

    def test_beta_decreases_outward(self):
        radii = np.array([3.0, 6.0, 12.0, 30.0])
        self.assertTrue(np.all(np.diff(orbital_beta_local(radii)) < 0))


class LocalDirectionTest(unittest.TestCase):
    def test_radial_direction_is_unchanged(self):
        pos = np.array([5.0, 0.0, 0.0])
        k = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(local_photon_direction(pos, k), k, atol=1e-12)

    def test_tangential_direction_is_unchanged(self):
        pos = np.array([5.0, 0.0, 0.0])
        k = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(local_photon_direction(pos, k), k, atol=1e-12)

    def test_oblique_angle_follows_metric(self):
        """tanψ_local = sqrt(1 − r_s/r) · tanψ_coord（ψ 为与径向夹角）。"""
        r = 4.0
        pos = np.array([r, 0.0, 0.0])
        psi_c = math.radians(40.0)
        k = np.array([math.cos(psi_c), math.sin(psi_c), 0.0])
        k_loc = local_photon_direction(pos, k)
        psi_l = math.atan2(k_loc[1], k_loc[0])
        self.assertAlmostEqual(math.tan(psi_l), math.sqrt(1 - 1 / r) * math.tan(psi_c), places=10)
        self.assertAlmostEqual(float(np.linalg.norm(k_loc)), 1.0, places=12)

    def test_broadcasts_over_leading_axes(self):
        pos = np.tile([6.0, 1.0, 0.2], (4, 1))
        k = np.tile([0.3, -0.8, 0.1], (4, 1))
        out = local_photon_direction(pos, k)
        self.assertEqual(out.shape, (4, 3))
        np.testing.assert_allclose(np.linalg.norm(out, axis=-1), 1.0, rtol=1e-12)


class DiskGFactorTest(unittest.TestCase):
    R_OBS = 60.0

    def _cases(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            r = rng.uniform(3.0, 30.0)
            phi = rng.uniform(0.0, 2 * math.pi)
            pos = np.array([r * math.cos(phi), r * math.sin(phi), 0.0])
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            # 只保留能到达无穷远的方向（冲击参数平方为正）
            if np.cross(pos, d) @ np.cross(pos, d) >= r ** 3:
                continue
            yield pos, d

    def test_matches_exact_gr_for_equatorial_orbits(self):
        """与严格 GR 解析解相对误差 < 1e-9（冲击参数由守恒量独立求得）。"""
        n = 0
        for pos, d in self._cases():
            r = float(np.linalg.norm(pos))
            b, lvec = _impact_parameter_from_invariants(pos, d)
            # 追踪方向 d 与光子传播方向相反：光子 L_z/E = −sign(lvec_z)·b·|lvec_z|/|lvec|
            lz_over_e = -b * lvec[2] / max(np.linalg.norm(lvec), 1e-30)
            g_ex = exact_equatorial_g_factor(r, lz_over_e, self.R_OBS)
            g = disk_g_factor(pos, d, self.R_OBS)
            self.assertAlmostEqual(g / g_ex, 1.0, delta=1e-9)
            n += 1
        self.assertGreater(n, 100)

    def test_approaching_side_blueshifted(self):
        """盘逆时针旋转（+z 看）：x>0 处物质朝 +y 运动，光子朝 +y 传播时 g>1。"""
        pos = np.array([8.0, 0.0, 0.0])
        trace_dir = np.array([0.0, -1.0, 0.0])   # 反向追踪方向 → 光子朝 +y
        self.assertGreater(disk_g_factor(pos, trace_dir, self.R_OBS), 1.0)
        self.assertLess(disk_g_factor(pos, -trace_dir, self.R_OBS), 1.0)

    def test_broadcasts(self):
        pos = np.array([[5.0, 0.0, 0.0], [0.0, 9.0, 0.0]])
        d = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertEqual(disk_g_factor(pos, d, self.R_OBS).shape, (2,))


class TaichiParityTest(unittest.TestCase):
    """Taichi helper 与 NumPy reference 一致（f32 容差）。"""

    @classmethod
    def setUpClass(cls):
        import taichi as ti

        ti.init(arch=ti.cpu, default_fp=ti.f32)
        from src.v2 import taichi_impl as T

        cls.ti = ti
        cls.T = T
        rng = np.random.default_rng(3)
        pos, dirs = [], []
        while len(pos) < 64:
            r = rng.uniform(3.0, 30.0)
            phi = rng.uniform(0.0, 2 * math.pi)
            p = np.array([r * math.cos(phi), r * math.sin(phi), rng.uniform(-0.3, 0.3)])
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            pos.append(p)
            dirs.append(d)
        cls.pos = np.array(pos)
        cls.dirs = np.array(dirs)

    def test_beta_parity(self):
        ti, T = self.ti, self.T
        radii = np.array([3.0, 4.0, 8.0, 30.0], dtype=np.float32)
        inp = ti.field(ti.f32, shape=4)
        out = ti.field(ti.f32, shape=4)
        inp.from_numpy(radii)

        @ti.kernel
        def k():
            for i in inp:
                out[i] = T.schwarzschild_orbital_beta_ti(inp[i], 1.0, 1e-6, 0.99)

        k()
        np.testing.assert_allclose(out.to_numpy(), orbital_beta_local(radii.astype(np.float64)), rtol=1e-5)

    def test_disk_g_parity(self):
        ti, T = self.ti, self.T
        n = len(self.pos)
        p_f = ti.Vector.field(3, ti.f32, shape=n)
        d_f = ti.Vector.field(3, ti.f32, shape=n)
        out = ti.field(ti.f32, shape=n)
        p_f.from_numpy(self.pos.astype(np.float32))
        d_f.from_numpy(self.dirs.astype(np.float32))

        @ti.kernel
        def k():
            for i in p_f:
                out[i] = T.disk_g_factor_ti(p_f[i], d_f[i], 60.0, 1.0)

        k()
        np.testing.assert_allclose(out.to_numpy(), disk_g_factor(self.pos, self.dirs, 60.0), rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
