#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""剪切级联（`src/v2/shear_cascade.py`、`DiskV2Taichi._casc_shear`）单元测试。

覆盖（docs/plans/v2_unified_gas_plan.md §8）：
- 几何构造：各八度长宽比、拖尾倾角与目标一致（由噪声坐标的雅可比反推）；倾角系数 0 时无剪切；
  方位周期为正整数；参数校验。
- `*_ti` 一致性：Taichi `_casc_shear` 与 NumPy 参考 `shear_cascade_np` 逐值一致（优化级别 0 / 1 两条噪声路径）。
- φ 方向无缝：φ 与 φ + 2π 的输出一致。
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import taichi as ti

from src.v2.params import DiskV2Params, DiskV2VolumeParams
from src.v2.shear_cascade import shear_cascade_geometry, shear_cascade_np
from tests.unit.test_disk_v2_noise_ti import _vnoise

_vnoise_vec = np.vectorize(_vnoise, otypes=[np.float64])


def _feature_shape(geom, k, spin):
    """由八度 k 的噪声坐标雅可比反推特征形状。

    Args:
        geom: `ShearCascadeGeometry`。
        k: 八度下标。
        spin: 盘旋转方向（+1 / −1）。

    Returns:
        `(aspect, tilt_rad, d_lnr, d_phi)`：长宽比、长轴相对方位方向的夹角（rad，≥ 0）、
        长轴方向在 (K·ln r, K·φ) 各向同性单位下的两个分量。

    Formula:
        以各向同性坐标 `X = (K·ln r, K·φ)` 表示时，噪声坐标 `(u, v) = J·X`，
        `J = [[|N₀₀|, 0], [spin·s, P/(2πK)]]`。噪声中的单位圆格对应特征 `J⁻¹·单位圆`，
        其奇异值之比为长宽比，最大奇异向量为长轴方向。
    """
    kr = geom.kr[k]
    jac = np.array([[geom.u_scale[k], 0.0],
                    [spin * geom.shear[k], geom.period[k] / (2.0 * math.pi * kr)]])
    u, sv, _ = np.linalg.svd(np.linalg.inv(jac))
    d = u[:, 0]
    tilt = math.atan2(abs(d[0]), abs(d[1]))
    return sv[0] / sv[1], tilt, d[0], d[1]


class ShearGeometryTest(unittest.TestCase):
    """解析构造给出的形状与目标长宽比 / 倾角一致。"""

    def test_aspect_and_tilt_match_target(self):
        for spin in (1.0, -1.0):
            g = shear_cascade_geometry(6.67, 4, 10.0, 2.5, 1.0, spin)
            for k in range(4):
                ar, tilt, dl, dp = _feature_shape(g, k, spin)
                # 方位周期取整使形状有小偏差（最大八度 P = 4 时最大）
                self.assertAlmostEqual(ar / g.aspect[k], 1.0, delta=0.12)
                self.assertAlmostEqual(tilt, g.tilt_rad[k], delta=math.radians(2.5))
                # 拖尾：沿长轴向外（ln r 增大）时，φ 向旋转反方向变化
                self.assertLess(np.sign(dl * dp) * spin, 0.0)

    def test_aspect_interpolates_geometrically(self):
        g = shear_cascade_geometry(6.67, 4, 10.0, 2.5, 0.01, 1.0)
        np.testing.assert_allclose(g.aspect, [10.0, 10.0 * 0.25 ** (1 / 3), 10.0 * 0.25 ** (2 / 3), 2.5])
        np.testing.assert_allclose(g.kr, [6.67, 20.01, 60.03, 180.09])

    def test_zero_tilt_has_no_shear(self):
        g = shear_cascade_geometry(6.67, 4, 10.0, 2.5, 0.0, -1.0)
        self.assertEqual(g.tilt_rad, (0.0, 0.0, 0.0, 0.0))
        for k in range(4):
            self.assertAlmostEqual(g.shear[k], 0.0, places=12)
            self.assertAlmostEqual(g.u_scale[k], 1.0, places=12)
            # 无倾角时方位周期 = round(2π·K / AR)
            self.assertEqual(g.period[k], max(1, round(2.0 * math.pi * g.kr[k] / g.aspect[k])))

    def test_periods_are_positive_integers(self):
        for tk in (0.0, 0.01, 0.5, 1.0):
            g = shear_cascade_geometry(6.67, 4, 10.0, 2.5, tk, 1.0)
            for p in g.period:
                self.assertIsInstance(p, int)
                self.assertGreaterEqual(p, 1)

    def test_default_tilt_is_tiny(self):
        g = shear_cascade_geometry(6.67, 4, 10.0, 2.5, 0.01, 1.0)
        self.assertLess(max(g.tilt_rad), math.radians(0.4))

    def test_invalid_arguments_raise(self):
        bad = [(0.0, 4, 10.0, 2.5, 0.01, 1.0), (6.67, 0, 10.0, 2.5, 0.01, 1.0),
               (6.67, 4, 0.5, 2.5, 0.01, 1.0), (6.67, 4, 10.0, 2.5, -0.1, 1.0),
               (6.67, 4, 10.0, 2.5, 0.01, 0.5)]
        for args in bad:
            with self.assertRaises(ValueError):
                shear_cascade_geometry(*args)

    def test_volume_params_validation(self):
        for kw in ({"shear_k0": 0.0}, {"shear_octaves": 0}, {"shear_ar_big": 0.9},
                   {"shear_con": 0.0}, {"shear_tilt_k": -1.0}):
            with self.assertRaises(ValueError):
                DiskV2VolumeParams(**kw)


class ShearParityTest(unittest.TestCase):
    """Taichi `_casc_shear` 与 NumPy 参考逐值一致；φ 方向无缝。"""

    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        rng = np.random.default_rng(11)
        n = 96
        cls.lnr = rng.uniform(math.log(3.0), math.log(30.0), n)
        cls.phi = rng.uniform(0.0, 2.0 * math.pi, n)
        cls.zr = rng.uniform(-0.01, 0.01, n)
        # 每个优化级别只构造一次（构造含 κ 标定，CPU 上约 40 s）
        from src.v2.taichi_impl import DiskV2Taichi
        cls.vp = DiskV2VolumeParams(shear_tilt_k=0.5)
        cls.disks = {opt: DiskV2Taichi(DiskV2Params(3.0, 30.0, disk_spin=-1.0), cls.vp, opt_level=opt)
                     for opt in (0, 1)}

    def _run_ti(self, opt, phi_shift=0.0):
        disk, vp = self.disks[opt], self.vp
        n = len(self.lnr)
        f_l, f_p, f_z, out = (ti.field(ti.f32, shape=n) for _ in range(4))
        f_l.from_numpy(self.lnr.astype(np.float32))
        f_p.from_numpy((self.phi + phi_shift).astype(np.float32))
        f_z.from_numpy(self.zr.astype(np.float32))

        @ti.kernel
        def k():
            for i in range(n):
                out[i] = disk._casc_shear(f_l[i], f_p[i], f_z[i], 3.7, 11.2)

        k()
        return out.to_numpy().astype(np.float64), disk, vp

    def test_matches_numpy_reference(self):
        for opt in (0, 1):
            got, disk, vp = self._run_ti(opt)
            geom = shear_cascade_geometry(vp.shear_k0, vp.shear_octaves, vp.shear_ar_big, vp.shear_ar_small,
                                          vp.shear_tilt_k, -1.0)
            # 与 Taichi 同精度输入（f32 取整后的坐标）
            ref = shear_cascade_np(self.lnr.astype(np.float32).astype(np.float64),
                                   self.phi.astype(np.float32).astype(np.float64),
                                   self.zr.astype(np.float32).astype(np.float64),
                                   3.7, 11.2, geom, vp.shear_con, vp.core_oct_gain, -1.0, _vnoise_vec)
            self.assertTrue(np.all(got >= 0.0))
            np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3, err_msg=f"opt_level={opt}")

    def test_phi_wraps_seamlessly(self):
        a, _, _ = self._run_ti(1)
        b, _, _ = self._run_ti(1, phi_shift=2.0 * math.pi)
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
