#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 颜色与亮度链路（v2.3 S2）单元测试。

覆盖：
- CIE 黑体色度表：与数值积分参考一致、低温偏红 / 高温偏蓝单调、色温方向
- 可见光亮度 Y(T)：单调、与数值积分一致、T = 0 或负温度行为
- 白平衡：von Kries 增益、T_wb 黑体输出中性、亮度（BT.709 加权）守恒
- palette_color 新语义（只剩 physical；cinematic 已删除）
- Taichi parity：blackbody_color_ti / blackbody_luminance_ti / white_balance_gain_ti
"""

import math
import unittest

import numpy as np

from disk_v2.params import DiskV2PaletteParams
from disk_v2.palette import (
    blackbody_color,
    blackbody_luminance,
    palette_color,
    white_balance_gain,
)

_LAM = np.linspace(380.0, 780.0, 801)


def _cie_cmf(lam):
    """CIE 1931 2° 色匹配函数（Wyman, Sloan & Shirley 2013 多瓣高斯近似）。"""
    def g(x, mu, s1, s2):
        sig = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / sig) ** 2)
    xb = 1.056 * g(lam, 599.8, 37.9, 31.0) + 0.362 * g(lam, 442.0, 16.0, 26.7) - 0.065 * g(lam, 501.1, 20.4, 26.2)
    yb = 0.821 * g(lam, 568.8, 46.9, 40.5) + 0.286 * g(lam, 530.9, 16.3, 31.1)
    zb = 1.217 * g(lam, 437.0, 11.8, 36.0) + 0.681 * g(lam, 459.0, 26.0, 13.8)
    return xb, yb, zb


def _planck(lam, t):
    x = np.minimum(1.4388e7 / (lam * t), 700.0)
    return lam ** -5 / np.expm1(x)


def _ref_rgb(t):
    xb, yb, zb = _cie_cmf(_LAM)
    spec = _planck(_LAM, t)
    xyz = np.array([(spec * xb).sum(), (spec * yb).sum(), (spec * zb).sum()])
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.maximum(m @ xyz, 0.0)
    return rgb / max(float(rgb @ [0.2126, 0.7152, 0.0722]), 1e-30)


def _ref_lum(t):
    return float((_planck(_LAM, t) * _cie_cmf(_LAM)[1]).sum())


class BlackbodyColorTest(unittest.TestCase):
    def test_matches_numerical_integration(self):
        """查表结果与直接数值积分的色度偏差 < 3%（log T 间距、f32 表）。"""
        for t in (1500.0, 3000.0, 4500.0, 6500.0, 9000.0, 20000.0):
            got = np.asarray(blackbody_color(t))
            np.testing.assert_allclose(got, _ref_rgb(t), rtol=0.03,
                                       err_msg=f"T={t}")

    def test_low_temp_reddish_high_temp_bluish(self):
        """2000 K B/R < 1 < 10000 K B/R（黑体色温方向）。"""
        c_lo = np.asarray(blackbody_color(2000.0))
        c_hi = np.asarray(blackbody_color(10000.0))
        self.assertLess(c_lo[2] / c_lo[0], 0.8)
        self.assertGreater(c_hi[2] / c_hi[0], 1.0)

    def test_saturation_monotonic_direction(self):
        """温度从低到高，色相沿黑体轨迹单调（用 G/R 比值抽查两个区间）。"""
        ts = np.array([2000.0, 3500.0, 5000.0, 7000.0])
        gr = [float(np.asarray(blackbody_color(t))[1] / np.asarray(blackbody_color(t))[0]) for t in ts]
        self.assertTrue(all(b > a for a, b in zip(gr, gr[1:])), msg=gr)

    def test_near_6500k_is_neutral(self):
        """6500 K（≈ D65 白点）应接近中性灰，各通道对 1 的偏差 < 5%。"""
        c = np.asarray(blackbody_color(6500.0))
        np.testing.assert_allclose(c, 1.0, rtol=0.05)

    def test_luminance_normalized_to_one(self):
        """色度表归一约定：BT.709 亮度 = 1（个别线性通道可 > 1）。"""
        for t in (3000.0, 4500.0, 8000.0):
            c = np.asarray(blackbody_color(t))
            self.assertAlmostEqual(float(c @ [0.2126, 0.7152, 0.0722]), 1.0, places=6, msg=str(t))

    def test_zero_and_negative_temperature_return_black(self):
        np.testing.assert_array_equal(np.asarray(blackbody_color(0.0)), [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(np.asarray(blackbody_color(-5.0)), [0.0, 0.0, 0.0])

    def test_array_input_broadcast(self):
        out = np.asarray(blackbody_color(np.array([3000.0, 6000.0])))
        self.assertEqual(out.shape, (2, 3))


class BlackbodyLuminanceTest(unittest.TestCase):
    def test_matches_numerical_integration(self):
        """ln Y 查表与数值积分的相对误差 < 1%。"""
        for t in (1000.0, 2500.0, 4500.0, 10000.0, 30000.0):
            got = blackbody_luminance(t)
            self.assertAlmostEqual(got / _ref_lum(t), 1.0, delta=0.01, msg=f"T={t}")

    def test_monotonic_increasing(self):
        ts = np.array([1000.0, 2000.0, 4000.0, 8000.0, 16000.0])
        ys = np.array([blackbody_luminance(t) for t in ts])
        self.assertTrue(np.all(np.diff(ys) > 0))

    def test_rayleigh_jeans_scaling(self):
        """T ≫ hc/(λk) 时 Y ∝ T（Wien 段之外，48000 K 处比值误差 < 10%）。"""
        y1, y2 = blackbody_luminance(40000.0), blackbody_luminance(48000.0)
        self.assertAlmostEqual(y2 / y1, 1.2, delta=0.1)

    def test_zero_temperature_returns_zero(self):
        self.assertEqual(blackbody_luminance(0.0), 0.0)
        self.assertEqual(blackbody_luminance(-1.0), 0.0)


class WhiteBalanceTest(unittest.TestCase):
    def test_twb_blackbody_is_neutral(self):
        """白平衡设为 T 时，该温度黑体的增益应为中性（三通道相等）。"""
        for twb in (3200.0, 4500.0, 5000.0, 6500.0):
            c = np.asarray(blackbody_color(twb))
            gain = white_balance_gain(twb)
            balanced = c * gain
            np.testing.assert_allclose(balanced, balanced.mean(), rtol=1e-9)

    def test_other_spectra_luma_drift_is_small(self):
        """von Kries 不逐谱守恒 BT.709 亮度：其他黑体谱平衡后 luma 漂移 < 6%。"""
        w = np.array([0.2126, 0.7152, 0.0722])
        for twb in (4000.0, 5000.0, 6500.0):
            for t_src in (3000.0, 4500.0, 6000.0):
                c = np.asarray(blackbody_color(t_src))
                self.assertLess(
                    abs(float(w @ (c * white_balance_gain(twb))) / float(w @ c) - 1.0),
                    0.06, msg=f"twb={twb}, T_src={t_src}",
                )

    def test_warm_twb_suppresses_red_channel(self):
        """暖白平衡（低于场景色温）：von Kries 压红提蓝。"""
        g = white_balance_gain(4000.0)
        self.assertLess(g[0], g[2])
        self.assertTrue(np.all(g > 0))

    def test_gain_never_negative_or_zero(self):
        g = white_balance_gain(1000.0)
        self.assertTrue(np.all(g > 0))


class PaletteSemanticsTest(unittest.TestCase):
    def test_cinematic_mode_removed(self):
        """palette_mode（含 'cinematic'）应已删除：字段不存在，传参被拒。"""
        self.assertNotIn("palette_mode", DiskV2PaletteParams.__dataclass_fields__)
        with self.assertRaises(TypeError):
            DiskV2PaletteParams(palette_mode="cinematic")

    def test_cinematic_helpers_removed(self):
        """cinematic 相关函数与参数应已删除。"""
        import disk_v2.palette as pal
        for name in ("cinematic_color", "cinematic_visual_temperature",
                     "physical_temperature_outer_K", "_rgb_saturation_boost"):
            self.assertFalse(hasattr(pal, name), msg=name)
        for field in ("cinematic_saturation", "cinematic_warm_shift",
                      "visual_temp_outer_K", "visual_temp_inner_K",
                      "cinematic_value_low_T", "cinematic_value_high_T"):
            self.assertNotIn(field, DiskV2PaletteParams.__dataclass_fields__)

    def test_palette_color_equals_blackbody_color(self):
        """palette_color 直接返回黑体色（无二级映射）。"""
        for t in (3000.0, 6000.0):
            np.testing.assert_allclose(
                np.asarray(palette_color(t, DiskV2PaletteParams())),
                np.asarray(blackbody_color(t)), rtol=1e-12)


class TaichiParityTest(unittest.TestCase):
    """Taichi 查表与 NumPy 参考一致（f32 容差）。"""

    @classmethod
    def setUpClass(cls):
        import taichi as ti

        ti.init(arch=ti.cpu, default_fp=ti.f32)
        from disk_v2 import taichi_impl as T

        cls.ti, cls.T = ti, T
        cls.pal = DiskV2PaletteParams()
        # 实例化 DiskV2Taichi 以构建 LUT field（cinematic 已删，is_cinematic 恒 False）
        from disk_v2.params import DiskV2Params, DiskV2StructureParams
        cls.disk = T.DiskV2Taichi(
            DiskV2Params(r_in=3.0, r_out=30.0, T_peak_K=4500.0),
            DiskV2StructureParams(),
            cls.pal,
        )

    def test_blackbody_color_parity(self):
        ti = self.ti
        temps = ti.field(ti.f32, shape=6)
        out = ti.Vector.field(3, ti.f32, shape=6)
        vals = np.array([1000.0, 2000.0, 4500.0, 6500.0, 12000.0, 40000.0], dtype=np.float32)
        temps.from_numpy(vals)

        @ti.kernel
        def k():
            for i in temps:
                out[i] = self.disk.blackbody_color_ti(temps[i])

        k()
        for i, t in enumerate(vals):
            np.testing.assert_allclose(out.to_numpy()[i],
                                       np.asarray(blackbody_color(float(t))), rtol=2e-3, atol=1e-4)

    def test_luminance_parity(self):
        ti = self.ti
        temps = ti.field(ti.f32, shape=5)
        out = ti.field(ti.f32, shape=5)
        vals = np.array([1000.0, 2500.0, 4500.0, 10000.0, 30000.0], dtype=np.float32)
        temps.from_numpy(vals)

        @ti.kernel
        def k():
            for i in temps:
                out[i] = self.disk.blackbody_luminance_ti(temps[i])

        k()
        for i, t in enumerate(vals):
            self.assertLessEqual(abs(out.to_numpy()[i] / blackbody_luminance(float(t)) - 1.0), 0.01)

    def test_white_balance_parity(self):
        """每档 white_balance_K 构造的 DiskV2Taichi，其 kernel 增益与 NumPy 一致。

        增益在 __init__ 按 white_balance_K 预计算并上传 field（Taichi kernel
        闭包不接受运行时向量参数），因此每个 T_wb 需要单独构造实例。
        """
        from disk_v2.params import DiskV2Params, DiskV2StructureParams
        for twb in (3500.0, 4500.0, 5500.0, 6600.0):
            disk = self.T.DiskV2Taichi(
                DiskV2Params(r_in=3.0, r_out=30.0, T_peak_K=4500.0),
                DiskV2StructureParams(),
                DiskV2PaletteParams(white_balance_K=twb),
            )
            out = self.ti.Vector.field(3, self.ti.f32, shape=1)

            @self.ti.kernel
            def k():
                out[0] = disk.white_balance_gain_ti(twb)

            k()
            np.testing.assert_allclose(
                out.to_numpy()[0], np.asarray(white_balance_gain(twb)),
                rtol=1e-6, atol=1e-7, err_msg=f"twb={twb}",
            )


class ContractAndBoundaryTest(unittest.TestCase):
    """白平衡 kernel 契约与 LUT 越界行为。"""

    @classmethod
    def setUpClass(cls):
        import taichi as ti
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        from disk_v2 import taichi_impl as T
        from disk_v2.params import (
            DiskV2PaletteParams,
            DiskV2Params,
            DiskV2StructureParams,
        )
        cls.ti, cls.T = ti, T
        # 构造温度与"调用温度"故意不同，锁定 kernel 增益取构造值的契约
        cls.disk = T.DiskV2Taichi(
            DiskV2Params(r_in=3.0, r_out=30.0, T_peak_K=4500.0),
            DiskV2StructureParams(),
            DiskV2PaletteParams(white_balance_K=5000.0),
        )

    def test_wb_kernel_ignores_runtime_argument(self):
        """`white_balance_gain_ti(T_wb)` 的 T_wb 仅作文档语义：传任何值都返回
        构造时 `white_balance_K`（5000 K）的增益——防止调用方误以为参数生效。"""
        out = self.ti.Vector.field(3, self.ti.f32, shape=3)

        @self.ti.kernel
        def k():
            out[0] = self.disk.white_balance_gain_ti(3000.0)
            out[1] = self.disk.white_balance_gain_ti(5000.0)
            out[2] = self.disk.white_balance_gain_ti(20000.0)

        k()
        expected = np.asarray(white_balance_gain(5000.0))
        got = out.to_numpy()
        np.testing.assert_allclose(got, np.stack([expected] * 3), rtol=1e-6, atol=1e-7)
        # 与"参数生效"的假设相反：3000/20000 K 的增益必须不同于实际值
        self.assertFalse(np.allclose(got[0], np.asarray(white_balance_gain(3000.0))))

    def test_color_lut_clamps_out_of_range_temperature(self):
        """色度表 [1000, 40000] K 外的温度被端点钳制（而非外推）。

        红移后 T·g 低于 1000 K 会取 1000 K 的颜色（比真实深红略浅），
        高于 40000 K 取 40000 K 的颜色。该行为由设计决定（§2.2 表范围），
        本测试锁定之，防止未来被改成外推。
        """
        from disk_v2.palette import _BB_T_MAX_K, _BB_T_MIN_K
        below = np.asarray(blackbody_color(300.0))
        at_min = np.asarray(blackbody_color(_BB_T_MIN_K))
        np.testing.assert_allclose(below, at_min, rtol=1e-12)
        above = np.asarray(blackbody_color(1.0e6))
        at_max = np.asarray(blackbody_color(_BB_T_MAX_K))
        np.testing.assert_allclose(above, at_max, rtol=1e-12)
        # 亮度表 [300, 60000] K 同样钳制
        from disk_v2.palette import _LNY_T_MAX_K, _LNY_T_MIN_K
        self.assertEqual(blackbody_luminance(1.0), blackbody_luminance(_LNY_T_MIN_K))
        self.assertEqual(blackbody_luminance(1.0e7), blackbody_luminance(_LNY_T_MAX_K))


if __name__ == "__main__":
    unittest.main()
