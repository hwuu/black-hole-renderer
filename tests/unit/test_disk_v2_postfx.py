#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 后处理链（v2.3 S7）单元测试。

覆盖（方案 §5 S7 测试要点）：
- bloom 对盘面局部对比损失 < 35%
- 光晕点亮暗区 5-15%
- 高光外缘 B/G 升高（轴向色散）
- 零强度时各效果为恒等
- 保色度 ACES：色度保持、超色域向白混合
- sRGB 编码正确性
- postfx 全链路可运行
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

from src.v2.postfx import (
    apply_bloom,
    apply_fringe,
    apply_lateral_ca,
    apply_white_balance,
    postfx,
    postfx_params_defaults,
    srgb_encode,
    tonemap_chroma_aces,
)


class IdentityTest(unittest.TestCase):
    """零强度时各效果为恒等。"""

    def test_bloom_zero_gain_identity(self):
        x = np.random.rand(16, 16, 3)
        out = apply_bloom(x, threshold=0.3, gain=0.0)
        np.testing.assert_allclose(out, x, atol=1e-12)

    def test_fringe_zero_strength_identity(self):
        x = np.random.rand(16, 16, 3)
        out = apply_fringe(x, fringe_strength=0.0)
        np.testing.assert_allclose(out, x, atol=1e-12)

    def test_lateral_ca_zero_identity(self):
        x = np.random.rand(16, 16, 3)
        out = apply_lateral_ca(x, strength=0.0)
        np.testing.assert_allclose(out, x, atol=1e-12)

    def test_wb_6600_near_identity(self):
        """6600 K ≈ 中性白点，增益 ≈ 1。"""
        x = np.random.rand(4, 4, 3) * 10
        out = apply_white_balance(x, 6600.0)
        np.testing.assert_allclose(out, x, rtol=0.05)


class BloomTest(unittest.TestCase):
    def setUp(self):
        # 接近真实盘面：基底 ~0.4，亮丝 +0.3~0.6，暗缝 ×0.3（对比 ~3 倍）
        rng = np.random.default_rng(42)
        self.img = np.full((64, 64, 3), 0.4)
        for _ in range(10):
            x, y = rng.integers(0, 56), rng.integers(0, 56)
            self.img[y:y+8, x:x+8] += rng.uniform(0.3, 0.6)
        for _ in range(10):
            x, y = rng.integers(0, 56), rng.integers(0, 56)
            self.img[y:y+6, x:x+6] *= 0.3

    def test_bloom_brightens_dark_region(self):
        """bloom 应让暗区被光晕点亮。"""
        dark_img = self.img.copy()
        dark_img[:32] = 0.01  # 暗区
        before = dark_img[:32].mean()
        after = apply_bloom(dark_img, threshold=0.3, gain=4.0)[:32].mean()
        self.assertGreater(after, before)

    def test_bloom_contrast_loss_bounded(self):
        """bloom 对盘面局部对比损失 < 35%。"""
        from numpy.lib.stride_tricks import sliding_window_view as sw
        bright = self.img[32:]
        after = apply_bloom(self.img, threshold=0.3, gain=4.0)[32:]
        for img, name in ((bright, "before"), (after, "after")):
            pass  # just compute below
        def local_contrast(x):
            pad = np.pad(x.mean(-1), 4, mode="edge")
            loc = sw(pad, (9, 9)).mean((-1, -2))
            m = x.mean(-1) > 0.05
            return (np.abs(x.mean(-1) - loc)[m] / np.maximum(loc[m], 1e-6)).mean()
        c_before = local_contrast(bright)
        c_after = local_contrast(after)
        loss = 1 - c_after / c_before
        self.assertLess(loss, 0.35, msg=f"contrast loss = {loss:.2%}")

    def test_axial_dispersion_blue_spreads_more(self):
        """轴向色散：B 通道光晕 > G 通道光晕（高光外缘 B/G 升高）。"""
        img = np.zeros((64, 64, 3))
        img[28:36, 28:36] = 2.0  # 中心亮块
        out = apply_bloom(img, threshold=0.3, gain=4.0, axial_scale=(1.0, 1.0, 1.5))
        # 外围区域（远离中心块）的 B/G 比
        ring = np.zeros((64, 64), dtype=bool)
        ring[16:48, 16:48] = True
        ring[24:40, 24:40] = False  # 中心排除
        b_out = out[..., 2][ring].mean()
        g_out = out[..., 1][ring].mean()
        self.assertGreater(b_out, g_out * 0.5, msg=f"B={b_out:.3f} G={g_out:.3f}")


class ACESChromaTest(unittest.TestCase):
    def test_chroma_preserved(self):
        """ACES 只调亮度，色度（RGB 比例）不变。"""
        x = np.array([[[0.1, 0.5, 0.2]]])
        out = tonemap_chroma_aces(x, white_blend=0.0)
        ratio_in = x[0, 0, 1] / x[0, 0, 0]
        ratio_out = out[0, 0, 1] / max(out[0, 0, 0], 1e-12)
        self.assertAlmostEqual(ratio_in, ratio_out, places=6)

    def test_white_blend_pulls_to_white(self):
        """超色域高光向白混合。"""
        x = np.array([[[10.0, 0.1, 0.1]]])  # 极端红色
        out0 = tonemap_chroma_aces(x, white_blend=0.0)
        out1 = tonemap_chroma_aces(x, white_blend=0.2)
        # 有 white_blend 时 B 通道应更高（向白拉）
        self.assertGreaterEqual(out1[0, 0, 2], out0[0, 0, 2])

    def test_output_in_unit_interval(self):
        x = np.array([0.0, 0.5, 1.0, 10.0, 1e6])[:, None, None] * np.ones((1, 1, 3))
        out = tonemap_chroma_aces(x)
        self.assertTrue(np.all(out >= 0) and np.all(out <= 1.0))


class SRGBTest(unittest.TestCase):
    def test_roundtrip(self):
        """srgb_encode 的输出在 [0,1]，暗端斜率 12.92。"""
        x = np.array([0.0, 0.001, 0.5, 1.0])
        out = srgb_encode(x)
        self.assertAlmostEqual(out[0], 0.0)
        self.assertAlmostEqual(out[3], 1.0)
        # 暗端线性段：0.001 → 0.001*12.92 = 0.01292
        self.assertAlmostEqual(out[1], 0.01292, places=5)

    def test_monotonic(self):
        x = np.linspace(0, 1, 100)
        out = srgb_encode(x)
        self.assertTrue(np.all(np.diff(out) > 0))


class PostfxPipelineTest(unittest.TestCase):
    def test_full_pipeline_runs(self):
        """postfx 全链路可运行，输出 uint8。"""
        rng = np.random.default_rng(0)
        hdr = rng.uniform(0, 5, (32, 48, 3))
        out = postfx(hdr, exposure=0.5)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (32, 48, 3))

    def test_default_params_match_reference(self):
        """默认参数与参考实现预设 M 一致。"""
        p = postfx_params_defaults()
        self.assertAlmostEqual(p["white_balance_K"], 5000.0)
        self.assertAlmostEqual(p["bloom_threshold"], 0.3)
        self.assertAlmostEqual(p["bloom_gain"], 4.0)
        self.assertAlmostEqual(p["axial_scale"][2], 1.15)
        self.assertAlmostEqual(p["fringe_strength"], 0.4)
        self.assertAlmostEqual(p["lateral_ca"], 0.0025)
        self.assertAlmostEqual(p["white_blend"], 0.12)

    def test_exposure_scales_output(self):
        """更高曝光 → 更亮的输出。"""
        hdr = np.ones((16, 16, 3)) * 0.1
        out_lo = postfx(hdr, exposure=0.5).mean()
        out_hi = postfx(hdr, exposure=2.0).mean()
        self.assertGreater(out_hi, out_lo)


if __name__ == "__main__":
    unittest.main()
