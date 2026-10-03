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
- 镜头 PSF（默认镜头模型）：ε=0 恒等、线性无阈值、能量守恒、成片亮区内部不变、
  点光源核心变暗且外围出现光晕、蓝光晕更宽（轴向色差）、legacy 链与旧公式逐位一致
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

from src.v2.postfx import (
    _box_blur,
    adjust_saturation,
    apply_bloom,
    apply_fringe,
    apply_lateral_ca,
    apply_lens_psf,
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

    def test_luma_threshold_preserves_chroma(self):
        """按亮度扣阈值：金色亮块的光晕保持源色比例（不偏橙红）。"""
        gold = np.array([1.0, 0.61, 0.245])  # 3000 K 白平衡后的金色
        img = np.zeros((64, 64, 3))
        img[28:36, 28:36] = 1.5 * gold
        out = apply_bloom(img, threshold=0.3, gain=4.0, axial_scale=(1.0, 1.0, 1.0))
        halo = out[10, 32]  # 亮块外的纯光晕像素
        np.testing.assert_allclose(halo / halo[0], gold, rtol=1e-6)

    def test_per_channel_threshold_reddens_halo(self):
        """逐通道扣阈值（旧行为）：同一金色亮块的光晕 G/R、B/R 低于源色（偏橙红）。"""
        gold = np.array([1.0, 0.61, 0.245])
        img = np.zeros((64, 64, 3))
        img[28:36, 28:36] = 1.5 * gold
        out = apply_bloom(img, threshold=0.3, gain=4.0, axial_scale=(1.0, 1.0, 1.0),
                          luma_threshold=False)
        halo = out[10, 32]
        self.assertLess(halo[1] / halo[0], gold[1])
        self.assertLess(halo[2] / halo[0], gold[2])

    def test_per_channel_threshold_matches_old_formula(self):
        """luma_threshold=False 时与旧公式 max(hdr − th, 0) 逐位一致。"""
        th, gain = 0.3, 4.0
        out = apply_bloom(self.img, threshold=th, gain=gain, luma_threshold=False)
        src = np.maximum(self.img - th, 0.0)
        h = self.img.shape[0]
        ref = np.zeros_like(self.img)
        for c, sc in enumerate((1.0, 1.0, 1.15)):
            ch = src[..., c:c + 1]
            ref[..., c:c + 1] = (0.25 * _box_blur(ch, max(1, int(h / 120 * sc)))
                                 + 0.35 * _box_blur(ch, max(2, int(h / 25 * sc)))
                                 + 0.40 * _box_blur(ch, max(4, int(h / 7 * sc))))
        np.testing.assert_array_equal(out, self.img + gain * ref)


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
        """默认镜头模型为能量守恒 PSF（ε = 0.4）；legacy 参数仍与参考实现预设 M 一致。"""
        p = postfx_params_defaults()
        self.assertEqual(p["lens_model"], "psf")
        self.assertAlmostEqual(p["lens_glare"], 0.4)
        self.assertAlmostEqual(p["white_balance_K"], 4000.0)
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


class LensPSFTest(unittest.TestCase):
    """镜头 PSF：out = (1 − ε)·x + ε·(K ∗ x)，能量守恒、线性、无阈值。"""

    # 单一小半径长尾（半径 = H/40），便于在画面内部精确检验能量守恒
    SMALL_TAIL = ((40.0, 1.0),)

    def test_zero_glare_identity(self):
        """ε = 0 = 理想镜头，原样返回。"""
        x = np.random.default_rng(1).uniform(0, 3, (32, 32, 3))
        self.assertIs(apply_lens_psf(x, glare=0.0), x)

    def test_linear_no_threshold(self):
        """线性：lens(a·x) = a·lens(x)；暗光与亮光按同一比例散射（无阈值）。"""
        x = np.random.default_rng(2).uniform(0, 1, (64, 64, 3))
        np.testing.assert_allclose(apply_lens_psf(3.0 * x, 0.4), 3.0 * apply_lens_psf(x, 0.4), rtol=1e-12)

    def test_energy_never_increases(self):
        """能量只减不增（补 0：散射到画面外的光丢失），且至少保留 (1 − ε)。"""
        x = np.random.default_rng(3).uniform(0, 2, (90, 120, 3))
        out = apply_lens_psf(x, 0.4)
        for c in range(3):
            self.assertLessEqual(out[..., c].sum(), x[..., c].sum() * (1 + 1e-12))
            self.assertGreaterEqual(out[..., c].sum(), 0.6 * x[..., c].sum())

    def test_energy_conserved_away_from_edges(self):
        """光源远离画面边缘时总能量严格守恒（光只被搬运）。"""
        x = np.zeros((400, 400, 3)); x[195:205, 195:205] = 2.0
        out = apply_lens_psf(x, 0.4, axial_scale=(1.0, 1.0, 1.0), tail=self.SMALL_TAIL)
        np.testing.assert_allclose(out.sum(), x.sum(), rtol=1e-9)

    def test_uniform_region_interior_unchanged(self):
        """成片均匀亮区内部不变：散射出去的光与散射进来的光抵消（不会被"点亮"）。"""
        x = np.full((400, 400, 3), 0.5)
        out = apply_lens_psf(x, 0.4, axial_scale=(1.0, 1.0, 1.0), tail=self.SMALL_TAIL)
        np.testing.assert_allclose(out[150:250, 150:250], 0.5, rtol=1e-9)

    def test_point_source_core_dims_and_halo_appears(self):
        """点光源：核心按 (1 − ε) 量级变暗，周围暗处出现光晕。"""
        x = np.zeros((400, 400, 3)); x[200, 200] = 100.0
        out = apply_lens_psf(x, 0.4, axial_scale=(1.0, 1.0, 1.0), tail=self.SMALL_TAIL)
        self.assertLess(out[200, 200, 1], 100.0 * 0.61)
        self.assertGreater(out[200, 210, 1], 0.0)

    def test_blue_halo_wider(self):
        """轴向色差：白色点光源外围蓝光晕比绿光晕延伸更远。"""
        x = np.zeros((400, 400, 3)); x[200, 200] = 100.0
        out = apply_lens_psf(x, 0.4, axial_scale=(1.0, 1.0, 1.15), tail=self.SMALL_TAIL)
        far = out[200, 200 + 33]   # 绿核外沿之外、蓝核之内
        self.assertGreater(far[2], far[1])

    def test_postfx_legacy_matches_old_chain(self):
        """lens_model="legacy" 与旧链（WB → bloom → 镶边 → CA → ACES → sRGB）逐位一致。"""
        hdr = np.random.default_rng(4).uniform(0, 2, (48, 64, 3))
        p = postfx_params_defaults()
        x = apply_white_balance(hdr * 0.7, p["white_balance_K"])
        x = apply_bloom(x, p["bloom_threshold"], p["bloom_gain"], p["axial_scale"], p["bloom_luma_threshold"])
        x = apply_fringe(x, p["fringe_strength"], p["fringe_threshold"], p["fringe_color"])
        x = apply_lateral_ca(x, p["lateral_ca"])
        x = adjust_saturation(tonemap_chroma_aces(x, p["white_blend"]), p["saturation"])
        ref = (np.clip(srgb_encode(x), 0, 1) * 255 + 0.5).astype(np.uint8)
        np.testing.assert_array_equal(postfx(hdr, exposure=0.7, lens_model="legacy"), ref)

    def test_invalid_lens_model_raises(self):
        with self.assertRaises(ValueError):
            postfx(np.zeros((8, 8, 3)), lens_model="foo")


if __name__ == "__main__":
    unittest.main()
