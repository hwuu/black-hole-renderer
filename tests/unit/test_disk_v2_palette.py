#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 调色与色调映射单元测试（v2.1 新增）。

覆盖：

- `blackbody_color` 在不同温度区间的方向性（高温偏蓝白、低温偏红）。
- `cinematic_color` 相对 physical 模式的饱和度差异。
- `tonemap` 把 `[0, ∞)` 输入压到 `[0, 1)`、对 0 输出 0、对极大输入逼近 1。
- `gamma_correct` 是 `x → x^(1/gamma)`、与逆运算近似自洽。
- `render_hdr_to_ldr` 组合一致。
- `apply_palette` 形状 + 数值。
- `DiskV2PaletteParams` 的参数校验。
"""

import unittest

import numpy as np

from disk_v2.palette import (
    apply_palette,
    blackbody_color,
    cinematic_color,
    gamma_correct,
    palette_color,
    render_hdr_to_ldr,
    tonemap,
)
from disk_v2.params import DiskV2PaletteParams


class DiskV2PaletteTest(unittest.TestCase):
    def setUp(self):
        self.physical = DiskV2PaletteParams(palette_mode="physical")
        self.cinematic = DiskV2PaletteParams(palette_mode="cinematic")

    # --- 参数校验 ---

    def test_invalid_palette_mode_raises(self):
        with self.assertRaises(ValueError):
            DiskV2PaletteParams(palette_mode="cinematicX")

    def test_aces_is_not_yet_implemented(self):
        with self.assertRaises(NotImplementedError):
            DiskV2PaletteParams(tonemap_mode="aces")

    def test_invalid_tonemap_mode_raises(self):
        with self.assertRaises(ValueError):
            DiskV2PaletteParams(tonemap_mode="foo")

    def test_invalid_gamma_raises(self):
        with self.assertRaises(ValueError):
            DiskV2PaletteParams(gamma=0.0)

    def test_invalid_opacity_scale_raises(self):
        with self.assertRaises(ValueError):
            DiskV2PaletteParams(opacity_scale=0.0)

    def test_invalid_cinematic_warm_shift_raises(self):
        with self.assertRaises(ValueError):
            DiskV2PaletteParams(cinematic_warm_shift=1.5)

    # --- blackbody_color 方向性 ---

    def test_blackbody_color_low_temperature_is_red_dominant(self):
        rgb = blackbody_color(2000.0)  # 冷红
        self.assertGreater(rgb[0], rgb[2])  # R > B

    def test_blackbody_color_high_temperature_is_blue_dominant(self):
        rgb = blackbody_color(20000.0)  # 蓝白
        # 高温下 B 应接近 1（饱和），R 由公式 1.292936·(t-60)^(-0.13) 在 200 处
        # 衰减到 ≈ 0.67，所以 B > R。
        self.assertGreater(rgb[2], rgb[0])
        self.assertGreater(rgb[2], 0.95)

    def test_blackbody_color_returns_zero_for_zero_temperature(self):
        rgb = blackbody_color(0.0)
        self.assertTrue(np.allclose(rgb, 0.0))

    def test_blackbody_color_works_for_array(self):
        T = np.array([2000.0, 6000.0, 10000.0])
        rgb = blackbody_color(T)
        self.assertEqual(rgb.shape, (3, 3))
        self.assertTrue(np.all(rgb >= 0.0))
        self.assertTrue(np.all(rgb <= 1.0))

    def test_blackbody_color_at_6500k_is_near_white(self):
        """6500K 是日光白点，三通道应接近 1。"""
        rgb = blackbody_color(6500.0)
        # 不要求严格 1.0；测每通道 > 0.7 已经能区分"白"和其他色。
        self.assertGreater(min(rgb), 0.7)

    # --- cinematic_color ---

    def test_cinematic_color_increases_saturation_vs_physical(self):
        """对中等物理温度，cinematic 相对 physical 应有更大的 R-B 通道差。"""
        T = 5.0e6
        rgb_phys = palette_color(T, self.physical, T_peak_K=1.0e7)
        rgb_cine = palette_color(T, self.cinematic, T_peak_K=1.0e7)
        diff_phys = rgb_phys[0] - rgb_phys[2]
        diff_cine = rgb_cine[0] - rgb_cine[2]
        self.assertGreater(diff_cine, diff_phys)

    def test_cinematic_color_at_1e7_not_saturated_white(self):
        """cinematic 重映射后，1e7 K 不应再变成纯白。"""
        rgb = cinematic_color(1.0e7, self.cinematic, T_peak_K=1.0e7)
        self.assertTrue(np.all(rgb >= 0.0))
        self.assertTrue(np.all(rgb <= 1.0))
        self.assertLess(float(np.max(rgb)), 0.98)
        self.assertGreater(float(np.std(rgb)), 0.01)

    def test_cinematic_color_returns_clamped_rgb(self):
        T = np.array([2000.0, 4000.0, 6500.0, 10000.0])
        rgb = cinematic_color(T, self.cinematic)
        self.assertTrue(np.all(rgb >= 0.0))
        self.assertTrue(np.all(rgb <= 1.0))

    def test_palette_color_dispatches_by_mode(self):
        rgb_phys = palette_color(5.0e6, self.physical, T_peak_K=1.0e7)
        rgb_cine = palette_color(5.0e6, self.cinematic, T_peak_K=1.0e7)
        # 两个应该不同（cinematic 改了饱和度和暖色偏移）。
        self.assertFalse(np.allclose(rgb_phys, rgb_cine))

    # --- tonemap ---

    def test_tonemap_maps_zero_to_zero(self):
        out = tonemap(np.zeros(5), self.physical)
        self.assertTrue(np.allclose(out, 0.0))

    def test_tonemap_maps_positive_input_to_open_unit_interval(self):
        inp = np.array([0.1, 1.0, 10.0, 100.0, 1e6])
        out = tonemap(inp, self.physical)
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out < 1.0))

    def test_tonemap_is_monotonic(self):
        inp = np.linspace(0.0, 100.0, 200)
        out = tonemap(inp, self.physical)
        self.assertTrue(np.all(np.diff(out) >= 0.0))

    def test_tonemap_approaches_unity_for_large_input(self):
        out_big = tonemap(np.array([1e6]), self.physical)
        self.assertGreater(float(out_big[0]), 0.999)
        self.assertLess(float(out_big[0]), 1.0)

    def test_tonemap_clips_negative_input_to_zero(self):
        out = tonemap(np.array([-1.0, -0.5]), self.physical)
        # 负输入应被 clip 到 0 后再 tonemap，结果为 0。
        self.assertTrue(np.allclose(out, 0.0))

    # --- gamma_correct ---

    def test_gamma_correct_is_identity_at_endpoints(self):
        out0 = gamma_correct(np.array([0.0]), self.physical)
        out1 = gamma_correct(np.array([1.0]), self.physical)
        self.assertAlmostEqual(float(out0[0]), 0.0, places=8)
        self.assertAlmostEqual(float(out1[0]), 1.0, places=8)

    def test_gamma_correct_inverse_via_power(self):
        """gamma_correct(x)^gamma 应 ≈ x，对所有 x ∈ [0, 1]。"""
        x = np.linspace(0.0, 1.0, 32)
        y = gamma_correct(x, self.physical)
        x_back = np.power(y, self.physical.gamma)
        self.assertTrue(np.allclose(x, x_back, atol=1e-10))

    def test_gamma_correct_brightens_midtones(self):
        """伽马校正应让 0.5 输入变得更亮（因为 0.5^(1/2.2) > 0.5）。"""
        out = gamma_correct(np.array([0.5]), self.physical)
        self.assertGreater(float(out[0]), 0.5)

    def test_gamma_correct_handles_negative_input(self):
        out = gamma_correct(np.array([-0.1]), self.physical)
        self.assertAlmostEqual(float(out[0]), 0.0, places=12)

    # --- render_hdr_to_ldr 组合 ---

    def test_render_hdr_to_ldr_is_composition(self):
        x = np.array([0.0, 0.5, 1.0, 10.0])
        manual = gamma_correct(tonemap(x, self.physical), self.physical)
        composed = render_hdr_to_ldr(x, self.physical)
        self.assertTrue(np.allclose(manual, composed))

    def test_render_hdr_to_ldr_output_in_unit_interval(self):
        x = np.array([0.0, 1e-3, 1.0, 1e6])
        out = render_hdr_to_ldr(x, self.physical)
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))

    # --- apply_palette ---

    def test_apply_palette_shape(self):
        T = np.array([3000.0, 5000.0, 8000.0])
        intensity = np.array([0.5, 1.0, 2.0])
        out = apply_palette(intensity, T, self.physical)
        self.assertEqual(out.shape, (3, 3))

    def test_apply_palette_zero_intensity_returns_zero(self):
        T = np.array([3000.0, 5000.0])
        intensity = np.zeros_like(T)
        out = apply_palette(intensity, T, self.physical)
        self.assertTrue(np.allclose(out, 0.0))

    def test_apply_palette_color_scales_linearly_with_intensity(self):
        T = np.full(4, 5000.0)
        intensity = np.array([0.1, 0.5, 1.0, 2.0])
        out = apply_palette(intensity, T, self.physical)
        # 颜色比例固定，强度线性。
        ratio_01 = out[1] / out[0]
        ratio_12 = out[2] / out[1]
        # 比例应分别为 intensity[1]/intensity[0] = 5, intensity[2]/intensity[1] = 2。
        self.assertTrue(np.allclose(ratio_01, 5.0, atol=1e-8))
        self.assertTrue(np.allclose(ratio_12, 2.0, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
