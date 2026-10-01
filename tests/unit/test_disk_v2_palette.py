#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 显示链（palette v2.3）单元测试。

S2 详细语义（CIE 表、ln Y、白平衡、cinematic 删除）见
`tests/unit/test_disk_v2_palette_s2.py`；本文件覆盖显示链：
tonemap / gamma / apply_palette 的边界与组合行为。
"""

import sys
import unittest
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from disk_v2.params import DiskV2PaletteParams
from disk_v2.palette import (
    apply_palette,
    blackbody_color,
    gamma_correct,
    render_hdr_to_ldr,
    tonemap,
)


class DiskV2PaletteTest(unittest.TestCase):
    def setUp(self):
        self.params = DiskV2PaletteParams()

    # --- 参数校验 ---

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

    def test_invalid_white_balance_raises(self):
        with self.assertRaises(ValueError):
            DiskV2PaletteParams(white_balance_K=0.0)

    # --- blackbody_color 方向性（值域断言见 _s2）---

    def test_blackbody_color_low_temperature_is_red_dominant(self):
        rgb = np.asarray(blackbody_color(2000.0))
        self.assertGreater(rgb[0], rgb[2])

    def test_blackbody_color_high_temperature_is_blue_dominant(self):
        rgb = np.asarray(blackbody_color(20000.0))
        self.assertGreater(rgb[2], rgb[0])

    def test_blackbody_color_returns_zero_for_zero_temperature(self):
        rgb = np.asarray(blackbody_color(0.0))
        self.assertTrue(np.allclose(rgb, 0.0))

    def test_blackbody_color_works_for_array(self):
        T = np.array([2000.0, 6000.0, 10000.0])
        rgb = np.asarray(blackbody_color(T))
        self.assertEqual(rgb.shape, (3, 3))
        self.assertTrue(np.all(rgb >= 0.0))

    # --- tonemap ---

    def test_tonemap_maps_zero_to_zero(self):
        out = tonemap(np.zeros(5), self.params)
        self.assertTrue(np.allclose(out, 0.0))

    def test_tonemap_maps_positive_input_to_open_unit_interval(self):
        inp = np.array([0.0, 0.1, 1.0, 10.0, 1000.0])
        out = tonemap(inp, self.params)
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out < 1.0))

    def test_tonemap_is_monotonic(self):
        inp = np.linspace(0.0, 100.0, 101)
        out = tonemap(inp, self.params)
        self.assertTrue(np.all(np.diff(out) > 0))

    def test_tonemap_approaches_unity_for_large_input(self):
        out_big = tonemap(np.array([1e6]), self.params)
        self.assertGreater(float(out_big[0]), 0.999)

    def test_tonemap_clips_negative_input_to_zero(self):
        out = tonemap(np.array([-1.0, -0.001]), self.params)
        self.assertTrue(np.allclose(out, 0.0))

    # --- gamma_correct ---

    def test_gamma_correct_is_identity_at_endpoints(self):
        out = gamma_correct(np.array([0.0, 1.0]), self.params)
        self.assertAlmostEqual(float(out[0]), 0.0)
        self.assertAlmostEqual(float(out[1]), 1.0)

    def test_gamma_correct_inverse_via_power(self):
        lin = np.array([0.2, 0.5])
        out = gamma_correct(lin, self.params)
        recovered = np.power(np.clip(out, 0, 1), self.params.gamma)
        self.assertTrue(np.allclose(recovered, lin, atol=1e-12))

    def test_gamma_correct_brightens_midtones(self):
        out = gamma_correct(np.array([0.5]), self.params)
        self.assertGreater(float(out[0]), 0.5)

    def test_gamma_correct_handles_negative_input(self):
        out = gamma_correct(np.array([-0.5]), self.params)
        self.assertEqual(float(out[0]), 0.0)

    # --- 显示链组合 ---

    def test_render_hdr_to_ldr_is_composition(self):
        hdr = np.array([0.0, 0.5, 2.0])
        composed = render_hdr_to_ldr(hdr, self.params)
        manual = gamma_correct(tonemap(hdr, self.params), self.params)
        self.assertTrue(np.allclose(composed, manual))

    def test_render_hdr_to_ldr_output_in_unit_interval(self):
        hdr = np.array([0.0, 1.0, 10.0, 1e5])
        out = render_hdr_to_ldr(hdr, self.params)
        self.assertTrue(np.all(out >= 0.0) and np.all(out <= 1.0))

    # --- apply_palette ---

    def test_apply_palette_shape(self):
        intensity = np.array([1.0, 2.0, 3.0])
        T = np.array([3000.0, 4500.0, 6000.0])
        out = np.asarray(apply_palette(intensity, T, self.params))
        self.assertEqual(out.shape, (3, 3))

    def test_apply_palette_zero_intensity_returns_zero(self):
        out = np.asarray(apply_palette(0.0, 4500.0, self.params))
        self.assertTrue(np.allclose(out, 0.0))

    def test_apply_palette_color_scales_linearly_with_intensity(self):
        T = 4500.0
        out1 = np.asarray(apply_palette(1.0, T, self.params))
        out2 = np.asarray(apply_palette(2.0, T, self.params))
        self.assertTrue(np.allclose(out2, 2.0 * out1))


if __name__ == "__main__":
    unittest.main()
