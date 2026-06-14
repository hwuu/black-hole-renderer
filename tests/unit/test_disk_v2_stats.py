#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 渲染统计单元测试。"""

import unittest

import numpy as np

from disk_v2.stats import compute_render_stats, hdr_luminance, ldr_luminance


class DiskV2StatsTest(unittest.TestCase):
    def test_hdr_luminance_weights(self):
        rgb = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.assertAlmostEqual(float(hdr_luminance(rgb)), 0.2126, places=6)

    def test_compute_render_stats_white_ratio(self):
        hdr = np.zeros((4, 4, 3), dtype=np.float32)
        ldr = np.zeros((4, 4, 3), dtype=np.uint8)
        ldr[0, 0] = [255, 255, 255]
        stats = compute_render_stats(hdr, ldr)
        self.assertAlmostEqual(stats.ldr_white_ratio, 1.0 / 16.0, places=6)
        self.assertAlmostEqual(stats.hdr_min, 0.0, places=9)

    def test_compute_render_stats_records_white_point(self):
        hdr = np.ones((2, 2, 3), dtype=np.float32)
        ldr = np.full((2, 2, 3), 128, dtype=np.uint8)
        stats = compute_render_stats(hdr, ldr, white_point=2.5)
        self.assertAlmostEqual(stats.white_point, 2.5, places=9)
        self.assertIn("white_point=2.5", stats.format_summary())

    def test_save_image_uint8_not_blown_to_white(self):
        """回归：uint8 LDR 不应被 save_image 二次 clip 成全白。"""
        import tempfile
        from PIL import Image

        from render import save_image

        mid_gray = np.full((8, 8, 3), 128, dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            save_image(mid_gray, tmp.name)
            saved = np.array(Image.open(tmp.name))
        self.assertTrue(np.allclose(saved, 128, atol=1))


if __name__ == "__main__":
    unittest.main()
