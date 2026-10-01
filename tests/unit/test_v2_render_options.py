#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2 渲染选项解析测试（v2.3 S2：--v2_visual_preset / --v2_palette_mode 已删除）。

覆盖：
- 默认值：bloom 三参数 None → 默认配方（关闭 / 阈值 1.0 / 半径 4.0）
- 显式 CLI 值优先于默认
- resolve 输出不再包含 palette_mode / v2_visual_preset
"""

import argparse
import unittest


def _make_args(**overrides) -> argparse.Namespace:
    """构造与 `parse_args()` 等价的 args 命名空间。"""
    defaults = {
        "v2_auto_exposure": False,
        "v2_bloom_threshold": None,
        "v2_bloom_intensity": None,
        "v2_bloom_radius": None,
        "v2_tonemap_mode": None,
        "v2_opacity_scale": 0.55,
        "v2_emission_scale": 1.0,
        "v2_lum_power": 4.0,
        "v2_volume_samples": 32,
        "v2_r_max": None,
        "r_max": 10.0,
        "v2_white_point_percentile": 99.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class ResolveV2RenderOptionsTest(unittest.TestCase):
    def test_defaults(self):
        """未指定时：bloom 关闭（None → 0.0），阈值 1.0，半径 4.0。"""
        from render import resolve_v2_render_options

        opts = resolve_v2_render_options(_make_args())
        self.assertEqual(opts["lum_power"], 4.0)
        self.assertFalse(opts["auto_exposure"])
        self.assertEqual(opts["bloom_intensity"], 0.0)
        self.assertEqual(opts["bloom_threshold"], 1.0)
        self.assertEqual(opts["bloom_radius"], 4.0)
        self.assertEqual(opts["tonemap_mode"], "reinhard")
        self.assertEqual(opts["opacity_scale"], 0.55)
        self.assertEqual(opts["emission_scale"], 1.0)
        self.assertEqual(opts["volume_samples"], 32)
        self.assertEqual(opts["r_max"], 10.0)
        self.assertEqual(opts["white_point_percentile"], 99.0)

    def test_explicit_values_win(self):
        """显式 CLI 值（含 0）优先于默认。"""
        from render import resolve_v2_render_options

        opts = resolve_v2_render_options(_make_args(
            v2_bloom_intensity=0.5,
            v2_bloom_threshold=0.2,
            v2_bloom_radius=9.0,
            v2_opacity_scale=2.0,
            v2_lum_power=3.0,
            v2_r_max=25.0,
        ))
        self.assertEqual(opts["bloom_intensity"], 0.5)
        self.assertEqual(opts["bloom_threshold"], 0.2)
        self.assertEqual(opts["bloom_radius"], 9.0)
        self.assertEqual(opts["opacity_scale"], 2.0)
        self.assertEqual(opts["lum_power"], 3.0)
        self.assertEqual(opts["r_max"], 25.0)

    def test_preset_and_palette_mode_removed(self):
        """v2.3：opts 不再包含 palette_mode；Namespace 不再有 v2_visual_preset。"""
        from render import resolve_v2_render_options

        opts = resolve_v2_render_options(_make_args())
        self.assertNotIn("palette_mode", opts)
        self.assertFalse(hasattr(_make_args(), "v2_visual_preset"))
        self.assertFalse(hasattr(_make_args(), "v2_palette_mode"))


if __name__ == "__main__":
    unittest.main()
