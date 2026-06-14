#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2 渲染参数 preset 测试（resolve_v2_render_options）。

修 B：撤销 interstellar preset 把 `lum_power=4.0` 改为 `2.5` 的偷改，
让 plan Step 4 严格 g^4 物理在 cinematic 主验收里保持成立。
"""

import argparse
import unittest


def _make_args(**overrides) -> argparse.Namespace:
    """构造一组完整的 args，模拟 `parse_args()` 输出，便于测试 preset。

    注意：bloom 三个参数 CLI 默认是 None（修 C），让 preset 能区分
    "用户未指定"与"用户显式传 0"。
    """
    defaults = {
        "v2_visual_preset": None,
        "v2_auto_exposure": False,
        "v2_bloom_threshold": None,
        "v2_bloom_intensity": None,
        "v2_bloom_radius": None,
        "v2_palette_mode": "cinematic",
        "v2_opacity_scale": 0.55,
        "v2_emission_scale": 1.0,
        "v2_lum_power": 4.0,
        "v2_volume_samples": 16,
        "v2_r_max": None,
        "r_max": 10.0,
        "v2_white_point_percentile": 99.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class ResolveV2RenderOptionsTest(unittest.TestCase):
    def test_no_preset_keeps_user_values(self):
        from render import resolve_v2_render_options

        args = _make_args(v2_visual_preset=None)
        opts = resolve_v2_render_options(args)
        # 无 preset 时所有值直接来自 args，bloom None 回落到"无 preset 默认"
        self.assertEqual(opts["lum_power"], 4.0)
        self.assertEqual(opts["palette_mode"], "cinematic")
        self.assertFalse(opts["auto_exposure"])
        self.assertEqual(opts["bloom_intensity"], 0.0)  # None → 0 (默认关闭)
        self.assertEqual(opts["bloom_threshold"], 1.0)  # None → 1.0
        self.assertEqual(opts["bloom_radius"], 4.0)     # None → 4.0

    # --- 修 B：interstellar preset 行为 ---

    def test_interstellar_preset_keeps_lum_power_at_four(self):
        """修 B：interstellar preset 不能覆盖 lum_power。

        Plan Step 4 line 176 要求 cinematic 主链使用严格 g^4 强度变换。
        D3 后曝光 reference 物理可控，HDR 由 Reinhard 自然压缩，不再需要
        把 lum_power 降到 2.5 来防止单帧饱和——降到 2.5 会让多普勒视觉
        显著性减少约 14 倍（6^4 / 6^2.5 ≈ 14.7）。
        """
        from render import resolve_v2_render_options

        # 用户未显式改 lum_power（默认 4.0），preset 应当保持 4.0。
        args = _make_args(v2_visual_preset="interstellar", v2_lum_power=4.0)
        opts = resolve_v2_render_options(args)
        self.assertEqual(
            opts["lum_power"], 4.0,
            "interstellar preset 不应再把 lum_power 降到 2.5；plan Step 4 严格 g^4 物理"
        )

    def test_interstellar_preset_enables_auto_exposure(self):
        """interstellar preset 默认开启 auto_exposure（D3 reference 路径才能用）。"""
        from render import resolve_v2_render_options

        args = _make_args(v2_visual_preset="interstellar")
        opts = resolve_v2_render_options(args)
        self.assertTrue(opts["auto_exposure"])

    def test_interstellar_preset_forces_cinematic_palette(self):
        """interstellar preset 必须用 cinematic palette，否则可见色温映射失效。"""
        from render import resolve_v2_render_options

        args = _make_args(v2_visual_preset="interstellar", v2_palette_mode="physical")
        opts = resolve_v2_render_options(args)
        self.assertEqual(opts["palette_mode"], "cinematic")

    def test_interstellar_preset_respects_user_lum_power_override(self):
        """显式 CLI `--v2_lum_power 3` 应优先于 preset，便于调试。"""
        from render import resolve_v2_render_options

        # 用户显式传 3.0
        args = _make_args(v2_visual_preset="interstellar", v2_lum_power=3.0)
        opts = resolve_v2_render_options(args)
        # 因为非默认 4.0，preset 不再覆盖（即使 preset 想覆盖也得让用户优先）
        self.assertEqual(opts["lum_power"], 3.0)

    # --- 修 C：interstellar preset bloom 配方 ---

    def test_interstellar_preset_sets_bloom_recipe_when_unspecified(self):
        """修 C：用户未指定 bloom 时，interstellar preset 应给出合理 bloom 配方。

        D3 reference 路径下 HDR max ≈ 0.025；旧默认 (threshold=0.15, intensity=0.45)
        完全过滤掉 bloom，让 acceptance 主验收的 _bloom / _no_bloom 输出完全相同。
        新配方 (threshold=5e-4, intensity=1.5, radius=8) 让 bloom 真正起作用。
        """
        from render import resolve_v2_render_options

        # 用户未指定 bloom（CLI 默认 None）
        args = _make_args(v2_visual_preset="interstellar")
        opts = resolve_v2_render_options(args)
        self.assertEqual(opts["bloom_intensity"], 1.5)
        self.assertEqual(opts["bloom_threshold"], 5e-4)
        self.assertEqual(opts["bloom_radius"], 8.0)

    def test_interstellar_preset_respects_user_bloom_intensity_zero(self):
        """修 C：用户显式 `--v2_bloom_intensity 0` 必须能覆盖 preset 的 1.5。

        旧的 `if args.v2_bloom_intensity == 0.0` 判定无法区分"用户显式 0"
        与"用户未指定"，导致用户关 bloom 的请求被 preset 默默忽略。
        新设计：CLI 默认 None，preset 用 `is None` 判定。
        """
        from render import resolve_v2_render_options

        # 用户显式传 0.0 关闭 bloom
        args = _make_args(v2_visual_preset="interstellar", v2_bloom_intensity=0.0)
        opts = resolve_v2_render_options(args)
        self.assertEqual(
            opts["bloom_intensity"], 0.0,
            "用户显式 `--v2_bloom_intensity 0` 必须能覆盖 preset 默认"
        )

    def test_interstellar_preset_respects_user_bloom_threshold_override(self):
        """修 C：用户显式 `--v2_bloom_threshold X` 应优先于 preset。"""
        from render import resolve_v2_render_options

        args = _make_args(v2_visual_preset="interstellar", v2_bloom_threshold=0.01)
        opts = resolve_v2_render_options(args)
        self.assertEqual(opts["bloom_threshold"], 0.01)


if __name__ == "__main__":
    unittest.main()
