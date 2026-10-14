#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""场景旋钮轨道（scene_anim）单元测试。

覆盖：
- `load_scene_anim` 的读取与校验（未知旋钮名、时刻非严格递增、值越界、空轨道等报错）
- `SceneAnim.values_at` 的求值：关键帧精确命中、端值保持、五次缓动中段插值
- `knob_values_at` 的轨道优先于 CLI 默认值合并
"""

import unittest

import numpy as np

from src.v2.camera_path import _smootherstep
from src.v2.scene_anim import KNOB_NAMES, SceneAnim, knob_values_at, load_scene_anim


class LoadSceneAnimTest(unittest.TestCase):
    def test_missing_block_returns_empty(self):
        """无 scene_anim 块 → 空轨道，求值结果为空 dict。"""
        anim = load_scene_anim({"duration": 10.0})
        self.assertEqual(anim.values_at(0.0), {})

    def test_loads_all_knobs(self):
        anim = load_scene_anim({"scene_anim": {
            "gravity_scale": [[0.0, 1.0], [4.0, 0.0]],
            "doppler_scale": [[0.0, 0.0]],
            "disk_weight": [[2.0, 0.0], [6.0, 1.0]],
            "disk_radius_scale": [[0.0, 0.06], [6.0, 1.0]],
        }})
        self.assertEqual(set(anim.values_at(0.0)), set(KNOB_NAMES))

    def test_unknown_knob_name_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            load_scene_anim({"scene_anim": {"gravity": [[0.0, 1.0]]}})
        self.assertIn("gravity", str(ctx.exception))

    def test_times_must_strictly_increase(self):
        with self.assertRaises(ValueError) as ctx:
            load_scene_anim({"scene_anim": {"gravity_scale": [[0.0, 1.0], [0.0, 0.5]]}})
        self.assertIn("gravity_scale", str(ctx.exception))
        with self.assertRaises(ValueError):
            load_scene_anim({"scene_anim": {"gravity_scale": [[4.0, 1.0], [2.0, 0.5]]}})

    def test_value_out_of_range_rejected(self):
        for bad in (1.5, -0.1):
            with self.assertRaises(ValueError):
                load_scene_anim({"scene_anim": {"gravity_scale": [[0.0, bad]]}})

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            load_scene_anim({"scene_anim": {"gravity_scale": [[float("nan"), 1.0]]}})
        with self.assertRaises(ValueError):
            load_scene_anim({"scene_anim": {"gravity_scale": [[0.0, float("inf")]]}})

    def test_empty_track_rejected(self):
        with self.assertRaises(ValueError):
            load_scene_anim({"scene_anim": {"gravity_scale": []}})

    def test_malformed_entries_rejected(self):
        for bad in ("x", [0.0], [[0.0, 1.0, 2.0]], [0.0, 1.0]):
            with self.assertRaises(ValueError):
                load_scene_anim({"scene_anim": {"gravity_scale": bad}})

    def test_single_keyframe_is_constant(self):
        anim = load_scene_anim({"scene_anim": {"doppler_scale": [[5.0, 0.3]]}})
        for t in (-1.0, 0.0, 5.0, 100.0):
            self.assertEqual(anim.values_at(t), {"doppler_scale": 0.3})


class EvalTest(unittest.TestCase):
    def _anim(self):
        return load_scene_anim({"scene_anim": {
            "gravity_scale": [[0.0, 1.0], [4.0, 0.0], [8.0, 1.0]],
        }})

    def test_keyframe_values_exact(self):
        anim = self._anim()
        self.assertEqual(anim.values_at(0.0)["gravity_scale"], 1.0)
        self.assertEqual(anim.values_at(4.0)["gravity_scale"], 0.0)
        self.assertEqual(anim.values_at(8.0)["gravity_scale"], 1.0)

    def test_hold_before_first_and_after_last(self):
        anim = self._anim()
        self.assertEqual(anim.values_at(-3.0)["gravity_scale"], 1.0)
        self.assertEqual(anim.values_at(99.0)["gravity_scale"], 1.0)

    def test_midpoint_interpolates(self):
        anim = self._anim()
        mid = anim.values_at(2.0)["gravity_scale"]
        self.assertGreater(mid, 0.0)
        self.assertLess(mid, 1.0)

    def test_quintic_matches_smootherstep(self):
        """关键帧之间为五次缓动：v(t) = v0 + (v1 − v0)·ss(u)，ss 与摇镜的 `_smootherstep` 一致。"""
        anim = load_scene_anim({"scene_anim": {"gravity_scale": [[0.0, 1.0], [4.0, 0.0]]}})
        for t in (0.5, 1.0, 2.5, 3.9):
            u = np.float64(t / 4.0)
            expected = 1.0 + (0.0 - 1.0) * float(_smootherstep(u))
            self.assertAlmostEqual(anim.values_at(t)["gravity_scale"], expected, places=12)

    def test_monotonic_within_segment(self):
        anim = self._anim()
        samples = [anim.values_at(t)["gravity_scale"] for t in np.linspace(0.0, 4.0, 41)]
        self.assertTrue(np.all(np.diff(samples) <= 1e-12))


class MergeTest(unittest.TestCase):
    def test_track_overrides_default(self):
        anim = load_scene_anim({"scene_anim": {"gravity_scale": [[0.0, 0.25]]}})
        values = knob_values_at(anim, 3.0, {"gravity_scale": 1.0, "doppler_scale": 0.0, "disk_weight": 1.0})
        self.assertEqual(values, {"gravity_scale": 0.25, "doppler_scale": 0.0, "disk_weight": 1.0})

    def test_empty_anim_keeps_defaults(self):
        anim = SceneAnim(tracks={})
        defaults = {"gravity_scale": 1.0, "doppler_scale": 0.0, "disk_weight": 1.0}
        self.assertEqual(knob_values_at(anim, 7.0, dict(defaults)), defaults)


if __name__ == "__main__":
    unittest.main()
