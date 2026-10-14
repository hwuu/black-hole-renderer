#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""制片层时间线（timeline）单元测试。

覆盖：时长公式（含无旁白镜头、最短时长）、帧网格量化、交叉淡化重叠、
句子时刻、render 镜头路径时长校验、fade 约束与 sfx 越界报错、JSON 序列化。
"""

import json
import os
import tempfile
import unittest

from src.film.manifest import load_manifest
from src.film.timeline import build_timeline, load_timeline
from tests.unit._film_fixture import valid_spec, write_manifest

FPS = 30


def _durations(spec):
    """按清单构造一组等长旁白时长（每句 2.0 s）。"""
    return {(shot["id"], k): 2.0
            for shot in spec["shots"] for k in range(len(shot.get("narration", [])))}


class BuildTimelineTest(unittest.TestCase):
    def _timeline(self, spec, durations, shot_paths=None):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(write_manifest(tmp, spec=spec, shot_paths=shot_paths))
            return build_timeline(manifest, durations, log=lambda msg: None)

    def test_duration_formula_and_quantization(self):
        """D = max(min_duration, lead + Σs + gap·(n−1) + tail)，向上取整到帧。"""
        spec = valid_spec()
        durations = {("s01", 0): 2.0, ("s01", 1): 3.0, ("s02", 0): 1.5}
        timeline = self._timeline(spec, durations, shot_paths={"shots/s01.json": 30.0})
        s01, s02 = timeline.shots
        # s01: 0.4 + (2.0 + 3.0) + 0.25 + 0.6 = 6.25 s → 187.5 帧向上取整 = 188 帧
        self.assertAlmostEqual(s01.duration, 188 / FPS, places=9)
        # s02: 0.4 + 1.5 + 0.6 = 2.5 s = 75 帧
        self.assertAlmostEqual(s02.duration, 75 / FPS, places=9)

    def test_min_duration_wins(self):
        spec = valid_spec()
        spec["shots"][0]["min_duration"] = 10.0
        timeline = self._timeline(spec, _durations(spec), shot_paths={"shots/s01.json": 30.0})
        self.assertAlmostEqual(timeline.shot("s01").duration, 10.0, places=9)

    def test_shot_without_narration(self):
        spec = valid_spec()
        spec["shots"].append({"id": "s03", "kind": "diagram",
                              "scene": "diagrams/x.py:X", "min_duration": 3.0})
        timeline = self._timeline(spec, _durations(spec), shot_paths={"shots/s01.json": 30.0})
        s03 = timeline.shot("s03")
        self.assertAlmostEqual(s03.duration, 3.0, places=9)
        self.assertEqual(s03.sentences, ())

    def test_crossfade_overlap(self):
        """start_i = start_{i−1} + D_{i−1} − x_i（x = round(fade·fps) 帧）。"""
        spec = valid_spec()
        durations = {("s01", 0): 2.0, ("s01", 1): 3.0, ("s02", 0): 1.5}
        timeline = self._timeline(spec, durations, shot_paths={"shots/s01.json": 30.0})
        s01, s02 = timeline.shots
        self.assertAlmostEqual(s01.start, 0.0, places=9)
        # 188 − 15 = 173 帧
        self.assertAlmostEqual(s02.start, 173 / FPS, places=9)
        self.assertAlmostEqual(timeline.total_duration, 248 / FPS, places=9)

    def test_sentence_times(self):
        spec = valid_spec()
        durations = {("s01", 0): 2.0, ("s01", 1): 3.0, ("s02", 0): 1.5}
        timeline = self._timeline(spec, durations, shot_paths={"shots/s01.json": 30.0})
        s01 = timeline.shot("s01")
        self.assertAlmostEqual(s01.sentences[0].start, 0.4, places=9)
        self.assertAlmostEqual(s01.sentences[0].end, 2.4, places=9)
        # 第 2 句起点 ≈ lead + 第 1 句 + gap = 2.65 s（帧量化 ±1.5 帧）
        self.assertAlmostEqual(s01.sentences[1].start, 2.65, delta=1.5 / FPS)
        self.assertAlmostEqual(s01.sentences[1].end, 5.65, delta=1.5 / FPS)
        s02 = timeline.shot("s02")
        self.assertAlmostEqual(s02.sentences[0].start, 173 / FPS + 0.4, places=9)
        self.assertAlmostEqual(s02.sentences[0].end, 173 / FPS + 1.9, places=9)

    def test_missing_duration_rejected(self):
        spec = valid_spec()
        durations = _durations(spec)
        del durations[("s01", 1)]
        with self.assertRaises(ValueError) as ctx:
            self._timeline(spec, durations, shot_paths={"shots/s01.json": 30.0})
        self.assertIn("s01", str(ctx.exception))
        self.assertIn("第 1 句", str(ctx.exception))

    def test_path_duration_must_cover_shot(self):
        spec = valid_spec()
        durations = {("s01", 0): 2.0, ("s01", 1): 3.0, ("s02", 0): 1.5}
        with self.assertRaises(ValueError) as ctx:
            self._timeline(spec, durations, shot_paths={"shots/s01.json": 6.0})
        self.assertIn("s01", str(ctx.exception))
        self.assertIn("188 帧", str(ctx.exception))

    def test_fade_must_not_exceed_neighbours(self):
        spec = valid_spec()
        spec["shots"][1]["transition"]["duration"] = 5.0
        with self.assertRaises(ValueError) as ctx:
            self._timeline(spec, _durations(spec), shot_paths={"shots/s01.json": 30.0})
        self.assertIn("fade", str(ctx.exception))

    def test_fade_shorter_than_one_frame_rejected(self):
        spec = valid_spec()
        spec["shots"][1]["transition"]["duration"] = 0.01
        with self.assertRaises(ValueError) as ctx:
            self._timeline(spec, _durations(spec), shot_paths={"shots/s01.json": 30.0})
        self.assertIn("不足一帧", str(ctx.exception))

    def test_sfx_outside_shot_rejected(self):
        spec = valid_spec()
        spec["sfx"][0]["at"] = 3.5
        with self.assertRaises(ValueError) as ctx:
            self._timeline(spec, _durations(spec), shot_paths={"shots/s01.json": 30.0})
        self.assertIn("at", str(ctx.exception))

    def test_json_roundtrip(self):
        spec = valid_spec()
        timeline = self._timeline(spec, _durations(spec), shot_paths={"shots/s01.json": 30.0})
        data = timeline.to_json_dict()
        self.assertEqual(data["fps"], FPS)
        self.assertAlmostEqual(data["total_duration"], timeline.total_duration, places=9)
        self.assertEqual([s["id"] for s in data["shots"]], ["s01", "s02"])
        self.assertEqual(len(data["shots"][0]["sentences"]), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "timeline.json")
            timeline.save(path)
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, data)


class BoundaryAndRoundtripTest(unittest.TestCase):
    """帧边界行为与 timeline.json 往返。"""

    def _timeline(self, spec, durations, shot_paths=None):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(write_manifest(tmp, spec=spec, shot_paths=shot_paths))
            return build_timeline(manifest, durations, log=lambda msg: None)

    def _spec(self, lead=0.0, gap=0.0, tail=0.0, min_duration=0.0):
        return {
            "title": "边界", "size": [640, 360], "fps": 30,
            "voice": {"engine": "say", "name": "T", "rate": "+0%"},
            "padding": {"lead": lead, "gap": gap, "tail": tail},
            "shots": [{"id": "a", "kind": "diagram", "scene": "d/x.py:X",
                       "narration": ["句"], "min_duration": min_duration}],
        }

    def test_duration_strictly_ceil_at_frame_boundary(self):
        """秒 × fps 刚好超过整帧边界（浮点尾差量级）时严格进位，不被误差吸收缩短。"""
        timeline = self._timeline(self._spec(), {("a", 0): 1.00000000002})
        # 1.00000000002 × 30 = 30.0000000006 帧 → 31 帧
        self.assertAlmostEqual(timeline.shot("a").duration, 31 / 30, places=9)

    def test_exact_frame_duration_kept(self):
        """恰好整帧时既不进位也不缩短。"""
        timeline = self._timeline(self._spec(), {("a", 0): 1.0})
        self.assertAlmostEqual(timeline.shot("a").duration, 1.0, places=9)

    def test_sentence_start_outside_shot_rejected(self):
        """留白 + 前句把镜头挤满：句子起点落在镜头外时报错（应提高 min_duration）。"""
        spec = self._spec(lead=0.02, min_duration=1 / 30)
        with self.assertRaises(ValueError) as ctx:
            self._timeline(spec, {("a", 0): 0.001})
        self.assertIn("a", str(ctx.exception))
        self.assertIn("min_duration", str(ctx.exception))

    def test_roundtrip_via_load_timeline(self):
        spec = valid_spec()
        durations = _durations(spec)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(write_manifest(tmp, spec=spec,
                                                    shot_paths={"shots/s01.json": 30.0}))
            timeline = build_timeline(manifest, durations, log=lambda msg: None)
            path = os.path.join(tmp, "timeline.json")
            timeline.save(path)
            loaded = load_timeline(path)
        self.assertEqual(loaded.fps, timeline.fps)
        self.assertAlmostEqual(loaded.total_duration, timeline.total_duration, places=9)
        self.assertEqual([s.shot_id for s in loaded.shots], [s.shot_id for s in timeline.shots])
        for a, b in zip(loaded.shots, timeline.shots):
            self.assertAlmostEqual(a.start, b.start, places=9)
            self.assertAlmostEqual(a.duration, b.duration, places=9)
            self.assertEqual([(s.index, s.text, s.start, s.end) for s in a.sentences],
                              [(s.index, s.text, s.start, s.end) for s in b.sentences])


if __name__ == "__main__":
    unittest.main()
