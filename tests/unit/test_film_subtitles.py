#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""制片层字幕（subtitles）单元测试。

覆盖：折行规则（标点优先、硬折兜底、不丢字符）、ASS 时间戳格式、
build_ass 的样式/事件结构（每句一条 Dialogue、image 镜头的右下角署名）。
"""

import tempfile
import unittest

from src.film.manifest import load_manifest
from src.film.subtitles import ass_time, build_ass, wrap_sentence
from src.film.timeline import build_timeline
from tests.unit._film_fixture import valid_spec, write_manifest


class WrapSentenceTest(unittest.TestCase):
    def test_short_sentence_single_line(self):
        self.assertEqual(wrap_sentence("短句。"), ["短句。"])

    def test_breaks_after_punctuation(self):
        text = "我们可以把光想象成，一颗滚动的小球，它会拐弯。"
        lines = wrap_sentence(text)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(len(line) <= 18 for line in lines))
        self.assertEqual("".join(lines), text)
        # 每个折行窗口内都含标点：断行跟随标点（行尾是标点而不是词中间）
        for line in lines[:-1]:
            self.assertEqual(line[-1], "，")

    def test_hard_wrap_when_window_has_no_punctuation(self):
        """窗口内没有标点时按字符硬折（首 18 字无标点，属设计行为）。"""
        text = "这是一句超过十八个字符长度的旁白句子，中间有标点，可以在这里折行。"
        lines = wrap_sentence(text)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(len(line) <= 18 for line in lines))
        self.assertEqual("".join(lines), text)
        self.assertEqual(lines[0][-1], "子")

    def test_hard_wrap_without_punctuation(self):
        self.assertEqual(wrap_sentence("a" * 45), ["a" * 18, "a" * 18, "a" * 9])

    def test_max_chars_respected(self):
        text = "一二三四五六七八九十一二三四五六七八九十一二三四五六七八九。"
        lines = wrap_sentence(text, max_chars=10)
        self.assertTrue(all(len(line) <= 10 for line in lines))
        self.assertEqual("".join(lines), text)


class AssTimeTest(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(ass_time(0), "0:00:00.00")
        self.assertEqual(ass_time(2.4), "0:00:02.40")
        self.assertEqual(ass_time(3661.29), "1:01:01.29")


class BuildAssTest(unittest.TestCase):
    def _ass(self, spec):
        durations = {(shot["id"], k): 2.0
                     for shot in spec["shots"] for k in range(len(shot.get("narration", [])))}
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(write_manifest(tmp, spec=spec,
                                                     shot_paths={"shots/s01.json": 30.0}))
            timeline = build_timeline(manifest, durations, log=lambda msg: None)
            return build_ass(manifest, timeline)

    def test_header_and_dialogues(self):
        ass = self._ass(valid_spec())
        self.assertIn("PlayResX: 1920", ass)
        self.assertIn("PlayResY: 1080", ass)
        self.assertIn("PingFang SC", ass)
        dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
        self.assertEqual(len(dialogues), 3)  # 每句一条，无 image 镜头 → 无署名事件
        self.assertTrue(all(",Default,," in line for line in dialogues))

    def test_long_sentence_wrapped_with_newline_marker(self):
        spec = valid_spec()
        spec["shots"][1]["narration"] = ["这是一句超过十八个字符长度的图解旁白句子，应当折行显示。"]
        ass = self._ass(spec)
        self.assertIn("\\N", ass)
        for line in ass.splitlines():
            if ",Default,," in line and "图解" in line:
                self.assertIn("\\N", line)

    def test_image_shot_credit_bottom_right(self):
        spec = valid_spec()
        spec["shots"].append({"id": "s03", "kind": "image",
                              "layers": [{"asset": "img1", "box": [0.1, 0.1, 0.5, 0.5]}],
                              "min_duration": 4.0})
        ass = self._ass(spec)
        dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
        credits = [line for line in dialogues if ",Credit,," in line]
        self.assertEqual(len(dialogues), 4)  # 3 句 + 1 署名
        self.assertEqual(len(credits), 1)
        self.assertIn("图片：EHT Collaboration，CC BY 4.0", credits[0])


if __name__ == "__main__":
    unittest.main()
