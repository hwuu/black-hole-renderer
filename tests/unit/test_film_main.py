#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""制片层命令行入口（python -m src.film）的分发测试。

重活（合成、渲染任务、音效）用替身验证调用与路径；字幕走一次真实流程
（chdir 到临时目录，避免污染仓库 output/）；S6/S7 命令应报 NotImplementedError。
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.film.__main__ import main
from src.film.manifest import load_manifest
from src.film.timeline import build_timeline
from tests.unit._film_fixture import valid_spec, write_manifest


class DispatchTest(unittest.TestCase):
    def _manifest_path(self, tmp, spec=None):
        return write_manifest(tmp, spec=spec, shot_paths={"shots/s01.json": 30.0})

    def test_narrate_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest_path(tmp)
            with patch("src.film.__main__.narrate_manifest", return_value={}) as mock:
                main([path, "narrate"])
            self.assertEqual(mock.call_count, 1)
            manifest, out_dir = mock.call_args[0]
            self.assertEqual(manifest.title, "测试片")
            self.assertEqual(out_dir, os.path.join("output", "scenes", manifest.film_id, "narration"))

    def test_timeline_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest_path(tmp)
            durations = {("s01", 0): 2.0, ("s01", 1): 2.0, ("s02", 0): 2.0}
            timeline = MagicMock(total_duration=8.27, shots=[1, 2, 3])
            with patch("src.film.__main__.narrate_manifest", return_value=durations), \
                    patch("src.film.__main__.build_timeline", return_value=timeline) as mock_build:
                main([path, "timeline"])
            manifest, got_durations = mock_build.call_args[0]
            self.assertEqual(got_durations, durations)
            expected = os.path.join("output", "scenes", manifest.film_id, "timeline.json")
            timeline.save.assert_called_once_with(expected)

    def test_jobs_local_and_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest_path(tmp)
            with patch("src.film.__main__.write_jobs_script") as mock:
                main([path, "jobs"])
            self.assertFalse(mock.call_args[1]["remote"])
            with patch("src.film.__main__.write_jobs_script") as mock:
                main([path, "jobs", "--remote"])
            self.assertTrue(mock.call_args[1]["remote"])

    def test_sfx_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest_path(tmp)
            with patch("src.film.__main__.render_sfx", return_value=[]) as mock:
                main([path, "sfx"])
            manifest, out_dir = mock.call_args[0]
            self.assertEqual(out_dir, os.path.join("output", "scenes", manifest.film_id, "sfx"))

    def test_future_commands_not_implemented(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest_path(tmp)
            for command in ("hud", "assets", "all"):
                with self.assertRaises(NotImplementedError, msg=command):
                    main([path, command])

    def test_invalid_command_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                main([self._manifest_path(tmp), "bogus"])


class SubtitlesFlowTest(unittest.TestCase):
    def test_subtitles_reads_timeline_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(write_manifest(tmp, shot_paths={"shots/s01.json": 30.0}))
            durations = {(shot.id, k): 2.0
                         for shot in manifest.shots for k in range(len(shot.narration))}
            timeline = build_timeline(manifest, durations, log=lambda msg: None)
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                timeline.save(os.path.join("output", "scenes", manifest.film_id, "timeline.json"))
                main([os.path.join(tmp, "film.json"), "subtitles"])
                path = os.path.join("output", "scenes", manifest.film_id, "subtitles.ass")
                self.assertTrue(os.path.exists(path))
                with open(path, encoding="utf-8") as f:
                    content = f.read()
                self.assertIn("Dialogue:", content)
                self.assertEqual(content.count("Dialogue:"), 3)  # 每句一条，无 image 镜头
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
