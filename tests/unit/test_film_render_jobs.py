#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""制片层渲染任务（render_jobs）单元测试。

覆盖：命令拼接（参数顺序、路径、分辨率预设）、本机 / 远端两种目标、
fade 非 0 警告、路径文件缺失提示、脚本写出与执行权限。
"""

import json
import os
import tempfile
import unittest

from src.film.manifest import load_manifest
from src.film.render_jobs import (REMOTE_OUTPUT_ROOT, REMOTE_PYTHON, REMOTE_REPO_DIR,
                                  build_command, build_jobs_script, write_jobs_script)
from tests.unit._film_fixture import valid_spec, write_manifest


def _manifest(tmp, spec=None):
    return load_manifest(write_manifest(tmp, spec=spec, shot_paths={"shots/s01.json": 30.0}))


def _set_path_content(tmp, rel, data):
    with open(os.path.join(tmp, rel), "w", encoding="utf-8") as f:
        json.dump(data, f)


class BuildCommandTest(unittest.TestCase):
    def test_local_command_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest(tmp)
            shot = manifest.shot("s01")
            out = os.path.join("output", "scenes", manifest.film_id, "shots", "s01.mp4")
            cmd = build_command(manifest, shot, out)
            self.assertTrue(cmd.startswith("python render.py --disk_model v2 --video "))
            self.assertIn("--ar1 3 --ar2 30 --device gpu", cmd)
            self.assertIn(f"--v2_camera_path scenes/{manifest.film_id}/shots/s01.json", cmd)
            self.assertIn("-r fhd", cmd)
            self.assertIn("--fps 30", cmd)
            self.assertTrue(cmd.endswith(f"-o {out}"))

    def test_shot_render_args_appended(self):
        spec = valid_spec()
        spec["shots"][0]["render_args"] = ["--v2_doppler_lum", "1"]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest(tmp, spec=spec)
            cmd = build_command(manifest, manifest.shot("s01"), "out.mp4")
            self.assertIn("--v2_doppler_lum 1", cmd)

    def test_remote_python_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest(tmp)
            cmd = build_command(manifest, manifest.shot("s01"), "out.mp4",
                                python_bin=REMOTE_PYTHON)
            self.assertTrue(cmd.startswith(REMOTE_PYTHON))


class BuildJobsScriptTest(unittest.TestCase):
    def test_local_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest(tmp)
            script = build_jobs_script(manifest, log=lambda msg: None)
            self.assertTrue(script.startswith("#!/bin/sh"))
            self.assertIn("set -e", script)
            self.assertIn(f"mkdir -p output/scenes/{manifest.film_id}/shots", script)
            self.assertEqual(script.count("render.py"), 1)  # diagram 镜头不生成命令
            self.assertIn(f"-o output/scenes/{manifest.film_id}/shots/s01.mp4", script)
            self.assertNotIn("cd /root", script)

    def test_remote_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest(tmp)
            script = build_jobs_script(manifest, remote=True, log=lambda msg: None)
            self.assertIn(f"cd {REMOTE_REPO_DIR}", script)
            self.assertIn(REMOTE_PYTHON, script)
            self.assertIn(f"-o {REMOTE_OUTPUT_ROOT}/{manifest.film_id}/shots/s01.mp4", script)

    def test_fade_warning_when_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest(tmp)
            _set_path_content(tmp, "shots/s01.json",
                              {"duration": 30.0, "fade": {"fade_in": 2.0, "fade_out": 3.0}})
            script = build_jobs_script(manifest, log=lambda msg: None)
            self.assertIn("会与清单 transition 叠加", script)

    def test_no_fade_warning_when_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest(tmp)
            _set_path_content(tmp, "shots/s01.json",
                              {"duration": 30.0, "fade": {"fade_in": 0.0, "fade_out": 0.0}})
            script = build_jobs_script(manifest, log=lambda msg: None)
            self.assertNotIn("会与清单 transition 叠加", script)

    def test_fade_default_warned_when_block_omitted(self):
        """省略 fade 块 → 按引擎默认淡入 2 s / 淡出 3 s 判定为叠加，给出警告。"""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest(tmp)  # 夹具路径文件只写 duration，无 fade 块
            script = build_jobs_script(manifest, log=lambda msg: None)
            self.assertIn("会与清单 transition 叠加", script)
            self.assertIn("2s / 3s", script)

    def test_fade_partial_block_uses_engine_defaults(self):
        """只写一个字段：缺省字段按引擎默认解析（fade_out 缺省 → 3 s），仍警告。"""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest(tmp)
            _set_path_content(tmp, "shots/s01.json",
                              {"duration": 30.0, "fade": {"fade_in": 0}})
            script = build_jobs_script(manifest, log=lambda msg: None)
            self.assertIn("0s / 3s", script)

    def test_missing_path_file_comment(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest(tmp)
            os.remove(os.path.join(tmp, "shots", "s01.json"))
            script = build_jobs_script(manifest, log=lambda msg: None)
            self.assertIn("路径文件不可读", script)

    def test_write_jobs_script_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest(tmp)
            path = os.path.join(tmp, "out", "render_jobs.sh")
            write_jobs_script(manifest, path, log=lambda msg: None)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.access(path, os.X_OK))
            with open(path, encoding="utf-8") as f:
                self.assertTrue(f.read().startswith("#!/bin/sh"))


if __name__ == "__main__":
    unittest.main()
