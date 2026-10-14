#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""制片层旁白合成（narration）单元测试（合成与测长均注入替身，不依赖网络与外部命令）。

覆盖：逐句合成与文件命名、缓存命中（文本/音色/语速任一变化触发重合成）、
引擎分派与缺失报错、时长字典完整性。
"""

import os
import tempfile
import unittest

import numpy as np

from src.film.manifest import load_manifest
from src.film.narration import _cache_key, narrate_manifest
from tests.unit._film_fixture import valid_spec, write_manifest


class _FakeSynth:
    """记录调用并把全局调用序号写入输出文件（跨实例递增，内容可区分每次合成）。"""

    total_calls = 0

    def __init__(self):
        self.calls = []

    def __call__(self, text, voice, rate, out_path):
        self.calls.append((text, voice, rate))
        _FakeSynth.total_calls += 1
        with open(out_path, "wb") as f:
            f.write(bytes([_FakeSynth.total_calls & 0xFF]))


def _probe(path):
    with open(path, "rb") as f:
        return 1.25


class NarrateManifestTest(unittest.TestCase):
    def _narrate(self, tmp, spec):
        manifest = load_manifest(write_manifest(tmp, spec=spec))
        out_dir = os.path.join(tmp, "out", "narration")
        synth = _FakeSynth()
        durations = narrate_manifest(manifest, out_dir, synth={"say": synth}, probe=_probe,
                                     log=lambda msg: None)
        return manifest, out_dir, synth, durations

    def test_synthesizes_each_sentence(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out_dir, synth, durations = self._narrate(tmp, valid_spec())
            self.assertEqual(len(synth.calls), 3)
            self.assertEqual(synth.calls[0], ("第一句话。", "Flo", "+0%"))
            self.assertEqual(durations, {("s01", 0): 1.25, ("s01", 1): 1.25, ("s02", 0): 1.25})
            for name in ("s01_0.mp3", "s01_1.mp3", "s02_0.mp3"):
                self.assertTrue(os.path.exists(os.path.join(out_dir, name)), name)
                self.assertTrue(os.path.exists(os.path.join(out_dir, name + ".key")), name)

    def test_cache_hit_skips_synth(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._narrate(tmp, valid_spec())
            _, _, synth2, _ = self._narrate(tmp, valid_spec())
            self.assertEqual(len(synth2.calls), 0)

    def test_changed_text_rewrites_only_that_sentence(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out_dir, synth1, _ = self._narrate(tmp, valid_spec())
            before = open(os.path.join(out_dir, "s01_0.mp3"), "rb").read()
            spec = valid_spec()
            spec["shots"][0]["narration"][0] = "改过的句子。"
            _, _, synth2, _ = self._narrate(tmp, spec)
            after = open(os.path.join(out_dir, "s01_0.mp3"), "rb").read()
            self.assertEqual(len(synth2.calls), 1)
            self.assertNotEqual(before, after)

    def test_key_file_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out_dir, _, _ = self._narrate(tmp, valid_spec())
            with open(os.path.join(out_dir, "s01_0.mp3.key"), encoding="utf-8") as f:
                key = f.read()
            self.assertEqual(key, _cache_key("say", "Flo", "+0%", "第一句话。"))

    def test_failed_synth_preserves_old_audio_and_cache(self):
        """重合成中途失败：正式 mp3 保持旧内容、无 tmp 残留；改回旧文本仍正确命中缓存。

        原子替换（.tmp → os.replace）保证：合成中途崩溃时正式产物与缓存键保持一致，
        不会出现"改回旧文本却命中到半截新音频"的脏缓存。
        """
        with tempfile.TemporaryDirectory() as tmp:
            _, out_dir, _, _ = self._narrate(tmp, valid_spec())
            mp3 = os.path.join(out_dir, "s01_0.mp3")
            with open(mp3, "rb") as f:
                old = f.read()

            class _Boom(Exception):
                pass

            def boom(text, voice, rate, out_path):
                with open(out_path, "wb") as f:
                    f.write(b"PARTIAL")
                raise _Boom()

            spec = valid_spec()
            spec["shots"][0]["narration"][0] = "新句子。"
            manifest = load_manifest(write_manifest(tmp, spec=spec))
            with self.assertRaises(_Boom):
                narrate_manifest(manifest, out_dir, synth={"say": boom}, probe=_probe)
            with open(mp3, "rb") as f:
                self.assertEqual(f.read(), old)
            self.assertFalse(os.path.exists(mp3 + ".tmp"))

            synth_after = _FakeSynth()
            narrate_manifest(load_manifest(write_manifest(tmp, spec=valid_spec())),
                             out_dir, synth={"say": synth_after}, probe=_probe,
                             log=lambda msg: None)
            self.assertEqual(len(synth_after.calls), 0)
            with open(mp3, "rb") as f:
                self.assertEqual(f.read(), old)

    def test_unknown_engine_rejected(self):
        spec = valid_spec()
        spec["voice"]["engine"] = "edge-tts"
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(write_manifest(tmp, spec=spec))
            with self.assertRaises(ValueError) as ctx:
                narrate_manifest(manifest, os.path.join(tmp, "n"),
                                 synth={"say": _FakeSynth()}, probe=_probe)
            self.assertIn("edge-tts", str(ctx.exception))

    def test_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_manifest(write_manifest(tmp))
            out_dir = os.path.join(tmp, "deep", "nested", "narration")
            narrate_manifest(manifest, out_dir, synth={"say": _FakeSynth()}, probe=_probe,
                             log=lambda msg: None)
            self.assertTrue(os.path.isdir(out_dir))


class SynthVoxcpmTest(unittest.TestCase):
    """voxcpm 引擎：人设包成 Voice Design 前缀、产物为 mp3、wav 中间件清理（假模型注入）。"""

    def test_persona_prompt_and_mp3_written(self):
        import src.film.narration as nar

        class _FakeModel:
            class tts_model:
                sample_rate = 48000

            calls = []

            def generate(self, text, cfg_value, inference_timesteps):
                self.calls.append((text, cfg_value, inference_timesteps))
                return (np.sin(2 * np.pi * 440.0 * np.arange(48000) / 48000.0) * 0.5).astype(np.float32)

        fake = _FakeModel()
        nar._VOXCPM_MODEL = fake
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "s01_hook_0.mp3")
                nar.synth_voxcpm("你好。", "沉稳的男讲解员", "+0%", out)
                self.assertEqual(fake.calls[0][0], "(沉稳的男讲解员)你好。")
                self.assertEqual(fake.calls[0][1], 2.0)
                self.assertEqual(fake.calls[0][2], 10)
                self.assertTrue(os.path.exists(out))
                self.assertFalse(os.path.exists(out + ".wav"))
        finally:
            nar._VOXCPM_MODEL = None


if __name__ == "__main__":
    unittest.main()
