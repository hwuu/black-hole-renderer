#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""制片层音效（sfx）单元测试。

覆盖：警笛多普勒的频率曲线（过零点法测频：驶近高于 f0、驶离低于 f0、
远场比值趋于 (c+v)/(c−v)）、首尾淡入淡出防爆音、幅度限幅、参数校验、
wav 写出与按清单渲染（含 `v` 别名与未知类型报错）。
"""

import os
import tempfile
import unittest
import wave

import numpy as np

from src.film.sfx import render_sfx, siren_doppler, write_wav
from src.film.manifest import load_manifest
from tests.unit._film_fixture import valid_spec, write_manifest

SR = 44100.0


def _measure_freq(samples, t_center, width=0.2):
    """过零点法测接收时刻 t_center 附近 ±width/2 的平均频率（Hz）。

    音频时间轴以 t = 0 为中心（数组下标 0 对应 t = −duration/2），
    时刻 → 下标：`i = (t + duration/2)·SR`。
    """
    duration = len(samples) / SR
    lo = max(0, int(round((t_center - width / 2 + duration / 2) * SR)))
    hi = min(len(samples), int(round((t_center + width / 2 + duration / 2) * SR)))
    window = samples[lo:hi]
    crossings = np.sum(window[:-1] * window[1:] < 0)
    return crossings / 2.0 / ((hi - lo) / SR)


class SirenDopplerTest(unittest.TestCase):
    def _siren(self, **overrides):
        params = dict(f0=700.0, speed=25.0, closest=10.0, duration=12.0, sample_rate=SR)
        params.update(overrides)
        return siren_doppler(**params)

    def test_frequency_curve(self):
        samples = self._siren()
        f_before = _measure_freq(samples, -5.0)
        f_mid = _measure_freq(samples, 0.0)
        f_after = _measure_freq(samples, 5.0)
        self.assertGreater(f_before, 735.0)
        self.assertLess(f_before, 775.0)
        self.assertGreater(f_mid, 690.0)
        self.assertLess(f_mid, 715.0)
        self.assertGreater(f_after, 635.0)
        self.assertLess(f_after, 670.0)
        self.assertGreater(f_before, 700.0)
        self.assertLess(f_after, 700.0)
        self.assertGreater(f_before / f_after, 1.14)
        self.assertLess(f_before / f_after, 1.17)

    def test_fade_in_out_no_click(self):
        samples = self._siren()
        self.assertEqual(abs(samples[0]), 0.0)
        self.assertEqual(abs(samples[-1]), 0.0)
        self.assertLess(abs(samples[5]), 0.01)

    def test_amplitude_bounded(self):
        samples = self._siren()
        self.assertGreater(np.abs(samples).max(), 0.4)
        self.assertLessEqual(np.abs(samples).max(), 0.8)

    def test_invalid_params_rejected(self):
        for kwargs in ({"speed": 400.0}, {"f0": 0.0}, {"closest": 0.0},
                       {"duration": 0.0}, {"amplitude": 1.5}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                siren_doppler(**kwargs)

    def test_write_wav_roundtrip(self):
        samples = self._siren(duration=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.wav")
            write_wav(path, samples, SR)
            with wave.open(path, "rb") as w:
                self.assertEqual(w.getnframes(), len(samples))
                self.assertEqual(w.getframerate(), int(SR))
                self.assertEqual(w.getsampwidth(), 2)


class RenderSfxTest(unittest.TestCase):
    def _manifest(self, tmp, spec):
        return load_manifest(write_manifest(tmp, spec=spec))

    def test_renders_fixture_event_with_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._manifest(tmp, valid_spec())  # params {"f0": 700, "v": 25}
            paths = render_sfx(manifest, os.path.join(tmp, "sfx"), log=lambda msg: None)
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].endswith("s02_0.wav"))
            with wave.open(paths[0], "rb") as w:
                self.assertGreater(w.getnframes(), 0)

    def test_unknown_kind_rejected(self):
        spec = valid_spec()
        spec["sfx"][0]["kind"] = "boom"
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._manifest(tmp, spec)
            with self.assertRaises(ValueError) as ctx:
                render_sfx(manifest, os.path.join(tmp, "sfx"))
            self.assertIn("boom", str(ctx.exception))

    def test_empty_sfx_returns_no_paths(self):
        spec = valid_spec()
        spec["sfx"] = []
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._manifest(tmp, spec)
            self.assertEqual(render_sfx(manifest, os.path.join(tmp, "sfx")), [])


if __name__ == "__main__":
    unittest.main()
