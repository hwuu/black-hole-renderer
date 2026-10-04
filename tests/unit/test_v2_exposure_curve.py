"""运镜曝光曲线与淡入淡出单测。"""

from __future__ import annotations

import unittest

import numpy as np

from src.v2.exposure_curve import (ExposureCurve, FadeSettings, MeteringSettings, auto_exposure, disk_level,
                                   fade_factor, load_fade_settings, load_metering_settings, meter_path)


class TestDiskLevel(unittest.TestCase):
    """盘区亮度水平与自动曝光。"""

    def test_uniform_disk(self):
        hdr = np.full((10, 10, 3), 2.0)
        self.assertAlmostEqual(disk_level(hdr), 2.0, places=9)
        self.assertAlmostEqual(auto_exposure(hdr), 0.45, places=9)

    def test_empty_frame(self):
        """画面中没有盘：亮度水平为 None，自动曝光回退到 1.0（与 DiskV2Renderer.finish 一致）。"""
        self.assertIsNone(disk_level(np.zeros((4, 4, 3))))
        self.assertEqual(auto_exposure(np.zeros((4, 4, 3))), 1.0)


class TestExposureCurve(unittest.TestCase):
    """曝光曲线：常数测光不补偿；部分补偿按比例；夹取生效；平滑后无跳变。"""

    s = MeteringSettings()
    t = np.arange(0.0, 20.0 + 1e-9, 0.5)

    def test_constant_levels_give_base_ev(self):
        c = ExposureCurve.from_levels(self.t, np.full(len(self.t), 3.0), 1.5, self.s)
        np.testing.assert_allclose(c.ev, 1.5, atol=1e-12)

    def test_partial_compensation_and_clip(self):
        """亮度在中段降 1 档 → 补偿 +0.6 档；降 4 档 → 夹在 +1.1 档。"""
        for drop, expected in ((1.0, 0.6), (4.0, 1.1)):
            levels = np.where(self.t < 10.0, 1.0, 2.0**-drop)
            c = ExposureCurve.from_levels(self.t, levels, 1.5, self.s)
            self.assertAlmostEqual(c.ev[-1] - 1.5, expected, places=6)
            self.assertAlmostEqual(c.ev[0], 1.5, places=6)

    def test_smooth_step_response(self):
        levels = np.where(self.t < 10.0, 1.0, 0.5)
        c = ExposureCurve.from_levels(self.t, levels, 1.5, self.s)
        self.assertLess(np.abs(np.diff(c.ev)).max(), 0.6 * 0.2)
        self.assertAlmostEqual(c.ev_at(10.25), 0.5 * (c.ev[20] + c.ev[21]), places=12)

    def test_missing_levels_interpolated(self):
        """缺失的测光点按相邻有效点插值；全部缺失时不补偿。"""
        levels = np.where(self.t < 10.0, 1.0, 0.5)
        gap = levels.copy()
        gap[[0, 5, 30]] = np.nan
        a = ExposureCurve.from_levels(self.t, levels, 1.5, self.s).ev
        b = ExposureCurve.from_levels(self.t, gap, 1.5, self.s).ev
        self.assertTrue(np.all(np.isfinite(b)))
        np.testing.assert_allclose(a, b, atol=1e-12)
        none = ExposureCurve.from_levels(self.t, np.full(len(self.t), np.nan), 1.5, self.s)
        np.testing.assert_allclose(none.ev, 1.5)

    def test_meter_path_samples_every_step(self):
        calls = []

        def render(t):
            calls.append(t)
            return np.full((4, 4, 3), 1.0)

        c = meter_path(render, 3.0, 1.0, self.s, log=lambda _m: None)
        np.testing.assert_allclose(calls, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        np.testing.assert_allclose(c.ev, 1.0)


class TestFade(unittest.TestCase):
    """淡入淡出：首末为 0、中段为 1、单调。"""

    def test_profile(self):
        f = FadeSettings(fade_in=2.0, fade_out=3.0)
        self.assertEqual(fade_factor(0.0, 10.0, f), 0.0)
        self.assertEqual(fade_factor(10.0, 10.0, f), 0.0)
        self.assertEqual(fade_factor(5.0, 10.0, f), 1.0)
        a = [fade_factor(t, 10.0, f) for t in np.linspace(0, 2, 21)]
        self.assertTrue(np.all(np.diff(a) >= 0))
        self.assertEqual(fade_factor(0.0, 10.0, FadeSettings(0.0, 0.0)), 1.0)


class TestLoad(unittest.TestCase):
    """路径文件中曝光与淡入淡出块的加载与校验。"""

    def test_defaults_and_invalid(self):
        self.assertEqual(load_metering_settings({}), MeteringSettings())
        self.assertEqual(load_fade_settings({"duration": 10.0}), FadeSettings())
        with self.assertRaises(ValueError):
            load_metering_settings({"exposure": {"metering_step": 0.0}})
        with self.assertRaises(ValueError):
            load_metering_settings({"exposure": {"ev_range": [1.0, -1.0]}})
        for block in ({"metering_size": [128]}, {"metering_size": [128.5, 72]}, {"ev_range": [1.0]},
                      {"gain": float("nan")}, {"smoothing": -1.0}, {"metering_size": None}, {"ev_range": None}):
            with self.assertRaises(ValueError, msg=str(block)):
                load_metering_settings({"exposure": block})
        with self.assertRaises(ValueError):
            load_fade_settings({"duration": 4.0, "fade": {"fade_in": 3.0, "fade_out": 3.0}})
        with self.assertRaises(ValueError):
            load_fade_settings({"duration": 4.0, "fade": {"fade_in": -1.0}})
        with self.assertRaises(ValueError):
            load_metering_settings({"exposure": None})
        with self.assertRaises(ValueError):
            load_fade_settings({"duration": float("nan")})


if __name__ == "__main__":
    unittest.main()
