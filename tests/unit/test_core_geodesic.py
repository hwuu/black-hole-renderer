#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NumPy 光线测地线积分（`src/core/geodesic.py`）单元测试。

覆盖：
- 临界冲击参数公式 `b_c = (3√3/2)·k·r_s` 与阴影随 k 单调缩小
- k = 1 时光线按冲击参数落入视界 / 逃逸（阴影边界在 b_c 附近）
- k = 0 时光走直线（方向不变、不落入视界）
"""

import math
import unittest

import numpy as np

from src.core.geodesic import critical_impact_parameter, integrate_ray


class CriticalImpactParameterTest(unittest.TestCase):
    def test_formula_at_k1(self):
        self.assertAlmostEqual(critical_impact_parameter(1.0), 3 * math.sqrt(3) / 2, places=12)

    def test_scales_linearly_with_k(self):
        b1 = critical_impact_parameter(1.0)
        self.assertAlmostEqual(critical_impact_parameter(0.5), 0.5 * b1, places=12)
        self.assertEqual(critical_impact_parameter(0.0), 0.0)


class IntegrateRayTest(unittest.TestCase):
    START = np.array([-30.0, 0.0, 0.0])
    DIR_X = np.array([1.0, 0.0, 0.0])

    def _ray(self, b, k, **kw):
        return integrate_ray(self.START + np.array([0.0, b, 0.0]), self.DIR_X,
                             gravity_scale=k, r_escape=45.0, **kw)

    def test_shadow_boundary_follows_b_c(self):
        """b 略小于 b_c 落入视界、略大于 b_c 逃逸（k = 1 与 k = 0.5 各验一次）。"""
        for k in (1.0, 0.5):
            b_c = critical_impact_parameter(k)
            _, outcome_in, _ = self._ray(0.97 * b_c, k, step=0.02)
            _, outcome_out, _ = self._ray(1.03 * b_c, k, step=0.02)
            self.assertEqual(outcome_in, "horizon", f"k={k}")
            self.assertEqual(outcome_out, "escaped", f"k={k}")

    def test_shadow_shrinks_with_k(self):
        """k 越小，同一冲击参数的光线越容易逃逸（阴影单调缩小）。"""
        b = 2.0
        outcomes = [self._ray(b, k, step=0.02)[1] for k in (1.0, 0.5, 0.2)]
        self.assertEqual(outcomes[0], "horizon")
        self.assertEqual(outcomes[2], "escaped")

    def test_k0_straight_line(self):
        start = self.START + np.array([0.0, 5.0, 0.0])
        pos, outcome, escape_dir = integrate_ray(start, self.DIR_X, gravity_scale=0.0, r_escape=45.0)
        self.assertEqual(outcome, "escaped")
        np.testing.assert_allclose(escape_dir, self.DIR_X, atol=1e-12)
        np.testing.assert_array_equal(pos[0], start)
        offsets = pos - start - np.outer(np.arange(len(pos)), self.DIR_X * 0.05)
        np.testing.assert_allclose(offsets, 0.0, atol=1e-9)

    def test_invalid_gravity_scale_rejected(self):
        with self.assertRaises(ValueError):
            integrate_ray(self.START, self.DIR_X, gravity_scale=-0.1)

    def test_step_over_origin_is_captured(self):
        """步长远大于吞光球时，光线整步跨过原点也必须被捕获（段检查，k = 0 与 k = 0.5 各验）。"""
        for k in (0.0, 0.5):
            start = np.array([-30.25, 0.0, 0.0])
            pos, outcome, _ = integrate_ray(start, self.DIR_X, gravity_scale=k, step=0.5, r_escape=45.0)
            self.assertEqual(outcome, "horizon", f"k={k}")
            self.assertTrue(np.all(np.isfinite(pos)), f"k={k}")

    def test_near_miss_with_large_step_not_captured(self):
        """与原点擦肩（最近距离远大于吞光半径）的大步长光线不误捕。"""
        start = np.array([-30.25, 0.5, 0.0])
        _, outcome, _ = integrate_ray(start, self.DIR_X, gravity_scale=0.0, step=0.5, r_escape=45.0)
        self.assertEqual(outcome, "escaped")

    def test_trajectory_contains_start_and_is_finite(self):
        start = self.START + np.array([0.0, 0.5, 0.0])
        pos, _, _ = integrate_ray(start, self.DIR_X, gravity_scale=1.0, step=0.1, r_escape=45.0)
        np.testing.assert_array_equal(pos[0], start)
        self.assertTrue(np.all(np.isfinite(pos)))


if __name__ == "__main__":
    unittest.main()
