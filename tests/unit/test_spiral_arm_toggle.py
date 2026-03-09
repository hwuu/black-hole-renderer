#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np

import render


class SpiralArmToggleTest(unittest.TestCase):
    def test_spiral_arms_return_zero_when_disabled(self):
        n_r, n_phi = 4, 8
        phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        r_norm = np.linspace(0.0, 1.0, n_r)
        phi_grid, r_norm_grid = np.meshgrid(phi, r_norm)

        original_flag = render.ENABLE_DISK_SPIRAL_ARMS
        try:
            render.ENABLE_DISK_SPIRAL_ARMS = False
            spiral, spiral_temp = render._generate_spiral_arms(
                np.random.default_rng(42),
                n_r,
                n_phi,
                phi_grid,
                r_norm_grid,
            )
        finally:
            render.ENABLE_DISK_SPIRAL_ARMS = original_flag

        np.testing.assert_array_equal(spiral, np.zeros((n_r, n_phi), dtype=np.float32))
        np.testing.assert_array_equal(spiral_temp, np.zeros((n_r, n_phi), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
