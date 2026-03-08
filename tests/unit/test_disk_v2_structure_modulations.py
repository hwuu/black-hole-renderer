#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from disk_v2.params import DiskV2Params, DiskV2StructureParams
from disk_v2.structure_modulations import (
    hotspot_modulation,
    shear_modulation,
    structure_modulation,
    weak_mode_modulation,
)


class DiskV2StructureModulationsTest(unittest.TestCase):
    def setUp(self):
        self.params = DiskV2Params(r_in=2.0, r_out=10.0, edge_softness=0.1)
        self.structure_params = DiskV2StructureParams()
        r_values = np.linspace(self.params.r_in + 1e-3, self.params.r_out - 1e-3, 48)
        phi_values = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        self.r_grid, self.phi_grid = np.meshgrid(r_values, phi_values, indexing="ij")

    def test_invalid_structure_params_raise_value_error(self):
        with self.assertRaises(ValueError):
            DiskV2StructureParams(shear_components=0)

        with self.assertRaises(ValueError):
            DiskV2StructureParams(hotspot_count=0)

        with self.assertRaises(ValueError):
            DiskV2StructureParams(hotspot_phi_sigma=0.0)

        with self.assertRaises(ValueError):
            DiskV2StructureParams(mode1_strength=0.6, mode2_strength=0.4)

        with self.assertRaises(ValueError):
            DiskV2StructureParams(shear_strength=1.0)

        with self.assertRaises(ValueError):
            DiskV2StructureParams(hotspot_strength=1.0)

    def test_weak_mode_modulation_is_positive_and_weak(self):
        mode = weak_mode_modulation(self.r_grid, self.phi_grid, self.params, self.structure_params)

        self.assertEqual(mode.shape, self.r_grid.shape)
        self.assertTrue(np.all(mode > 0.0))
        self.assertLess(float(np.max(mode) - np.min(mode)), 0.25)

    def test_shear_modulation_is_deterministic_for_same_seed(self):
        shear_a = shear_modulation(self.r_grid, self.phi_grid, self.params, self.structure_params, seed=123)
        shear_b = shear_modulation(self.r_grid, self.phi_grid, self.params, self.structure_params, seed=123)
        shear_c = shear_modulation(self.r_grid, self.phi_grid, self.params, self.structure_params, seed=456)

        self.assertTrue(np.allclose(shear_a, shear_b))
        self.assertFalse(np.allclose(shear_a, shear_c))

    def test_hotspot_modulation_is_signed_and_biases_variation_toward_inner_disk(self):
        hotspots = hotspot_modulation(self.r_grid, self.phi_grid, self.params, self.structure_params, seed=123)
        inner_band = hotspots[: self.r_grid.shape[0] // 3]
        outer_band = hotspots[-self.r_grid.shape[0] // 3 :]

        self.assertTrue(np.any(hotspots > 1.0))
        self.assertTrue(np.any(hotspots < 1.0))
        self.assertGreater(float(np.mean(np.abs(inner_band - 1.0))), float(np.mean(np.abs(outer_band - 1.0))))

    def test_modulations_return_neutral_factor_outside_disk(self):
        outside_r = self.params.r_out * 1.1

        self.assertEqual(weak_mode_modulation(outside_r, 0.0, self.params, self.structure_params), 1.0)
        self.assertEqual(shear_modulation(outside_r, 0.0, self.params, self.structure_params, seed=123), 1.0)
        self.assertEqual(hotspot_modulation(outside_r, 0.0, self.params, self.structure_params, seed=123), 1.0)

    def test_structure_modulation_is_finite_positive_and_neutral_outside_disk(self):
        structure = structure_modulation(self.r_grid, self.phi_grid, self.params, self.structure_params, seed=7)
        outside = structure_modulation(self.params.r_out * 1.1, 0.0, self.params, self.structure_params, seed=7)

        self.assertEqual(structure.shape, self.r_grid.shape)
        self.assertTrue(np.all(np.isfinite(structure)))
        self.assertTrue(np.all(structure > 0.0))
        self.assertEqual(outside, 1.0)


if __name__ == "__main__":
    unittest.main()
