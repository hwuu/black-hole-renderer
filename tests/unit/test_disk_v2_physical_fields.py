#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from disk_v2.physical_fields import (
    angular_velocity_field,
    density_field,
    midplane_density_field,
    midplane_temperature_field,
    temperature_field,
)
from disk_v2.geometry import (
    disk_half_thickness,
    disk_radial_mask,
    disk_radial_weight,
    disk_vertical_weight,
    disk_volume_mask,
)
from disk_v2.params import DiskV2Params


class DiskV2PhysicalFieldsTest(unittest.TestCase):
    def setUp(self):
        self.params = DiskV2Params(
            r_in=2.0,
            r_out=10.0,
            h0=0.05,
            beta_h=0.05,
            rho_power=1.0,
            temp_scale=1.0,
            omega_scale=1.0,
            edge_softness=0.1,
        )

    def test_invalid_params_raise_value_error(self):
        with self.assertRaises(ValueError):
            DiskV2Params(r_in=2.0, r_out=2.0)

        with self.assertRaises(ValueError):
            DiskV2Params(h0=0.0)

        with self.assertRaises(ValueError):
            DiskV2Params(edge_softness=0.5)

    def test_disk_half_thickness_is_positive_and_smooth(self):
        radii = np.linspace(self.params.r_in, self.params.r_out, 64)
        thickness = disk_half_thickness(radii, self.params)

        self.assertTrue(np.all(thickness > 0.0))
        self.assertTrue(np.all(np.diff(thickness) > 0.0))
        self.assertFalse(np.any(np.isnan(thickness)))

    def test_disk_radial_mask_respects_inner_and_outer_bounds(self):
        radii = np.array(
            [
                self.params.r_in * 0.9,
                self.params.r_in,
                0.5 * (self.params.r_in + self.params.r_out),
                self.params.r_out,
                self.params.r_out * 1.1,
            ],
            dtype=np.float64,
        )
        mask = disk_radial_mask(radii, self.params)

        self.assertFalse(mask[0])
        self.assertTrue(mask[1])
        self.assertTrue(mask[2])
        self.assertTrue(mask[3])
        self.assertFalse(mask[4])

    def test_disk_volume_mask_respects_radial_and_vertical_bounds(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)

        self.assertTrue(disk_volume_mask(r_mid, 0.0, self.params))
        self.assertFalse(disk_volume_mask(self.params.r_in * 0.9, 0.0, self.params))
        self.assertFalse(disk_volume_mask(self.params.r_out * 1.1, 0.0, self.params))
        self.assertFalse(disk_volume_mask(r_mid, h_mid * 1.01, self.params))

    def test_geometry_masks_use_closed_boundary_convention(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)

        self.assertTrue(disk_radial_mask(self.params.r_in, self.params))
        self.assertTrue(disk_radial_mask(self.params.r_out, self.params))
        self.assertTrue(disk_volume_mask(r_mid, h_mid, self.params))
        self.assertTrue(disk_volume_mask(r_mid, -h_mid, self.params))

    def test_weights_and_fields_vanish_on_exact_boundaries(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)

        self.assertEqual(disk_radial_weight(self.params.r_in, self.params), 0.0)
        self.assertEqual(disk_radial_weight(self.params.r_out, self.params), 0.0)
        self.assertEqual(disk_vertical_weight(r_mid, h_mid, self.params), 0.0)
        self.assertEqual(disk_vertical_weight(r_mid, -h_mid, self.params), 0.0)
        self.assertEqual(midplane_density_field(self.params.r_in, self.params), 0.0)
        self.assertEqual(midplane_temperature_field(self.params.r_in, self.params), 0.0)
        self.assertEqual(density_field(r_mid, h_mid, self.params), 0.0)
        self.assertEqual(temperature_field(r_mid, h_mid, self.params), 0.0)

    def test_disk_vertical_weight_is_symmetric_and_closes_on_surface(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)
        z_samples = np.array([0.0, 0.25 * h_mid, -0.25 * h_mid, h_mid, 1.1 * h_mid], dtype=np.float64)

        weight = disk_vertical_weight(r_mid, z_samples, self.params)

        self.assertAlmostEqual(weight[0], 1.0, places=8)
        self.assertAlmostEqual(weight[1], weight[2], places=8)
        self.assertGreater(weight[1], 0.0)
        self.assertEqual(weight[3], 0.0)
        self.assertEqual(weight[4], 0.0)

    def test_disk_radial_weight_is_flat_in_middle_and_zero_outside(self):
        radii = np.array(
            [
                self.params.r_in - 0.1,
                self.params.r_in + 0.05,
                0.5 * (self.params.r_in + self.params.r_out),
                self.params.r_out - 0.05,
                self.params.r_out + 0.1,
            ],
            dtype=np.float64,
        )
        weight = disk_radial_weight(radii, self.params)

        self.assertEqual(weight[0], 0.0)
        self.assertGreater(weight[1], 0.0)
        self.assertAlmostEqual(weight[2], 1.0, places=6)
        self.assertGreater(weight[3], 0.0)
        self.assertEqual(weight[4], 0.0)

    def test_angular_velocity_field_monotonically_decreases_with_radius(self):
        radii = np.linspace(self.params.r_in, self.params.r_out, 128)
        omega = angular_velocity_field(radii, self.params)

        self.assertTrue(np.all(omega > 0.0))
        self.assertTrue(np.all(np.diff(omega) < 0.0))
        self.assertGreater(omega[0], omega[-1])

    def test_density_is_symmetric_and_midplane_dominant(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)

        rho_mid = density_field(r_mid, 0.0, self.params)
        rho_up = density_field(r_mid, 0.5 * h_mid, self.params)
        rho_down = density_field(r_mid, -0.5 * h_mid, self.params)

        self.assertGreater(rho_mid, rho_up)
        self.assertAlmostEqual(rho_up, rho_down, places=8)
        self.assertEqual(density_field(r_mid, 1.1 * h_mid, self.params), 0.0)
        self.assertEqual(density_field(self.params.r_out * 1.1, 0.0, self.params), 0.0)
        self.assertGreater(midplane_density_field(r_mid, self.params), 0.0)

    def test_density_field_vanishes_on_geometric_surface(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)

        self.assertGreater(density_field(r_mid, 0.9 * h_mid, self.params), 0.0)
        self.assertEqual(density_field(r_mid, h_mid, self.params), 0.0)

    def test_temperature_peaks_outside_inner_edge_and_decays_outward(self):
        radii = np.linspace(self.params.r_in, self.params.r_out, 1024)
        temp_mid = midplane_temperature_field(radii, self.params)
        peak_idx = int(np.argmax(temp_mid))
        peak_radius = radii[peak_idx]

        self.assertEqual(temp_mid[0], 0.0)
        self.assertGreater(peak_radius, self.params.r_in)
        self.assertLess(peak_radius, self.params.r_in + 0.5 * (self.params.r_out - self.params.r_in))
        self.assertTrue(np.all(np.diff(temp_mid[peak_idx:]) <= 1e-8))

    def test_temperature_is_symmetric_and_midplane_dominant(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)

        temp_mid = temperature_field(r_mid, 0.0, self.params)
        temp_up = temperature_field(r_mid, 0.5 * h_mid, self.params)
        temp_down = temperature_field(r_mid, -0.5 * h_mid, self.params)

        self.assertGreater(temp_mid, temp_up)
        self.assertAlmostEqual(temp_up, temp_down, places=8)
        self.assertEqual(temperature_field(r_mid, 1.1 * h_mid, self.params), 0.0)

    def test_temperature_field_vanishes_on_geometric_surface(self):
        r_mid = 0.5 * (self.params.r_in + self.params.r_out)
        h_mid = disk_half_thickness(r_mid, self.params)

        self.assertGreater(temperature_field(r_mid, 0.9 * h_mid, self.params), 0.0)
        self.assertEqual(temperature_field(r_mid, h_mid, self.params), 0.0)


if __name__ == "__main__":
    unittest.main()
