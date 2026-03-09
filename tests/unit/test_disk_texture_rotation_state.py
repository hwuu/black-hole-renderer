#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

import numpy as np

import render


class DiskTextureRotationStateTest(unittest.TestCase):
    def test_prebuilt_state_skips_heavy_component_regeneration(self):
        n_r, n_phi = 4, 8

        def make_field(value: float) -> np.ndarray:
            return np.full((n_r, n_phi), value, dtype=np.float32)

        call_counts = {
            "temp_base": 0,
            "spiral": 0,
            "turbulence": 0,
            "filaments": 0,
            "rt": 0,
            "hotspots": 0,
            "az_hotspot": 0,
            "disturb": 0,
        }

        def fake_temp_base(rng, n_r_arg, n_phi_arg, r_norm_grid, *args, **kwargs):
            self.assertEqual((n_r_arg, n_phi_arg), (n_r, n_phi))
            call_counts["temp_base"] += 1
            return make_field(0.2)

        def fake_spiral(rng, n_r_arg, n_phi_arg, phi_grid, r_norm_grid, *args, **kwargs):
            self.assertEqual((n_r_arg, n_phi_arg), (n_r, n_phi))
            call_counts["spiral"] += 1
            return make_field(0.1), make_field(0.03)

        def fake_turbulence(rng, n_r_arg, n_phi_arg, r_norm_grid, *args, **kwargs):
            self.assertEqual((n_r_arg, n_phi_arg), (n_r, n_phi))
            call_counts["turbulence"] += 1
            return make_field(0.2), np.zeros((n_r, n_phi), dtype=np.int32), make_field(0.01)

        def fake_filaments(rng, n_r_arg, n_phi_arg, phi_grid, r_norm_grid, disk_area, *args, **kwargs):
            self.assertEqual((n_r_arg, n_phi_arg), (n_r, n_phi))
            call_counts["filaments"] += 1
            return make_field(0.15), make_field(0.02)

        def fake_rt(rng, n_r_arg, n_phi_arg, phi_grid, r_norm_grid, disk_area, enable_rt, *args, **kwargs):
            self.assertEqual((n_r_arg, n_phi_arg), (n_r, n_phi))
            call_counts["rt"] += 1
            return make_field(0.05), make_field(0.01)

        def fake_hotspots(rng, n_r_arg, n_phi_arg, phi_grid, r_norm_grid, disk_area, *args, **kwargs):
            self.assertEqual((n_r_arg, n_phi_arg), (n_r, n_phi))
            call_counts["hotspots"] += 1
            return make_field(0.12), make_field(0.04)

        def fake_az_hotspot(rng, n_r_arg, n_phi_arg, phi_grid, r_norm_grid, *args, **kwargs):
            self.assertEqual((n_r_arg, n_phi_arg), (n_r, n_phi))
            call_counts["az_hotspot"] += 1
            return make_field(0.08)

        def fake_disturbance_mod(rng, n_r_arg, n_phi_arg, kep_shift_pixels, r_norm_grid, *args, **kwargs):
            self.assertEqual((n_r_arg, n_phi_arg), (n_r, n_phi))
            call_counts["disturb"] += 1
            return make_field(0.9)

        with (
            patch.object(render, "_generate_temperature_base", side_effect=fake_temp_base),
            patch.object(render, "_generate_spiral_arms", side_effect=fake_spiral),
            patch.object(render, "_generate_turbulence", side_effect=fake_turbulence),
            patch.object(render, "_generate_filaments", side_effect=fake_filaments),
            patch.object(render, "_generate_rt_spikes", side_effect=fake_rt),
            patch.object(render, "_generate_hotspots", side_effect=fake_hotspots),
            patch.object(render, "_generate_azimuthal_hotspot", side_effect=fake_az_hotspot),
            patch.object(render, "_generate_disturbance_mod", side_effect=fake_disturbance_mod),
        ):
            state = render.build_disk_texture_rotating_state(n_phi=n_phi, n_r=n_r, enable_rt=True)

        self.assertEqual(
            call_counts,
            {
                "temp_base": 1,
                "spiral": 1,
                "turbulence": 1,
                "filaments": 1,
                "rt": 1,
                "hotspots": 1,
                "az_hotspot": 1,
                "disturb": 1,
            },
        )

        with (
            patch.object(render, "_generate_temperature_base", side_effect=AssertionError("temp_base should be cached")),
            patch.object(render, "_generate_spiral_arms", side_effect=AssertionError("spiral should be cached")),
            patch.object(render, "_generate_turbulence", side_effect=AssertionError("turbulence should be cached")),
            patch.object(render, "_generate_filaments", side_effect=AssertionError("filaments should be cached")),
            patch.object(render, "_generate_rt_spikes", side_effect=AssertionError("rt should be cached")),
            patch.object(render, "_generate_hotspots", side_effect=AssertionError("hotspots should be cached")),
            patch.object(render, "_generate_azimuthal_hotspot", side_effect=AssertionError("az_hotspot should be cached")),
            patch.object(render, "_generate_disturbance_mod", side_effect=AssertionError("disturbance should be cached")),
        ):
            tex = render.generate_disk_texture_rotating(
                n_phi=n_phi,
                n_r=n_r,
                enable_rt=True,
                t_offset=np.pi,
                state=state,
            )

        self.assertEqual(tex.shape, (n_r, n_phi, 4))


if __name__ == "__main__":
    unittest.main()
