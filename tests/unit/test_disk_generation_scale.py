#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import render


class DiskGenerationScaleTest(unittest.TestCase):
    def test_parse_args_accepts_disk_generation_scale(self):
        argv = ["render.py", "--disk_generation_scale", "4"]
        with patch.object(sys, "argv", argv):
            args = render.parse_args()

        self.assertEqual(args.disk_generation_scale, 4)

    def test_load_cached_disk_texture_forwards_generation_scale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(render.os, "makedirs"),
                patch.object(render.np, "save") as save_mock,
                patch.object(render.os.path, "exists", return_value=False),
                patch.object(render, "generate_disk_texture", return_value=np.zeros((8, 16, 4), dtype=np.float32)) as gen_mock,
            ):
                cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    tex = render.load_cached_disk_texture(
                        width=640,
                        height=360,
                        cam_pos=[6.0, 0.0, 0.5],
                        fov=60.0,
                        generation_scale=4,
                        force=True,
                    )
                finally:
                    os.chdir(cwd)

        self.assertEqual(tex.shape, (8, 16, 4))
        self.assertEqual(gen_mock.call_args.kwargs["generation_scale"], 4)
        cache_path = save_mock.call_args.args[0]
        self.assertIn("scale4", cache_path)

    def test_build_rotating_state_forwards_generation_scale(self):
        n_r, n_phi = 4, 8

        def make_field(value: float) -> np.ndarray:
            return np.full((n_r, n_phi), value, dtype=np.float32)

        def fake_zero_pair(*args, **kwargs):
            self.assertEqual(kwargs["generation_scale"], 4)
            return make_field(0.1), make_field(0.01)

        def fake_turbulence(*args, **kwargs):
            self.assertEqual(kwargs["generation_scale"], 4)
            return make_field(0.2), np.zeros((n_r, n_phi), dtype=np.int32), make_field(0.02)

        def fake_az(*args, **kwargs):
            self.assertEqual(kwargs["generation_scale"], 4)
            return make_field(0.05)

        def fake_disturbance(*args, **kwargs):
            self.assertEqual(kwargs["generation_scale"], 4)
            return make_field(0.9)

        with (
            patch.object(render, "_generate_temperature_base", return_value=make_field(0.2)),
            patch.object(render, "_generate_spiral_arms", side_effect=fake_zero_pair),
            patch.object(render, "_generate_turbulence", side_effect=fake_turbulence),
            patch.object(render, "_generate_filaments", side_effect=fake_zero_pair),
            patch.object(render, "_generate_rt_spikes", side_effect=fake_zero_pair),
            patch.object(render, "_generate_hotspots", return_value=(make_field(0.12), make_field(0.04))),
            patch.object(render, "_generate_azimuthal_hotspot", side_effect=fake_az),
            patch.object(render, "_generate_disturbance_mod", side_effect=fake_disturbance),
        ):
            state = render.build_disk_texture_rotating_state(
                n_phi=n_phi,
                n_r=n_r,
                generation_scale=4,
            )

        self.assertEqual(state.generation_scale, 4)


if __name__ == "__main__":
    unittest.main()
