#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import render


class _FakeWriter:
    def init_video_stream(self, *args, **kwargs):
        return None

    def write_frame(self, *args, **kwargs):
        return None


class _FakeRenderer:
    def __init__(self):
        self.calls = []

    def render(self, cam_pos, fov, frame=0):
        self.calls.append((list(cam_pos), fov, frame))
        return np.zeros((2, 2, 3), dtype=np.float32)

    def update_disk_texture(self, tex):
        return None


class OrbitDegreesTest(unittest.TestCase):
    def test_parse_args_accepts_negative_orbit_degrees(self):
        argv = [
            "render.py",
            "--video",
            "--orbit",
            "--orbit_degrees",
            "-90",
        ]
        with patch.object(sys, "argv", argv):
            args = render.parse_args()

        self.assertEqual(args.orbit_degrees, -90.0)

    def test_render_video_uses_configured_orbit_degrees(self):
        renderer = _FakeRenderer()

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(render.iio, "imopen", return_value=_FakeWriter()),
                patch.object(render.iio, "imread", return_value=np.zeros((2, 2, 3), dtype=np.uint8)),
            ):
                output_path = os.path.join(tmpdir, "orbit.mp4")
                render.render_video(
                    renderer=renderer,
                    width=2,
                    height=2,
                    n_frames=4,
                    fps=24,
                    output_path=output_path,
                    fov=60.0,
                    static_cam_pos=[10.0, 0.0, 2.0],
                    orbit=True,
                    orbit_degrees=-90.0,
                    disk_rotation_algorithm="baseline",
                )

        orbit_radius = float(np.linalg.norm([10.0, 0.0, 2.0]))
        expected_positions = [
            [orbit_radius, 0.0, 2.0],
            [orbit_radius * np.cos(np.radians(-22.5)), orbit_radius * np.sin(np.radians(-22.5)), 2.0],
            [orbit_radius * np.cos(np.radians(-45.0)), orbit_radius * np.sin(np.radians(-45.0)), 2.0],
            [orbit_radius * np.cos(np.radians(-67.5)), orbit_radius * np.sin(np.radians(-67.5)), 2.0],
        ]

        self.assertEqual(len(renderer.calls), 4)
        for (cam_pos, fov, frame), expected in zip(renderer.calls, expected_positions):
            self.assertEqual(fov, 60.0)
            self.assertIsInstance(frame, int)
            np.testing.assert_allclose(cam_pos, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
