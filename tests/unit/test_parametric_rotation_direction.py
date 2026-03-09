#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

import numpy as np

import render


class _FakeRng:
    def integers(self, low, high=None, size=None, endpoint=False):
        if size is None:
            return 2
        return np.full(size, 2, dtype=np.int64)

    def uniform(self, low=0.0, high=1.0, size=None):
        value = (low + high) / 2.0
        if size is None:
            return value
        return np.full(size, value, dtype=np.float32)


def _signed_best_shift(reference: np.ndarray, shifted: np.ndarray) -> int:
    size = reference.shape[0]
    scores = [float(np.dot(reference, np.roll(shifted, -shift))) for shift in range(size)]
    best_shift = int(np.argmax(scores))
    if best_shift > size // 2:
        best_shift -= size
    return best_shift


class ParametricRotationDirectionTest(unittest.TestCase):
    def test_temperature_base_rotates_same_direction_as_phi_grid(self):
        def fake_zero_pair(*args, **kwargs):
            n_r = kwargs.get("n_r", args[1])
            n_phi = kwargs.get("n_phi", args[2])
            zeros = np.zeros((n_r, n_phi), dtype=np.float32)
            return zeros, zeros

        def fake_zero_triple(*args, **kwargs):
            n_r = kwargs.get("n_r", args[1])
            n_phi = kwargs.get("n_phi", args[2])
            zeros = np.zeros((n_r, n_phi), dtype=np.float32)
            return zeros, zeros.astype(np.int32), zeros

        def fake_zero_hotspot(*args, **kwargs):
            n_r = kwargs.get("n_r", args[1])
            n_phi = kwargs.get("n_phi", args[2])
            return np.zeros((n_r, n_phi), dtype=np.float32)

        def fake_identity_disturbance(rng, n_r, n_phi, density, temp_struct, *args, **kwargs):
            return density, temp_struct

        original_np_roll = np.roll
        captured_shifts = []

        def tracking_roll(array, shift, axis=None):
            captured_shifts.append(shift)
            return original_np_roll(array, shift, axis=axis)

        with (
            patch.object(render, "_generate_spiral_arms", side_effect=fake_zero_pair),
            patch.object(render, "_generate_turbulence", side_effect=fake_zero_triple),
            patch.object(render, "_generate_filaments", side_effect=fake_zero_pair),
            patch.object(render, "_generate_rt_spikes", side_effect=fake_zero_pair),
            patch.object(render, "_generate_hotspots", side_effect=fake_zero_pair),
            patch.object(render, "_generate_azimuthal_hotspot", side_effect=fake_zero_hotspot),
            patch.object(render, "_apply_disturbance", side_effect=fake_identity_disturbance),
            patch.object(render.np, "roll", side_effect=tracking_roll),
        ):
            n_r, n_phi = 4, 16
            t_offset = np.pi / 2.0
            render.generate_disk_texture_rotating(n_phi=n_phi, n_r=n_r, t_offset=t_offset)

            r_norm = np.linspace(0.0, 1.0, n_r)
            r_vals = 2.0 + (3.5 - 2.0) * r_norm
            omega_vals = np.sqrt(0.5 / (r_vals ** 3 + 1e-6))
            expected_shifts = [
                -int(t_offset * omega / (2.0 * np.pi) * n_phi)
                for omega in omega_vals
                for _ in range(2)
            ]

            self.assertEqual(
                captured_shifts,
                expected_shifts,
                msg=(
                    "temperature base 动态旋转方向应与 phi_grid 一致，"
                    f"当前 captured_shifts={captured_shifts}, expected_shifts={expected_shifts}"
                ),
            )

    def test_turbulence_rotates_same_direction_as_phi_grid(self):
        def fake_tileable_noise(shape, rng, freq_u=6, freq_v=6):
            noise = np.zeros(shape, dtype=np.float32)
            noise[:, 0] = 1.0
            return noise

        def fake_periodic_pixel_noise(shape, rng):
            noise = np.zeros(shape, dtype=np.float32)
            noise[:, 0] = 1.0
            return noise

        with (
            patch.object(render, "_tileable_noise", side_effect=fake_tileable_noise),
            patch.object(render, "_periodic_pixel_noise", side_effect=fake_periodic_pixel_noise),
        ):
            n_r, n_phi = 4, 16
            t_offset = np.pi / 2.0
            r_norm_grid = np.full((n_r, n_phi), 10.0, dtype=np.float32)
            omega_grid = np.ones((n_r, n_phi), dtype=np.float32)

            turbulence_t0, _, _ = render._generate_turbulence(
                _FakeRng(), n_r, n_phi, r_norm_grid, 0.0, omega_grid
            )
            turbulence_t1, _, _ = render._generate_turbulence(
                _FakeRng(), n_r, n_phi, r_norm_grid, t_offset, omega_grid
            )

            phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
            phi_shift = _signed_best_shift(np.cos(phi), np.cos(phi + t_offset))
            turbulence_shift = _signed_best_shift(turbulence_t0[0], turbulence_t1[0])

            self.assertEqual(
                turbulence_shift,
                phi_shift,
                msg=(
                    f"turbulence 动态旋转方向应与 phi_grid 一致，"
                    f"当前 phi_shift={phi_shift}, turbulence_shift={turbulence_shift}"
                ),
            )

    def test_azimuthal_hotspot_noise_rotates_same_direction_as_phi_grid(self):
        original_fbm_noise = render._fbm_noise

        def fake_fbm_noise(shape, rng, octaves=4, persistence=0.5, base_scale=1, wrap_u=False):
            noise = np.zeros(shape, dtype=np.float32)
            noise[:, 0] = 1.0
            return noise

        render._fbm_noise = fake_fbm_noise
        try:
            n_r, n_phi = 4, 8
            phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
            phi_grid_base, _ = np.meshgrid(phi, np.zeros(n_r, dtype=np.float32))
            r_norm_grid = np.zeros((n_r, n_phi), dtype=np.float32)
            omega_grid = np.ones((n_r, n_phi), dtype=np.float32)
            t_offset = np.pi / 2.0

            hotspot_t0 = render._generate_azimuthal_hotspot(
                _FakeRng(), n_r, n_phi, phi_grid_base, r_norm_grid, 0.0, omega_grid
            )
            hotspot_t1 = render._generate_azimuthal_hotspot(
                _FakeRng(),
                n_r,
                n_phi,
                phi_grid_base + t_offset * omega_grid,
                r_norm_grid,
                t_offset,
                omega_grid,
            )

            phi_reference_t0 = np.cos(phi_grid_base[0])
            phi_reference_t1 = np.cos((phi_grid_base + t_offset * omega_grid)[0])

            phi_shift = _signed_best_shift(phi_reference_t0, phi_reference_t1)
            hotspot_shift = _signed_best_shift(hotspot_t0[0], hotspot_t1[0])

            self.assertEqual(
                hotspot_shift,
                phi_shift,
                msg=(
                    f"azimuthal hotspot 动态噪声旋转方向应与 phi_grid 一致，"
                    f"当前 phi_shift={phi_shift}, hotspot_shift={hotspot_shift}"
                ),
            )
        finally:
            render._fbm_noise = original_fbm_noise


if __name__ == "__main__":
    unittest.main()
