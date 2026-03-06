#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

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
