#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 Schwarzschild g-factor reference 单元测试。"""

import unittest

import numpy as np

from disk_v2.params import DiskV2Params
from disk_v2.physical_fields import angular_velocity_field
from disk_v2.relativity import (
    doppler_g_factor,
    gravitational_g_factor,
    omega_kep,
    omega_norm,
    orbital_beta_local,
    total_g_factor,
)


class DiskV2RelativityTest(unittest.TestCase):
    def test_gravitational_redshift_is_less_than_one_for_distant_observer(self):
        g = gravitational_g_factor(r_em=3.2, r_obs=1000.0)
        self.assertGreater(g, 0.0)
        self.assertLess(g, 1.0)

    def test_gravitational_factor_increases_with_emission_radius(self):
        r_obs = 1000.0
        g_inner = gravitational_g_factor(r_em=3.2, r_obs=r_obs)
        g_outer = gravitational_g_factor(r_em=20.0, r_obs=r_obs)
        self.assertLess(g_inner, g_outer)
        self.assertLess(g_outer, 1.0)

    def test_doppler_approaching_side_is_blueshifted_vs_receding(self):
        beta = 0.35
        g_approach = doppler_g_factor(beta, cos_theta=1.0)
        g_recede = doppler_g_factor(beta, cos_theta=-1.0)
        self.assertGreater(g_approach, 1.0)
        self.assertLess(g_recede, 1.0)
        self.assertGreater(g_approach, g_recede)

    def test_orbital_beta_is_capped_near_isco(self):
        beta_isco = orbital_beta_local(3.0)
        beta_safe = orbital_beta_local(3.2)
        self.assertLessEqual(beta_isco, 0.99)
        self.assertLessEqual(beta_safe, 0.99)
        self.assertGreater(beta_safe, 0.0)

    def test_total_g_factor_applies_engineering_cap(self):
        g = total_g_factor(r_em=3.0, r_obs=1000.0, cos_theta=1.0, g_cap=6.0)
        self.assertLessEqual(g, 6.0)

    def test_omega_norm_matches_legacy_angular_velocity_profile(self):
        params = DiskV2Params(r_in=3.0, r_out=50.0, omega_scale=2.5)
        radii = np.array([3.0, 6.0, 12.0, 24.0])
        expected = params.omega_scale * omega_norm(radii, params.r_in)
        actual = angular_velocity_field(radii, params)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_omega_kep_and_norm_share_reference(self):
        r_in = 3.0
        radii = np.array([3.0, 4.5, 9.0])
        expected = omega_kep(radii) / omega_kep(r_in)
        np.testing.assert_allclose(omega_norm(radii, r_in), expected)


if __name__ == "__main__":
    unittest.main()
