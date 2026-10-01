#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 v2.2 成像方程 reference 单元测试。"""

import unittest

import numpy as np

from disk_v2.imaging import (
    physical_baseline_flux,
    physical_baseline_volume_flux,
    reference_exposure,
    tau_effective_midplane,
)
from disk_v2.params import DiskV2PaletteParams, DiskV2Params


class DiskV2ImagingTest(unittest.TestCase):
    """验证 Step 1 冻结的成像方程语义。"""

    def setUp(self):
        self.params = DiskV2Params()
        self.opacity_scale = 0.55

    def test_tau_effective_is_non_negative_and_inner_weighted(self):
        radii = np.linspace(self.params.r_in + 1e-3, self.params.r_out - 1e-3, 4096)
        tau = np.asarray(tau_effective_midplane(radii, self.params, self.opacity_scale))

        self.assertTrue(np.all(np.isfinite(tau)))
        self.assertTrue(np.all(tau >= 0.0))
        self.assertGreater(float(np.max(tau)), 0.0)

        inner = tau[radii < self.params.r_in * 3.0]
        outer = tau[radii > self.params.r_out * 0.75]
        self.assertGreater(float(np.mean(inner)), float(np.mean(outer)))

    def test_physical_flux_has_inner_peak_and_decays_outward(self):
        radii = np.linspace(self.params.r_in + 1e-3, self.params.r_out - 1e-3, 4096)
        flux = np.asarray(physical_baseline_flux(radii, self.params, self.opacity_scale))

        self.assertTrue(np.all(np.isfinite(flux)))
        self.assertTrue(np.all(flux >= 0.0))

        peak_idx = int(np.argmax(flux))
        peak_r = radii[peak_idx]
        self.assertGreater(peak_r, self.params.r_in)
        self.assertLess(peak_r, self.params.r_in * 4.0)

        outer_band = flux[radii > self.params.r_out * 0.75]
        self.assertGreater(float(np.max(flux)), float(np.mean(outer_band)) * 10.0)

    def test_physical_flux_internal_band_does_not_collapse_under_soft_edge(self):
        """Bug A1：`W_r` 只能作为 support 乘一次，不能被密度/温度重复放大成黑环。"""
        radii = np.linspace(self.params.r_in + 0.3, self.params.r_in + 2.0, 64)
        flux = np.asarray(physical_baseline_flux(radii, self.params, self.opacity_scale))
        finite = flux[np.isfinite(flux) & (flux > 0.0)]

        self.assertGreater(finite.size, 0)
        self.assertLess(float(np.max(finite) / max(np.min(finite), 1e-20)), 100.0)

    def test_reference_exposure_is_camera_independent_profile_quantity(self):
        exposure_a = reference_exposure(self.params, self.opacity_scale, target_ldr=0.7)
        exposure_b = reference_exposure(self.params, self.opacity_scale, target_ldr=0.7, sample_count=8192)

        self.assertTrue(np.isfinite(exposure_a))
        self.assertGreater(exposure_a, 0.0)
        self.assertAlmostEqual(exposure_a, exposure_b, delta=0.05 * exposure_a)

        radii = np.linspace(self.params.r_in + 1e-6, self.params.r_out - 1e-6, 4096)
        flux = np.asarray(physical_baseline_volume_flux(radii, self.params, self.opacity_scale))
        f_ref = float(np.percentile(flux[flux > 0.0], 99.0))
        hdr_ref = exposure_a * f_ref
        ldr_ref = hdr_ref / (1.0 + hdr_ref)
        self.assertAlmostEqual(ldr_ref, 0.7, delta=0.03)

    def test_volume_flux_reference_keeps_same_radial_trend(self):
        radii = np.linspace(self.params.r_in + 1e-3, self.params.r_out - 1e-3, 1024)
        surface_flux = np.asarray(physical_baseline_flux(radii, self.params, self.opacity_scale))
        volume_flux = np.asarray(physical_baseline_volume_flux(radii, self.params, self.opacity_scale))

        self.assertTrue(np.all(np.isfinite(volume_flux)))
        self.assertTrue(np.all(volume_flux >= 0.0))
        self.assertGreater(float(np.max(volume_flux)), 0.0)
        self.assertLess(abs(float(radii[np.argmax(surface_flux)] - radii[np.argmax(volume_flux)])), 2.0)


if __name__ == "__main__":
    unittest.main()
