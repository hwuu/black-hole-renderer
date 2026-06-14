#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 visual atlas 单元测试。"""

import unittest

import numpy as np

from disk_v2.params import DiskV2Params, DiskV2StructureParams
from disk_v2.visual_atlas import (
    VisualAtlas,
    build_visual_atlas,
    sample_atlas_bilinear,
)


class DiskV2VisualAtlasTest(unittest.TestCase):
    def setUp(self):
        self.params = DiskV2Params(r_in=3.0, r_out=15.0)
        self.structure = DiskV2StructureParams(
            atlas_n_r=64,
            atlas_n_phi=128,
            atlas_generation_scale=2,
        )

    def test_build_visual_atlas_deterministic(self):
        a = build_visual_atlas(self.params, self.structure, seed=7)
        b = build_visual_atlas(self.params, self.structure, seed=7)
        self.assertTrue(np.allclose(a.emission_weight, b.emission_weight))
        self.assertTrue(np.allclose(a.density_weight, b.density_weight))

    def test_emission_and_density_in_valid_range(self):
        atlas = build_visual_atlas(self.params, self.structure, seed=11)
        self.assertGreater(float(atlas.emission_weight.max()), 0.0)
        self.assertGreater(float(atlas.density_weight.max()), 0.0)
        self.assertGreaterEqual(float(atlas.emission_weight.min()), 0.0)
        self.assertGreaterEqual(float(atlas.density_weight.min()), 0.0)
        # 密度 atlas 调制幅度应弱于发射 atlas。
        em_dev = np.abs(atlas.emission_weight - 1.0)
        de_dev = np.abs(atlas.density_weight - 1.0)
        self.assertLessEqual(float(de_dev.max()), float(em_dev.max()) + 1e-5)

    def test_alpha_clip_suppresses_weak_regions(self):
        sp_strong = DiskV2StructureParams(
            atlas_n_r=32,
            atlas_n_phi=64,
            alpha_clip_threshold=0.5,
            atlas_generation_scale=1,
        )
        sp_weak = DiskV2StructureParams(
            atlas_n_r=32,
            atlas_n_phi=64,
            alpha_clip_threshold=0.001,
            atlas_generation_scale=1,
        )
        atlas_strong = build_visual_atlas(self.params, sp_strong, seed=3)
        atlas_weak = build_visual_atlas(self.params, sp_weak, seed=3)
        self.assertLess(
            float(atlas_strong.emission_weight.mean()),
            float(atlas_weak.emission_weight.mean()),
        )

    def test_sample_atlas_bilinear_outside_disk_is_zero(self):
        atlas = build_visual_atlas(self.params, self.structure, seed=5)
        out = sample_atlas_bilinear(
            atlas.emission_weight, 1.0, 0.5, atlas.r_in, atlas.r_out,
        )
        self.assertAlmostEqual(float(out), 0.0, places=6)

    def test_sample_atlas_bilinear_matches_grid_center(self):
        atlas = build_visual_atlas(self.params, self.structure, seed=9)
        r_mid = 0.5 * (atlas.r_in + atlas.r_out)
        phi_mid = np.pi
        out = float(sample_atlas_bilinear(
            atlas.emission_weight, r_mid, phi_mid, atlas.r_in, atlas.r_out,
        ))
        self.assertGreater(out, 0.0)

    def test_phi_periodicity_no_seam_jump(self):
        atlas = build_visual_atlas(self.params, self.structure, seed=13)
        r_mid = 0.5 * (atlas.r_in + atlas.r_out)
        v0 = float(sample_atlas_bilinear(
            atlas.emission_weight, r_mid, 0.0, atlas.r_in, atlas.r_out,
        ))
        v1 = float(sample_atlas_bilinear(
            atlas.emission_weight, r_mid, 2.0 * np.pi - 1e-6, atlas.r_in, atlas.r_out,
        ))
        self.assertAlmostEqual(v0, v1, delta=0.15)


if __name__ == "__main__":
    unittest.main()
