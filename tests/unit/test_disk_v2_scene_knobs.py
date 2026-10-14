#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""场景旋钮在渲染器与路径视频层的接线测试。

覆盖：
- `DiskV2Renderer.set_scene_knobs` 的取值校验
- w = 0：盘发射 HDR 全零、天空透射不受影响、视界命中照常（真实黑体渲染，单一小分辨率实例）
- k = 0：光线走直线，天空路与 NumPy 针孔投影参考逐值一致，无视界命中
- 多普勒方向（AGENTS.md 踩坑第 22 条的历史疑问）：盘逆时针旋转、相机在 −y 方向时，
  d = 1 使画面左侧（朝相机运动的一侧）亮度高于右侧
- `render_path_hdr` / `reference_exposure` 的旋钮合并与参考帧测光（渲染器注入替身，纯函数级）
"""

import math
import tempfile
import unittest

import numpy as np
import taichi as ti

from src.v2.exposure_curve import auto_exposure
from src.v2.scene_anim import load_scene_anim
from src.v2.path_video import reference_exposure, render_path_hdr


class _StubRenderer:
    """`render_path_hdr` / `reference_exposure` 的渲染器替身（不触发 Taichi）。"""

    class _Params:
        r_in = 3.0

    def __init__(self, hdr, hdr_for_knobs=None):
        self.hdr = hdr
        self.hdr_for_knobs = hdr_for_knobs
        self.params = self._Params()
        self.width, self.height = 32, 16
        self.jitter_seed = {}
        self.knobs = None

    def set_scene_knobs(self, **knobs):
        self.knobs = knobs

    def render_hdr(self, cam_pos, fov, t, forward=None, roll_deg=None):
        if self.hdr_for_knobs is not None:
            return self.hdr_for_knobs(self.knobs or {}), None
        return self.hdr, None


def _spec_path(duration=20.0):
    """最小合法路径文件内容（两个关键帧；时长须容纳默认节奏的 ramp_in + ramp_out = 14 s）。"""
    kf = {"r": 40.0, "phi_deg": -90.0, "z": 5.0, "fov": 38.0, "roll": 0.0, "subject_uv": [0.5, 0.5]}
    return {"duration": duration, "keyframes": [dict(kf), dict(kf, phi_deg=-89.0)]}


class KnobWiringTest(unittest.TestCase):
    """旋钮合并与参考帧测光（纯函数级，渲染器注入替身）。"""

    def _path(self):
        from src.v2.camera_path import load_camera_path

        return load_camera_path(_spec_path())

    def test_render_path_hdr_merges_track_over_defaults(self):
        anim = load_scene_anim({"scene_anim": {"gravity_scale": [[0.0, 0.5]]}})
        stub = _StubRenderer(np.zeros((16, 32, 3)))
        render_path_hdr(stub, self._path(), 1.0, 16.0, seed=1, anim=anim,
                        knob_defaults={"gravity_scale": 1.0, "doppler_scale": 0.7, "disk_weight": 1.0})
        self.assertEqual(stub.knobs, {"gravity_scale": 0.5, "doppler_scale": 0.7, "disk_weight": 1.0})

    def test_render_path_hdr_without_anim_keeps_knobs(self):
        stub = _StubRenderer(np.zeros((16, 32, 3)))
        render_path_hdr(stub, self._path(), 1.0, 16.0, seed=1)
        self.assertIsNone(stub.knobs)

    def test_reference_exposure_measures_stub_hdr(self):
        stub = _StubRenderer(np.full((16, 32, 3), 0.5))
        e0 = reference_exposure(stub, self._path(), 16.0, 2.0)
        self.assertAlmostEqual(e0, auto_exposure(np.full((16, 32, 3), 0.5)), places=12)

    def test_reference_exposure_without_disk_falls_back_to_one(self):
        stub = _StubRenderer(np.zeros((16, 32, 3)))
        self.assertEqual(reference_exposure(stub, self._path(), 16.0, 0.0), 1.0)

    def test_reference_exposure_applies_track_knobs(self):
        """参考帧必须在参考时刻的旋钮状态下测光：w 轨道为 0 时 E₀ 取暗帧，而非默认亮帧。"""
        bright = np.full((16, 32, 3), 0.5)
        dim = np.full((16, 32, 3), 0.05)
        stub = _StubRenderer(bright, hdr_for_knobs=lambda k: bright if k.get("disk_weight", 1.0) > 0.5 else dim)
        path = self._path()
        e0_default = reference_exposure(stub, path, 16.0, 2.0)
        anim = load_scene_anim({"scene_anim": {"disk_weight": [[0.0, 0.0]]}})
        e0_track = reference_exposure(stub, path, 16.0, 2.0, anim=anim,
                                      knob_defaults={"disk_weight": 1.0})
        self.assertAlmostEqual(e0_default, auto_exposure(bright), places=12)
        self.assertAlmostEqual(e0_track, auto_exposure(dim), places=12)

    def test_reference_time_out_of_range_rejected(self):
        import json
        import os
        import tempfile

        from src.v2.path_video import load_camera_plan

        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, "path.json")
            spec = _spec_path(duration=20.0)
            spec["exposure"] = {"reference_time": 20.5}
            with open(file_path, "w") as f:
                json.dump(spec, f)
            with self.assertRaises(ValueError):
                load_camera_plan(file_path)
            spec["exposure"] = {"reference_time": 20.0}
            with open(file_path, "w") as f:
                json.dump(spec, f)
            plan = load_camera_plan(file_path)
            self.assertEqual(plan.metering.reference_time, 20.0)


class SceneKnobsRenderTest(unittest.TestCase):
    """旋钮对真实渲染输出的作用（单一小分辨率渲染器，全部 Taichi 编译只在 setUpClass 发生一次）。"""

    CAM = [0.0, -40.0, 4.87]
    FOV = 38.0

    @classmethod
    def setUpClass(cls):
        from src.core.skybox import load_or_generate_skybox
        from src.v2.params import DiskV2Params
        from src.v2.taichi_render import DiskV2Renderer

        ti.init(arch=ti.gpu, default_fp=ti.f32)
        sky, _, _ = load_or_generate_skybox(None, 1024, 512, 4000)
        cls.renderer = DiskV2Renderer(96, 48, DiskV2Params(r_in=3.0, r_out=30.0), sky,
                                      r_max=50.0, opt_level=0, ss=1,
                                      doppler_lum=1.0, doppler_color=1.0, device="gpu")

    def setUp(self):
        self.renderer.set_scene_knobs()
        self.renderer.jitter_seed[None] = 0

    def _render(self):
        return self.renderer.render_hdr(cam_pos=self.CAM, fov=self.FOV, t=2000.0)

    def test_set_knobs_rejects_out_of_range(self):
        for bad in (-0.1, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                self.renderer.set_scene_knobs(gravity_scale=bad)
            with self.assertRaises(ValueError):
                self.renderer.set_scene_knobs(doppler_scale=bad)
            with self.assertRaises(ValueError):
                self.renderer.set_scene_knobs(disk_weight=bad)

    def test_w_zero_removes_disk_but_keeps_shadow(self):
        self.renderer.set_scene_knobs(disk_weight=0.0)
        hdr, sky = self._render()
        self.assertTrue(np.all(hdr == 0.0))
        self.assertGreater(float(sky.max()), 0.0)
        self.assertGreater(int(self.renderer.event_horizon_field.to_numpy().sum()), 0)

    def test_k0_rays_straight_and_sky_matches_pinhole(self):
        """k = 0、w = 0：天空路与 NumPy 针孔投影（直线光线）逐值一致，无视界命中。"""
        from src.v2.camera import build_camera_v1_compatible

        self.renderer.set_scene_knobs(gravity_scale=0.0, disk_weight=0.0)
        _, sky = self._render()
        self.assertEqual(int(self.renderer.event_horizon_field.to_numpy().sum()), 0)

        cam_pos, right, up, forward, pw, ph, _ = build_camera_v1_compatible(
            self.CAM, self.FOV, self.renderer._iw, self.renderer._ih, 0.0)
        cp, cr, cu, cf = (cam_pos, right, up, forward)
        center = cp + cf * 1.0
        tl = center - cr * (pw * self.renderer._iw / 2.0) + cu * (ph * self.renderer._ih / 2.0)
        i_idx = np.arange(self.renderer._iw)[:, None, None]
        j_idx = np.arange(self.renderer._ih)[None, :, None]
        pixel_pos = (tl[None, None, :]
                     + (i_idx + 0.5) * pw * cr[None, None, :]
                     - (j_idx + 0.5) * ph * cu[None, None, :])
        dirs = pixel_pos - cp[None, None, :]
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        theta = np.arccos(np.clip(dirs[..., 2], -1.0, 1.0))
        phi = np.mod(np.arctan2(dirs[..., 1], dirs[..., 0]), 2 * math.pi)
        sky_tex = self.renderer.skybox_field.to_numpy()
        w, h = sky_tex.shape[:2]
        u = phi / (2 * math.pi) * w
        v = theta / math.pi * h
        u0 = np.floor(u).astype(int)
        v0 = np.clip(np.floor(v).astype(int), 0, h - 1)
        fu, fv = u - u0, v - v0
        u0w, u1w = u0 % w, (u0 + 1) % w
        v1 = np.clip(v0 + 1, 0, h - 1)
        expected = (sky_tex[u0w, v0] * ((1 - fu) * (1 - fv))[..., None]
                    + sky_tex[u1w, v0] * (fu * (1 - fv))[..., None]
                    + sky_tex[u0w, v1] * ((1 - fu) * fv)[..., None]
                    + sky_tex[u1w, v1] * (fu * fv)[..., None])
        # kernel 的 sky_field 按 [x, y] 存储后转置为 (H, W, 3)
        np.testing.assert_allclose(sky, expected.transpose(1, 0, 2), atol=2e-3, rtol=1e-2)

    def test_doppler_brightens_approaching_side(self):
        """盘逆时针（spin = +1）、相机在 −y：朝相机运动的是画面左侧；d = 1 使左侧更亮。"""
        half = self.renderer.width // 2
        self.renderer.set_scene_knobs(doppler_scale=0.0)
        hdr0, _ = self._render()
        ratio0 = float(hdr0[:, :half].mean()) / max(float(hdr0[:, half:].mean()), 1e-30)
        self.renderer.set_scene_knobs(doppler_scale=1.0)
        hdr1, _ = self._render()
        ratio1 = float(hdr1[:, :half].mean()) / max(float(hdr1[:, half:].mean()), 1e-30)
        self.assertGreater(ratio1, max(1.15, 1.25 * ratio0))

    def test_w_zero_keeps_horizon_and_unblocks_sky(self):
        """w = 0 只去掉气体：视界命中（几何量）与 w = 1 完全相同，盘所在区域的透射天空恢复。"""
        self.renderer.set_scene_knobs(disk_weight=1.0)
        _, sky1 = self._render()
        horizon1 = self.renderer.event_horizon_field.to_numpy().copy()
        self.renderer.set_scene_knobs(disk_weight=0.0)
        _, sky0 = self._render()
        horizon0 = self.renderer.event_horizon_field.to_numpy()
        np.testing.assert_array_equal(horizon0, horizon1)
        h = self.renderer.height
        band = slice(int(0.35 * h), int(0.65 * h))
        self.assertGreater(float(sky0[band].mean()), 1.2 * float(sky1[band].mean()))

    def test_half_weight_scales_gas_conserving_source_function(self):
        """0 < w < 1：气体量减半 —— 发射整体下降、透过率升高，而光学厚区的亮度（由 j/α 决定）不变。

        相机贴盘面平视时全程光学厚，w = 0.5 与 w = 1 的盘区亮度比接近 1 正是
        `j_total / α` 不随 w 缩放的证据（发射与消光同步乘 w）。
        """
        self.renderer.set_scene_knobs(disk_weight=1.0)
        hdr1, sky1 = self._render()
        self.renderer.set_scene_knobs(disk_weight=0.5)
        hdr05, sky05 = self._render()
        h = self.renderer.height
        band = slice(int(0.35 * h), int(0.65 * h))
        thick_ratio = float(hdr05[band].mean()) / float(hdr1[band].mean())
        self.assertGreater(thick_ratio, 0.85)
        self.assertLess(thick_ratio, 1.15)
        self.assertLess(float(hdr05.mean()), float(hdr1.mean()))
        self.assertGreater(float(sky05[band].mean()), float(sky1[band].mean()))


if __name__ == "__main__":
    unittest.main()
