#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk V2 Phase 5 相对论亮度与颜色单元测试。

Phase 5 在 `DiskV2Renderer` 中加入 g-factor 修正（Doppler + 引力红移），
按 `g^lum_power` 缩放亮度、按 Wien 偏移调颜色。

这里的单测**不**驱动完整渲染（CPU 太慢），而是用一个小尺寸 GPU 渲染
作为 smoke + 启用 g-factor 与禁用 g-factor 的差异检查。

如果 GPU 不可用，测试会被跳过（不视为失败）。
"""

import unittest

import numpy as np


def _gpu_available() -> bool:
    """检测 GPU 是否可用且能 init。若当前进程已经 init 过其他 arch，
    Taichi 1.x 会忽略二次 init，本函数返回 True 并尝试复用。"""
    try:
        import taichi as ti
        # 如果已经 init 过，跳过；否则尝试 init GPU。
        if not ti.lang.impl.get_runtime().materialized:
            ti.init(arch=ti.gpu, default_fp=ti.f32)
        return True
    except Exception:
        return False


class DiskV2GFactorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _gpu_available():
            raise unittest.SkipTest("GPU 不可用，跳过 V2 g-factor smoke 测试")

        from render import generate_skybox
        cls.skybox = generate_skybox(tex_w=256, tex_h=128, n_stars=100).astype(np.float32)

    def _make_renderer(
        self,
        enable_g: bool,
        palette_mode: str = "cinematic",
        *,
        auto_exposure: bool = False,
    ):
        from disk_v2.params import DiskV2Params, DiskV2StructureParams, DiskV2PaletteParams
        from disk_v2.taichi_render import DiskV2Renderer

        p = DiskV2Params()
        sp = DiskV2StructureParams(clump_count=20)
        pp = DiskV2PaletteParams(palette_mode=palette_mode)
        return DiskV2Renderer(
            width=64,
            height=64,
            params=p,
            structure_params=sp,
            palette_params=pp,
            skybox=self.skybox,
            disk_tilt_deg=80.0,  # 高倾角，让 doppler 方向性可见
            volume_samples=4,
            device="gpu",
            enable_g_factor=enable_g,
            lum_power=4.0,
            auto_exposure=auto_exposure,
        )

    def test_renderer_runs_with_g_factor(self):
        """V2 渲染器在 g-factor 启用时能正常出图（smoke）。"""
        r = self._make_renderer(enable_g=True)
        img = r.render(cam_pos=[20, 0, 1], fov=90)
        self.assertEqual(img.shape, (64, 64, 3))
        self.assertTrue(np.issubdtype(img.dtype, np.floating))
        # 至少有一个非黑像素。
        self.assertGreater(float(img.max()), 0.0)

    def test_g_factor_changes_output(self):
        """启用 g-factor 与禁用应产生明显不同的输出。"""
        r_on = self._make_renderer(enable_g=True)
        img_on = r_on.render(cam_pos=[20, 0, 1], fov=90)

        # 注意：Taichi 在同一进程内无法切换 init，因此第二次渲染器要复用同一 ti.init。
        # 这里再构造一个 enable_g=False 的渲染器。
        r_off = self._make_renderer(enable_g=False)
        img_off = r_off.render(cam_pos=[20, 0, 1], fov=90)

        # 两图应不完全相同（g-factor 调整了发射率方向性）。方向性物理正确性由
        # test_disk_v2_relativity 的解析单测覆盖；这里仅保留 GPU smoke。
        diff = float(np.mean(np.abs(img_on.astype(np.float64) - img_off.astype(np.float64))))
        self.assertGreater(
            diff,
            1.0e-4,
            msg=f"启用与禁用 g-factor 平均像素差仅 {diff:.6f}，预期 > 1e-4",
        )

    def test_auto_exposure_uses_reference_within_factor_of_three(self):
        """Bug A2/A4：auto exposure 的 white point 应与 reference white point 同量级。"""
        r = self._make_renderer(enable_g=True, auto_exposure=True)
        r.render(cam_pos=[20, 0, 1], fov=90)

        wp = float(r.last_white_point)
        ref = float(r.reference_white_point)
        self.assertGreater(wp, 0.0)
        self.assertGreater(ref, 0.0)
        self.assertGreaterEqual(wp, 0.3 * ref)
        self.assertLessEqual(wp, 3.0 * ref)


if __name__ == "__main__":
    unittest.main()
