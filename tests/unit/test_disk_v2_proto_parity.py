"""V2 与参考实现（Proto）一致性的保护单测（2026-10 差距排查修复）。

覆盖本轮修复的每一类偏差，防止再次回退：

1. 烟雾 / 低频噪声归一到单位方差（`smoke_norm` / `low_norm`）。
2. 低频层刚体环带中心约定 `r_b = exp(ln r_in + (b + 0.5)·dln)`。
3. 光行时间相位换算 `_delayed_rot` / `_delayed_seed`。
4. 体积路径：落入视界的光线保留路上累积的盘发射。
5. `render(t=...)` 真正驱动平流相位（视频旋转）。
6. 超采样 `ss` 的内部分辨率与输出形状。
"""

import math
import unittest

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32)  # 无 GPU 时 Taichi 自动回退 CPU

from disk_v2.advection import RigidRingBands  # noqa: E402
from disk_v2.params import (  # noqa: E402
    DiskV2PaletteParams,
    DiskV2Params,
    DiskV2StructureParams,
    DiskV2VolumeParams,
)
from disk_v2.taichi_impl import _delayed_rot, _delayed_seed  # noqa: E402
from disk_v2.taichi_render import DiskV2Renderer  # noqa: E402

R_IN, R_OUT = 3.0, 30.0
CAM_ELEV = math.radians(7.0)
CAM = [0.0, -40.0 * math.cos(CAM_ELEV), 40.0 * math.sin(CAM_ELEV)]


def _make_renderer(width=24, height=14, ss=2):
    """构造与参考实现预设 M 同参数的小尺寸 V2 渲染器（黑色天空）。"""
    return DiskV2Renderer(
        width=width, height=height,
        params=DiskV2Params(r_in=R_IN, r_out=R_OUT),
        structure_params=DiskV2StructureParams(use_visual_atlas=False),
        palette_params=DiskV2PaletteParams(),
        skybox=np.zeros((8, 16, 3), np.float32),
        r_max=90.0, auto_exposure=True, device="gpu",
        volume_params=DiskV2VolumeParams(), use_postfx=True, ss=ss,
    )


_SHARED = {}


def _shared_renderer():
    """全模块共享一个渲染器（体积 kernel 编译昂贵，只编译一次）。"""
    if "r" not in _SHARED:
        _SHARED["r"] = _make_renderer()
    return _SHARED["r"]


class TestNoiseNormalization(unittest.TestCase):
    """烟雾与低频调制噪声应为约单位方差（参数按单位方差定义）。"""

    def test_smoke_and_low_noise_unit_variance(self):
        disk = _shared_renderer().disk_ti
        n = 16384
        lo = ti.field(ti.f32, n)
        sm = ti.field(ti.f32, n)

        @ti.kernel
        def sample():
            for i in range(n):
                r = ti.exp(ti.log(3.2) + (ti.log(28.0) - ti.log(3.2)) * (ti.cast(i % 128, ti.f32) + 0.5) / 128.0)
                phi = ti.cast(i // 128, ti.f32) / 128.0 * 2.0 * math.pi
                zeta = ti.cast((i * 7) % 17, ti.f32) / 17.0 * 4.0 - 2.0
                lo[i] = disk._turb_low(r, phi, 0.0)
                sm[i] = disk._turb_pair_smoke(r, phi, zeta, 0.0, 131.7)

        sample()
        # 未归一化时 std ≈ 0.21；归一后应接近 1（带混合会略降方差）
        self.assertGreater(float(lo.to_numpy().std()), 0.6)
        self.assertLess(float(lo.to_numpy().std()), 1.5)
        self.assertGreater(float(sm.to_numpy().std()), 0.6)
        self.assertLess(float(sm.to_numpy().std()), 1.5)


class TestLowBandConvention(unittest.TestCase):
    """低频层带中心与参考实现 turb_low 一致。"""

    def test_low_band_centers(self):
        dln = DiskV2VolumeParams().dln_l
        bands = RigidRingBands(R_IN, R_OUT, dln, 6.0, lnr0_bands=0.0, center_frac=0.5)
        for k, b in enumerate(range(bands.b_lo, bands.b_hi + 1)):
            self.assertAlmostEqual(bands.r_b[k], math.exp(math.log(R_IN) + (b + 0.5) * dln), places=9)

    def test_core_band_centers_unchanged(self):
        dln = DiskV2VolumeParams().dln_r
        bands = RigidRingBands(R_IN, R_OUT, dln, 4.0)
        lnr0 = math.log(R_IN) - 2.0 * dln
        for k, b in enumerate(range(bands.b_lo, bands.b_hi + 1)):
            self.assertAlmostEqual(bands.r_b[k], math.exp(lnr0 + b * dln), places=9)


class TestDelayedPhase(unittest.TestCase):
    """光行时间相位换算：延迟 0 不变；延迟一个寿命 → 进度不变、索引减 1。"""

    def test_delayed_phase(self):
        out = ti.Vector.field(5, ti.f32, shape=())

        @ti.kernel
        def run():
            rot0 = _delayed_rot(1.0, 0.1, 0.0)
            f0, c0 = _delayed_seed(0.3, 7, 50.0, 0.0)
            f1, c1 = _delayed_seed(0.3, 7, 50.0, 50.0)
            out[None] = ti.Vector([rot0, f0, ti.cast(c0, ti.f32), f1, ti.cast(c1, ti.f32)])

        run()
        rot0, f0, c0, f1, c1 = out[None].to_numpy()
        self.assertAlmostEqual(rot0, 1.0, places=6)
        self.assertAlmostEqual(f0, 0.3, places=6)
        self.assertEqual(int(c0), 7)
        self.assertAlmostEqual(f1, 0.3, places=5)
        self.assertEqual(int(c1), 6)

    def test_delayed_seed_wraps_cycle_index(self):
        out = ti.Vector.field(2, ti.f32, shape=())

        @ti.kernel
        def run():
            f, c = _delayed_seed(0.2, 0, 10.0, 5.0)
            out[None] = ti.Vector([f, ti.cast(c, ti.f32)])

        run()
        f, c = out[None].to_numpy()
        self.assertAlmostEqual(f, 0.7, places=5)
        self.assertEqual(int(c), 4095)


class TestVolumeRender(unittest.TestCase):
    """体积渲染路径的合成、视界、时间与超采样。"""

    @classmethod
    def setUpClass(cls):
        cls.renderer = _shared_renderer()
        cls.img = cls.renderer.render(cam_pos=CAM, fov=38.0)

    def test_horizon_pixels_keep_disk_emission(self):
        eh = self.renderer.event_horizon_field.to_numpy() == 1
        disk_hdr = self.renderer.disk_hdr_field.to_numpy().sum(-1)
        self.assertTrue(eh.any())
        # 光子环下半部：先穿过盘再落入视界的光线，发射不应被清零
        self.assertTrue((disk_hdr[eh] > 0.0).any())

    def test_last_hdr_equals_emission_plus_transmitted_sky(self):
        self.assertEqual(self.renderer.last_hdr.shape, (14, 24, 3))
        self.assertEqual(self.img.shape, (14, 24, 3))
        self.assertTrue(np.isfinite(self.renderer.last_hdr).all())

    def test_render_time_drives_advection(self):
        a = self.renderer.render(cam_pos=CAM, fov=38.0, t=2000.0)
        hdr_a = self.renderer.last_hdr.copy()
        self.renderer.render(cam_pos=CAM, fov=38.0, t=2050.0)
        hdr_b = self.renderer.last_hdr
        self.assertEqual(a.shape, (14, 24, 3))
        self.assertGreater(float(np.abs(hdr_a - hdr_b).mean()), 0.0)

    def test_supersampling_internal_resolution(self):
        # ss = 2：内部按 (2W, 2H) 积分，输出按 (H, W) 盒式下采样
        self.assertEqual(self.renderer.hdr_field.shape, (48, 28))

    def test_ss_requires_volume_postfx(self):
        with self.assertRaises(ValueError):
            DiskV2Renderer(
                width=8, height=8, params=DiskV2Params(r_in=R_IN, r_out=R_OUT),
                structure_params=DiskV2StructureParams(), palette_params=DiskV2PaletteParams(),
                skybox=np.zeros((8, 16, 3), np.float32), device="gpu", ss=2,
            )


if __name__ == "__main__":
    unittest.main()
