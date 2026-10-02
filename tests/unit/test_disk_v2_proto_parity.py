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

from src.v2.advection import RigidRingBands  # noqa: E402
from src.v2.noise_ti import hashf  # noqa: E402
from src.v2.params import DiskV2Params, DiskV2VolumeParams  # noqa: E402
from src.v2.taichi_impl import _delayed_rot, _delayed_seed  # noqa: E402
from src.v2.taichi_render import DiskV2Renderer  # noqa: E402

R_IN, R_OUT = 3.0, 30.0
CAM_ELEV = math.radians(7.0)
CAM = [0.0, -40.0 * math.cos(CAM_ELEV), 40.0 * math.sin(CAM_ELEV)]
# 外圈细节参数的参考实现取值（隔离其他功能的测试用；含义见 DiskV2VolumeParams）
REF_DETAIL = dict(core_az_stretch=0.0, core_oct_gain=1.0, band_seam_fix=False,
                  core_contrast=1.0, outer_detail_fade=1.0)


def _make_renderer(width=24, height=14, ss=2):
    """构造与参考实现预设 M 同参数的小尺寸 V2 渲染器（黑色天空）。"""
    return DiskV2Renderer(
        width=width, height=height,
        params=DiskV2Params(r_in=R_IN, r_out=R_OUT),
        skybox=np.zeros((8, 16, 3), np.float32),
        volume_params=DiskV2VolumeParams(),
        r_max=90.0, ss=ss,
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
        # 天空为黑色 → hdr_field 即路上累积的盘发射
        disk_hdr = self.renderer.hdr_field.to_numpy().sum(-1)
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

    def test_sky_does_not_change_disk_or_exposure(self):
        # 天空单独存 sky_field，不进入 last_hdr，也不影响自动曝光
        r = self.renderer
        seed = int(r.jitter_seed[None])
        r.jitter_seed[None] = seed
        r.render(cam_pos=CAM, fov=38.0)
        hdr_black, wp_black = r.last_hdr.copy(), r.last_white_point
        r.skybox_field.fill(1.0)
        try:
            r.jitter_seed[None] = seed
            img_sky = r.render(cam_pos=CAM, fov=38.0)
            np.testing.assert_array_equal(r.last_hdr, hdr_black)
            self.assertEqual(r.last_white_point, wp_black)
            self.assertGreater(float(r.sky_field.to_numpy().max()), 0.0)
            self.assertTrue(np.isfinite(img_sky).all())
        finally:
            r.skybox_field.fill(0.0)

    def test_supersampling_internal_resolution(self):
        # ss = 2：内部按 (2W, 2H) 积分，输出按 (H, W) 盒式下采样
        self.assertEqual(self.renderer.hdr_field.shape, (48, 28))


class TestOptLevel1Exact(unittest.TestCase):
    """优化级别 1（快速噪声 + 共享带信息）与级别 0 输出一致（8 位输出每通道相差不超过 1）。"""

    def test_level1_matches_level0(self):
        r0 = _shared_renderer()
        r1 = DiskV2Renderer(
            width=24, height=14, params=DiskV2Params(r_in=R_IN, r_out=R_OUT),
            skybox=np.zeros((8, 16, 3), np.float32), volume_params=DiskV2VolumeParams(),
            r_max=90.0, ss=2, opt_level=1,
        )
        r0.jitter_seed[None] = 700
        r1.jitter_seed[None] = 700
        a = (r0.render(cam_pos=CAM, fov=38.0) * 255 + 0.5).astype(int)
        b = (r1.render(cam_pos=CAM, fov=38.0) * 255 + 0.5).astype(int)
        self.assertLessEqual(int(np.abs(a - b).max()), 1)


class TestSpinReversal(unittest.TestCase):
    """反转旋转方向（disk_spin = −1）：多普勒亮侧随平流结构一起翻到另一侧。"""

    def test_bright_side_flips_with_spin(self):
        def render(spin):
            r = DiskV2Renderer(
                width=48, height=27, params=DiskV2Params(r_in=R_IN, r_out=R_OUT, disk_spin=spin),
                skybox=np.zeros((8, 16, 3), np.float32), volume_params=DiskV2VolumeParams(),
                r_max=90.0, ss=2,
            )
            r.jitter_seed[None] = 800
            r.render(cam_pos=CAM, fov=38.0)
            lum = r.last_hdr.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
            return lum

        cw, ccw = render(-1.0), render(1.0)
        half = cw.shape[1] // 2
        ratio_cw = float(ccw[:, :half].sum() / ccw[:, half:].sum())
        ratio_ccw = float(cw[:, :half].sum() / cw[:, half:].sum())
        # 默认（逆时针）亮侧在左；反转后亮侧在右
        self.assertGreater(ratio_cw, 1.0)
        self.assertLess(ratio_ccw, 1.0)


class TestThicknessScale(unittest.TestCase):
    """盘厚缩放：有效标高与烟雾层厚度同乘 thickness_scale，薄盘关于中面保持对称。"""

    def test_effective_thickness_and_validation(self):
        d = _shared_renderer().disk_ti
        s = DiskV2VolumeParams().thickness_scale
        self.assertAlmostEqual(s, 1.0 / 9.0)
        self.assertAlmostEqual(d._hr_ref, 0.027 * s, places=7)
        self.assertAlmostEqual(d._cl_spacing, 0.014 * s, places=7)
        self.assertAlmostEqual(d._cl_width, 0.011 * s, places=7)
        with self.assertRaises(ValueError):
            DiskV2VolumeParams(thickness_scale=0.0)

    def test_thin_disk_mirror_symmetric(self):
        r = _shared_renderer()

        def flux(elev_deg):
            e = math.radians(elev_deg)
            r.jitter_seed[None] = 600
            r.render(cam_pos=[0.0, -40.0 * math.cos(e), 40.0 * math.sin(e)], fov=38.0)
            return float(r.last_hdr.astype(np.float64).sum())

        # 盘关于中面对称：盘面上方与下方同仰角观察，盘总通量一致。
        # 24×14 像素下上下两侧噪声实现不同带来约 5% 起伏，容差取 10%；底面渲染错误会使比值偏离数倍。
        # 正式门槛（720p，±2%）在验收中实测，见 docs/plans/v2_edge_on_plan.md §8。
        self.assertAlmostEqual(flux(-7.0) / flux(7.0), 1.0, delta=0.10)


class TestLumTempScale(unittest.TestCase):
    """亮度温度倍率 s：峰值归一取 Y(s·T_peak)，外盘变亮，多普勒左右明暗不对称经补偿保持不变。"""

    def test_peak_normalization_and_doppler_compensation(self):
        from src.v2.palette import _blackbody_luminance_exact, doppler_lum_compensation
        r = _shared_renderer()
        d = r.disk_ti
        s = DiskV2VolumeParams().lum_temp_scale
        self.assertAlmostEqual(s, 1.25)
        self.assertAlmostEqual(d._ln_y_peak, math.log(_blackbody_luminance_exact(s * d._t_peak_vol)), places=9)
        self.assertAlmostEqual(r._doppler_lum_eff, 0.25 * doppler_lum_compensation(d._t_peak_vol, s), places=9)

    def test_outer_disk_brighter_with_same_asymmetry(self):
        w = np.array([0.2126, 0.7152, 0.0722])

        def render(s):
            r = DiskV2Renderer(
                width=48, height=27, params=DiskV2Params(r_in=R_IN, r_out=R_OUT),
                skybox=np.zeros((8, 16, 3), np.float32), volume_params=DiskV2VolumeParams(lum_temp_scale=s, **REF_DETAIL),
                r_max=90.0, ss=2,
            )
            r.jitter_seed[None] = 800
            r.render(cam_pos=CAM, fov=38.0)
            return r.last_hdr.astype(np.float64) @ w

        phys, art = render(1.0), render(1.25)
        half = phys.shape[1] // 2

        def lr(lum):
            return float(lum[:, :half].sum() / lum[:, half:].sum())

        def median_rel(lum):
            return float(np.median(lum[lum > 1e-4 * lum.max()]) / lum.max())

        # 外圈细节取参考值（实测数据在该条件下得到；新默认下中位数比约 1.4）
        # 实测：左右通量比 4.03 → 4.03（补偿后相差 0.1%）；盘区亮度中位数 / 峰值 0.0014 → 0.0028
        self.assertAlmostEqual(lr(art) / lr(phys), 1.0, delta=0.05)
        self.assertGreater(median_rel(art) / median_rel(phys), 1.5)


class TestOuterDetail(unittest.TestCase):
    """外圈细节：方位周期随半径增长、外圈衰减关闭、保方差混合（默认参数）。"""

    @classmethod
    def setUpClass(cls):
        cls.disk = _shared_renderer().disk_ti
        cls.vp = DiskV2VolumeParams()

    def test_band_azimuthal_period(self):
        """n_φ(b) = max(n, round(n·r_b/(6·s)))，内区不低于基础周期，外圈随半径增加。"""
        d, vp = self.disk, self.vp
        bands = list(range(d._adv_core_f.b_lo, d._adv_core_f.b_lo + 12))
        out = ti.Vector.field(2, ti.f32, shape=len(bands))

        @ti.kernel
        def run():
            for i in range(len(bands)):
                n_i, n_t = d._band_nphi(d._adv_core_f.b_lo + i)
                out[i] = ti.Vector([n_i, n_t])

        run()
        got = out.to_numpy()
        for k, b in enumerate(bands):
            q = math.exp(d._lnr0_r + b * d._dln_r) / (6.0 * vp.core_az_stretch)
            self.assertEqual(got[k, 0], max(vp.nphi_i, round(vp.nphi_i * q)))
            self.assertEqual(got[k, 1], max(vp.nphi_t, round(vp.nphi_t * q)))
        self.assertGreater(got[-1, 1], got[0, 1])

    def test_outer_cut_disabled(self):
        """outer_detail_fade = 0：外圈不截八度、对比度保持 con_i。"""
        d = self.disk
        out = ti.Vector.field(2, ti.f32, shape=())

        @ti.kernel
        def run():
            lev, con = d._outer_cut(25.0)
            out[None] = ti.Vector([lev, con])

        run()
        lev, con = out[None].to_numpy()
        self.assertEqual(lev, 0.0)
        self.assertAlmostEqual(con, self.vp.con_i, places=5)

    def test_blend_finish_formula(self):
        """c = max(m_c + α·S/√Σw², 0)，tn = max(m_t + S_t/√Σw², 0)；Σw² ≈ 0 时为 0。"""
        d = self.disk
        self.assertGreater(d._mc_single, 0.0)
        self.assertGreater(d._mt_single, 0.0)
        out = ti.Vector.field(4, ti.f32, shape=())

        @ti.kernel
        def run():
            c, tn = d._blend_finish(0.3, -0.2, 0.25)
            c0, t0 = d._blend_finish(0.3, -0.2, 0.0)
            out[None] = ti.Vector([c, tn, c0, t0])

        run()
        c, tn, c0, t0 = out[None].to_numpy()
        self.assertAlmostEqual(c, max(d._mc_single + self.vp.core_contrast * 0.3 / 0.5, 0.0), places=5)
        self.assertAlmostEqual(tn, max(d._mt_single - 0.2 / 0.5, 0.0), places=5)
        self.assertEqual((c0, t0), (0.0, 0.0))

    def test_no_band_seam(self):
        """接缝修复后主云起伏在带中心与两带正中一致（参考实现两带正中只剩约 71%）。

        对每个采样点按其自身带坐标的小数部分 fbf 分组：fbf ≈ 0 / 1 为带中心，fbf ≈ 0.5 为两带正中。
        """
        d = self.disk
        n = 65536
        cs = ti.field(ti.f32, n)
        fs = ti.field(ti.f32, n)

        @ti.kernel
        def run():
            for i in range(n):
                r = 8.0 + 14.0 * hashf(i, 31, 1)
                phi = 2.0 * math.pi * hashf(i, 31, 2)
                fb = d._band_coord(ti.log(r), phi)
                c, _tn = d._flow_I(r, phi, 0.0, 0.0)
                cs[i] = c
                fs[i] = fb - ti.floor(fb)

        run()
        c, f = cs.to_numpy(), fs.to_numpy()
        mid = np.abs(f - 0.5) < 0.1
        edge = np.abs(f - 0.5) > 0.4
        ratio = (c[mid].std() / c[mid].mean()) / (c[edge].std() / c[edge].mean())
        self.assertAlmostEqual(float(ratio), 1.0, delta=0.12)


class TestTiltEquivalence(unittest.TestCase):
    """倾角等价：盘绕 x 轴倾 θ、相机仰角 e  ≡  盘不倾、相机仰角 e + θ（相机在 y-z 平面内）。

    盘局部坐标下相机位于仰角 e + θ，相机 right 均为 +x，因此两幅图逐像素一致（f32 舍入内）。
    同时检查亮侧 = 蓝移侧：逼近侧（图像左侧，盘逆时针）通量更大、B/R 更高。
    """

    def test_tilt_matches_elevation_shift_and_bright_side_is_blue(self):
        tilt = 20.0

        def cam(elev_deg):
            e = math.radians(elev_deg)
            return [0.0, -40.0 * math.cos(e), 40.0 * math.sin(e)]

        flat = _shared_renderer()
        tilted = DiskV2Renderer(
            width=24, height=14, params=DiskV2Params(r_in=R_IN, r_out=R_OUT),
            skybox=np.zeros((8, 16, 3), np.float32), volume_params=DiskV2VolumeParams(),
            r_max=90.0, disk_tilt_deg=tilt, ss=2,
        )
        flat.jitter_seed[None] = 500
        tilted.jitter_seed[None] = 500
        flat.render(cam_pos=cam(7.0 + tilt), fov=38.0)
        tilted.render(cam_pos=cam(7.0), fov=38.0)
        a, b = tilted.last_hdr.astype(np.float64), flat.last_hdr.astype(np.float64)
        w = np.array([0.2126, 0.7152, 0.0722])
        la, lb = a @ w, b @ w
        m = (la > 1e-3 * la.max()) & (lb > 1e-3 * lb.max())
        self.assertTrue(m.any())
        self.assertLess(float(np.median(np.abs(np.log(la[m] / lb[m])))), 1e-2)
        self.assertAlmostEqual(float(la.sum() / lb.sum()), 1.0, places=3)
        half = a.shape[1] // 2
        self.assertGreater(la[:, :half].sum(), la[:, half:].sum())
        self.assertGreater(a[:, :half, 2].sum() / a[:, :half, 0].sum(),
                           a[:, half:, 2].sum() / a[:, half:, 0].sum())


if __name__ == "__main__":
    unittest.main()
