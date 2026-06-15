"""Disk V2 独立 Taichi 渲染器（v2.1 Phase 4）。

`DiskV2Renderer` 是一个最小可用的 V2 渲染器：

- 复用 Schwarzschild 笛卡尔等效势 + RK4 测地线积分（与 `render.py` 同公式）。
- 命中盘体包围体时做有限步长发射-吸收积分。
- HDR 累积 → 可选 HDR-domain Bloom → Reinhard → sRGB 伽马。

支持：

- V2 有限厚度发射-吸收积分。
- Phase 5 g-factor 相对论亮度与颜色修正。
- Phase 7 HDR-domain Bloom（参数仍需视觉验收重校）。

暂不支持：

- lens flare / 色散等 V1 后处理。
- 视频 / 交互模式（视觉验收专用，单帧足够）。

调用方式（由 `render.py --disk_model v2` 触发）：

```python
renderer = DiskV2Renderer(width=1280, height=720, ...)
img = renderer.render(cam_pos=[6, 0, 2], fov=90.0)
```
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import taichi as ti

from .camera import build_camera_v1_compatible
from .imaging import reference_exposure
from .params import DiskV2PaletteParams, DiskV2Params, DiskV2StructureParams
from .stats import RenderStats, compute_render_stats, hdr_luminance
from .structure_modulations import _ClumpCenters
from .taichi_impl import (
    DiskV2Taichi,
    doppler_g_factor_ti,
    disk_half_thickness_ti,
    disk_volume_mask_ti,
    schwarzschild_gravitational_g_ti,
    schwarzschild_orbital_beta_ti,
)


# Schwarzschild 半径，与 render.py 保持一致（无量纲单位）。
_RS: float = 1.0


@ti.data_oriented
class DiskV2Renderer:
    """V2 最小可用 Taichi 渲染器。

    Args:
        width: 输出图像宽度（像素）。
        height: 输出图像高度（像素）。
        params: `DiskV2Params`。
        structure_params: `DiskV2StructureParams`。
        palette_params: `DiskV2PaletteParams`。
        skybox: 程序生成的天空盒 `(tex_h, tex_w, 3)` 数组（float32, [0,1]）。
        step_size: 远场光线积分基础步长。
        r_max: 逃逸半径下限；实际 `r_escape = max(r_max, 2*cam_distance)`。
        disk_tilt_deg: 盘倾角（度）。
        volume_samples: 盘内体积积分步数。
        opacity_scale: 盘内吸收系数 = `opacity_scale * ρ`。
        seed: 团块随机种子。
        centers: 可选预生成 `_ClumpCenters`。
        device: `"cpu"` 或 `"gpu"`。
        ignore_taichi_cache: 是否清除 Taichi 编译缓存。
    """

    def __init__(
        self,
        width: int,
        height: int,
        params: DiskV2Params,
        structure_params: DiskV2StructureParams,
        palette_params: DiskV2PaletteParams,
        skybox: np.ndarray,
        step_size: float = 0.1,
        r_max: float = 10.0,
        disk_tilt_deg: float = 0.0,
        volume_samples: int = 16,
        opacity_scale: float = 0.5,
        emission_scale: float = 1.0,
        lum_power: float = 4.0,
        g_cap: float = 6.0,
        enable_g_factor: bool = True,
        bloom_threshold: float = 1.0,
        bloom_intensity: float = 0.0,
        bloom_radius: float = 4.0,
        auto_exposure: bool = False,
        white_point_percentile: float = 99.0,
        print_stats: bool = False,
        seed: int = 42,
        centers: Optional[_ClumpCenters] = None,
        device: str = "cpu",
        ignore_taichi_cache: bool = False,
    ) -> None:
        """初始化 V2 渲染器。

        Args:
            width, height: 输出图像尺寸。
            params, structure_params, palette_params: V2 三类参数。
            skybox: 程序生成天空盒。
            step_size: 远场 RK4 基础步长。
            r_max: 逃逸半径下限。
            disk_tilt_deg: 盘倾角（度）。
            volume_samples: 盘内体积积分步数。
            opacity_scale: 盘内吸收系数 = `opacity_scale * ρ`。
            emission_scale: HDR 发射率整体曝光缩放。默认 `1.0` 保持原始积分亮度；
                视觉验收时可调低，避免盘面在 tonemap 前整体饱和。
            lum_power: g-factor 亮度指数（Phase 5 默认 4，对应 `I_obs / I_em = g^4`
                的相对论强度变换）。
            g_cap: g-factor 上限，避免极端蓝移侧饱和到无穷。
            enable_g_factor: 是否启用 Phase 5 相对论亮度/颜色处理。`False` 时
                只输出无方向性的发射率。
            bloom_threshold: Phase 7 Bloom 亮度阈值（在 HDR 域提取超过此值的像素）。
                推荐 `1.0`（tonemap 前 HDR 强度 > 1 视为高亮）。
            bloom_intensity: Phase 7 Bloom 强度，`0.0` 关闭。推荐 `0.3 ~ 0.6` 为
                弱 Bloom，`0.8 ~ 1.2` 为电影感预设。
            bloom_radius: Phase 7 Bloom 高斯模糊半径（像素）。推荐 `3 ~ 8`。
            auto_exposure: 是否根据 HDR 亮度分位数自动设置 white point。
            white_point_percentile: auto exposure 使用的 HDR 亮度分位数（默认 99）。
            print_stats: 渲染完成后是否打印 HDR/LDR 诊断统计。
            seed: 团块随机种子。
            centers: 可选预生成 `_ClumpCenters`。
            device: `"cpu"` 或 `"gpu"`。
            ignore_taichi_cache: 是否清除 Taichi 编译缓存。
        """
        # 这里假设 Taichi 已经被 render.py 主入口或外部调用方初始化过；
        # 如果尚未初始化则用 device 触发一次。
        if not ti.lang.impl.get_runtime().materialized:
            arch = ti.cpu if device == "cpu" else ti.gpu
            ti.init(arch=arch, default_fp=ti.f32)

        self.width = width
        self.height = height
        self.params = params
        self.structure_params = structure_params
        self.palette_params = palette_params
        self.step_size = float(step_size)
        self.r_max = float(r_max)
        self.disk_tilt_rad = math.radians(disk_tilt_deg)
        self.volume_samples = int(volume_samples)
        self.opacity_scale = float(opacity_scale)
        self.emission_scale = float(emission_scale)
        self.lum_power = float(lum_power)
        self.g_cap = float(g_cap)
        self.enable_g_factor = bool(enable_g_factor)
        self.bloom_threshold = float(bloom_threshold)
        self.bloom_intensity = float(bloom_intensity)
        self.bloom_radius = float(bloom_radius)
        self.auto_exposure = bool(auto_exposure)
        self.white_point_percentile = float(white_point_percentile)
        self.print_stats = bool(print_stats)
        self.last_stats: RenderStats | None = None
        self.last_white_point: float = 1.0
        self.last_actual_hdr_white_point: float | None = None
        exposure_ref = reference_exposure(params, opacity_scale)
        self.reference_white_point: float = 1.0 / max(float(exposure_ref), 1e-12)

        # 把基础场和 palette 包装为 Taichi 句柄。
        self.disk_ti = DiskV2Taichi(
            params=params,
            structure_params=structure_params,
            palette_params=palette_params,
            emission_opacity_scale=opacity_scale,
            seed=seed,
            centers=centers,
        )

        # 输出图像 field（HDR 浮点；Bloom + tonemap 在 Python 端 / 简化 kernel 完成）。
        self.hdr_field = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self.image_field = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        # 方向 1（2026-06-14）：背景与盘分离处理
        # - disk_hdr_field: 盘的物理通量（参与曝光 + tonemap + bloom）
        # - disk_alpha_field: 盘 + 事件视界的累积不透明度（合成时分配 disk vs sky）
        # - sky_hdr_field: 背景天空原始颜色（**不**参与曝光、tonemap、bloom）
        # 解决 cinematic auto_exposure 把暗背景 (luma~0.02) ×exposure_scale(~1400)
        # 抬到 HDR~30 → Reinhard 0.97 → 全图灰白的问题。
        self.disk_hdr_field = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self.disk_alpha_field = ti.field(dtype=ti.f32, shape=(width, height))
        self.sky_hdr_field = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        # V1 风格 LDR bloom 所需 fields：
        # - disk_ldr_field: 盘经 exposure+tonemap+gamma 后的 LDR，bloom 输入输出
        # - bright_field: 高亮提取缓冲
        # - blur_field: separable 高斯中间缓冲
        self.disk_ldr_field = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self.bright_field = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self.blur_field = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        # V1 bloom 参数 fields（Taichi 闭包 kernel 不接受任何参数注解，
        # 只能通过 field 传递 runtime 值）
        self.bloom_threshold_field = ti.field(dtype=ti.f32, shape=())
        self.bloom_intensity_field = ti.field(dtype=ti.f32, shape=())
        self.bloom_kernel_radius_field = ti.field(dtype=ti.i32, shape=())
        self.bloom_sigma_scale_field = ti.field(dtype=ti.f32, shape=())
        # event_horizon mask：bloom 后强制黑洞内部为黑色
        self.event_horizon_field = ti.field(dtype=ti.i32, shape=(width, height))

        # 上传 skybox。
        sky_h, sky_w = skybox.shape[:2]
        self.sky_w = int(sky_w)
        self.sky_h = int(sky_h)
        self.skybox_field = ti.Vector.field(3, dtype=ti.f32, shape=(sky_w, sky_h))
        # Taichi field 下标为 [x, y]，而 NumPy skybox 是 [y, x, rgb]。
        # 上传前转置，避免采样到错位的天空盒数据。
        self.skybox_field.from_numpy(np.transpose(skybox.astype(np.float32), (1, 0, 2)))

        # 相机参数 field（每帧更新）。
        self.cam_pos_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_right_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_up_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_forward_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.pixel_width_field = ti.field(dtype=ti.f32, shape=())
        self.pixel_height_field = ti.field(dtype=ti.f32, shape=())
        self.top_left_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.r_escape_field = ti.field(dtype=ti.f32, shape=())
        self.white_point_field = ti.field(dtype=ti.f32, shape=())
        self.white_point_field[None] = 1.0

        # 编译主内核。
        self._compile_kernels()

    def _compile_kernels(self) -> None:
        disk = self.disk_ti
        tilt = float(self.disk_tilt_rad)
        h_base = float(self.step_size)
        rs = float(_RS)
        volume_samples = int(self.volume_samples)
        opacity_scale = float(self.opacity_scale)
        emission_scale = float(self.emission_scale)
        sky_w = int(self.sky_w)
        sky_h = int(self.sky_h)
        lum_power = float(self.lum_power)
        g_cap = float(self.g_cap)
        enable_g = bool(self.enable_g_factor)

        # Phase 5 g-factor 公式集中在 taichi_impl.py helper 中，避免体积/薄层路径漂移。

        @ti.func
        def _compute_acceleration(pos, L2):
            """Schwarzschild 笛卡尔等效势的加速度：a = -1.5 L² x / r⁵。"""
            r2 = pos.dot(pos)
            r = ti.sqrt(r2)
            r5 = r2 * r2 * r
            return -1.5 * L2 / r5 * pos

        @ti.func
        def _sample_skybox(direction):
            """根据光线方向采样天空盒。"""
            d = direction.normalized()
            phi = ti.atan2(d[1], d[0])
            theta = ti.acos(ti.min(ti.max(d[2], -1.0), 1.0))
            u = (phi + ti.math.pi) / (2.0 * ti.math.pi)
            v = theta / ti.math.pi
            x = ti.cast(u * sky_w, ti.i32) % sky_w
            y = ti.cast(v * sky_h, ti.i32)
            if y < 0:
                y = 0
            if y >= sky_h:
                y = sky_h - 1
            return self.skybox_field[x, y]

        @ti.func
        def _world_to_local_disk(pos):
            """世界坐标 → 盘体局部坐标（盘面绕 x 轴倾斜 tilt 弧度）。"""
            # 盘法向 (0, -sin tilt, cos tilt)；局部 z' 是世界 (cos tilt) z' = pos · 盘法向。
            sin_t = ti.sin(tilt)
            cos_t = ti.cos(tilt)
            # 旋转：把世界 (y, z) 倒转回盘的 (y', z')。
            x_local = pos[0]
            y_local = pos[1] * cos_t + pos[2] * sin_t
            z_local = -pos[1] * sin_t + pos[2] * cos_t
            return ti.Vector([x_local, y_local, z_local], dt=ti.f32)

        tan_t = ti.tan(tilt)
        use_visual_atlas = bool(disk._use_visual_atlas)
        img_w = int(self.width)
        img_h = int(self.height)
        min_fac = ti.cast(0.2, ti.f32)
        max_fac = ti.cast(10.0, ti.f32)
        alpha_gain = ti.cast(6.0, ti.f32)

        @ti.kernel
        def _ray_march_kernel():
            """V2 主光追内核：Schwarzschild 测地线 + V2 盘体体积积分。"""
            r_esc = self.r_escape_field[None]
            max_iter = ti.cast(r_esc * 40.0 / h_base, ti.i32)
            max_affine = r_esc * 40.0
            r_cap = 1.0 * rs

            cp = self.cam_pos_field[None]
            cr = self.cam_right_field[None]
            cu = self.cam_up_field[None]
            cf = self.cam_forward_field[None]
            pw = self.pixel_width_field[None]
            ph = self.pixel_height_field[None]
            # 与 V1 `render._ray_march_kernel` 完全一致：垂直 FOV + aspect 像素步长。
            center = cp + cf * 1.0
            tl = center - cr * (pw * img_w / 2.0) + cu * (ph * img_h / 2.0)

            for i, j in self.image_field:
                px_f = ti.cast(i, ti.f32)
                py_f = ti.cast(j, ti.f32)
                pixel_pos = tl + (px_f + 0.5) * pw * cr - (py_f + 0.5) * ph * cu
                ray_dir = (pixel_pos - cp).normalized()

                pos = cp
                dir_ = ray_dir

                # 角动量平方 L² = |r × dir|²。
                L_vec = pos.cross(dir_)
                L2_val = L_vec.dot(L_vec)

                escaped = False
                escape_dir = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                event_horizon_hit = False
                hdr_accum = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                transmittance = 1.0
                plane_disk = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                plane_alpha_total = 0.0
                step_count = 0
                affine = 0.0

                while step_count < max_iter and transmittance > 1e-4:
                    old_pos = pos
                    r_cur = pos.norm()
                    r_safe = ti.max(r_cur, r_cap + 1e-3)
                    far_scale = ti.sqrt(r_safe / r_cap)
                    if far_scale > max_fac:
                        far_scale = max_fac
                    near_damp = 1.0 / (1.0 + 2.0 * (ti.pow(r_cap / r_safe, 3)))
                    dt_fac = far_scale * near_damp
                    if dt_fac < min_fac:
                        dt_fac = min_fac
                    if dt_fac > max_fac:
                        dt_fac = max_fac
                    h = h_base * dt_fac

                    # RK4 主光线。
                    k1p = h * dir_
                    k1d = h * _compute_acceleration(pos, L2_val)
                    k2p = h * (dir_ + 0.5 * k1d)
                    k2d = h * _compute_acceleration(pos + 0.5 * k1p, L2_val)
                    k3p = h * (dir_ + 0.5 * k2d)
                    k3d = h * _compute_acceleration(pos + 0.5 * k2p, L2_val)
                    k4p = h * (dir_ + k3d)
                    k4d = h * _compute_acceleration(pos + k3p, L2_val)
                    new_pos = pos + (k1p + 2 * k2p + 2 * k3p + k4p) / 6.0
                    new_dir = dir_ + (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0

                    r = new_pos.norm()
                    affine += h

                    if r < r_cap:
                        event_horizon_hit = True
                        break
                    elif r > r_esc:
                        escaped = True
                        escape_dir = new_dir.normalized()
                        break
                    elif affine > max_affine:
                        escaped = True
                        escape_dir = new_dir.normalized()
                        break

                    if ti.static(not use_visual_atlas):
                        old_local = _world_to_local_disk(old_pos)
                        new_local = _world_to_local_disk(new_pos)
                        old_r = ti.sqrt(old_local[0] ** 2 + old_local[1] ** 2)
                        new_r = ti.sqrt(new_local[0] ** 2 + new_local[1] ** 2)
                        old_h = disk_half_thickness_ti(
                            old_r, disk._h0, disk._beta_h, disk._r_in,
                        )
                        new_h = disk_half_thickness_ti(
                            new_r, disk._h0, disk._beta_h, disk._r_in,
                        )
                        max_h = ti.max(old_h, new_h, 1e-4)
                        old_in = disk_volume_mask_ti(
                            old_r, old_local[2],
                            disk._h0, disk._beta_h, disk._r_in, disk._r_out,
                        )
                        new_in = disk_volume_mask_ti(
                            new_r, new_local[2],
                            disk._h0, disk._beta_h, disk._r_in, disk._r_out,
                        )
                        crosses_midplane = old_local[2] * new_local[2] < 0.0
                        near_midplane = ti.min(
                            ti.abs(old_local[2]), ti.abs(new_local[2]),
                        ) < max_h * 2.5
                        cross_r = old_r
                        if crosses_midplane:
                            t_cross = ti.abs(old_local[2]) / (
                                ti.abs(old_local[2]) + ti.abs(new_local[2]) + 1e-8
                            )
                            cross_y = old_local[1] + t_cross * (new_local[1] - old_local[1])
                            cross_x = old_local[0] + t_cross * (new_local[0] - old_local[0])
                            cross_r = ti.sqrt(cross_x ** 2 + cross_y ** 2)
                        in_radial_span = (
                            (old_r >= disk._r_in and old_r <= disk._r_out)
                            or (new_r >= disk._r_in and new_r <= disk._r_out)
                            or (
                                crosses_midplane
                                and cross_r >= disk._r_in
                                and cross_r <= disk._r_out
                            )
                        )
                        passes_disk = (
                            old_in == 1
                            or new_in == 1
                            or ((crosses_midplane or near_midplane) and in_radial_span)
                        )
                        if passes_disk:
                            n_vol = volume_samples
                            if crosses_midplane or near_midplane:
                                n_vol = volume_samples * 2
                            for s_idx in range(n_vol):
                                t = (ti.cast(s_idx, ti.f32) + 0.5) / n_vol
                                sample_pos = old_pos + t * (new_pos - old_pos)
                                sl = _world_to_local_disk(sample_pos)
                                r_local = ti.sqrt(sl[0] ** 2 + sl[1] ** 2)
                                z_local = sl[2]
                                phi_local = ti.atan2(sl[1], sl[0])
                                in_disk = disk_volume_mask_ti(
                                    r_local, z_local,
                                    disk._h0, disk._beta_h, disk._r_in, disk._r_out,
                                )
                                if in_disk == 1:
                                    j_em = disk.sample_emission(r_local, phi_local, z_local)
                                    T_K = disk.sample_temperature(r_local, z_local)
                                    atlas_color = disk.sample_atlas_color_mod_ti(r_local, phi_local)
                                    rho = disk.sample_density(r_local, phi_local, z_local)
                                    alpha_k = opacity_scale * ti.max(rho, 0.0)
                                    g_total = 1.0
                                    if ti.static(enable_g):
                                        r_safe_em = ti.max(r_local, 3.0 * rs)
                                        inv_r = 1.0 / ti.max(r_local, 1e-6)
                                        v_hat_local = ti.Vector([
                                            -sl[1] * inv_r, sl[0] * inv_r, 0.0,
                                        ], dt=ti.f32)
                                        sin_t = ti.sin(tilt)
                                        cos_t = ti.cos(tilt)
                                        v_hat_world = ti.Vector([
                                            v_hat_local[0],
                                            v_hat_local[1] * cos_t - v_hat_local[2] * sin_t,
                                            v_hat_local[1] * sin_t + v_hat_local[2] * cos_t,
                                        ], dt=ti.f32)
                                        ray_to_cam = -dir_
                                        cos_theta = v_hat_world.dot(ray_to_cam.normalized())
                                        beta = schwarzschild_orbital_beta_ti(r_safe_em, rs, 0.01, 0.99)
                                        g_doppler = doppler_g_factor_ti(beta, cos_theta)
                                        r_obs = cp.norm()
                                        g_grav = schwarzschild_gravitational_g_ti(r_safe_em, r_obs, rs)
                                        g_total = ti.min(g_doppler * g_grav, g_cap)
                                    intensity_factor = ti.pow(ti.max(g_total, 0.1), lum_power)
                                    color = disk.sample_observed_palette_color(T_K, g_total)
                                    ds = (new_pos - old_pos).norm() / n_vol
                                    shifted_color = ti.Vector([
                                        color[0] * atlas_color,
                                        color[1] * atlas_color,
                                        color[2] * atlas_color,
                                    ], dt=ti.f32)
                                    hdr_accum += transmittance * emission_scale * j_em * shifted_color * intensity_factor * ds
                                    transmittance *= ti.exp(-alpha_k * ds)
                    else:
                        # Thin-layer visual path:
                        # use_visual_atlas=True 时，为保留快速视觉验收路径，这里退化为
                        # 倾斜中面单次命中 + 表面发射/alpha 合成。它不是真正的有限厚度
                        # 体积积分；但发射、颜色、g-factor 与 HDR/tonemap 仍走同一物理 helper。
                        f_old = old_pos[2] - old_pos[1] * tan_t
                        f_new = new_pos[2] - new_pos[1] * tan_t
                        if f_old * f_new < 0.0:
                            t_frac = f_old / (f_old - f_new + 1e-8)
                            hit_x = old_pos[0] + t_frac * (new_pos[0] - old_pos[0])
                            hit_y = old_pos[1] + t_frac * (new_pos[1] - old_pos[1])
                            hit_r = ti.sqrt(hit_x * hit_x + hit_y * hit_y)
                            if hit_r >= disk._r_in and hit_r <= disk._r_out:
                                phi_hit = ti.atan2(hit_y, hit_x)
                                j_em = disk.sample_emission(hit_r, phi_hit, 0.0)
                                rho = disk.sample_density(hit_r, phi_hit, 0.0)
                                T_K = disk.sample_temperature(hit_r, 0.0)
                                g_total = 1.0
                                if ti.static(enable_g):
                                    r_safe_em = ti.max(hit_r, 3.0 * rs)
                                    inv_r = 1.0 / ti.max(hit_r, 1e-6)
                                    v_hat_local = ti.Vector([
                                        -hit_y * inv_r, hit_x * inv_r, 0.0,
                                    ], dt=ti.f32)
                                    sin_t = ti.sin(tilt)
                                    cos_t = ti.cos(tilt)
                                    v_hat_world = ti.Vector([
                                        v_hat_local[0],
                                        v_hat_local[1] * cos_t - v_hat_local[2] * sin_t,
                                        v_hat_local[1] * sin_t + v_hat_local[2] * cos_t,
                                    ], dt=ti.f32)
                                    ray_to_cam = -dir_
                                    cos_theta = v_hat_world.dot(ray_to_cam.normalized())
                                    beta = schwarzschild_orbital_beta_ti(r_safe_em, rs, 0.01, 0.99)
                                    g_doppler = doppler_g_factor_ti(beta, cos_theta)
                                    r_obs = cp.norm()
                                    g_grav = schwarzschild_gravitational_g_ti(r_safe_em, r_obs, rs)
                                    g_total = ti.min(g_doppler * g_grav, g_cap)
                                intensity_factor = ti.pow(ti.max(g_total, 0.1), lum_power)
                                color = disk.sample_observed_palette_color(T_K, g_total)
                                shifted_color = ti.Vector([
                                    color[0],
                                    color[1],
                                    color[2],
                                ], dt=ti.f32)
                                radiance = (
                                    shifted_color * intensity_factor * emission_scale * j_em
                                )
                                base_alpha = ti.min(opacity_scale * ti.max(rho, 0.0), 0.999)
                                disk_alpha = 1.0 - ti.pow(1.0 - base_alpha, alpha_gain)
                                front = 1.0 - plane_alpha_total
                                plane_disk += radiance * disk_alpha * front
                                plane_alpha_total = 1.0 - front * (1.0 - disk_alpha)

                    pos = new_pos
                    dir_ = new_dir
                    step_count += 1

                # 背景。
                bg_color = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                if event_horizon_hit:
                    bg_color = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                elif escaped:
                    bg_color = _sample_skybox(escape_dir)

                # 方向 1：分通道写入，sky 不参与曝光，disk 走 cinematic 链
                # disk_hdr: 盘的纯发射（不含 sky 也不含 sky × transmittance）
                # disk_alpha: 盘（+ event horizon）的累积不透明度
                #   - event_horizon_hit: alpha=1，完全遮挡 sky
                #   - use_visual_atlas:  alpha=plane_alpha_total（atlas 累积透明度）
                #   - volume 路径:       alpha=1-transmittance（积分累积透明度）
                # sky_hdr: 原始 sky 颜色，不参与曝光
                disk_hdr_value = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                disk_alpha_value = 0.0
                sky_hdr_value = bg_color  # 已经处理过 event_horizon_hit（黑色）和 escape
                if event_horizon_hit:
                    # 黑洞剪影：alpha=1，背景完全被遮挡
                    disk_alpha_value = 1.0
                    self.event_horizon_field[i, j] = 1
                elif ti.static(use_visual_atlas):
                    disk_hdr_value = plane_disk
                    disk_alpha_value = plane_alpha_total
                    self.event_horizon_field[i, j] = 0
                else:
                    disk_hdr_value = hdr_accum
                    disk_alpha_value = 1.0 - transmittance
                    self.event_horizon_field[i, j] = 0

                self.disk_hdr_field[i, j] = disk_hdr_value
                self.disk_alpha_field[i, j] = ti.min(ti.max(disk_alpha_value, 0.0), 1.0)
                self.sky_hdr_field[i, j] = sky_hdr_value

                # 保留 hdr_field 作为"近似传统合成"用于向后兼容（stats / fallback）。
                # 注意：bloom 现在只跑在 disk_hdr_field 上，不再修改 hdr_field。
                hdr_total = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                if ti.static(use_visual_atlas):
                    hdr_total = plane_disk + bg_color * (1.0 - plane_alpha_total)
                else:
                    hdr_total = hdr_accum + bg_color * transmittance
                self.hdr_field[i, j] = hdr_total

        self._ray_march_kernel = _ray_march_kernel

        @ti.kernel
        def _disk_tonemap_kernel():
            """disk_hdr → disk_ldr：exposure × tonemap × gamma，写入 disk_ldr_field。

            bloom 将在 disk_ldr_field（LDR 域）上做，与 V1 行为一致。
            event_horizon 像素强制输出黑色。
            """
            wp = self.white_point_field[None]
            inv_wp = 1.0 / ti.max(wp, 1e-6)
            for i, j in self.disk_ldr_field:
                if self.event_horizon_field[i, j] == 1:
                    self.disk_ldr_field[i, j] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                else:
                    disk_hdr_raw = self.disk_hdr_field[i, j]
                    disk_hdr = disk.apply_exposure_ti(disk_hdr_raw, inv_wp)
                    disk_ldr = disk.tonemap_reinhard(disk_hdr)
                    disk_ldr = disk.gamma_correct_ti(disk_ldr)
                    self.disk_ldr_field[i, j] = disk_ldr

        self._disk_tonemap_kernel = _disk_tonemap_kernel

        @ti.kernel
        def _bloom_kernel():
            """V1 风格 LDR bloom：逐通道独立 sigma 模拟相机色散。

            直接从 V1 `_bloom_kernel` 复制，在 disk_ldr_field 上操作。
            三通道 sigma²：R=25, G=80, B=1600（×sigma_scale），
            让蓝光扩散远 8× → halo 边缘呈明显蓝色。

            流程：亮度提取 → 水平高斯 → 垂直高斯 → 加回 disk_ldr_field。

            参数通过 field 传入（Taichi 闭包 kernel 不接受运行时参数注解）：
            bloom_threshold_field, bloom_intensity_field, bloom_kernel_radius_field,
            bloom_sigma_scale_field。
            """
            threshold = self.bloom_threshold_field[None]
            intensity = self.bloom_intensity_field[None]
            kernel_radius = self.bloom_kernel_radius_field[None]
            sigma_scale = self.bloom_sigma_scale_field[None]

            w = ti.cast(self.disk_ldr_field.shape[0], ti.i32)
            h = ti.cast(self.disk_ldr_field.shape[1], ti.i32)

            # 1. 提取高亮像素
            for i, j in self.disk_ldr_field:
                col = self.disk_ldr_field[i, j]
                lum = col[0] * 0.2126 + col[1] * 0.7152 + col[2] * 0.0722
                if lum > threshold:
                    self.bright_field[i, j] = col
                else:
                    self.bright_field[i, j] = ti.Vector([0.0, 0.0, 0.0])

            # 2. 水平方向模糊（逐通道独立 sigma）
            for i, j in self.blur_field:
                sum_r = 0.0
                sum_g = 0.0
                sum_b = 0.0
                weight_r = 0.0
                weight_g = 0.0
                weight_b = 0.0
                dx = -kernel_radius
                while dx <= kernel_radius:
                    ni = i + dx
                    if 0 <= ni < w:
                        dist_sq = ti.cast(dx * dx, ti.f32)
                        col = self.bright_field[ni, j]
                        w_r = ti.exp(-dist_sq / (25.0 * sigma_scale))
                        w_g = ti.exp(-dist_sq / (80.0 * sigma_scale))
                        w_b = ti.exp(-dist_sq / (1600.0 * sigma_scale))
                        sum_r += col[0] * w_r
                        sum_g += col[1] * w_g
                        sum_b += col[2] * w_b
                        weight_r += w_r
                        weight_g += w_g
                        weight_b += w_b
                    dx += 1
                if weight_r > 0.0:
                    self.blur_field[i, j] = ti.Vector([
                        sum_r / weight_r, sum_g / weight_g, sum_b / weight_b,
                    ])
                else:
                    self.blur_field[i, j] = ti.Vector([0.0, 0.0, 0.0])

            # 3. 复制到 bright_field 供垂直 pass 读取
            for i, j in self.bright_field:
                self.bright_field[i, j] = self.blur_field[i, j]

            # 4. 垂直方向模糊
            for i, j in self.blur_field:
                sum_r = 0.0
                sum_g = 0.0
                sum_b = 0.0
                weight_r = 0.0
                weight_g = 0.0
                weight_b = 0.0
                dy = -kernel_radius
                while dy <= kernel_radius:
                    nj = j + dy
                    if 0 <= nj < h:
                        dist_sq = ti.cast(dy * dy, ti.f32)
                        col = self.bright_field[i, nj]
                        w_r = ti.exp(-dist_sq / (25.0 * sigma_scale))
                        w_g = ti.exp(-dist_sq / (80.0 * sigma_scale))
                        w_b = ti.exp(-dist_sq / (1600.0 * sigma_scale))
                        sum_r += col[0] * w_r
                        sum_g += col[1] * w_g
                        sum_b += col[2] * w_b
                        weight_r += w_r
                        weight_g += w_g
                        weight_b += w_b
                    dy += 1
                if weight_r > 0.0:
                    self.blur_field[i, j] = ti.Vector([
                        sum_r / weight_r, sum_g / weight_g, sum_b / weight_b,
                    ])
                else:
                    self.blur_field[i, j] = ti.Vector([0.0, 0.0, 0.0])

            # 5. 加回 disk_ldr_field（逐通道加性合成）
            # 三通道 sigma 差距大（R=10, G≈18, B≈80），归一化加权平均让
            # G/B 在核心区跟 R 一样强 → bloom 变绿/蓝而不是暖白。
            # 逐通道 intensity 衰减让核心保持暖色，外缘才显现淡蓝：
            #   R × 1.0:  红光 sigma 最小，只在核心亮——主导暖色
            #   G × 0.35: 绿光 sigma 中等，压制避免偏绿
            #   B × 0.15: 蓝光 sigma 最大，极轻量——只在远处淡淡可见
            for i, j in self.disk_ldr_field:
                orig = self.disk_ldr_field[i, j]
                bloom_val = self.blur_field[i, j]
                r = orig[0] + bloom_val[0] * intensity * 1.0
                g = orig[1] + bloom_val[1] * intensity * 0.35
                b = orig[2] + bloom_val[2] * intensity * 0.15
                self.disk_ldr_field[i, j] = ti.Vector([
                    ti.min(ti.max(r, 0.0), 1.0),
                    ti.min(ti.max(g, 0.0), 1.0),
                    ti.min(ti.max(b, 0.0), 1.0),
                ], dt=ti.f32)

        self._bloom_kernel = _bloom_kernel

        @ti.kernel
        def _compose_kernel():
            """disk_ldr（含 bloom）+ sky + alpha → 最终 image_field。

            sky 不做 gamma（与 V1 一致，背景 PNG 已经是 sRGB encoded）。
            event_horizon 像素 disk_alpha=1 + disk_ldr=黑色 → 最终黑色。
            """
            for i, j in self.image_field:
                alpha = self.disk_alpha_field[i, j]
                disk_ldr = self.disk_ldr_field[i, j]
                sky_ldr = self.sky_hdr_field[i, j]
                self.image_field[i, j] = disk_ldr * alpha + sky_ldr * (1.0 - alpha)

        self._compose_kernel = _compose_kernel

    def _setup_camera(self, cam_pos: List[float], fov: float) -> None:
        """计算相机基向量并填到 Taichi field 中（与 V1 `build_camera` 一致）。"""
        (
            cam_pos_arr,
            right,
            up,
            forward,
            pixel_width,
            pixel_height,
            _top_left,
        ) = build_camera_v1_compatible(cam_pos, fov, self.width, self.height)

        self.cam_pos_field[None] = cam_pos_arr.astype(np.float32).tolist()
        self.cam_right_field[None] = right.astype(np.float32).tolist()
        self.cam_up_field[None] = up.astype(np.float32).tolist()
        self.cam_forward_field[None] = forward.astype(np.float32).tolist()
        self.pixel_width_field[None] = float(pixel_width)
        self.pixel_height_field[None] = float(pixel_height)
        self.top_left_field[None] = _top_left.astype(np.float32).tolist()

        distance = float(np.linalg.norm(cam_pos_arr))
        r_out = float(self.disk_ti._r_out)
        r_escape = max(self.r_max, distance * 2.0, r_out * 1.6)
        self.r_escape_field[None] = r_escape

    # D3：物理 reference 与 actual HDR p99 的合理偏差窗口。
    # ratio = actual_hdr_p99 / reference_white_point ∈ [_RATIO_TRUSTED_LO, _RATIO_TRUSTED_HI]
    # 时，使用 reference（cinematic 曝光基线物理可控、跨场景一致）。
    # 落在该窗口之外才视为 reference 不可信，fallback 到 HDR p99 兜底。
    #
    # D2 实测：默认主验收参数下 ratio ≈ 4.3（g-factor·tilt·palette·transmittance
    # 共同贡献），完全在合理范围内。把判定窗口放宽到 [0.1, 10] 让 reference
    # 在正常物理偏差下都生效；
    # 进一步落到外层 [0.01, 100] 区间时打 warning，便于发现物理参数异常；
    # 超过 [0.01, 100] 才完全 fallback 到 HDR p99，避免单帧黑掉或全白。
    _RATIO_TRUSTED_LO: float = 0.1
    _RATIO_TRUSTED_HI: float = 10.0
    _RATIO_WARN_LO: float = 0.01
    _RATIO_WARN_HI: float = 100.0

    def _compute_white_point(self, hdr_np: np.ndarray) -> tuple[float, float]:
        """从 disk-only HDR buffer 计算 tonemap 用的 white point。

        Args:
            hdr_np: 形状 `(W, H, 3)` 的 disk HDR RGB（方向 1：不含 sky）。

        Returns:
            `(used_white_point, actual_hdr_white_point)`：

            - `used_white_point`：fallback 判定后实际用于 tonemap 的 white point。
            - `actual_hdr_white_point`：盘内像素 p{n} 原始候选值（fallback 判定前）；
              与 `reference_white_point` 之比即 D3 ratio。

        Notes:
            判定逻辑（D3 修订 + 方向 1 修订）：

            - **方向 1 (2026-06-14)**：percentile 只在 `disk_alpha > 0` 的像素上算，
              避免大量背景 0 像素把 p96 拖到接近 0。否则 r_out=50 大盘下，
              背景占屏 > 95%，p96 = 0 → ratio = 0 → 进入 warn 区，但用户场景
              其实是正常的。
            - 若 `actual_hdr_p{n}` 与 `reference_white_point` 比值落在
              `[_RATIO_TRUSTED_LO, _RATIO_TRUSTED_HI]`，使用 reference。
              这是 cinematic 曝光基线，物理可控、跨场景稳定。
            - 比值落在 `[_RATIO_WARN_LO, _RATIO_WARN_HI]` 但越出 trusted
              窗口时，打印 warning，但仍优先使用 reference，保持曝光稳定。
            - 比值越出 warn 窗口才完全 fallback 到 HDR p{n}，避免极端
              情况下出图全黑或全白。

            注：`n = white_point_percentile`，默认 99；`interstellar` preset 用 96。
        """
        luma_full = hdr_luminance(hdr_np).ravel()
        # 方向 1：只统计盘像素，避免背景 0 像素污染分位数。
        # disk_alpha_field 拷回 CPU 做 mask。若 mask 后无任何盘像素（测试场景
        # 或 disk_alpha 全 0），fallback 到整张 HDR 统计。
        try:
            alpha_np = self.disk_alpha_field.to_numpy().ravel()
        except AttributeError:
            alpha_np = None
        luma: np.ndarray
        if alpha_np is not None and alpha_np.size == luma_full.size:
            mask = (alpha_np > 1e-4) & np.isfinite(luma_full)
            if mask.any():
                luma = luma_full[mask]
            else:
                # 测试场景：disk_alpha 全 0 但 HDR 非 0 → 用整图，保持旧行为
                luma = luma_full[np.isfinite(luma_full)]
        else:
            luma = luma_full[np.isfinite(luma_full)]
        if luma.size == 0:
            return 1.0, 1.0
        pct = float(np.clip(self.white_point_percentile, 50.0, 99.99))
        hdr_wp = max(float(np.percentile(luma, pct)), 1e-6)
        ref_wp = max(float(self.reference_white_point), 1e-6)
        ratio = hdr_wp / ref_wp

        if self._RATIO_TRUSTED_LO <= ratio <= self._RATIO_TRUSTED_HI:
            return ref_wp, hdr_wp
        if self._RATIO_WARN_LO <= ratio <= self._RATIO_WARN_HI:
            print(
                f"[V2 exposure] warning: actual HDR p{pct:g} (disk-only) / reference = {ratio:.2f}, "
                f"偏离 trusted 窗口 [{self._RATIO_TRUSTED_LO}, {self._RATIO_TRUSTED_HI}]；"
                f"仍使用 reference 保持曝光基线稳定，请检查物理参数。"
            )
            return ref_wp, hdr_wp
        # 极端偏离：reference 不可信，兜底 HDR p{n}，避免单帧异常。
        print(
            f"[V2 exposure] reference 不可信: actual/ref = {ratio:.2g}（HDR p{pct:g} disk-only），"
            f"超过 [{self._RATIO_WARN_LO}, {self._RATIO_WARN_HI}]；fallback 到 HDR p{pct:g}。"
        )
        return hdr_wp, hdr_wp

    def render(self, cam_pos: List[float], fov: float) -> np.ndarray:
        """渲染单帧（含可选 V1 风格 LDR bloom）。

        Args:
            cam_pos: 相机位置 `[x, y, z]`，单位为 r_s。
            fov: 视野角（度）。

        Returns:
            `(height, width, 3)` uint8 RGB 数组。

        Notes:
            管线（方向 1 + V1 bloom）：

            1. `_ray_march_kernel`：HDR 体积积分 → disk_hdr / disk_alpha / sky_hdr
            2. `_compute_white_point`：disk-only HDR 统计 → wp
            3. `_disk_tonemap_kernel`：disk_hdr → disk_ldr（exposure + tonemap + gamma）
            4. `_bloom_kernel`（可选）：V1 风格 LDR 域 bloom，逐通道独立 sigma
               模拟相机色散，B 通道 sigma 8× → 蓝色 halo 边缘
            5. `_compose_kernel`：disk_ldr（含 bloom）+ sky + alpha → 最终 image_field
        """
        self._setup_camera(cam_pos, fov)
        self._ray_march_kernel()

        # stats / white_point 只看 disk_hdr_field（不含 sky）
        disk_hdr_for_stats = self.disk_hdr_field.to_numpy()
        if self.auto_exposure:
            white_point, actual_hdr_wp = self._compute_white_point(disk_hdr_for_stats)
        else:
            white_point = 1.0
            actual_hdr_wp = None
        self.last_white_point = white_point
        self.last_actual_hdr_white_point = actual_hdr_wp
        self.white_point_field[None] = float(white_point)

        # disk_hdr → disk_ldr（LDR 域）
        self._disk_tonemap_kernel()

        # V1 风格 LDR bloom（可选）
        if self.bloom_intensity > 0.0:
            kernel_radius = int(self.width * 0.02)
            sigma_scale = (self.width / 640.0) ** 2
            self.bloom_threshold_field[None] = 0.0
            self.bloom_intensity_field[None] = float(self.bloom_intensity)
            self.bloom_kernel_radius_field[None] = kernel_radius
            self.bloom_sigma_scale_field[None] = float(sigma_scale)
            self._bloom_kernel()

        # disk_ldr + sky + alpha → 最终合成
        self._compose_kernel()

        img = self.image_field.to_numpy()  # (W, H, 3)
        img = np.transpose(img, (1, 0, 2))
        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        stats = compute_render_stats(
            disk_hdr_for_stats,
            img,
            white_point=white_point if self.auto_exposure else None,
            reference_white_point=self.reference_white_point if self.auto_exposure else None,
            actual_hdr_white_point=actual_hdr_wp,
            white_point_percentile=(
                float(self.white_point_percentile) if self.auto_exposure else None
            ),
        )
        self.last_stats = stats
        if self.print_stats:
            print(stats.format_summary())

        # 与 V1 `TaichiRenderer.render` 一致：返回 float32 LDR `[0, 1]`，由 `save_image` 量化。
        return img

    # _apply_bloom 已删除（2026-06-14）：替换为 V1 风格 LDR 域 Taichi _bloom_kernel。
