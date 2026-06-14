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
from .params import DiskV2PaletteParams, DiskV2Params, DiskV2StructureParams
from .stats import RenderStats, compute_render_stats, hdr_luminance
from .structure_modulations import _ClumpCenters
from .taichi_impl import (
    DiskV2Taichi,
    disk_half_thickness_ti,
    disk_volume_mask_ti,
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

        # 把基础场和 palette 包装为 Taichi 句柄。
        self.disk_ti = DiskV2Taichi(
            params=params,
            structure_params=structure_params,
            palette_params=palette_params,
            seed=seed,
            centers=centers,
        )

        # 输出图像 field（HDR 浮点；Bloom + tonemap 在 Python 端 / 简化 kernel 完成）。
        self.hdr_field = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self.image_field = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

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

        # Phase 5 物理常数：Schwarzschild M = 0.5（让 r_s = 2M = 1）。
        # Schwarzschild 圆轨道角速度 Ω(r) = sqrt(M/r³) / sqrt(1 - 3M/r)。
        # 注意 r_isco = 3 r_s = 6M = 3.0 (单位 r_s)；ISCO 内 1 - 3M/r ≤ 0，公式发散。
        # 我们要求 r_in ≥ 3 r_s（params.py 强制），所以这里恒为正。
        sch_M = 0.5

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
                                    color = disk.sample_palette_color(T_K)
                                    atlas_color = disk.sample_atlas_color_mod_ti(r_local, phi_local)
                                    rho = disk.sample_density(r_local, phi_local, z_local)
                                    alpha_k = opacity_scale * ti.max(rho, 0.0)
                                    g_total = 1.0
                                    if ti.static(enable_g):
                                        r_safe_em = ti.max(r_local, 3.0 * rs)
                                        denom_om = ti.sqrt(ti.max(1.0 - 3.0 * sch_M / r_safe_em, 1e-6))
                                        omega_em = ti.sqrt(sch_M / (r_safe_em ** 3)) / denom_om
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
                                        lapse = ti.sqrt(ti.max(1.0 - rs / r_safe_em, 1e-6))
                                        beta = ti.min(
                                            r_safe_em * omega_em / ti.max(lapse, 1e-6), 0.99,
                                        )
                                        gamma = 1.0 / ti.sqrt(ti.max(1.0 - beta * beta, 1e-6))
                                        ray_to_cam = -dir_
                                        cos_theta = v_hat_world.dot(ray_to_cam.normalized())
                                        denom = ti.max(1.0 - beta * cos_theta, 1e-3)
                                        g_doppler = 1.0 / (gamma * denom)
                                        r_obs = cp.norm()
                                        grav_num = ti.sqrt(ti.max(1.0 - rs / ti.max(r_obs, rs + 1e-3), 1e-6))
                                        grav_den = ti.sqrt(ti.max(1.0 - rs / ti.max(r_safe_em, rs + 1e-3), 1e-6))
                                        g_grav = grav_num / grav_den
                                        g_total = ti.min(g_doppler * g_grav, g_cap)
                                    intensity_factor = ti.pow(ti.max(g_total, 0.1), lum_power)
                                    wien_color_factor = ti.Vector([1.0, 1.0, 1.0], dt=ti.f32)
                                    if ti.static(enable_g):
                                        g_safe = ti.max(g_total, 0.1)
                                        wien_arg = 1.0 - 1.0 / g_safe
                                        T_for_wien = ti.max(T_K, 1.0e3)
                                        x_r = 0.01439 / (650.0e-9 * T_for_wien)
                                        x_g = 0.01439 / (530.0e-9 * T_for_wien)
                                        x_b = 0.01439 / (460.0e-9 * T_for_wien)
                                        wien_color_factor = ti.Vector([
                                            ti.exp(x_r * wien_arg),
                                            ti.exp(x_g * wien_arg),
                                            ti.exp(x_b * wien_arg),
                                        ], dt=ti.f32)
                                        norm_g = wien_color_factor[1]
                                        wien_color_factor = wien_color_factor / ti.max(norm_g, 1e-6)
                                        wien_color_factor = ti.Vector([
                                            ti.min(wien_color_factor[0], 3.0),
                                            ti.min(wien_color_factor[1], 3.0),
                                            ti.min(wien_color_factor[2], 3.0),
                                        ], dt=ti.f32)
                                    ds = (new_pos - old_pos).norm() / n_vol
                                    shifted_color = ti.Vector([
                                        color[0] * wien_color_factor[0] * atlas_color,
                                        color[1] * wien_color_factor[1] * atlas_color,
                                        color[2] * wien_color_factor[2] * atlas_color,
                                    ], dt=ti.f32)
                                    hdr_accum += transmittance * emission_scale * j_em * shifted_color * intensity_factor * ds
                                    transmittance *= ti.exp(-alpha_k * ds)
                    else:
                        f_old = old_pos[2] - old_pos[1] * tan_t
                        f_new = new_pos[2] - new_pos[1] * tan_t
                        if f_old * f_new < 0.0:
                            t_frac = f_old / (f_old - f_new + 1e-8)
                            hit_x = old_pos[0] + t_frac * (new_pos[0] - old_pos[0])
                            hit_y = old_pos[1] + t_frac * (new_pos[1] - old_pos[1])
                            hit_r = ti.sqrt(hit_x * hit_x + hit_y * hit_y)
                            if hit_r >= disk._r_in and hit_r <= disk._r_out:
                                phi_hit = ti.atan2(hit_y, hit_x)
                                ew = disk.sample_emission_atlas_ti(hit_r, phi_hit)
                                j_em = disk.sample_emission(hit_r, phi_hit, 0.0)
                                color = disk.sample_visual_disk_color_ti(hit_r, phi_hit, 0.0)
                                g_total = 1.0
                                if ti.static(enable_g):
                                    r_safe_em = ti.max(hit_r, 3.0 * rs)
                                    denom_om = ti.sqrt(ti.max(1.0 - 3.0 * sch_M / r_safe_em, 1e-6))
                                    omega_em = ti.sqrt(sch_M / (r_safe_em ** 3)) / denom_om
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
                                    lapse = ti.sqrt(ti.max(1.0 - rs / r_safe_em, 1e-6))
                                    beta = ti.min(
                                        r_safe_em * omega_em / ti.max(lapse, 1e-6), 0.99,
                                    )
                                    gamma = 1.0 / ti.sqrt(ti.max(1.0 - beta * beta, 1e-6))
                                    ray_to_cam = -dir_
                                    cos_theta = v_hat_world.dot(ray_to_cam.normalized())
                                    denom = ti.max(1.0 - beta * cos_theta, 1e-3)
                                    g_doppler = 1.0 / (gamma * denom)
                                    r_obs = cp.norm()
                                    grav_num = ti.sqrt(ti.max(1.0 - rs / ti.max(r_obs, rs + 1e-3), 1e-6))
                                    grav_den = ti.sqrt(ti.max(1.0 - rs / ti.max(r_safe_em, rs + 1e-3), 1e-6))
                                    g_grav = grav_num / grav_den
                                    g_total = ti.min(g_doppler * g_grav, g_cap)
                                intensity_factor = ti.pow(ti.max(g_total, 0.1), lum_power)
                                wien_color_factor = ti.Vector([1.0, 1.0, 1.0], dt=ti.f32)
                                if ti.static(enable_g):
                                    g_safe = ti.max(g_total, 0.1)
                                    wien_arg = 1.0 - 1.0 / g_safe
                                    T_for_wien = 8000.0
                                    x_r = 0.01439 / (650.0e-9 * T_for_wien)
                                    x_g = 0.01439 / (530.0e-9 * T_for_wien)
                                    x_b = 0.01439 / (460.0e-9 * T_for_wien)
                                    wien_color_factor = ti.Vector([
                                        ti.exp(x_r * wien_arg),
                                        ti.exp(x_g * wien_arg),
                                        ti.exp(x_b * wien_arg),
                                    ], dt=ti.f32)
                                    norm_g = wien_color_factor[1]
                                    wien_color_factor = wien_color_factor / ti.max(norm_g, 1e-6)
                                    wien_color_factor = ti.Vector([
                                        ti.min(wien_color_factor[0], 3.0),
                                        ti.min(wien_color_factor[1], 3.0),
                                        ti.min(wien_color_factor[2], 3.0),
                                    ], dt=ti.f32)
                                shifted_color = ti.Vector([
                                    color[0] * wien_color_factor[0],
                                    color[1] * wien_color_factor[1],
                                    color[2] * wien_color_factor[2],
                                ], dt=ti.f32)
                                r_ratio = disk._r_in / ti.max(hit_r, disk._r_in)
                                radial_falloff = ti.pow(ti.max(r_ratio, 0.0), 1.25)
                                # Visual atlas 只提供纹理细节；半径方向仍需遵守内热外冷的发射梯度。
                                radial_emission = 0.05 + 1.35 * radial_falloff
                                emissivity = ti.max(
                                    radial_emission,
                                    0.25 * ti.pow(ti.max(j_em, 0.0), 0.55),
                                )
                                radiance = (
                                    shifted_color * intensity_factor * emission_scale * emissivity
                                )
                                radiance = ti.Vector([
                                    ti.min(radiance[0], 1.0),
                                    ti.min(radiance[1], 1.0),
                                    ti.min(radiance[2], 1.0),
                                ], dt=ti.f32)
                                radial_alpha = 0.22 + 0.78 * ti.sqrt(radial_falloff)
                                base_alpha = ti.min(ew * radial_alpha, 0.999)
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

                hdr_total = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                if ti.static(use_visual_atlas):
                    plane_disk = ti.Vector([
                        ti.min(ti.max(plane_disk[0], 0.0), 1.0),
                        ti.min(ti.max(plane_disk[1], 0.0), 1.0),
                        ti.min(ti.max(plane_disk[2], 0.0), 1.0),
                    ], dt=ti.f32)
                    bg_color = bg_color * (1.0 - plane_alpha_total)
                    hdr_total = plane_disk + bg_color
                else:
                    bg_color = bg_color * transmittance
                    hdr_total = hdr_accum + bg_color
                # 存 HDR；后续 Python 端做可选 Bloom，再 tonemap + gamma。
                self.hdr_field[i, j] = hdr_total

        self._ray_march_kernel = _ray_march_kernel

        @ti.kernel
        def _tonemap_kernel():
            """把 hdr_field 经 white point 缩放 + Reinhard + sRGB 伽马后写入 image_field。"""
            wp = self.white_point_field[None]
            inv_wp = 1.0 / ti.max(wp, 1e-6)
            for i, j in self.image_field:
                hdr = self.hdr_field[i, j] * inv_wp
                ldr = disk.tonemap_reinhard(hdr)
                ldr = disk.gamma_correct_ti(ldr)
                self.image_field[i, j] = ldr

        self._tonemap_kernel = _tonemap_kernel

        @ti.kernel
        def _gamma_only_kernel():
            """Visual atlas LDR 路径：跳过 Reinhard，仅 clamp + sRGB 伽马（对齐 V1 盘层合成）。"""
            for i, j in self.image_field:
                hdr = self.hdr_field[i, j]
                ldr = ti.Vector([
                    ti.min(ti.max(hdr[0], 0.0), 1.0),
                    ti.min(ti.max(hdr[1], 0.0), 1.0),
                    ti.min(ti.max(hdr[2], 0.0), 1.0),
                ], dt=ti.f32)
                ldr = disk.gamma_correct_ti(ldr)
                self.image_field[i, j] = ldr

        self._gamma_only_kernel = _gamma_only_kernel

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

    def _compute_white_point(self, hdr_np: np.ndarray) -> float:
        """从 HDR buffer 计算 tonemap 用的 white point。

        Args:
            hdr_np: 形状 `(W, H, 3)` 的 HDR RGB。

        Returns:
            非负 white point 标量。
        """
        luma = hdr_luminance(hdr_np).ravel()
        luma = luma[np.isfinite(luma)]
        if luma.size == 0:
            return 1.0
        pct = float(np.clip(self.white_point_percentile, 50.0, 99.99))
        return max(float(np.percentile(luma, pct)), 1e-6)

    def render(self, cam_pos: List[float], fov: float) -> np.ndarray:
        """渲染单帧（含可选 Bloom）。

        Args:
            cam_pos: 相机位置 `[x, y, z]`，单位为 r_s。
            fov: 视野角（度）。

        Returns:
            `(height, width, 3)` uint8 RGB 数组。

        Notes:
            管线为：HDR 体积积分 → 可选 HDR Bloom → tonemap → gamma → LDR 输出。
            Bloom 在 HDR 域做，符合 docs/design_ad_v2.md §3.6 的要求。
        """
        self._setup_camera(cam_pos, fov)
        self._ray_march_kernel()

        if self.bloom_intensity > 0.0:
            # Python 端 Bloom：在 HDR 域做亮度提取 + 高斯模糊 + 加回。
            hdr_np = self.hdr_field.to_numpy()  # (W, H, 3)
            hdr_np = self._apply_bloom(hdr_np)
            self.hdr_field.from_numpy(hdr_np.astype(np.float32))

        hdr_for_stats = self.hdr_field.to_numpy()
        use_atlas_ldr = bool(self.disk_ti._use_visual_atlas)
        if use_atlas_ldr:
            white_point = 1.0
        else:
            white_point = self._compute_white_point(hdr_for_stats) if self.auto_exposure else 1.0
        self.last_white_point = white_point
        self.white_point_field[None] = float(white_point)

        if use_atlas_ldr:
            self._gamma_only_kernel()
        else:
            self._tonemap_kernel()
        img = self.image_field.to_numpy()  # (W, H, 3)
        img = np.transpose(img, (1, 0, 2))
        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        stats = compute_render_stats(
            hdr_for_stats,
            img,
            white_point=white_point if self.auto_exposure else None,
        )
        self.last_stats = stats
        if self.print_stats:
            print(stats.format_summary())

        # 与 V1 `TaichiRenderer.render` 一致：返回 float32 LDR `[0, 1]`，由 `save_image` 量化。
        return img

    def _apply_bloom(self, hdr: np.ndarray) -> np.ndarray:
        """在 HDR 域做 Bloom：提取超过阈值的亮度 → 高斯模糊 → 加回原图。

        Args:
            hdr: 形状 `(W, H, 3)` 的 HDR RGB 数组。

        Returns:
            同形状的 HDR RGB 数组，叠加 Bloom 后。

        Notes:
            高斯模糊用 separable 1D 卷积（scipy.ndimage 风格的纯 NumPy 实现）。
        """
        threshold = self.bloom_threshold
        intensity = self.bloom_intensity
        radius = max(self.bloom_radius, 0.5)

        luma = (
            0.2126 * hdr[..., 0]
            + 0.7152 * hdr[..., 1]
            + 0.0722 * hdr[..., 2]
        )
        bright_mask = np.maximum(luma - threshold, 0.0) / (luma + 1e-6)
        bright = hdr * bright_mask[..., None]

        # 1D 高斯核。
        size = int(max(3, math.ceil(radius * 3.0)) | 1)  # 奇数
        x = np.arange(size) - size // 2
        kernel = np.exp(-(x ** 2) / (2.0 * radius * radius))
        kernel /= kernel.sum()

        # Separable 1D 卷积：先 X 后 Y。
        blurred = np.empty_like(bright)
        for c in range(3):
            tmp = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, bright[..., c])
            blurred[..., c] = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, tmp)

        return hdr + intensity * blurred
