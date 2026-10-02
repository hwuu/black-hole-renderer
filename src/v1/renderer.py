from typing import Tuple, List, Optional
import math
import numpy as np
import taichi as ti
from src.core.camera import build_camera
from src.core.constants import DISK_ALPHA_GAIN, DISK_COLOR_TEMPERATURE, DISK_RADIAL_BRIGHTNESS_MAX, DISK_RADIAL_BRIGHTNESS_MIN, DISK_RADIAL_BRIGHTNESS_POWER, FILAMENT_BIRTH_FADE_DUR, FILAMENT_DEATH_THRESHOLD, G_BRIGHTNESS_GAIN, G_FACTOR_CAP, G_LUMINOSITY_POWER, RS, R_DISK_INNER_DEFAULT, R_DISK_OUTER_DEFAULT
from src.v1.texture import compute_edge_alpha, generate_disk_mipmaps


class TaichiRenderer:
    """
    Taichi 渲染器类，kernel 仅编译一次，支持多帧渲染。

    用法:
        renderer = TaichiRenderer(width, height, skybox, disk_tex, ...)
        img1 = renderer.render(cam_pos=[6, 0, 0.5], fov=90)
        img2 = renderer.render(cam_pos=[7, 1, 0.3], fov=90)
    """

    def __init__(self, width, height, skybox, disk_tex,
                 step_size=0.1, r_max=10.0, device="cpu",
                 r_disk_inner=R_DISK_INNER_DEFAULT,
                 r_disk_outer=R_DISK_OUTER_DEFAULT,
                 disk_tilt=0.0,
                 lens_flare=False,
                 anti_alias="disabled",
                 aa_strength=1.0,
                 disk_rotation_speed=0.1,
                 ignore_taichi_cache=False):
        # ti is imported at module level as "import taichi as ti"
        self.width = width
        self.height = height
        self.step_size = step_size
        self.r_max = r_max
        self.r_disk_inner = r_disk_inner
        self.r_disk_outer = r_disk_outer
        self.disk_tilt = disk_tilt
        self.lens_flare = lens_flare
        self.anti_alias = anti_alias
        self.aa_strength = aa_strength
        self.disk_rotation_speed = disk_rotation_speed

        use_cache = not ignore_taichi_cache
        ti.init(arch=ti.cpu if device == "cpu" else ti.gpu, offline_cache=use_cache)

        tex_h, tex_w = skybox.shape[:2]
        dtex_h, dtex_w = disk_tex.shape[:2]
        self.tex_w = tex_w
        self.tex_h = tex_h
        self.dtex_w = dtex_w
        self.dtex_h = dtex_h

        self.texture_field = ti.Vector.field(3, dtype=ti.f32, shape=(tex_h, tex_w))
        self.texture_field.from_numpy(skybox)

        self.disk_texture_field = ti.Vector.field(4, dtype=ti.f32, shape=(dtex_h, dtex_w))
        self.disk_texture_field.from_numpy(disk_tex)

        # Mipmap 纹理（始终生成，用于抗锯齿）
        mips = generate_disk_mipmaps(disk_tex, levels=4)
        self.num_mip_levels = len(mips)
        # 所有 mipmap 填充到相同尺寸
        max_h = max(m.shape[0] for m in mips)
        max_w = max(m.shape[1] for m in mips)
        self.disk_mips_field = ti.Vector.field(4, dtype=ti.f32, shape=(self.num_mip_levels, max_h, max_w))

        # 逐级填充（用 numpy 预处理后一次性写入，避免逐像素循环）
        mips_padded = np.zeros((len(mips), max_h, max_w, 4), dtype=np.float32)
        for i, m in enumerate(mips):
            h, w = m.shape[:2]
            mips_padded[i, :h, :w] = m
        self.disk_mips_field.from_numpy(mips_padded)

        self.image_field = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self.disk_layer_field = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

        self.cam_pos_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_right_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_up_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_forward_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.pixel_width_field = ti.field(dtype=ti.f32, shape=())
        self.pixel_height_field = ti.field(dtype=ti.f32, shape=())
        self.r_escape_field = ti.field(dtype=ti.f32, shape=())

        self.bright_field = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self.blur_field = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self.final_field = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

        # Simplex noise 排列表（标准 Ken Perlin 排列，重复一次以处理溢出）
        _perm = [
            151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
            140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,
            247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,
            57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
            74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,
            60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,
            65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,
            200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,
            52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,
            207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,
            119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
            129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,
            218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,
            81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,
            184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,
            222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
        ]
        self.perm_field = ti.field(ti.i32, shape=(512,))
        self.perm_field.from_numpy(np.array(_perm + _perm, dtype=np.int32))

        self._compile_kernels()

    def update_disk_texture(self, new_disk_tex: np.ndarray) -> None:
        """更新吸积盘纹理（用于动态纹理生成）。

        Args:
            new_disk_tex: 新的纹理数组 (n_r, n_phi, 4) float32
        """
        dtex_h, dtex_w = new_disk_tex.shape[:2]
        assert dtex_h == self.dtex_h and dtex_w == self.dtex_w, \
            f"Texture size mismatch: expected {self.dtex_h}x{self.dtex_w}, got {dtex_h}x{dtex_w}"

        self.disk_texture_field.from_numpy(new_disk_tex)

        # 重新生成 mipmap
        mips = generate_disk_mipmaps(new_disk_tex, levels=4)
        max_h = max(m.shape[0] for m in mips)
        max_w = max(m.shape[1] for m in mips)
        mips_padded = np.zeros((len(mips), max_h, max_w, 4), dtype=np.float32)
        for i, m in enumerate(mips):
            h, w = m.shape[:2]
            mips_padded[i, :h, :w] = m
        self.disk_mips_field.from_numpy(mips_padded)

    def upload_parametric_state(self, state: 'DiskTextureRotatingState') -> None:
        """上传 parametric 旋转状态到 GPU，预计算归一化统计量。

        将 DiskTextureRotatingState 的 13 个组件场和辅助数据上传到 Taichi fields，
        预计算密度和温度的归一化统计量。当 generation_scale=1 时，这些统计量
        在所有 t_offset 下精确不变，GPU 合成结果与 CPU 路径像素级等价。

        Args:
            state: DiskTextureRotatingState，预计算的旋转状态对象

        Notes:
            此方法应在 render_video 的 parametric 模式循环前调用一次。
            调用后可使用 update_disk_texture_gpu(t_offset) 替代 CPU 纹理生成路径。

        组件打包顺序（索引 0-12）:
            0: temp_base, 1: spiral, 2: spiral_temp, 3: turbulence, 4: turb_temp,
            5: arcs, 6: arcs_temp, 7: rt_spikes, 8: rt_temp, 9: hotspot,
            10: hotspot_temp, 11: az_hotspot, 12: disturb_mod
        """
        n_r = state.n_r
        n_phi = state.n_phi

        packed = np.stack([
            state.temp_base,       # 0
            state.spiral,          # 1
            state.spiral_temp,     # 2
            state.turbulence,      # 3
            state.turb_temp,       # 4
            state.arcs,            # 5
            state.arcs_temp,       # 6
            state.rt_spikes,       # 7
            state.rt_temp,         # 8
            state.hotspot,         # 9
            state.hotspot_temp,    # 10
            state.az_hotspot,      # 11
            state.disturb_mod,     # 12
        ], axis=0).astype(np.float32)

        self._comp_field = ti.field(dtype=ti.f32, shape=(13, n_r, n_phi))
        self._comp_field.from_numpy(packed)

        self._omega_rows_field = ti.field(dtype=ti.f32, shape=(n_r,))
        self._omega_rows_field.from_numpy(state.omega_rows)

        self._edge_field = ti.field(dtype=ti.f32, shape=(n_r,))
        self._edge_field.from_numpy(state.edge)

        # 预计算归一化统计量（t=0 无旋转，直接用原始组件计算）
        rt_weight = 0.20 if state.enable_rt else 0.0
        density = (0.15 + 0.10 * state.spiral + 0.30 * state.turbulence
                   + 0.20 * state.hotspot + 0.30 * state.arcs
                   + rt_weight * state.rt_spikes) * state.disturb_mod
        density *= state.edge[:, None]
        density_p98 = float(np.percentile(density, 98))

        temp_struct = (state.spiral_temp + state.turb_temp + state.arcs_temp
                       + state.rt_temp + state.hotspot_temp) * state.disturb_mod
        pos_mask = temp_struct > 0
        struct_scale = float(np.percentile(temp_struct[pos_mask], 95)) if np.any(pos_mask) else 1.0

        temp_struct_scaled = np.clip(temp_struct / (struct_scale + 1e-6) * 0.8, 0, 1.2)
        struct_max_per_r = np.max(temp_struct_scaled, axis=1).astype(np.float32)
        struct_p70_per_r = np.quantile(temp_struct_scaled, 0.7, axis=1).astype(np.float32)

        self._param_stats_field = ti.field(dtype=ti.f32, shape=(2,))
        self._param_stats_field.from_numpy(np.array([density_p98, struct_scale], dtype=np.float32))

        row_stats = np.stack([struct_max_per_r, struct_p70_per_r], axis=1).astype(np.float32)
        self._param_row_stats_field = ti.Vector.field(2, dtype=ti.f32, shape=(n_r,))
        self._param_row_stats_field.from_numpy(row_stats)

        self._param_enable_rt = 1 if state.enable_rt else 0
        self._param_color_temp = float(state.color_temp)
        self._parametric_gpu_ready = True

    def _compile_kernels(self):
        # ti is module-level import
        width, height = self.width, self.height
        tex_w, tex_h = self.tex_w, self.tex_h
        dtex_w, dtex_h = self.dtex_w, self.dtex_h
        texture_field = self.texture_field
        disk_texture_field = self.disk_texture_field
        disk_mips_field = self.disk_mips_field
        num_mip_levels = self.num_mip_levels
        anti_alias_mode = 0 if self.anti_alias == "disabled" else 1
        aa_strength = self.aa_strength

        g_cap = ti.cast(G_FACTOR_CAP, ti.f32)
        lum_power = ti.cast(G_LUMINOSITY_POWER, ti.f32)
        gain = ti.cast(G_BRIGHTNESS_GAIN, ti.f32)
        alpha_gain = ti.cast(DISK_ALPHA_GAIN, ti.f32)
        color_temp = ti.cast(DISK_COLOR_TEMPERATURE, ti.f32)

        @ti.func
        def _color_temp_to_tint(temp):
            """Convert color temperature (K) to RGB tint using Tanner Helland approximation.

            Reference: http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
            temp: temperature in Kelvin
            Returns: RGB vector with values in [0, 1]
            """
            t = temp / 100.0

            # Red calculation
            r = 1.0
            if t > 66.0:
                r = ti.min(ti.max(1.292936 * ti.pow(ti.max(t - 60.0, 0.0001), -0.1332047592), 0.0), 1.0)

            # Green calculation
            g = 0.0
            if t <= 66.0:
                g = ti.min(ti.max(0.390082 * ti.log(ti.max(t, 0.0001)) - 0.631841, 0.0), 1.0)
            else:
                g = ti.min(ti.max(1.129891 * ti.pow(ti.max(t - 60.0, 0.0001), -0.0755148492), 0.0), 1.0)

            # Blue calculation
            b = 1.0
            if t < 66.0:
                if t <= 19.0:
                    b = 0.0
                else:
                    b = ti.min(ti.max(0.543207 * ti.log(ti.max(t - 10.0, 0.0001)) - 1.19625, 0.0), 1.0)

            return ti.Vector([r, g, b])

        @ti.func
        def _apply_g_factor(base_color, hit_pos, hit_r, ray_dir_to_cam, cam_pos,
                            r_inner, r_outer, tilt_rad):
            """Apply relativistic g-factor to disk color.

            Computes Doppler shift and gravitational redshift for accretion disk emission.
            Returns color modulated by g-factor with radial brightness profile.
            """
            rs_f = ti.cast(RS, ti.f32)
            r_obs = cam_pos.norm()
            r_em = hit_pos.norm()
            r_safe = ti.max(r_em, rs_f + 1e-3)

            omega = ti.sqrt(0.5 / (r_safe ** 3 + 1e-6))
            lorentz = ti.sqrt(ti.max(1.0 - rs_f / r_safe, 1e-6))
            beta = ti.min(r_safe * omega / ti.max(lorentz, 1e-6), 0.99)
            gamma = 1.0 / ti.sqrt(ti.max(1.0 - beta * beta, 1e-6))

            sin_t = ti.sin(tilt_rad)
            cos_t = ti.cos(tilt_rad)
            disk_normal = ti.Vector([0.0, -sin_t, cos_t])
            r_hat = hit_pos.normalized()
            v_hat = r_hat.cross(disk_normal)
            v_norm = v_hat.norm()
            if v_norm > 1e-6:
                v_hat = v_hat / v_norm
            else:
                v_hat = ti.Vector([0.0, 1.0, 0.0])

            ray_hat = ray_dir_to_cam.normalized()
            cos_theta = v_hat.dot(ray_hat)
            denom = ti.max(1.0 - beta * cos_theta, 1e-3)
            g_doppler = 1.0 / (gamma * denom)

            grav_num = ti.sqrt(ti.max(1.0 - rs_f / ti.max(r_obs, rs_f + 1e-3), 1e-6))
            grav_den = ti.sqrt(ti.max(1.0 - rs_f / ti.max(r_em, rs_f + 1e-3), 1e-6))
            g_grav = grav_num / grav_den

            g = ti.min(g_doppler * g_grav, g_cap)
            intensity = ti.max(ti.pow(g, lum_power), 0.0)
            brightness = gain * intensity / (1.0 + intensity / g_cap)

            radial_span = ti.max(r_outer - r_inner, 1e-3)
            radial_t = (ti.max(hit_r, r_inner) - r_inner) / radial_span
            radial_t = ti.min(ti.max(radial_t, 0.0), 1.0)
            radial_profile = ti.pow(
                1.0 - radial_t,
                ti.cast(DISK_RADIAL_BRIGHTNESS_POWER, ti.f32)
            )
            min_boost = ti.cast(DISK_RADIAL_BRIGHTNESS_MIN, ti.f32)
            max_boost = ti.cast(DISK_RADIAL_BRIGHTNESS_MAX, ti.f32)
            radial_boost = min_boost + (max_boost - min_boost) * radial_profile
            brightness *= radial_boost

            # 黑体辐射颜色偏移（Wien 近似）
            # B(λ, gT)/B(λ, T) ≈ exp(x(1 - 1/g))，x = hc/(λkT)
            # 基准温度 ~10000K，代表波长 R=650nm G=530nm B=460nm
            # x_R = 0.01439/(650e-9*10000) ≈ 2.21
            # x_G = 0.01439/(530e-9*10000) ≈ 2.72
            # x_B = 0.01439/(460e-9*10000) ≈ 3.13
            g_safe = ti.max(g, 0.1)
            wien_arg = 1.0 - 1.0 / g_safe
            r_scale = ti.exp(2.21 * wien_arg)
            g_scale = ti.exp(2.72 * wien_arg)
            b_scale = ti.exp(3.13 * wien_arg)
            # 归一化：让绿色通道保持不变，只看相对偏移
            norm = g_scale
            r_scale = ti.min(r_scale / norm, 3.0)
            g_scale = 1.0
            b_scale = ti.min(b_scale / norm, 3.0)

            shifted = ti.Vector([
                base_color[0] * r_scale,
                base_color[1] * g_scale,
                base_color[2] * b_scale,
            ])
            tint = _color_temp_to_tint(color_temp)
            return ti.math.clamp(shifted * tint * brightness, 0.0, 10.0)

        @ti.func
        def _compute_acceleration(pos, L2):
            """Compute gravitational acceleration for Schwarzschild metric."""
            r2 = pos.dot(pos)
            r = ti.sqrt(r2)
            r5 = r2 * r2 * r
            return -1.5 * L2 / r5 * pos

        @ti.func
        def _compute_acc_jacobian(pos, d_pos, L2):
            """Compute Jacobian of acceleration w.r.t. position.

            For variational equation: d(acc)/d(pos) = -1.5*L2 * (I/r^5 - 5*pos*pos^T/r^7)
            Applied to perturbation vector d_pos.
            """
            r2 = pos.dot(pos)
            r = ti.sqrt(r2)
            r5 = r2 * r2 * r
            r7 = r5 * r2
            factor = -1.5 * L2 / r5
            proj = pos.dot(d_pos) / r2
            return factor * (d_pos - 5.0 * pos * proj)

        @ti.func
        def _sample_skybox(d):
            """Sample skybox texture with bilinear interpolation."""
            x, y, z = d[0], d[1], d[2]
            theta = ti.acos(ti.min(ti.max(z, -1.0), 1.0))
            phi = ti.atan2(y, x)
            if phi < 0:
                phi += 2 * ti.math.pi
            u = phi / (2 * ti.math.pi) * tex_w
            v = theta / ti.math.pi * tex_h
            u0 = ti.cast(ti.floor(u), ti.i32)
            v0 = ti.cast(ti.floor(v), ti.i32)
            fu = u - ti.cast(u0, ti.f32)
            fv = v - ti.cast(v0, ti.f32)
            u0_w = u0 % tex_w
            u1_w = (u0 + 1) % tex_w
            v0_h = ti.min(ti.max(v0, 0), tex_h - 1)
            v1_h = ti.min(ti.max(v0 + 1, 0), tex_h - 1)
            c00 = texture_field[v0_h, u0_w]
            c10 = texture_field[v0_h, u1_w]
            c01 = texture_field[v1_h, u0_w]
            c11 = texture_field[v1_h, u1_w]
            return (c00 * (1 - fu) * (1 - fv) +
                    c10 * fu * (1 - fv) +
                    c01 * (1 - fu) * fv +
                    c11 * fu * fv)

        @ti.func
        def _sample_disk(hit_x, hit_y, r_inner, r_outer, t_offset):
            """Sample accretion disk texture with bilinear interpolation."""
            r = ti.sqrt(hit_x ** 2 + hit_y ** 2)
            phi = ti.atan2(hit_y, hit_x)
            r_safe = ti.max(r, 1e-3)
            omega = ti.sqrt(0.5 / (r_safe ** 3 + 1e-6))
            phi = phi + t_offset * omega
            while phi < 0:
                phi += 2 * ti.math.pi
            while phi >= 2 * ti.math.pi:
                phi -= 2 * ti.math.pi
            u = phi / (2 * ti.math.pi) * dtex_w
            v = (r - r_inner) / (r_outer - r_inner) * dtex_h
            # 双线性插值
            u0 = ti.cast(ti.floor(u), ti.i32)
            v0 = ti.cast(ti.floor(v), ti.i32)
            fu = u - ti.cast(u0, ti.f32)
            fv = v - ti.cast(v0, ti.f32)
            u0_w = u0 % dtex_w
            u1_w = (u0 + 1) % dtex_w
            v0_h = ti.min(ti.max(v0, 0), dtex_h - 1)
            v1_h = ti.min(ti.max(v0 + 1, 0), dtex_h - 1)
            c00 = disk_texture_field[v0_h, u0_w]
            c10 = disk_texture_field[v0_h, u1_w]
            c01 = disk_texture_field[v1_h, u0_w]
            c11 = disk_texture_field[v1_h, u1_w]
            return (c00 * (1 - fu) * (1 - fv) +
                    c10 * fu * (1 - fv) +
                    c01 * (1 - fu) * fv +
                    c11 * fu * fv)

        @ti.func
        def _sample_disk_mip(hit_x, hit_y, r_inner, r_outer, t_offset, lod):
            """Sample accretion disk texture with mipmap LOD."""
            r = ti.sqrt(hit_x ** 2 + hit_y ** 2)
            phi = ti.atan2(hit_y, hit_x)
            r_safe = ti.max(r, 1e-3)
            omega = ti.sqrt(0.5 / (r_safe ** 3 + 1e-6))
            phi = phi + t_offset * omega
            while phi < 0:
                phi += 2 * ti.math.pi
            while phi >= 2 * ti.math.pi:
                phi -= 2 * ti.math.pi

            lod_i = ti.cast(ti.min(ti.max(lod, 0.0), ti.cast(num_mip_levels - 1, ti.f32)), ti.i32)

            # 根据 lod 计算实际纹理尺寸
            tex_w_lod = ti.cast(dtex_w, ti.f32) / ti.pow(2.0, ti.cast(lod_i, ti.f32))
            tex_h_lod = ti.cast(dtex_h, ti.f32) / ti.pow(2.0, ti.cast(lod_i, ti.f32))

            u = phi / (2 * ti.math.pi) * tex_w_lod
            v = (r - r_inner) / (r_outer - r_inner) * tex_h_lod

            u0 = ti.cast(ti.floor(u), ti.i32)
            v0 = ti.cast(ti.floor(v), ti.i32)
            fu = u - ti.cast(u0, ti.f32)
            fv = v - ti.cast(v0, ti.f32)
            u0_w = ti.cast(u0 % ti.cast(tex_w_lod, ti.i32), ti.i32)
            u1_w = ti.cast((u0 + 1) % ti.cast(tex_w_lod, ti.i32), ti.i32)
            v0_h = ti.min(ti.max(v0, 0), ti.cast(tex_h_lod - 1, ti.i32))
            v1_h = ti.min(ti.max(v0 + 1, 0), ti.cast(tex_h_lod - 1, ti.i32))
            c00 = disk_mips_field[lod_i, v0_h, u0_w]
            c10 = disk_mips_field[lod_i, v0_h, u1_w]
            c01 = disk_mips_field[lod_i, v1_h, u0_w]
            c11 = disk_mips_field[lod_i, v1_h, u1_w]
            return (c00 * (1 - fu) * (1 - fv) +
                    c10 * fu * (1 - fv) +
                    c01 * (1 - fu) * fv +
                    c11 * fu * fv)

        # ---- 3D Simplex Noise + FBM ----
        perm_field = self.perm_field

        @ti.func
        def _grad3_dot(hash_val, x, y, z):
            """Compute dot product of gradient vector selected by hash with (x, y, z).

            Selects one of 12 gradient directions lying along cube edges,
            then returns the dot product with the offset vector.

            Args:
                hash_val: integer hash selecting one of 12 gradient directions
                x, y, z: offset vector components from simplex corner
            Returns:
                dot product (float), contributes to final noise in approx [-1, 1]
            """
            h = hash_val % 12
            u = x if h < 8 else y
            v = y if h < 4 else (x if h == 12 or h == 14 else z)
            r1 = u if h & 1 == 0 else -u
            r2 = v if h & 2 == 0 else -v
            return r1 + r2

        @ti.func
        def _simplex_noise_3d(x, y, z):
            """3D simplex noise based on Stefan Gustavson's implementation.

            Evaluates coherent gradient noise on a simplex (tetrahedral) lattice.
            The simplex grid is obtained by skewing the input coordinate space.

            Args:
                x, y, z: input coordinates (float, any range)
            Returns:
                noise value in [-1, 1]

            Formula:
                n = 32 * sum_i( max(0.6 - |d_i|^2, 0)^4 * grad_i . d_i )
                where d_i is the offset from simplex corner i,
                grad_i is a pseudo-random gradient from a permutation table.

            Physical Meaning:
                Provides spatially coherent pseudo-random values for procedural
                texture generation. Used as building block for FBM.
            """
            F3 = 1.0 / 3.0
            G3 = 1.0 / 6.0

            s = (x + y + z) * F3
            i = ti.cast(ti.floor(x + s), ti.i32)
            j = ti.cast(ti.floor(y + s), ti.i32)
            k = ti.cast(ti.floor(z + s), ti.i32)

            t = ti.cast(i + j + k, ti.f32) * G3
            x0 = x - (ti.cast(i, ti.f32) - t)
            y0 = y - (ti.cast(j, ti.f32) - t)
            z0 = z - (ti.cast(k, ti.f32) - t)

            # 确定所在单纯形（6 种排列之一）
            i1 = 0; j1 = 0; k1 = 0
            i2 = 0; j2 = 0; k2 = 0
            if x0 >= y0:
                if y0 >= z0:
                    i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0
                elif x0 >= z0:
                    i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1
                else:
                    i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1
            else:
                if y0 < z0:
                    i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1
                elif x0 < z0:
                    i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1
                else:
                    i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0

            x1 = x0 - ti.cast(i1, ti.f32) + G3
            y1 = y0 - ti.cast(j1, ti.f32) + G3
            z1 = z0 - ti.cast(k1, ti.f32) + G3
            x2 = x0 - ti.cast(i2, ti.f32) + 2.0 * G3
            y2 = y0 - ti.cast(j2, ti.f32) + 2.0 * G3
            z2 = z0 - ti.cast(k2, ti.f32) + 2.0 * G3
            x3 = x0 - 1.0 + 3.0 * G3
            y3 = y0 - 1.0 + 3.0 * G3
            z3 = z0 - 1.0 + 3.0 * G3

            ii = i & 255
            jj = j & 255
            kk = k & 255
            gi0 = perm_field[ii + perm_field[jj + perm_field[kk]]]
            gi1 = perm_field[ii + i1 + perm_field[jj + j1 + perm_field[kk + k1]]]
            gi2 = perm_field[ii + i2 + perm_field[jj + j2 + perm_field[kk + k2]]]
            gi3 = perm_field[ii + 1 + perm_field[jj + 1 + perm_field[kk + 1]]]

            n = 0.0
            t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
            if t0 >= 0.0:
                t0 = t0 * t0
                n += t0 * t0 * _grad3_dot(gi0, x0, y0, z0)
            t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
            if t1 >= 0.0:
                t1 = t1 * t1
                n += t1 * t1 * _grad3_dot(gi1, x1, y1, z1)
            t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
            if t2 >= 0.0:
                t2 = t2 * t2
                n += t2 * t2 * _grad3_dot(gi2, x2, y2, z2)
            t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
            if t3 >= 0.0:
                t3 = t3 * t3
                n += t3 * t3 * _grad3_dot(gi3, x3, y3, z3)

            return 32.0 * n

        @ti.func
        def _fbm_3d(x, y, z, octaves, persistence, lacunarity):
            """Fractal Brownian motion using 3D simplex noise.

            Accumulates multiple octaves of simplex noise with geometrically
            decaying amplitude and increasing frequency, producing self-similar
            fractal patterns at multiple scales.

            Args:
                x, y, z: input coordinates (float, any range)
                octaves: number of noise layers to sum (int, typically 3-6)
                persistence: amplitude decay per octave (float, typically 0.4-0.6;
                    higher = more high-frequency detail)
                lacunarity: frequency multiplier per octave (float, typically 2.0)
            Returns:
                accumulated noise value (float). Range depends on octaves and
                persistence; for persistence=0.5 and 4 octaves, approx [-1.87, 1.87].

            Formula:
                fbm = sum_{i=0}^{octaves-1} persistence^i * noise(x * lac^i, y * lac^i, z * lac^i)

            Physical Meaning:
                Generates natural-looking turbulent patterns for accretion disk
                background layer. The time coordinate (z or dedicated t axis)
                provides smooth temporal evolution without Keplerian wrap artifacts.
            """
            value = 0.0
            amplitude = 1.0
            freq = 1.0
            for _ in range(octaves):
                value += amplitude * _simplex_noise_3d(x * freq, y * freq, z * freq)
                amplitude *= persistence
                freq *= lacunarity
            return value

        @ti.kernel
        def _ray_march_kernel(image_field: ti.template(), disk_layer_field: ti.template(),
                              cam_pos_field: ti.template(), cam_right_field: ti.template(),
                              cam_up_field: ti.template(), cam_forward_field: ti.template(),
                              pixel_width_field: ti.template(),
                              pixel_height_field: ti.template(), r_escape_field: ti.template(),
                              h_base: ti.f32, r_inner: ti.f32, r_outer: ti.f32, t_offset: ti.f32,
                              disk_tilt: ti.f32, skip_diff: ti.i32):
            """Ray marching kernel for Schwarzschild black hole rendering.

            Traces rays from camera through each pixel, integrating accretion disk
            emission along the path with relativistic effects.
            """
            cp = cam_pos_field[None]
            cr = cam_right_field[None]
            cu = cam_up_field[None]
            cf = cam_forward_field[None]
            pw = pixel_width_field[None]
            ph = pixel_height_field[None]

            # 吸积盘倾斜角度（弧度）
            tilt_rad = disk_tilt * ti.math.pi / 180.0
            min_fac = ti.cast(0.2, ti.f32)

            center = cp + cf * 1.0
            tl = center - cr * (pw * width / 2) + cu * (ph * height / 2)

            max_fac = ti.cast(10.0, ti.f32)
            r_cap = ti.cast(RS, ti.f32)
            r_esc = r_escape_field[None]
            max_iter = ti.cast(r_esc * 40 / h_base, ti.i32)
            max_affine = r_esc * 40.0

            for i, j in image_field:
                px_f = ti.cast(i, ti.f32)
                py_f = ti.cast(j, ti.f32)
                pixel_pos = tl + (px_f + 0.5) * pw * cr - (py_f + 0.5) * ph * cu
                ray_dir = (pixel_pos - cp).normalized()

                pos = cp
                dir_ = ray_dir
                L2_val = dir_.cross(pos).norm() ** 2

                d_pos_dx = ti.Vector([0.0, 0.0, 0.0])
                d_dir_dx = ti.Vector([0.0, 0.0, 0.0])
                d_pos_dy = ti.Vector([0.0, 0.0, 0.0])
                d_dir_dy = ti.Vector([0.0, 0.0, 0.0])
                if skip_diff == 0:
                    pixel_pos_x1 = tl + (px_f + 1.5) * pw * cr - (py_f + 0.5) * ph * cu
                    ray_dir_x1 = (pixel_pos_x1 - cp).normalized()
                    d_dir_dx = ray_dir_x1 - ray_dir
                    pixel_pos_y1 = tl + (px_f + 0.5) * pw * cr - (py_f + 1.5) * ph * cu
                    ray_dir_y1 = (pixel_pos_y1 - cp).normalized()
                    d_dir_dy = ray_dir_y1 - ray_dir

                escaped = False
                escape_dir = ti.Vector([0.0, 0.0, 0.0])
                event_horizon_hit = False
                accum_disk = ti.Vector([0.0, 0.0, 0.0])
                disk_alpha_total = 0.0
                step_count = 0
                affine = 0.0
                # 记录击中时的微分状态
                hit_d_pos_dx = ti.Vector([0.0, 0.0, 0.0])
                hit_d_pos_dy = ti.Vector([0.0, 0.0, 0.0])
                tan_t = ti.tan(tilt_rad)

                while step_count < max_iter:
                    old_pos = pos
                    old_z = pos[2]
                    old_y = pos[1]
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

                    # 主光线 RK4
                    k1p = h * dir_
                    k1d = h * _compute_acceleration(pos, L2_val)
                    k2p = h * (dir_ + 0.5 * k1d)
                    k2d = h * _compute_acceleration(pos + 0.5 * k1p, L2_val)
                    k3p = h * (dir_ + 0.5 * k2d)
                    k3d = h * _compute_acceleration(pos + 0.5 * k2p, L2_val)
                    k4p = h * (dir_ + k3d)
                    k4d = h * _compute_acceleration(pos + k3p, L2_val)

                    new_pos = pos + (k1p + 2 * k2p + 2 * k3p + k4p) / 6
                    new_dir = dir_ + (k1d + 2 * k2d + 2 * k3d + k4d) / 6

                    new_d_pos_dx = d_pos_dx
                    new_d_dir_dx = d_dir_dx
                    new_d_pos_dy = d_pos_dy
                    new_d_dir_dy = d_dir_dy
                    if skip_diff == 0:
                        k1p_dx = h * d_dir_dx
                        k1d_dx = h * _compute_acc_jacobian(pos, d_pos_dx, L2_val)
                        k2p_dx = h * (d_dir_dx + 0.5 * k1d_dx)
                        k2d_dx = h * _compute_acc_jacobian(pos + 0.5 * k1p, d_pos_dx + 0.5 * k1p_dx, L2_val)
                        k3p_dx = h * (d_dir_dx + 0.5 * k2d_dx)
                        k3d_dx = h * _compute_acc_jacobian(pos + 0.5 * k2p, d_pos_dx + 0.5 * k2p_dx, L2_val)
                        k4p_dx = h * (d_dir_dx + k3d_dx)
                        k4d_dx = h * _compute_acc_jacobian(pos + k3p, d_pos_dx + k3p_dx, L2_val)

                        new_d_pos_dx = d_pos_dx + (k1p_dx + 2 * k2p_dx + 2 * k3p_dx + k4p_dx) / 6
                        new_d_dir_dx = d_dir_dx + (k1d_dx + 2 * k2d_dx + 2 * k3d_dx + k4d_dx) / 6

                        k1p_dy = h * d_dir_dy
                        k1d_dy = h * _compute_acc_jacobian(pos, d_pos_dy, L2_val)
                        k2p_dy = h * (d_dir_dy + 0.5 * k1d_dy)
                        k2d_dy = h * _compute_acc_jacobian(pos + 0.5 * k1p, d_pos_dy + 0.5 * k1p_dy, L2_val)
                        k3p_dy = h * (d_dir_dy + 0.5 * k2d_dy)
                        k3d_dy = h * _compute_acc_jacobian(pos + 0.5 * k2p, d_pos_dy + 0.5 * k2p_dy, L2_val)
                        k4p_dy = h * (d_dir_dy + k3d_dy)
                        k4d_dy = h * _compute_acc_jacobian(pos + k3p, d_pos_dy + k3p_dy, L2_val)

                        new_d_pos_dy = d_pos_dy + (k1p_dy + 2 * k2p_dy + 2 * k3p_dy + k4p_dy) / 6
                        new_d_dir_dy = d_dir_dy + (k1d_dy + 2 * k2d_dy + 2 * k3d_dy + k4d_dy) / 6

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

                    if skip_diff == 0:
                        d_pos_dx = new_d_pos_dx
                        d_dir_dx = new_d_dir_dx
                        d_pos_dy = new_d_pos_dy
                        d_dir_dy = new_d_dir_dy

                    new_z = new_pos[2]
                    new_y = new_pos[1]

                    # 吸积盘检测：穿过倾斜平面 z = y * tan(tilt)
                    # 平面方程: z - y * tan_t = 0
                    f_old = old_z - old_y * tan_t
                    f_new = new_z - new_y * tan_t
                    if f_old * f_new < 0:
                        t_frac = f_old / (f_old - f_new + 1e-8)
                        hit_x = old_pos[0] + t_frac * (new_pos[0] - old_pos[0])
                        hit_y = old_pos[1] + t_frac * (new_pos[1] - old_pos[1])
                        hit_r = ti.sqrt(hit_x ** 2 + hit_y ** 2)

                        if skip_diff == 0:
                            hit_d_pos_dx = d_pos_dx + t_frac * (new_d_pos_dx - d_pos_dx)
                            hit_d_pos_dy = d_pos_dy + t_frac * (new_d_pos_dy - d_pos_dy)

                        if r_outer >= hit_r >= r_inner:
                            hit_z = hit_y * tan_t
                            hit_pos_vec = ti.Vector([hit_x, hit_y, hit_z])
                            ray_to_cam = -dir_

                            disk_rgba = ti.Vector([0.0, 0.0, 0.0, 0.0])
                            if anti_alias_mode == 0 or skip_diff == 1:
                                # disabled: 直接采样
                                disk_rgba = _sample_disk(hit_x, hit_y, r_inner, r_outer, t_offset)
                            else:
                                # ray_differentials: 根据光线微分计算纹理梯度
                                # 计算击中点处的纹理坐标梯度
                                # 纹理坐标: u = phi/(2pi) * dtex_w, v = (r-r_inner)/(r_outer-r_inner) * dtex_h
                                hit_r_cyl = ti.sqrt(hit_x ** 2 + hit_y ** 2 + 1e-6)

                                # du/dpixel_x 和 dv/dpixel_x
                                # r 对 d_pos_dx 的导数
                                dr_dx = (hit_x * hit_d_pos_dx[0] + hit_y * hit_d_pos_dx[1]) / hit_r_cyl
                                # phi 对 d_pos_dx 的导数
                                dphi_dx = (-hit_y * hit_d_pos_dx[0] + hit_x * hit_d_pos_dx[1]) / (hit_r_cyl ** 2 + 1e-6)

                                # 纹理坐标梯度
                                dudx = dphi_dx * dtex_w / (2.0 * ti.math.pi)
                                dvdx = dr_dx * dtex_h / (r_outer - r_inner)

                                # Y 方向梯度
                                dr_dy = (hit_x * hit_d_pos_dy[0] + hit_y * hit_d_pos_dy[1]) / hit_r_cyl
                                dphi_dy = (-hit_y * hit_d_pos_dy[0] + hit_x * hit_d_pos_dy[1]) / (hit_r_cyl ** 2 + 1e-6)
                                dudy = dphi_dy * dtex_w / (2.0 * ti.math.pi)
                                dvdy = dr_dy * dtex_h / (r_outer - r_inner)

                                # 计算梯度幅值用于 LOD（取 X 和 Y 方向的最大值）
                                grad_sq_x = dudx * dudx + dvdx * dvdx
                                grad_sq_y = dudy * dudy + dvdy * dvdy
                                grad_sq = ti.max(grad_sq_x, grad_sq_y)

                                lod_diff = ti.log(ti.max(grad_sq, 1.0)) / ti.log(2.0) * aa_strength
                                lod_diff = ti.min(ti.max(lod_diff, 0.0), 3.0)

                                disk_rgba = _sample_disk_mip(hit_x, hit_y, r_inner, r_outer, t_offset, lod_diff)

                            disk_col = ti.Vector([disk_rgba[0], disk_rgba[1], disk_rgba[2]])
                            base_alpha = ti.min(disk_rgba[3], 0.999)
                            disk_alpha = 1.0 - ti.pow(1.0 - base_alpha, alpha_gain)

                            col_shifted = _apply_g_factor(
                                disk_col, hit_pos_vec, hit_r, ray_to_cam, cp, r_inner, r_outer, tilt_rad
                            )

                            front_factor = 1.0 - disk_alpha_total
                            accum_disk += col_shifted * disk_alpha * front_factor
                            disk_alpha_total = 1.0 - front_factor * (1.0 - disk_alpha)

                    pos = new_pos
                    dir_ = new_dir
                    step_count += 1

                # 分离式渲染：背景和吸积盘分开存储
                bg_color = ti.Vector([0.0, 0.0, 0.0])
                if event_horizon_hit:
                    bg_color = ti.Vector([0.0, 0.0, 0.0])
                elif escaped:
                    bg_color = _sample_skybox(escape_dir)

                bg_color = bg_color * (1.0 - disk_alpha_total)

                image_field[i, j] = bg_color
                disk_layer_field[i, j] = ti.math.clamp(accum_disk, 0.0, 1.0)

        self._ray_march_kernel = _ray_march_kernel

        @ti.kernel
        def _bloom_kernel(image_field: ti.template(), bright_field: ti.template(),
                          blur_field: ti.template(), threshold: ti.f32, intensity: ti.f32,
                          kernel_radius: ti.i32, sigma_scale: ti.f32):
            """Bloom post-processing kernel.

            Extracts bright regions, applies separable Gaussian blur,
            and adds back to image for glow effect.
            """
            w = ti.cast(image_field.shape[0], ti.i32)
            h = ti.cast(image_field.shape[1], ti.i32)

            for i, j in image_field:
                col = image_field[i, j]
                lum = col[0] * 0.2126 + col[1] * 0.7152 + col[2] * 0.0722
                if lum > threshold:
                    bright_field[i, j] = col
                else:
                    bright_field[i, j] = ti.Vector([0.0, 0.0, 0.0])

            # 水平方向模糊（sigma 按 sigma_scale 缩放）
            for i, j in blur_field:
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
                        col = bright_field[ni, j]

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
                    blur_field[i, j] = ti.Vector([sum_r / weight_r, sum_g / weight_g, sum_b / weight_b])
                else:
                    blur_field[i, j] = ti.Vector([0.0, 0.0, 0.0])

            # 复制回 bright_field
            for i, j in bright_field:
                bright_field[i, j] = blur_field[i, j]

            # 垂直方向模糊（sigma 按 sigma_scale 缩放）
            for i, j in blur_field:
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
                        col = bright_field[i, nj]

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
                    blur_field[i, j] = ti.Vector([sum_r / weight_r, sum_g / weight_g, sum_b / weight_b])
                else:
                    blur_field[i, j] = ti.Vector([0.0, 0.0, 0.0])

            for i, j in image_field:
                image_field[i, j] = ti.math.clamp(
                    image_field[i, j] + blur_field[i, j] * intensity, 0.0, 1.0)

        self._bloom_kernel = _bloom_kernel

        @ti.kernel
        def _lens_flare_kernel(image_field: ti.template(),
                               disk_center_x: ti.f32, disk_center_y: ti.f32,
                               screen_center_x: ti.f32, screen_center_y: ti.f32,
                               intensity: ti.f32, scale: ti.f32):
            """Lens flare effect kernel.

            Renders ghost images and diffraction rings along the line
            from bright source to screen center.
            """
            w = ti.cast(image_field.shape[0], ti.i32)
            h = ti.cast(image_field.shape[1], ti.i32)

            for i, j in image_field:
                dx = ti.cast(i, ti.f32) - disk_center_x
                dy = ti.cast(j, ti.f32) - disk_center_y
                dist = ti.sqrt(dx * dx + dy * dy)

                flare = ti.Vector([0.0, 0.0, 0.0])

                for g in range(6):
                    t = ti.cast(g + 1, ti.f32) * 0.10
                    ghost_x = disk_center_x + (screen_center_x - disk_center_x) * t
                    ghost_y = disk_center_y + (screen_center_y - disk_center_y) * t
                    gdx = ti.cast(i, ti.f32) - ghost_x
                    gdy = ti.cast(j, ti.f32) - ghost_y
                    gdist = ti.sqrt(gdx * gdx + gdy * gdy)
                    gsize = ti.cast(20 + g * 15, ti.f32) * scale
                    if gdist < gsize:
                        galpha = (1.0 - gdist / gsize) * (1.0 - ti.cast(g, ti.f32) * 0.12) * 0.4
                        ghost_col = ti.Vector([1.0, 0.9, 0.7]) * galpha
                        flare += ghost_col

                ring_t = 0.3
                ring_x = disk_center_x + (screen_center_x - disk_center_x) * ring_t
                ring_y = disk_center_y + (screen_center_y - disk_center_y) * ring_t
                rdx = ti.cast(i, ti.f32) - ring_x
                rdy = ti.cast(j, ti.f32) - ring_y
                rdist = ti.sqrt(rdx * rdx + rdy * rdy)
                ring_r = 80.0 * scale
                ring_w = 8.0 * scale
                ring_alpha = 0.0
                if ti.abs(rdist - ring_r) < ring_w:
                    ring_alpha = (1.0 - ti.abs(rdist - ring_r) / ring_w) * 0.15
                if ring_alpha > 0:
                    flare += ti.Vector([0.6, 0.7, 1.0]) * ring_alpha

                image_field[i, j] = ti.math.clamp(image_field[i, j] + flare * intensity, 0.0, 1.0)

        self._lens_flare_kernel = _lens_flare_kernel

        @ti.kernel
        def _compose_disk_texture_kernel(
                disk_tex: ti.template(),
                comp: ti.template(),
                omega: ti.template(),
                edge: ti.template(),
                stats: ti.template(),
                row_stats: ti.template(),
                t_offset: ti.f32,
                enable_rt: ti.i32,
                color_temp_val: ti.f32):
            """GPU 纹理合成 kernel：滚动 13 个组件 + 合成最终 RGBA 纹理。

            精确复现 _generate_disk_texture_rotating_from_state +
            _compose_disk_texture_from_fields 的完整逻辑。
            当 generation_scale=1 时与 CPU 路径像素级等价。
            """
            n_r = disk_tex.shape[0]
            n_phi = disk_tex.shape[1]

            density_p98 = stats[0]
            struct_scale = stats[1]

            t_factor = (color_temp_val - 4500.0) / (6500.0 - 2700.0)
            T_min = 2000.0 + t_factor * 1000.0
            T_max = 9000.0 + t_factor * 3000.0

            rt_w = 0.20
            if enable_rt == 0:
                rt_w = 0.0

            for ri, phi_i in disk_tex:
                omega_val = omega[ri]
                shift = ti.cast(
                    t_offset * omega_val / (2.0 * ti.math.pi) * ti.cast(n_phi, ti.f32),
                    ti.i32)
                src = (phi_i + shift) % n_phi
                if src < 0:
                    src += n_phi

                tb = comp[0, ri, src]
                sp = comp[1, ri, src]
                sp_t = comp[2, ri, src]
                turb = comp[3, ri, src]
                turb_t = comp[4, ri, src]
                arc = comp[5, ri, src]
                arc_t = comp[6, ri, src]
                rt = comp[7, ri, src]
                rt_t = comp[8, ri, src]
                hs = comp[9, ri, src]
                hs_t = comp[10, ri, src]
                az = comp[11, ri, src]
                dm = comp[12, ri, src]

                # density = weighted sum * disturb_mod * edge
                density = (0.15 + 0.10 * sp + 0.15 * turb + 0.20 * hs
                           + 0.30 * arc + rt_w * rt) * dm * edge[ri]
                density = ti.min(ti.max(density / (density_p98 + 1e-6), 0.0), 1.0)

                # temp_struct = sum of temp components * disturb_mod
                temp_struct = (sp_t + turb_t + arc_t + rt_t + hs_t) * dm
                ts_scaled = ti.min(ti.max(
                    temp_struct / (struct_scale + 1e-6) * 0.8, 0.0), 1.2)

                # clamp temp_base
                max_r = row_stats[ri][0]
                p70_r = row_stats[ri][1]
                ceiling = ti.max(p70_r, 0.05)
                tb_clamped = ti.min(tb, ceiling)
                tb_clamped = ti.min(tb_clamped, max_r)

                temperature = ti.min(ti.max(
                    ti.max(tb_clamped, ts_scaled), 0.0), 1.0)

                # anisotropic temperature + blackbody
                temp_aniso = ti.min(ti.max(
                    temperature * (0.9 + 0.25 * az), 0.0), 1.0)
                T_K = T_min + temp_aniso * (T_max - T_min)
                bb = _color_temp_to_tint(T_K)
                bb_b = ti.min(bb[2], bb[0])

                lum = ti.min(ti.max(ti.sqrt(temp_aniso), 0.0), 1.0)

                disk_tex[ri, phi_i] = ti.Vector([
                    ti.min(ti.max(bb[0] * lum, 0.0), 1.0),
                    ti.min(ti.max(bb[1] * lum, 0.0), 1.0),
                    ti.min(ti.max(bb_b * lum, 0.0), 1.0),
                    density
                ])

        self._compose_disk_texture_kernel = _compose_disk_texture_kernel

        @ti.kernel
        def _mipmap_copy_base_kernel(mips: ti.template(), base: ti.template()):
            """将 disk_texture_field (level 0) 复制到 mipmap field。"""
            for ri, phi_i in base:
                mips[0, ri, phi_i] = base[ri, phi_i]

        self._mipmap_copy_base_kernel = _mipmap_copy_base_kernel

        @ti.kernel
        def _mipmap_downsample_kernel(mips: ti.template(),
                                      level: ti.i32,
                                      src_h: ti.i32, src_w: ti.i32):
            """2×2 box filter 下采样生成 mipmap 的第 level 级。"""
            dst_h = src_h // 2
            dst_w = src_w // 2
            for ri, phi_i in ti.ndrange(dst_h, dst_w):
                c = (mips[level - 1, ri * 2, phi_i * 2]
                     + mips[level - 1, ri * 2, phi_i * 2 + 1]
                     + mips[level - 1, ri * 2 + 1, phi_i * 2]
                     + mips[level - 1, ri * 2 + 1, phi_i * 2 + 1]) / 4.0
                mips[level, ri, phi_i] = c

        self._mipmap_downsample_kernel = _mipmap_downsample_kernel

        @ti.kernel
        def _compose_final_kernel(final: ti.template(),
                                  bg: ti.template(),
                                  disk: ti.template(),
                                  bloom: ti.template(),
                                  use_bloom: ti.i32):
            """合成最终图像：背景 + 吸积盘 + 可选 bloom，Y 轴翻转适配 ti.GUI。"""
            h = final.shape[1]
            for i, j in final:
                jf = h - 1 - j
                if use_bloom == 1:
                    final[i, j] = ti.math.clamp(
                        bg[i, jf] + disk[i, jf] + bloom[i, jf], 0.0, 1.0)
                else:
                    final[i, j] = ti.math.clamp(
                        bg[i, jf] + disk[i, jf], 0.0, 1.0)

        self._compose_final_kernel = _compose_final_kernel

        # ---- 噪声评估 kernel（供测试和调试使用）----
        @ti.kernel
        def _noise_eval_kernel(out: ti.template(), coords: ti.template(),
                               mode: ti.i32, octaves: ti.i32,
                               persistence: ti.f32, lacunarity: ti.f32):
            """Evaluate simplex noise or FBM at given coordinates.

            Args:
                out: output field, shape (N,), stores noise values
                coords: input field, shape (N, 3), xyz coordinates
                mode: 0 = simplex_noise_3d, 1 = fbm_3d
                octaves: FBM octave count (only used when mode=1)
                persistence: FBM persistence (only used when mode=1)
                lacunarity: FBM lacunarity (only used when mode=1)
            """
            for i in out:
                cx = coords[i, 0]
                cy = coords[i, 1]
                cz = coords[i, 2]
                if mode == 0:
                    out[i] = _simplex_noise_3d(cx, cy, cz)
                else:
                    out[i] = _fbm_3d(cx, cy, cz, octaves, persistence, lacunarity)

        self._noise_eval_kernel = _noise_eval_kernel

        # ---- 背景层实时生成 kernel（宽 r 组件）----

        @ti.kernel
        def _generate_background_kernel(
                comp: ti.template(),
                az_freq: ti.i32,
                az_shear: ti.f32,
                r_inner: ti.f32,
                r_outer: ti.f32,
                t: ti.f32):
            """Generate wide-r background components using time-varying 3D noise.

            Writes to comp_field at indices [0,1,2,3,4,11,12] for the 5 wide-r
            components: temp_base, MAD asymmetry/temp, turbulence/turb_temp,
            az_hotspot, disturb_mod.

            Noise coordinates use Keplerian-rotated phi: phi_rot = phi + omega(r)*t,
            mapped via (cos(phi_rot), sin(phi_rot)) for seamless wrapping. This gives
            differential rotation without pre-computed array roll or wrap artifacts.

            Args:
                comp: component field (13, n_r, n_phi) — output for indices 0-4, 11, 12
                az_freq: azimuthal hotspot frequency (integer, typically 2-4)
                az_shear: azimuthal hotspot shear strength (float, typically 2-4)
                r_inner: inner disk radius (physical units)
                r_outer: outer disk radius (physical units)
                t: wall-clock time in seconds for temporal evolution
            """
            n_r = comp.shape[1]
            n_phi = comp.shape[2]
            pi2 = 2.0 * ti.math.pi

            for ri, phi_i in ti.ndrange(n_r, n_phi):
                r = ti.cast(ri, ti.f32) / ti.cast(n_r, ti.f32)
                phi = ti.cast(phi_i, ti.f32) / ti.cast(n_phi, ti.f32) * pi2

                # 开普勒旋转：每行以自身角速度旋转，内快外慢
                # phi + omega*t 使模式沿 -phi 方向移动，与实体层 np.roll(-shift) 一致
                r_phys = r_inner + (r_outer - r_inner) * r
                omega = ti.sqrt(0.5 / (r_phys * r_phys * r_phys + 1e-6))
                phi_rot = phi + omega * t
                cx = ti.cos(phi_rot)
                cy = ti.sin(phi_rot)

                # --- idx 0: temp_base ---
                # 径向衰减 + 慢速 FBM 噪声调制
                # 噪声映射到 [0,1] 以匹配 CPU _fbm_noise 的归一化输出
                decay = ti.pow(ti.max(1.0 - r, 0.0), 1.3)
                tb_noise = ti.min(ti.max(
                    0.5 + 0.5 * _fbm_3d(cx * 8.0, cy * 8.0,
                                         r * 8.0 + t * 0.05,
                                         4, 0.6, 2.0),
                    0.0), 1.0)
                comp[0, ri, phi_i] = decay * (0.85 + 0.15 * tb_noise) * 0.25

                # --- idx 1,2: spiral 已移除，置零 ---
                comp[1, ri, phi_i] = 0.0
                comp[2, ri, phi_i] = 0.0

                # --- idx 3,4: turbulence / turb_temp ---
                # 多尺度时变噪声叠加（每层 clamp 到 [0,1] 匹配 CPU _tileable_noise）
                t_coarse = ti.min(ti.max(0.5 + 0.5 * _fbm_3d(
                    cx * 8.0, cy * 8.0, r * 4.0 + t * 0.06,
                    3, 0.45, 2.0), 0.0), 1.0) * 0.08
                t_mid = ti.min(ti.max(0.5 + 0.5 * _fbm_3d(
                    cx * 24.0, cy * 24.0, r * 12.0 + t * 0.08,
                    4, 0.45, 2.0), 0.0), 1.0) * 0.15
                t_fine = ti.min(ti.max(0.5 + 0.5 * _fbm_3d(
                    cx * 80.0, cy * 80.0, r * 40.0 + t * 0.1,
                    5, 0.45, 2.0), 0.0), 1.0) * 0.25
                t_extra = ti.min(ti.max(0.5 + 0.5 * _fbm_3d(
                    cx * 200.0, cy * 200.0, r * 100.0 + t * 0.12,
                    4, 0.4, 2.0), 0.0), 1.0) * 0.22
                t_ultra = ti.min(ti.max(0.5 + 0.5 * _fbm_3d(
                    cx * 400.0, cy * 400.0, r * 200.0 + t * 0.15,
                    3, 0.35, 2.0), 0.0), 1.0) * 0.18
                t_pixel = ti.min(ti.max(
                    _simplex_noise_3d(cx * 800.0, cy * 800.0,
                                      r * 400.0 + t * 0.2),
                    0.0), 1.0) * 0.12
                turb = ti.min(ti.max(
                    t_coarse + t_mid + t_fine + t_extra + t_ultra + t_pixel,
                    0.0), 1.0)
                comp[3, ri, phi_i] = turb
                comp[4, ri, phi_i] = 0.05 * turb

                # --- idx 11: az_hotspot ---
                # 低频正弦方位波 * 噪声调制（使用旋转后的 phi）
                shear = ti.pow(r, 1.2) * az_shear
                az_wave = 0.5 + 0.5 * ti.sin(
                    (phi_rot + shear) * ti.cast(az_freq, ti.f32))
                az_n = ti.min(ti.max(
                    0.5 + 0.5 * _fbm_3d(cx * 3.0, cy * 3.0,
                                         r * 3.0 + t * 0.04,
                                         3, 0.5, 2.0),
                    0.0), 1.0)
                comp[11, ri, phi_i] = az_wave * az_n

                # --- idx 12: disturb_mod ---
                # 多层扰动调制场（t 演化极慢，旋转由 phi_rot 主导）
                # 每层 clamp 到 [0,1] 匹配 CPU _tileable_noise
                d_coarse = ti.min(ti.max(0.5 + 0.5 * _fbm_3d(
                    cx * 8.0, cy * 8.0, r * 4.0 + t * 0.003,
                    3, 0.5, 2.0), 0.0), 1.0) * 0.05
                d_mid = ti.min(ti.max(0.5 + 0.5 * _fbm_3d(
                    cx * 32.0, cy * 32.0, r * 16.0 + t * 0.005,
                    3, 0.5, 2.0), 0.0), 1.0) * 0.15
                d_fine = ti.min(ti.max(0.5 + 0.5 * _fbm_3d(
                    cx * 100.0, cy * 100.0, r * 50.0 + t * 0.006,
                    4, 0.45, 2.0), 0.0), 1.0) * 0.30
                d_extra = ti.min(ti.max(0.5 + 0.5 * _fbm_3d(
                    cx * 250.0, cy * 250.0, r * 125.0 + t * 0.008,
                    4, 0.4, 2.0), 0.0), 1.0) * 0.30
                d_pixel = ti.min(ti.max(
                    _simplex_noise_3d(cx * 500.0, cy * 500.0,
                                      r * 250.0 + t * 0.01),
                    0.0), 1.0) * 0.20
                disturb_raw = (d_coarse + d_mid + d_fine + d_extra + d_pixel) * 1.4
                disturb_raw = ti.min(ti.max(disturb_raw, 0.05), 1.0)
                radial_preserve = 0.6 + 0.4 * r
                comp[12, ri, phi_i] = ti.min(ti.max(
                    disturb_raw * radial_preserve, 0.1), 1.0)

        self._generate_background_kernel = _generate_background_kernel

        @ti.kernel
        def _copy_entity_staging_to_comp(comp: ti.template(),
                                         staging: ti.template()):
            """Copy entity staging field (6, n_r, n_phi) to comp[5:10].

            Args:
                comp: component field (13, n_r, n_phi)
                staging: entity staging field (6, n_r, n_phi), maps to:
                    staging[0] -> comp[5]  (arcs / filaments density)
                    staging[1] -> comp[6]  (arcs_temp)
                    staging[2] -> comp[7]  (rt_spikes)
                    staging[3] -> comp[8]  (rt_temp)
                    staging[4] -> comp[9]  (hotspot)
                    staging[5] -> comp[10] (hotspot_temp)
            """
            for idx, ri, phi_i in staging:
                comp[5 + idx, ri, phi_i] = staging[idx, ri, phi_i]

        self._copy_entity_staging_to_comp = _copy_entity_staging_to_comp

        @ti.kernel
        def _zero_comp_slice(comp: ti.template(), idx: ti.i32):
            """Zero out a single component slice of comp_field."""
            for ri, phi_i in ti.ndrange(comp.shape[1], comp.shape[2]):
                comp[idx, ri, phi_i] = 0.0

        self._zero_comp_slice = _zero_comp_slice

        @ti.kernel
        def _fill_comp_slice(comp: ti.template(), idx: ti.i32, val: ti.f32):
            """Fill a single component slice of comp_field with a constant."""
            for ri, phi_i in ti.ndrange(comp.shape[1], comp.shape[2]):
                comp[idx, ri, phi_i] = val

        self._fill_comp_slice = _fill_comp_slice

    def init_background_layer(self, n_r: int, n_phi: int, seed: int = 42) -> None:
        """Initialize background layer parameters for interactive mode.

        Generates randomized spiral arm geometry and azimuthal hotspot parameters,
        creates the component field for both background (GPU noise) and entity
        (CPU lifecycle) layers.

        Args:
            n_r: radial resolution of disk texture
            n_phi: azimuthal resolution of disk texture
            seed: random seed for reproducible parameter generation
        """
        rng = np.random.default_rng(seed)

        # 组件场（background kernel 写 [0-4,11,12]，entity 层写 [5-10]）
        if not hasattr(self, '_comp_field') or self._comp_field.shape != (13, n_r, n_phi):
            self._comp_field = ti.field(dtype=ti.f32, shape=(13, n_r, n_phi))

        # 方位热点参数
        self._bg_az_freq = int(rng.integers(2, 5))
        self._bg_az_shear = float(rng.uniform(2.0, 4.0))

        # 边缘和 omega 数据（compose kernel 需要）
        self._edge_field = ti.field(dtype=ti.f32, shape=(n_r,))
        self._edge_field.from_numpy(compute_edge_alpha(n_r).astype(np.float32))

        r_norm = np.linspace(0, 1, n_r)
        r_vals = self.r_disk_inner + (self.r_disk_outer - self.r_disk_inner) * r_norm
        omega_rows = np.sqrt(0.5 / (r_vals ** 3 + 1e-6)).astype(np.float32)
        self._omega_rows_field = ti.field(dtype=ti.f32, shape=(n_r,))
        self._omega_rows_field.from_numpy(omega_rows)
        self._bg_omega_all_np = omega_rows
        self._bg_r_norm_all = r_norm

        self._bg_n_r = n_r
        self._bg_n_phi = n_phi

        # 实体层 staging field（CPU 累加 → from_numpy → copy kernel → comp[5:10]）
        self._entity_staging_field = ti.field(ti.f32, shape=(6, n_r, n_phi))

        # compose kernel 所需统计量（初始值足够宽松，不过度钳制 temp_base）
        self._param_stats_field = ti.field(dtype=ti.f32, shape=(2,))
        self._param_stats_field.from_numpy(np.array([0.5, 0.5], dtype=np.float32))

        r_norm_init = np.linspace(0, 1, n_r)
        tb_init = np.clip(1.0 - r_norm_init, 0, 1) ** 1.3 * 0.25
        self._param_row_stats_field = ti.Vector.field(2, dtype=ti.f32, shape=(n_r,))
        init_row_stats = np.column_stack([
            np.maximum(tb_init, 0.25).astype(np.float32),
            np.maximum(tb_init * 0.8, 0.10).astype(np.float32),
        ])
        self._param_row_stats_field.from_numpy(init_row_stats)

        self._param_enable_rt = 1
        self._param_color_temp = float(DISK_COLOR_TEMPERATURE)

        self._bg_ready = True

    def generate_background(self, t: float) -> None:
        """Generate background layer components on GPU for current time.

        Writes time-evolved noise patterns to comp_field indices [0,1,2,3,4,11,12].
        Must call init_background_layer() first.

        Args:
            t: wall-clock time in seconds
        """
        assert hasattr(self, '_bg_ready') and self._bg_ready, \
            "Must call init_background_layer() first"
        self._generate_background_kernel(
            self._comp_field, self._bg_az_freq, self._bg_az_shear,
            float(self.r_disk_inner), float(self.r_disk_outer), float(t))

    def accumulate_entity_layer(self, factories: dict, now: float) -> None:
        """Accumulate entity contributions and upload to comp_field[5:10].

        For each entity type, iterates over alive entities, applies Keplerian
        rotation (np.roll) and fade factor, then sums contributions into the
        component arrays. Results are uploaded via staging field + copy kernel.

        Args:
            factories: dict with keys 'filament', 'hotspot', 'rt_spike',
                values are EntityFactory instances
            now: current wall-clock time in seconds

        Notes:
            Mapping to comp_field indices:
                staging[0] -> comp[5]  = filaments density (arcs)
                staging[1] -> comp[6]  = filaments temperature (arcs_temp)
                staging[2] -> comp[7]  = RT spikes density
                staging[3] -> comp[8]  = RT spikes temperature
                staging[4] -> comp[9]  = hotspot density
                staging[5] -> comp[10] = hotspot temperature
        """
        n_r = self._bg_n_r
        n_phi = self._bg_n_phi

        staging = np.zeros((6, n_r, n_phi), dtype=np.float32)

        component_map = [
            ('filament', 0, 1),   # density → staging[0], temp → staging[1]
            ('rt_spike', 2, 3),   # density → staging[2], temp → staging[3]
            ('hotspot',  4, 5),   # density → staging[4], temp → staging[5]
        ]

        omega_np = self._bg_omega_all_np
        r_norm_all = self._bg_r_norm_all if hasattr(self, '_bg_r_norm_all') else np.linspace(0, 1, n_r)
        phi_arr = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        two_pi = 2 * np.pi

        for key, d_idx, t_idx in component_map:
            factory = factories.get(key)
            if factory is None:
                continue
            for entity in factory.alive_entities:
                age = now - entity.birth_time

                if entity.entity_type == 'filament':
                    decay = entity.density_factor(age)
                    if decay < FILAMENT_DEATH_THRESHOLD:
                        continue

                    s0 = max(entity.blob_sigma_phi0, 1e-6)
                    sigma_phi_t = s0 + entity.alpha_shear * age
                    amplitude_d = entity.blob_peak_density * s0 / sigma_phi_t
                    amplitude_t = entity.blob_peak_temp * s0 / sigma_phi_t

                    fade_in_dur = FILAMENT_BIRTH_FADE_DUR
                    birth_alpha = min(age / fade_in_dur, 1.0) if fade_in_dur > 0 else 1.0

                    cool_factor = math.exp(-age / entity.tau_cool) if entity.tau_cool > 0 else 1.0
                    scale_d = amplitude_d * birth_alpha * cool_factor
                    scale_t = amplitude_t * birth_alpha * cool_factor

                    inv_2sigma_phi_sq = 0.5 / (sigma_phi_t * sigma_phi_t)
                    sigma_r = max(entity.blob_sigma_r, 1e-6)
                    inv_2sigma_r_sq = 0.5 / (sigma_r * sigma_r)

                    for k, ri in enumerate(entity.row_indices):
                        if 0 <= ri < n_r:
                            r_w = math.exp(-(r_norm_all[ri] - entity.blob_base_r) ** 2
                                           * inv_2sigma_r_sq)
                            center = (entity.source_phi - omega_np[ri] * age) % two_pi
                            d_phi = phi_arr - center
                            d_phi = d_phi - two_pi * np.round(d_phi / two_pi)
                            phi_profile = np.exp(-d_phi * d_phi * inv_2sigma_phi_sq)
                            staging[d_idx, ri] += phi_profile * (scale_d * r_w)
                            staging[t_idx, ri] += phi_profile * (scale_t * r_w)
                else:
                    alpha = entity.fade_factor(now)
                    if alpha <= 0:
                        continue
                    for k, ri in enumerate(entity.row_indices):
                        if 0 <= ri < n_r:
                            shift_ri = int(age * omega_np[ri] / (2 * np.pi) * n_phi)
                            staging[d_idx, ri] += np.roll(
                                entity.phi_density[k], -shift_ri) * alpha
                            staging[t_idx, ri] += np.roll(
                                entity.phi_temp[k], -shift_ri) * alpha

        self._entity_staging_field.from_numpy(staging)
        self._copy_entity_staging_to_comp(self._comp_field,
                                          self._entity_staging_field)

    def recompute_interactive_stats(self) -> None:
        """Recompute normalization stats from current comp_field content.

        Reads comp_field from GPU, computes density_p98, struct_scale, and
        per-row stats, then uploads to stats fields. Should be called after
        both background and entity layers have been written to comp_field.

        Notes:
            Uses the same normalization logic as upload_parametric_state /
            _compose_disk_texture_from_fields to ensure visual consistency.
        """
        comp = self._comp_field.to_numpy()  # (13, n_r, n_phi)
        edge = self._edge_field.to_numpy()  # (n_r,)

        sp = comp[1]
        turb = comp[3]
        arc = comp[5]
        hs = comp[9]
        rt = comp[7]
        dm = comp[12]

        rt_w = 0.20 if self._param_enable_rt else 0.0
        density = (0.15 + 0.10 * sp + 0.15 * turb + 0.20 * hs
                   + 0.30 * arc + rt_w * rt) * dm
        density *= edge[:, None]
        density_p98 = float(np.percentile(density, 98))
        density_p98 = max(density_p98, 0.01)

        sp_t = comp[2]
        turb_t = comp[4]
        arc_t = comp[6]
        rt_t = comp[8]
        hs_t = comp[10]
        temp_struct = (sp_t + turb_t + arc_t + rt_t + hs_t) * dm
        pos_mask = temp_struct > 0
        struct_scale = (float(np.percentile(temp_struct[pos_mask], 95))
                        if np.any(pos_mask) else 1.0)
        struct_scale = max(struct_scale, 0.01)

        temp_struct_scaled = np.clip(
            temp_struct / (struct_scale + 1e-6) * 0.8, 0, 1.2)
        struct_max_per_r = np.max(temp_struct_scaled, axis=1).astype(np.float32)
        struct_p70_per_r = np.quantile(
            temp_struct_scaled, 0.7, axis=1).astype(np.float32)

        # 生命周期模式下实体层更稀疏，很多行的 struct 统计量接近 0，
        # 导致 compose kernel 将 temp_base 过度钳制（ceiling=max(p70,0.05)≈0.05）。
        # 设置下限让 temp_base 在无结构区域仍保持内盘基础亮度。
        tb = self._comp_field.to_numpy()[0]  # temp_base
        tb_max_per_r = np.max(tb, axis=1).astype(np.float32)
        struct_max_per_r = np.maximum(struct_max_per_r, tb_max_per_r)
        struct_p70_per_r = np.maximum(struct_p70_per_r, tb_max_per_r * 0.8)

        self._param_stats_field.from_numpy(
            np.array([density_p98, struct_scale], dtype=np.float32))
        row_stats = np.column_stack(
            [struct_max_per_r, struct_p70_per_r]).astype(np.float32)
        self._param_row_stats_field.from_numpy(row_stats)

    def compose_interactive_texture(self, solo_idx: int = -1) -> None:
        """Compose disk texture from comp_field and update mipmaps.

        Runs the compose kernel with t_offset=0 (both layers already handle
        rotation), then regenerates mipmaps. Called each frame in interactive mode.

        Args:
            solo_idx: -1 = show all components (normal mode).
                >= 0: solo display the component at this index, zeroing all others.
                Component indices: 0=temp_base, 1=spiral, 2=spiral_temp,
                3=turbulence, 4=turb_temp, 5=arcs, 6=arcs_temp,
                7=rt_spikes, 8=rt_temp, 9=hotspot, 10=hotspot_temp,
                11=az_hotspot, 12=disturb_mod
        """
        if solo_idx >= 0:
            # 配对关系：密度组件和对应温度组件一起保留
            _DENSITY_TEMP_PAIRS = {
                0: [],        # temp_base 独立
                1: [2],       # spiral → spiral_temp
                2: [1],       # spiral_temp → spiral
                3: [4],       # turbulence → turb_temp
                4: [3],       # turb_temp → turbulence
                5: [6],       # arcs → arcs_temp
                6: [5],       # arcs_temp → arcs
                7: [8],       # rt_spikes → rt_temp
                8: [7],       # rt_temp → rt_spikes
                9: [10],      # hotspot → hotspot_temp
                10: [9],      # hotspot_temp → hotspot
                11: [],       # az_hotspot 独立
                12: [],       # disturb_mod 独立
            }
            keep = {solo_idx} | set(_DENSITY_TEMP_PAIRS.get(solo_idx, []))
            for i in range(13):
                if i not in keep:
                    if i == 12:
                        # disturb_mod 设为 1.0（中性乘子）以免密度/温度被清零
                        self._fill_comp_slice(self._comp_field, 12, 1.0)
                    else:
                        self._zero_comp_slice(self._comp_field, i)
            self.recompute_interactive_stats()

        self._compose_disk_texture_kernel(
            self.disk_texture_field, self._comp_field,
            self._omega_rows_field, self._edge_field,
            self._param_stats_field, self._param_row_stats_field,
            0.0, self._param_enable_rt, self._param_color_temp)

        self._mipmap_copy_base_kernel(self.disk_mips_field,
                                      self.disk_texture_field)
        h, w = self._bg_n_r, self._bg_n_phi
        for lev in range(1, self.num_mip_levels):
            self._mipmap_downsample_kernel(self.disk_mips_field, lev, h, w)
            h //= 2
            w //= 2

    def eval_noise(self, coords: np.ndarray, mode: str = "simplex",
                   octaves: int = 4, persistence: float = 0.5,
                   lacunarity: float = 2.0) -> np.ndarray:
        """Evaluate noise at given coordinates (for testing/debugging).

        Args:
            coords: (N, 3) float32 array of xyz coordinates
            mode: "simplex" for raw simplex noise, "fbm" for fractal Brownian motion
            octaves: FBM octave count (ignored for simplex mode)
            persistence: FBM amplitude decay per octave
            lacunarity: FBM frequency multiplier per octave
        Returns:
            (N,) float32 array of noise values
        """
        n = coords.shape[0]
        coords_field = ti.field(ti.f32, shape=(n, 3))
        coords_field.from_numpy(coords.astype(np.float32))
        out_field = ti.field(ti.f32, shape=(n,))
        mode_i = 0 if mode == "simplex" else 1
        self._noise_eval_kernel(out_field, coords_field, mode_i,
                                octaves, persistence, lacunarity)
        return out_field.to_numpy()

    def update_disk_texture_gpu(self, t_offset: float) -> None:
        """在 GPU 上合成旋转纹理并更新 mipmap（替代 CPU 路径）。

        调用前需先调用 upload_parametric_state() 上传组件数据。
        完整替代 generate_disk_texture_rotating() + update_disk_texture() 的 CPU 路径，
        当 generation_scale=1 时与 CPU 路径像素级等价。

        Args:
            t_offset: 旋转时间偏移量，决定各行的开普勒旋转角度
        """
        assert hasattr(self, '_parametric_gpu_ready') and self._parametric_gpu_ready, \
            "Must call upload_parametric_state() before update_disk_texture_gpu()"

        self._compose_disk_texture_kernel(
            self.disk_texture_field, self._comp_field,
            self._omega_rows_field, self._edge_field,
            self._param_stats_field, self._param_row_stats_field,
            float(t_offset), self._param_enable_rt, self._param_color_temp
        )

        self._mipmap_copy_base_kernel(self.disk_mips_field, self.disk_texture_field)
        h, w = self.dtex_h, self.dtex_w
        for lev in range(1, self.num_mip_levels):
            self._mipmap_downsample_kernel(self.disk_mips_field, lev, h, w)
            h //= 2
            w //= 2

    def render_to_field(self, cam_pos: List[float], fov: float, frame: int = 0,
                        skip_differentials: bool = False, skip_bloom: bool = False) -> None:
        """渲染单帧到 GPU final_field（不做 GPU→CPU 传输，供交互模式使用）。

        渲染结果写入 self.final_field，可直接传给 ti.GUI.set_image()。
        """
        cam_pos_arr, cam_right, cam_up, cam_forward, pw, ph = build_camera(
            np.array(cam_pos, dtype=np.float64), fov, self.width, self.height
        )
        distance = float(np.linalg.norm(cam_pos_arr))
        r_escape = max(self.r_max, distance * 2)

        self.cam_pos_field[None] = list(cam_pos_arr.astype(np.float32))
        self.cam_right_field[None] = list(cam_right.astype(np.float32))
        self.cam_up_field[None] = list(cam_up.astype(np.float32))
        self.cam_forward_field[None] = list(cam_forward.astype(np.float32))
        self.pixel_width_field[None] = float(pw)
        self.pixel_height_field[None] = float(ph)
        self.r_escape_field[None] = float(r_escape)

        h_base = float(self.step_size)
        r_inner = float(self.r_disk_inner)
        r_outer = float(self.r_disk_outer)
        t_offset = float(frame) * self.disk_rotation_speed
        disk_tilt = float(self.disk_tilt)

        skip_diff_i = 1 if skip_differentials else 0
        self._ray_march_kernel(
            self.image_field, self.disk_layer_field, self.cam_pos_field, self.cam_right_field,
            self.cam_up_field, self.cam_forward_field, self.pixel_width_field,
            self.pixel_height_field, self.r_escape_field, h_base, r_inner, r_outer, t_offset,
            disk_tilt, skip_diff_i
        )

        use_bloom = 0
        if not skip_bloom:
            kernel_radius = int(self.width * 0.02)
            sigma_scale = (self.width / 640.0) ** 2
            self._bloom_kernel(self.disk_layer_field, self.bright_field, self.blur_field, 0, 0.4, kernel_radius, sigma_scale)
            use_bloom = 1

        self._compose_final_kernel(
            self.final_field, self.image_field, self.disk_layer_field,
            self.blur_field, use_bloom
        )

    def render(self, cam_pos: List[float], fov: float, frame: int = 0,
               skip_differentials: bool = False, skip_bloom: bool = False) -> np.ndarray:
        """
        渲染单帧图像。

        参数:
            cam_pos: 相机位置 [x, y, z]
            fov: 视野角度
            frame: 帧编号（用于吸积盘自转动画）
            skip_differentials: 跳过微分光线计算（~3x 加速，禁用抗锯齿 LOD）
            skip_bloom: 跳过 bloom 后处理

        返回:
            (height, width, 3) RGB 图像
        """
        cam_pos_arr, cam_right, cam_up, cam_forward, pw, ph = build_camera(
            np.array(cam_pos, dtype=np.float64), fov, self.width, self.height
        )
        distance = float(np.linalg.norm(cam_pos_arr))
        r_escape = max(self.r_max, distance * 2)

        self.cam_pos_field[None] = list(cam_pos_arr.astype(np.float32))
        self.cam_right_field[None] = list(cam_right.astype(np.float32))
        self.cam_up_field[None] = list(cam_up.astype(np.float32))
        self.cam_forward_field[None] = list(cam_forward.astype(np.float32))
        self.pixel_width_field[None] = float(pw)
        self.pixel_height_field[None] = float(ph)
        self.r_escape_field[None] = float(r_escape)

        h_base = float(self.step_size)
        r_inner = float(self.r_disk_inner)
        r_outer = float(self.r_disk_outer)
        t_offset = float(frame) * self.disk_rotation_speed
        disk_tilt = float(self.disk_tilt)

        skip_diff_i = 1 if skip_differentials else 0
        self._ray_march_kernel(
            self.image_field, self.disk_layer_field, self.cam_pos_field, self.cam_right_field,
            self.cam_up_field, self.cam_forward_field, self.pixel_width_field,
            self.pixel_height_field, self.r_escape_field, h_base, r_inner, r_outer, t_offset,
            disk_tilt, skip_diff_i
        )

        img = self.image_field.to_numpy()
        disk = self.disk_layer_field.to_numpy()

        if skip_bloom:
            final = np.clip(img + disk, 0, 1)
        else:
            kernel_radius = int(self.width * 0.02)
            sigma_scale = (self.width / 640.0) ** 2
            self._bloom_kernel(self.disk_layer_field, self.bright_field, self.blur_field, 0, 0.4, kernel_radius, sigma_scale)
            disk_bloom = self.blur_field.to_numpy()
            final = np.clip(img + disk + disk_bloom, 0, 1)

        # Lens flare（CPU 实现）
        if self.lens_flare:
            final = self._apply_lens_flare(final, disk)
        return final.transpose(1, 0, 2)

    def _apply_lens_flare(self, final, disk):
        """应用 lens flare 效果，final 和 disk 都是 (width, height, 3)"""
        w, h, _ = final.shape
        # 分辨率缩放因子（基准 SD 360p）
        scale = min(w, h) / 360.0

        # 找吸积盘亮度中心
        disk_brightness = np.max(disk, axis=2)  # shape: (w, h)
        total_brightness = np.sum(disk_brightness)
        if total_brightness < 0.01:
            return final

        x_coords, y_coords = np.mgrid[0:w, 0:h]  # shape: (w, h)
        light_x = np.sum(x_coords * disk_brightness) / total_brightness
        light_y = np.sum(y_coords * disk_brightness) / total_brightness
        screen_cx, screen_cy = w / 2, h / 2

        intensity = min(total_brightness / (w * h * 0.3), 1.0) * 1.5

        flare = np.zeros((w, h, 3), dtype=np.float32)

        # 多个 ghost 光斑
        for g in range(8):
            t = (g + 1) * 0.15
            ghost_x = light_x + (screen_cx - light_x) * t
            ghost_y = light_y + (screen_cy - light_y) * t
            ghost_size = (25 + g * 30) * scale

            dx = x_coords - ghost_x
            dy = y_coords - ghost_y
            dist = np.sqrt(dx**2 + dy**2)

            mask = dist < ghost_size
            alpha = np.zeros((w, h), dtype=np.float32)
            alpha[mask] = (1 - dist[mask] / ghost_size) ** 2 * (1 - g * 0.08) * intensity

            ghost_color = np.array([1.0, 0.9, 0.7])
            for c in range(3):
                flare[:, :, c] += alpha * ghost_color[c]

        # 多层环形光环（光圈衍射效果）
        for ring_idx in range(3):
            ring_t = 0.35 + ring_idx * 0.15
            ring_x = light_x + (screen_cx - light_x) * ring_t
            ring_y = light_y + (screen_cy - light_y) * ring_t
            ring_r = (60 + ring_idx * 40) * scale
            ring_w = (6 + ring_idx * 3) * scale

            dx = x_coords - ring_x
            dy = y_coords - ring_y
            dist = np.sqrt(dx**2 + dy**2)
            ring_dist = np.abs(dist - ring_r)
            ring_alpha = np.clip(1 - ring_dist / ring_w, 0, 1) ** 2 * 0.5 * intensity * (1 - ring_idx * 0.25)

            # 环的颜色略有差异，模拟色散
            ring_colors = [
                np.array([0.3, 0.4, 1.0]),   # 内环偏蓝
                np.array([0.5, 0.5, 0.9]),   # 中环偏紫
                np.array([0.7, 0.5, 0.8]),   # 外环偏暖
            ]
            for c in range(3):
                flare[:, :, c] += ring_alpha * ring_colors[ring_idx][c]

        # 六边形光环（光圈叶片效果）
        hex_t = 0.5
        hex_x = light_x + (screen_cx - light_x) * hex_t
        hex_y = light_y + (screen_cy - light_y) * hex_t
        hex_r = 100 * scale

        dx = x_coords - hex_x
        dy = y_coords - hex_y
        angle = np.arctan2(dy, dx)
        dist = np.sqrt(dx**2 + dy**2)

        # 六边形边缘检测
        hex_edge = np.abs(np.mod(angle, np.pi/3) - np.pi/6)
        hex_factor = np.clip(1 - hex_edge / 0.2, 0, 1)
        ring_dist = np.abs(dist - hex_r)
        ring_alpha = np.clip(1 - ring_dist / (15 * scale), 0, 1) ** 2 * hex_factor * 0.3 * intensity

        hex_color = np.array([0.6, 0.7, 1.0])
        for c in range(3):
            flare[:, :, c] += ring_alpha * hex_color[c]

        # 横向光斑条纹（星芒效果）
        streak_len = min(w, h) * 0.4
        streak_alpha = intensity * 0.3

        dx = x_coords - light_x
        dy = y_coords - light_y
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)

        # 4 条主星芒
        for main_angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            angle_diff = np.abs(np.mod(angle - main_angle + np.pi, 2*np.pi) - np.pi)
            streak_mask = angle_diff < 0.05
            falloff = np.exp(-dist / streak_len)

            streak_color = np.array([1.0, 0.95, 0.9])
            for c in range(3):
                flare[:, :, c] += np.where(streak_mask, falloff * streak_alpha * streak_color[c], 0)

        return np.clip(final + flare, 0, 1)
