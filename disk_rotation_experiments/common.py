#!/usr/bin/env python3
"""
共享代码模块 - 从 render.py 复制必要的函数

支持参数化旋转：预先生成所有随机参数，保证不同 t_offset 下是同一结构的旋转
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
import os
from typing import Dict, Any

# 添加父目录到路径，以便导入 render.py 的函数
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render import (
    _tileable_noise,
    _fbm_noise,
    _blackbody_rgb,
    compute_edge_alpha,
)


# ============================================================
# 全局参数缓存 - 避免每帧重新生成
# ============================================================
_disk_texture_cache: Dict[str, Any] = {}


def _generate_spiral_arms_fixed(params: Dict, n_r: int, n_phi: int,
                                 phi_grid: np.ndarray, r_norm_grid: np.ndarray) -> tuple:
    """生成螺旋臂 - 使用预先生成的参数"""
    spiral = np.zeros((n_r, n_phi), dtype=np.float32)
    temp_contribution = np.zeros((n_r, n_phi), dtype=np.float32)

    for arm in params['arms']:
        for j in range(arm['sub_arm_count']):
            sr = arm['sub_r_starts'][j]
            sr_len = arm['sub_arm_lengths'][j]
            sr_width = arm['sub_widths'][j]
            sr_int = arm['sub_intensities'][j]
            sr_end = sr + sr_len

            arm_angle = phi_grid - arm['base_angle'] + r_norm_grid * arm['rotations'] * 2 * np.pi

            arm_noise = params['arm_noise']
            width_mod = 0.2 + 1.5 * arm_noise
            width_mod = np.clip(width_mod, 0.15, 3.0)

            arm_kappa = 1.0 / (sr_width ** 2) * 1.5
            arm_val = np.exp(arm_kappa * (np.cos(arm_angle) - 1) * width_mod)

            mask = (r_norm_grid >= sr) & (r_norm_grid <= sr_end)
            arm_val = np.where(mask, arm_val, 0)

            intensity_mod = 0.1 + 0.9 * (arm_noise ** 0.15)

            fade_edge = 0.02
            fade_in = np.clip((r_norm_grid - sr) / fade_edge, 0, 1)
            fade_out = np.clip((sr_end - r_norm_grid) / fade_edge, 0, 1)
            arm_val *= fade_in * fade_out * sr_int * intensity_mod

            spiral += arm_val
            temp_contribution += arm_val * arm['arm_delta_T']

    spiral = np.clip(spiral / (np.max(spiral) + 1e-6), 0, 1)
    return spiral, temp_contribution


def _generate_turbulence_fixed(params: Dict, n_r: int, n_phi: int,
                                r_norm_grid: np.ndarray, t_offset: float,
                                omega_grid: np.ndarray) -> tuple:
    """生成湍流 - 使用预先生成的噪声，动态应用旋转"""
    shear_strength = params['shear_strength']
    kep_shear = shear_strength * (1.0 / (r_norm_grid + 0.3) ** 1.5 - 0.8)
    kep_shear = np.clip(kep_shear, 0, shear_strength * 8)
    kep_shift_pixels = (kep_shear / (2 * np.pi) * n_phi).astype(int)
    max_shift = n_phi // 4
    kep_shift_pixels = np.clip(kep_shift_pixels, -max_shift, max_shift)

    # 复制噪声并应用旋转
    turbulence_coarse = params['noise_coarse'].copy()
    turbulence_mid = params['noise_mid'].copy()
    turbulence_fine = params['noise_fine'].copy()
    turbulence_extra = params['noise_extra'].copy()
    turbulence_ultra = params['noise_ultra'].copy()
    pixel_noise = params['pixel_noise'].copy()

    if t_offset != 0.0:
        for ri in range(n_r):
            rot = int(t_offset * omega_grid[ri, 0] / (2 * np.pi) * n_phi)
            turbulence_coarse[ri, :] = np.roll(turbulence_coarse[ri, :], rot)
            turbulence_mid[ri, :] = np.roll(turbulence_mid[ri, :], rot)
            turbulence_fine[ri, :] = np.roll(turbulence_fine[ri, :], rot)
            turbulence_extra[ri, :] = np.roll(turbulence_extra[ri, :], rot)
            turbulence_ultra[ri, :] = np.roll(turbulence_ultra[ri, :], rot)
            pixel_noise[ri, :] = np.roll(pixel_noise[ri, :], rot)

    turbulence = (0.03 * turbulence_coarse + 0.05 * turbulence_mid
                  + 0.08 * turbulence_fine + 0.05 * turbulence_extra
                  + 0.03 * turbulence_ultra + 0.02 * np.clip(pixel_noise, 0, 1))

    temp_contribution = 0.04 * np.clip(turbulence, 0, 1)
    return turbulence, kep_shift_pixels, temp_contribution


def _generate_filaments_fixed(params: Dict, n_r: int, n_phi: int,
                               phi_grid: np.ndarray, r_norm_grid: np.ndarray) -> tuple:
    """生成细丝 - 使用预先生成的参数"""
    arcs = np.zeros((n_r, n_phi), dtype=np.float32)
    temp_contribution = np.zeros((n_r, n_phi), dtype=np.float32)

    for i in range(params['arc_count']):
        base_phi = params['arc_phi_starts'][i]
        base_r = params['arc_rs'][i]
        base_width = params['arc_r_widths'][i]
        total_length = params['arc_lengths'][i]
        intensity = params['arc_intensities'][i]
        delta_T = params['arc_delta_Ts'][i]

        sub_count = params['sub_filament_counts'][i]
        sub_lengths = params['sub_lengths_list'][i]
        sub_starts = params['sub_starts_list'][i]
        sub_widths = params['sub_widths_list'][i]
        sub_intensities = params['sub_intensities_list'][i]

        for j in range(sub_count):
            sub_phi = sub_starts[j]
            sub_len = sub_lengths[j]
            sub_w = sub_widths[j]
            sub_int = sub_intensities[j]

            phi_range = sub_len / (base_r + 0.01)
            phi_half_width = np.maximum(phi_range * 0.7, 0.2)
            kappa = 1.5 / (phi_half_width ** 2)

            sub_val = np.exp(kappa * (np.cos(phi_grid - sub_phi) - 1))

            r_diff = r_norm_grid - base_r
            r_prof = np.exp(-0.5 * (r_diff / sub_w) ** 2)

            arcs += sub_val * r_prof * sub_int
            temp_contribution += sub_val * r_prof * sub_int * delta_T * 0.7

    arcs = np.clip(arcs, 0, 1)
    temp_contribution = np.clip(temp_contribution, 0, arcs * 0.5)
    return arcs, temp_contribution


def _generate_hotspots_fixed(params: Dict, n_r: int, n_phi: int,
                              phi_grid: np.ndarray, r_norm_grid: np.ndarray) -> tuple:
    """生成热点 - 使用预先生成的参数"""
    hotspot = np.zeros((n_r, n_phi), dtype=np.float32)

    for i in range(params['hotspot_count']):
        h_phi = params['h_phis'][i]
        h_r = params['h_rs'][i]
        h_phi_width = params['h_phi_widths'][i]
        h_r_width = params['h_r_widths'][i]
        h_int = params['h_intensities'][i]

        kappa = 1.0 / (h_phi_width ** 2) * 1.5
        h_batch = np.exp(kappa * (np.cos(phi_grid - h_phi) - 1.0))
        r_diff = r_norm_grid - h_r
        h_batch *= np.exp(-0.5 * (r_diff / h_r_width) ** 2)
        h_batch *= h_int

    hotspot = np.clip(hotspot, 0, 1)
    temp_contribution = 0.12 * hotspot
    return hotspot, temp_contribution


def _generate_rt_spikes_fixed(params: Dict, n_r: int, n_phi: int,
                               phi_grid: np.ndarray, r_norm_grid: np.ndarray,
                               enable_rt: bool) -> tuple:
    """生成 RT 不稳定性 - 使用预先生成的参数"""
    rt_spikes = np.zeros((n_r, n_phi), dtype=np.float32)
    temp_contribution = np.zeros((n_r, n_phi), dtype=np.float32)

    if not enable_rt:
        return rt_spikes, temp_contribution

    for i in range(params['rt_count']):
        rt_phi = params['rt_phis'][i]
        rt_r_base = params['rt_r_bases'][i]
        rt_phi_width = params['rt_phi_widths'][i]
        rt_r_length = params['rt_r_lengths'][i]
        rt_int = params['rt_intensities'][i]
        rt_delta_T = params['rt_delta_Ts'][i]

        rt_phi_kappa = 1.0 / (rt_phi_width ** 2) * 1.5
        rt_val = np.exp(rt_phi_kappa * (np.cos(phi_grid - rt_phi) - 1))

        rt_r_diff = r_norm_grid - rt_r_base
        r_fade_out = np.clip(rt_r_length * 2 - rt_r_diff, 0, 1)
        r_fade_in = np.clip((r_norm_grid - rt_r_base) / (rt_r_length * 0.3), 0, 1)
        rt_r_profile = np.exp(-0.5 * (rt_r_diff / (rt_r_length * 0.4)) ** 2) * r_fade_out * r_fade_in

        rt_val *= rt_r_profile * rt_int
        rt_spikes += rt_val
        temp_contribution += rt_val * rt_delta_T

    rt_spikes = np.clip(rt_spikes, 0, 1)
    return rt_spikes, temp_contribution


def _generate_azimuthal_hotspot_fixed(params: Dict, n_r: int, n_phi: int,
                                       phi_grid: np.ndarray, r_norm_grid: np.ndarray,
                                       t_offset: float, omega_grid: np.ndarray) -> np.ndarray:
    """生成方位热点 - 使用预先生成的参数，动态应用旋转"""
    az_wave = 0.5 + 0.5 * np.sin((phi_grid + params['shear']) * params['az_freq'])
    az_noise = params['az_noise'].copy()

    if t_offset != 0.0:
        for ri in range(n_r):
            rot = int(t_offset * omega_grid[ri, 0] / (2 * np.pi) * n_phi)
            az_noise[ri, :] = np.roll(az_noise[ri, :], rot)

    az_hotspot = np.clip(0.6 * az_wave + 0.4 * az_noise, 0, 1) ** 1.2
    return az_hotspot


def _apply_disturbance_fixed(params: Dict, n_r: int, n_phi: int,
                              density: np.ndarray, temp_struct: np.ndarray,
                              kep_shift_pixels: np.ndarray, r_norm_grid: np.ndarray,
                              t_offset: float, omega_grid: np.ndarray) -> tuple:
    """应用扰动 - 使用预先生成的噪声，动态应用旋转"""
    disturb_coarse = params['noise_coarse'].copy()
    disturb_mid = params['noise_mid'].copy()
    disturb_fine = params['noise_fine'].copy()
    disturb_extra = params['noise_extra'].copy()
    disturb_pixel = params['pixel_noise'].copy()

    if t_offset != 0.0:
        for ri in range(n_r):
            rot = int(t_offset * omega_grid[ri, 0] / (2 * np.pi) * n_phi)
            disturb_coarse[ri, :] = np.roll(disturb_coarse[ri, :], rot)
            disturb_mid[ri, :] = np.roll(disturb_mid[ri, :], rot)
            disturb_fine[ri, :] = np.roll(disturb_fine[ri, :], rot)
            disturb_extra[ri, :] = np.roll(disturb_extra[ri, :], rot)
            disturb_pixel[ri, :] = np.roll(disturb_pixel[ri, :], rot)

    disturb_mod = (0.01 * disturb_coarse + 0.02 * disturb_mid + 0.04 * disturb_fine
                   + 0.03 * disturb_extra + 0.02 * disturb_pixel)
    disturb_mod = np.clip(disturb_mod * 0.5 + 0.5, 0.5, 1.0)
    radial_preserve = 0.6 + 0.4 * r_norm_grid
    disturb_mod = np.clip(disturb_mod * radial_preserve, 0.5, 1.0)

    density = density * disturb_mod
    temp_struct = temp_struct * disturb_mod
    return density, temp_struct


def _precompute_disk_params(n_phi: int, n_r: int, seed: int,
                            r_inner: float, r_outer: float) -> Dict[str, Any]:
    """预计算吸积盘的所有随机参数"""
    rng = np.random.default_rng(seed)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r_norm = np.linspace(0, 1, n_r)
    phi_grid, r_norm_grid = np.meshgrid(phi, r_norm)
    r_vals = r_inner + (r_outer - r_inner) * r_norm_grid
    disk_area = (r_outer ** 2 - r_inner ** 2) / 10.0

    params = {}

    # 螺旋臂参数
    n_arms = rng.integers(2, 5)
    n_from_center = rng.integers(2, 4)
    arms = []
    arm_noise = _tileable_noise((n_r, n_phi), rng, freq_u=3, freq_v=2)

    for arm_idx in range(n_arms):
        if arm_idx < n_from_center:
            r_start = 0.0
            base_angle = arm_idx * 2 * np.pi / n_from_center
        else:
            r_start = rng.uniform(0.05, 0.5)
            base_angle = rng.uniform(0, 2 * np.pi)

        rotations = rng.uniform(2.5, 5.0)
        base_width = rng.uniform(0.2, 0.4)
        arm_delta_T = rng.uniform(0.1, 0.3)
        r_length = min(rotations / 6.0 * (1.0 - r_start), 1.0 - r_start)

        sub_arm_count = rng.integers(4, 9)
        sub_arm_fill = rng.uniform(0.4, 0.6)
        sub_arm_lengths = rng.uniform(0.08, 0.20, sub_arm_count)
        sub_arm_lengths = sub_arm_lengths / sub_arm_lengths.sum() * r_length * sub_arm_fill

        sub_r_starts = np.zeros(sub_arm_count)
        for j in range(1, sub_arm_count):
            gap = rng.uniform(0.08, 0.15)
            sub_r_starts[j] = sub_r_starts[j-1] + sub_arm_lengths[j-1] + gap
        sub_r_starts += r_start

        sub_widths = np.clip(base_width * rng.uniform(0.3, 2.5, sub_arm_count), 0.06, 1.2)
        sub_intensities = rng.uniform(0.1, 0.7, sub_arm_count)

        arms.append({
            'r_start': r_start, 'base_angle': base_angle, 'rotations': rotations,
            'base_width': base_width, 'arm_delta_T': arm_delta_T,
            'sub_arm_count': sub_arm_count, 'sub_arm_lengths': sub_arm_lengths,
            'sub_r_starts': sub_r_starts, 'sub_widths': sub_widths,
            'sub_intensities': sub_intensities
        })

    params['spiral'] = {'arms': arms, 'arm_noise': arm_noise}

    # 湍流参数
    shear_strength = rng.uniform(3.0, 6.0)
    params['turbulence'] = {
        'shear_strength': shear_strength,
        'noise_coarse': _tileable_noise((n_r, n_phi), rng, freq_u=8, freq_v=4),
        'noise_mid': _tileable_noise((n_r, n_phi), rng, freq_u=24, freq_v=12),
        'noise_fine': _tileable_noise((n_r, n_phi), rng, freq_u=80, freq_v=40),
        'noise_extra': _tileable_noise((n_r, n_phi), rng, freq_u=200, freq_v=100),
        'noise_ultra': _tileable_noise((n_r, n_phi), rng, freq_u=400, freq_v=200),
        'pixel_noise': (_tileable_noise((n_r, n_phi), rng, freq_u=600, freq_v=300) - 0.5) * 2
    }

    # 细丝参数
    arc_count = int(rng.uniform(80, 150))
    sub_filament_counts = rng.integers(2, 5, arc_count)
    arc_phi_starts = rng.uniform(0, 2 * np.pi, arc_count)
    arc_rs = 0.05 + rng.uniform(0.05, 0.95, arc_count) ** 0.6 * 0.9
    arc_r_widths = rng.uniform(0.002, 0.008, arc_count)
    arc_lengths = rng.uniform(0.5, 1.2, arc_count)
    arc_intensities = rng.uniform(0.7, 1.0, arc_count)
    arc_delta_Ts = 0.3 + 0.6 * rng.power(0.3, arc_count)

    sub_lengths_list = []
    sub_starts_list = []
    sub_widths_list = []
    sub_intensities_list = []

    for i in range(arc_count):
        sub_count = sub_filament_counts[i]
        total_length = arc_lengths[i]
        sub_fill = rng.uniform(0.35, 0.55)
        sub_lengths = rng.uniform(0.08, 0.20, sub_count)
        sub_lengths = sub_lengths / sub_lengths.sum() * total_length * sub_fill

        sub_starts = np.zeros(sub_count)
        sub_starts[0] = arc_phi_starts[i]
        for j in range(1, sub_count):
            gap = rng.uniform(0.08, 0.20)
            sub_starts[j] = sub_starts[j-1] + sub_lengths[j-1] + gap

        base_width = arc_r_widths[i]
        sub_widths = np.clip(base_width * rng.uniform(0.3, 3.0, sub_count), 0.001, 0.025)
        sub_intensities = arc_intensities[i] * rng.uniform(0.15, 1.0, sub_count)

        sub_lengths_list.append(sub_lengths)
        sub_starts_list.append(sub_starts)
        sub_widths_list.append(sub_widths)
        sub_intensities_list.append(sub_intensities)

    params['filaments'] = {
        'arc_count': arc_count,
        'sub_filament_counts': sub_filament_counts,
        'arc_phi_starts': arc_phi_starts,
        'arc_rs': arc_rs,
        'arc_r_widths': arc_r_widths,
        'arc_lengths': arc_lengths,
        'arc_intensities': arc_intensities,
        'arc_delta_Ts': arc_delta_Ts,
        'sub_lengths_list': sub_lengths_list,
        'sub_starts_list': sub_starts_list,
        'sub_widths_list': sub_widths_list,
        'sub_intensities_list': sub_intensities_list
    }

    # 热点参数
    hotspot_count = int(rng.uniform(20, 40))
    h_phis = rng.uniform(0, 2 * np.pi, hotspot_count)
    h_rs = 0.1 + rng.uniform(0, 1, hotspot_count) ** 0.6 * 0.85
    h_phi_widths = rng.uniform(0.08, 0.20, hotspot_count)
    h_r_widths = 0.02 + rng.uniform(0, 0.03, hotspot_count)
    h_intensities = 0.3 + (1 - h_rs) * 0.6 + rng.uniform(0, 0.1, hotspot_count)

    params['hotspots'] = {
        'hotspot_count': hotspot_count,
        'h_phis': h_phis,
        'h_rs': h_rs,
        'h_phi_widths': h_phi_widths,
        'h_r_widths': h_r_widths,
        'h_intensities': h_intensities
    }

    # RT 不稳定性参数
    rt_count = int(rng.uniform(15, 30) * disk_area * 0.8)
    rt_phis = rng.uniform(0, 2 * np.pi, rt_count)
    rt_r_bases = np.power(rng.uniform(0.01, 0.15, rt_count), 1.5)
    rt_phi_widths = rng.uniform(0.08, 0.20, rt_count)
    rt_r_lengths = rng.uniform(0.08, 0.20, rt_count)
    rt_intensities = rng.uniform(0.8, 1.0, rt_count)
    rt_delta_Ts = rng.uniform(0.5, 1.2, rt_count)

    params['rt'] = {
        'rt_count': rt_count,
        'rt_phis': rt_phis,
        'rt_r_bases': rt_r_bases,
        'rt_phi_widths': rt_phi_widths,
        'rt_r_lengths': rt_r_lengths,
        'rt_intensities': rt_intensities,
        'rt_delta_Ts': rt_delta_Ts
    }

    # 方位热点参数
    az_freq = rng.integers(2, 5)
    params['azimuthal'] = {
        'az_freq': az_freq,
        'shear': r_norm_grid ** 1.2 * rng.uniform(2.0, 4.0),
        'az_noise': _fbm_noise((n_r, n_phi), rng, octaves=3, persistence=0.5, base_scale=3, wrap_u=True)
    }

    # 温度基底噪声参数
    params['temp_base'] = {
        'temp_coarse': _fbm_noise((n_r, n_phi), rng, octaves=4, persistence=0.6, base_scale=8, wrap_u=True),
        'temp_fine': _fbm_noise((n_r, n_phi), rng, octaves=5, persistence=0.45, base_scale=3, wrap_u=True)
    }

    # 扰动密度参数
    params['disturbance'] = {
        'noise_coarse': _tileable_noise((n_r, n_phi), rng, freq_u=8, freq_v=4),
        'noise_mid': _tileable_noise((n_r, n_phi), rng, freq_u=32, freq_v=16),
        'noise_fine': _tileable_noise((n_r, n_phi), rng, freq_u=100, freq_v=50),
        'noise_extra': _tileable_noise((n_r, n_phi), rng, freq_u=250, freq_v=125),
        'pixel_noise': (_tileable_noise((n_r, n_phi), rng, freq_u=400, freq_v=200) - 0.5) * 2
    }

    # 存储网格参数
    params['grids'] = {
        'phi_grid_base': phi_grid,
        'r_norm_grid': r_norm_grid,
        'r_vals': r_vals,
        'omega_grid': np.sqrt(0.5 / (r_vals + 0.01)),
        'disk_area': disk_area
    }

    return params


def _generate_texture_from_params(params: Dict, t_offset: float,
                                   enable_rt: bool) -> np.ndarray:
    """从预计算的参数生成纹理，应用指定的旋转偏移"""
    grids = params['grids']
    n_r, n_phi = grids['r_norm_grid'].shape

    # 计算旋转后的 phi_grid
    phi_grid = grids['phi_grid_base'] - t_offset * grids['omega_grid']

    # 温度基底
    radial_decay = np.clip(1.0 - grids['r_norm_grid'], 0, 1) ** 1.3
    temp_coarse = params['temp_base']['temp_coarse'].copy()
    temp_fine = params['temp_base']['temp_fine'].copy()

    if t_offset != 0.0:
        for ri in range(n_r):
            rot = int(t_offset * grids['omega_grid'][ri, 0] / (2 * np.pi) * n_phi)
            temp_coarse[ri, :] = np.roll(temp_coarse[ri, :], rot)
            temp_fine[ri, :] = np.roll(temp_fine[ri, :], rot)

    temp_noise = 0.6 * temp_coarse + 0.4 * temp_fine
    temp_base = np.clip(radial_decay * (0.85 + 0.15 * temp_noise), 0, 1) * 0.25

    temp_struct = np.zeros((n_r, n_phi), dtype=np.float32)

    # 生成各组件
    spiral, spiral_temp = _generate_spiral_arms_fixed(params['spiral'], n_r, n_phi, phi_grid, grids['r_norm_grid'])
    temp_struct += spiral_temp

    turbulence, kep_shift_pixels, turb_temp = _generate_turbulence_fixed(
        params['turbulence'], n_r, n_phi, grids['r_norm_grid'], t_offset, grids['omega_grid'])
    temp_struct += turb_temp

    arcs, arcs_temp = _generate_filaments_fixed(params['filaments'], n_r, n_phi, phi_grid, grids['r_norm_grid'])
    temp_struct += arcs_temp

    # RT 不稳定性
    rt_spikes, rt_temp = _generate_rt_spikes_fixed(params['rt'], n_r, n_phi, phi_grid, grids['r_norm_grid'], enable_rt)
    temp_struct += rt_temp

    hotspot, hotspot_temp = _generate_hotspots_fixed(params['hotspots'], n_r, n_phi, phi_grid, grids['r_norm_grid'])
    temp_struct += hotspot_temp

    az_hotspot = _generate_azimuthal_hotspot_fixed(params['azimuthal'], n_r, n_phi,
                                                    phi_grid, grids['r_norm_grid'],
                                                    t_offset, grids['omega_grid'])

    # 组合密度
    rt_weight = 0.15 if enable_rt else 0.0
    density = 0.15 + 0.22 * spiral + 0.30 * turbulence + 0.16 * hotspot + 0.12 * arcs + rt_weight * rt_spikes

    # 湍流扰动
    density, temp_struct = _apply_disturbance_fixed(params['disturbance'], n_r, n_phi,
                                                     density, temp_struct, kep_shift_pixels,
                                                     grids['r_norm_grid'], t_offset, grids['omega_grid'])

    # 边缘软化
    edge = compute_edge_alpha(n_r)
    density *= edge[:, None]

    # 归一化
    density = np.clip(density / (np.percentile(density, 98) + 1e-6), 0, 1)

    # 合成温度场
    if np.any(temp_struct > 0):
        struct_scale = np.percentile(temp_struct[temp_struct > 0], 95)
        temp_struct_scaled = temp_struct / (struct_scale + 1e-6)
    else:
        temp_struct_scaled = temp_struct
    temp_struct_scaled = np.clip(temp_struct_scaled * 0.8, 0, 1.2)

    struct_max_per_r = np.max(temp_struct_scaled, axis=1)
    struct_p70_per_r = np.quantile(temp_struct_scaled, 0.7, axis=1)
    struct_ceiling = np.maximum(struct_p70_per_r, 0.05)
    temp_base = np.minimum(temp_base, struct_ceiling[:, None])
    temp_base = np.minimum(temp_base, struct_max_per_r[:, None])

    temperature_field = np.clip(np.maximum(temp_base, temp_struct_scaled), 0, 1)

    # 颜色
    temp_aniso = np.clip(temperature_field * (0.9 + 0.25 * az_hotspot), 0, 1)
    T_min, T_max = 1200.0, 10000.0
    T_K = T_min + temp_aniso * (T_max - T_min)
    bb_color = _blackbody_rgb(T_K)
    bb_color[:, :, 2] = np.minimum(bb_color[:, :, 2], bb_color[:, :, 0])

    luminosity = np.clip(np.sqrt(temp_aniso), 0, 1)

    tex = np.zeros((n_r, n_phi, 4), dtype=np.float32)
    tex[:, :, 0] = np.clip(bb_color[:, :, 0] * luminosity, 0, 1)
    tex[:, :, 1] = np.clip(bb_color[:, :, 1] * luminosity, 0, 1)
    tex[:, :, 2] = np.clip(bb_color[:, :, 2] * luminosity, 0, 1)
    tex[:, :, 3] = np.clip(density, 0, 1)

    return tex


def clear_disk_texture_cache():
    """清除纹理缓存"""
    global _disk_texture_cache
    _disk_texture_cache = {}


def generate_disk_texture_topview(n_phi=512, n_r=128, seed=42, r_inner=2.0, r_outer=15.0,
                                   enable_rt=True, t_offset=0.0, use_cache=True):
    """
    生成吸积盘纹理（极坐标），支持参数化旋转

    关键特性：
    - 首次调用时预先生成所有随机参数并缓存
    - 后续调用使用相同的结构参数，只改变旋转偏移
    - 保证不同 t_offset 下是同一结构的旋转，而非重新生成

    Args:
        n_phi: 角度方向分辨率
        n_r: 径向方向分辨率
        seed: 随机种子
        r_inner: 内半径
        r_outer: 外半径
        enable_rt: 是否启用 RT 不稳定性
        t_offset: 旋转偏移（用于参数化旋转）
        use_cache: 是否使用缓存（默认 True，设为 False 可强制重新生成参数）

    Returns:
        (n_r, n_phi, 4) float32 纹理，RGBA
    """
    cache_key = f"{n_phi}_{n_r}_{seed}_{r_inner}_{r_outer}"

    # 检查缓存
    if use_cache and cache_key in _disk_texture_cache:
        params = _disk_texture_cache[cache_key]
    else:
        # 预计算参数
        params = _precompute_disk_params(n_phi, n_r, seed, r_inner, r_outer)
        if use_cache:
            _disk_texture_cache[cache_key] = params

    # 从参数生成纹理
    return _generate_texture_from_params(params, t_offset, enable_rt)


def polar_to_cartesian(texture, size=512, r_inner=2.0, r_outer=15.0):
    """
    将极坐标纹理转换为笛卡尔坐标（俯视图）

    Args:
        texture: (n_r, n_phi, 4) 极坐标纹理
        size: 输出图像尺寸
        r_inner: 内半径
        r_outer: 外半径

    Returns:
        (size, size, 3) RGB 图像
    """
    n_r, n_phi = texture.shape[:2]

    # 创建笛卡尔网格
    lin = np.linspace(-r_outer, r_outer, size)
    xv, yv = np.meshgrid(lin, lin[::-1])
    r = np.sqrt(xv**2 + yv**2)
    phi = np.mod(np.arctan2(yv, xv) + 2*np.pi, 2*np.pi)

    # 创建输出图像
    canvas = np.zeros((size, size, 3), dtype=np.float32)

    # 只在吸积盘范围内采样
    mask = (r >= r_inner) & (r <= r_outer)

    # 计算纹理坐标
    phi_idx = phi / (2 * np.pi) * n_phi
    r_idx = (r - r_inner) / (r_outer - r_inner) * (n_r - 1)

    # 双线性插值
    phi0 = np.floor(phi_idx).astype(int) % n_phi
    r0 = np.clip(np.floor(r_idx).astype(int), 0, n_r - 1)
    phi1 = (phi0 + 1) % n_phi
    r1 = np.clip(r0 + 1, 0, n_r - 1)

    f_phi = (phi_idx - np.floor(phi_idx)).astype(np.float32)
    f_r = (r_idx - np.floor(r_idx)).astype(np.float32)

    for c in range(3):
        c00 = texture[r0, phi0, c]
        c10 = texture[r0, phi1, c]
        c01 = texture[r1, phi0, c]
        c11 = texture[r1, phi1, c]
        val = c00*(1-f_phi)*(1-f_r) + c10*f_phi*(1-f_r) + c01*(1-f_phi)*f_r + c11*f_phi*f_r
        canvas[..., c] = val

    # 应用 alpha 混合
    a00 = texture[r0, phi0, 3]
    a10 = texture[r0, phi1, 3]
    a01 = texture[r1, phi0, 3]
    a11 = texture[r1, phi1, 3]
    alpha = a00*(1-f_phi)*(1-f_r) + a10*f_phi*(1-f_r) + a01*(1-f_phi)*f_r + a11*f_phi*f_r
    canvas *= alpha[..., None]

    # 只保留吸积盘区域
    canvas = np.where(mask[..., None], canvas, 0.0)

    return canvas


def add_text_overlay(image, text_lines, position='bottom-right'):
    """
    在图像上添加文字覆盖层

    Args:
        image: (H, W, 3) numpy array, float32, [0, 1]
        text_lines: 文字行列表
        position: 位置 ('bottom-right', 'top-left', etc.)

    Returns:
        (H, W, 3) numpy array with text overlay
    """
    # 转换为 PIL Image
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, 'RGB')

    # 创建绘图对象
    draw = ImageDraw.Draw(pil_img)

    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    # 计算文字位置
    h, w = image.shape[:2]
    line_height = 18
    padding = 10

    if position == 'bottom-right':
        x = w - 250
        y = h - len(text_lines) * line_height - padding
    elif position == 'top-left':
        x = padding
        y = padding
    else:
        x = padding
        y = padding

    # 绘制半透明背景
    bbox_x0 = x - 5
    bbox_y0 = y - 5
    bbox_x1 = x + 240
    bbox_y1 = y + len(text_lines) * line_height + 5
    draw.rectangle([bbox_x0, bbox_y0, bbox_x1, bbox_y1], fill=(0, 0, 0, 180))

    # 绘制文字
    for i, line in enumerate(text_lines):
        draw.text((x, y + i * line_height), line, fill=(255, 255, 255), font=font)

    # 转换回 numpy
    return np.array(pil_img).astype(np.float32) / 255.0


def sample_disk_with_rotation(texture, t_offset, r_inner, r_outer, size=512):
    """
    采样吸积盘纹理并应用旋转（Baseline 方案）

    Args:
        texture: (n_r, n_phi, 4) 极坐标纹理
        t_offset: 旋转偏移
        r_inner: 内半径
        r_outer: 外半径
        size: 输出尺寸

    Returns:
        (size, size, 3) RGB 图像
    """
    n_r, n_phi = texture.shape[:2]

    # 创建笛卡尔网格
    lin = np.linspace(-r_outer, r_outer, size)
    xv, yv = np.meshgrid(lin, lin[::-1])
    r = np.sqrt(xv**2 + yv**2)
    phi = np.arctan2(yv, xv)

    # 应用开普勒旋转
    omega = np.sqrt(0.5 / (r + 0.01))
    phi_rotated = phi + t_offset * omega
    phi_rotated = np.mod(phi_rotated + 2*np.pi, 2*np.pi)

    # 创建输出图像
    canvas = np.zeros((size, size, 3), dtype=np.float32)

    # 只在吸积盘范围内采样
    mask = (r >= r_inner) & (r <= r_outer)

    # 计算纹理坐标
    phi_idx = phi_rotated / (2 * np.pi) * n_phi
    r_idx = (r - r_inner) / (r_outer - r_inner) * (n_r - 1)

    # 双线性插值
    phi0 = np.floor(phi_idx).astype(int) % n_phi
    r0 = np.clip(np.floor(r_idx).astype(int), 0, n_r - 1)
    phi1 = (phi0 + 1) % n_phi
    r1 = np.clip(r0 + 1, 0, n_r - 1)

    f_phi = (phi_idx - np.floor(phi_idx)).astype(np.float32)
    f_r = (r_idx - np.floor(r_idx)).astype(np.float32)

    for c in range(3):
        c00 = texture[r0, phi0, c]
        c10 = texture[r0, phi1, c]
        c01 = texture[r1, phi0, c]
        c11 = texture[r1, phi1, c]
        val = c00*(1-f_phi)*(1-f_r) + c10*f_phi*(1-f_r) + c01*(1-f_phi)*f_r + c11*f_phi*f_r
        canvas[..., c] = val

    # 应用 alpha
    a00 = texture[r0, phi0, 3]
    a10 = texture[r0, phi1, 3]
    a01 = texture[r1, phi0, 3]
    a11 = texture[r1, phi1, 3]
    alpha = a00*(1-f_phi)*(1-f_r) + a10*f_phi*(1-f_r) + a01*(1-f_phi)*f_r + a11*f_phi*f_r
    canvas *= alpha[..., None]

    canvas = np.where(mask[..., None], canvas, 0.0)

    return canvas
