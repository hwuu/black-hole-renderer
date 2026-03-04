#!/usr/bin/env python3
"""
共享代码模块 - 从 render.py 复制必要的函数
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
import os

# 添加父目录到路径，以便导入 render.py 的函数
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 render.py 导入必要的函数
from render import (
    _tileable_noise,
    _fbm_noise,
    _blackbody_rgb,
    compute_edge_alpha,
    _generate_spiral_arms,
    _generate_turbulence,
    _generate_filaments,
    _generate_rt_spikes,
    _generate_azimuthal_hotspot,
    _generate_hotspots,
    _apply_disturbance,
)


def generate_disk_texture_topview(n_phi=512, n_r=128, seed=42, r_inner=2.0, r_outer=15.0,
                                   enable_rt=True, t_offset=0.0):
    """
    生成吸积盘纹理（极坐标），可选旋转偏移

    Args:
        n_phi: 角度方向分辨率
        n_r: 径向方向分辨率
        seed: 随机种子
        r_inner: 内半径
        r_outer: 外半径
        enable_rt: 是否启用 RT 不稳定性
        t_offset: 旋转偏移（用于参数化旋转）

    Returns:
        (n_r, n_phi, 4) float32 纹理，RGBA
    """
    rng = np.random.default_rng(seed)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r_norm = np.linspace(0, 1, n_r)
    phi_grid, r_norm_grid = np.meshgrid(phi, r_norm)

    # 计算 omega_grid（用于湍流旋转）
    r_vals = r_inner + (r_outer - r_inner) * r_norm_grid
    omega_grid = np.sqrt(0.5 / (r_vals + 0.01))
    
    # 如果有旋转偏移，应用开普勒旋转到 phi_grid
    if t_offset != 0.0:
        phi_grid = phi_grid + t_offset * omega_grid

    r_vals = r_inner + (r_outer - r_inner) * r_norm_grid
    disk_area = (r_outer ** 2 - r_inner ** 2) / 10.0

    # 温度基底
    radial_decay = np.clip(1.0 - r_norm_grid, 0, 1) ** 1.3
    temp_coarse = _fbm_noise((n_r, n_phi), rng, octaves=4, persistence=0.6, base_scale=8, wrap_u=True)
    temp_fine = _fbm_noise((n_r, n_phi), rng, octaves=5, persistence=0.45, base_scale=3, wrap_u=True)
    
    # 如果 t_offset != 0，对温度基底应用开普勒旋转
    if t_offset != 0.0:
        for ri in range(n_r):
            rotation_pixels = int(t_offset * omega_grid[ri, 0] / (2 * np.pi) * n_phi)
            temp_coarse[ri, :] = np.roll(temp_coarse[ri, :], rotation_pixels)
            temp_fine[ri, :] = np.roll(temp_fine[ri, :], rotation_pixels)
    
    temp_noise = 0.6 * temp_coarse + 0.4 * temp_fine
    temp_base = np.clip(radial_decay * (0.85 + 0.15 * temp_noise), 0, 1)
    temp_base *= 0.25

    temp_struct = np.zeros((n_r, n_phi), dtype=np.float32)

# 生成各种结构
    spiral, spiral_temp = _generate_spiral_arms(rng, n_r, n_phi, phi_grid, r_norm_grid, t_offset, omega_grid)
    temp_struct += spiral_temp

    turbulence, kep_shift_pixels, turb_temp = _generate_turbulence(rng, n_r, n_phi, r_norm_grid,
                                                                    t_offset, omega_grid)
    temp_struct += turb_temp

    arcs, arcs_temp = _generate_filaments(rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, t_offset, omega_grid)
    temp_struct += arcs_temp

    rt_spikes, rt_temp = _generate_rt_spikes(rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, enable_rt, t_offset, omega_grid)
    temp_struct += rt_temp

    hotspot, hotspot_temp = _generate_hotspots(rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, t_offset, omega_grid)
    temp_struct += hotspot_temp

    az_hotspot = _generate_azimuthal_hotspot(rng, n_r, n_phi, phi_grid, r_norm_grid, t_offset, omega_grid)

    # 组合密度
    rt_weight = 0.15 if enable_rt else 0.0
    density = 0.15 + 0.22 * spiral + 0.30 * turbulence + 0.16 * hotspot + 0.12 * arcs + rt_weight * rt_spikes

    # 湍流扰动
    density, temp_struct = _apply_disturbance(rng, n_r, n_phi, density, temp_struct,
                                               kep_shift_pixels, r_norm_grid, t_offset, omega_grid)

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
