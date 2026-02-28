#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# type: ignore[attr-defined, misc, valid-type]
"""
Schwarzschild 黑洞光线追踪渲染器

基于广义相对论的史瓦西度规，使用笛卡尔等效形式的光线方程：
    d²x/dλ² = -1.5 * L² * x / r⁵

支持两种渲染框架：
- numpy: 纯 NumPy 实现，compact 算法
- taichi: Taichi 框架（CPU/GPU），while_loop 算法

Reference: https://github.com/JaeHyunLee94/BlackHoleRendering
"""

import numpy as np
from PIL import Image
import os
import time
import argparse
import hashlib

# ============================================================================
# 公共常量
# ============================================================================

# 核心常量（按需调参，可参考注释范围）
RS = 1.0
EPS = 1e-6

# —— g 因子着色相关 —— 影响吸积盘自身的亮度/颜色，背景天空不受这些参数影响。
# g 因子亮度压缩的软上限，推荐 0.5~6（默认 3.0），值越小盘面整体越暗
G_FACTOR_CAP = 6
# g 的幂次，决定亮度随 g 变化的敏感度，建议 1.5~3（默认 2.2）
G_LUMINOSITY_POWER = 6
# 亮度缩放系数，常用 0.2~0.6（默认 0.38），越大盘面全局越亮
G_BRIGHTNESS_GAIN = 0.6

# —— 吸积盘透明度与色温 —— 决定盘层遮挡背景与整体暖色偏移。
# DISK_ALPHA_GAIN > 1 会让盘体更实心，推荐 1~20（默认 1.2）
DISK_ALPHA_GAIN = 1.5
# DISK_BASE_TINT 拉伸 RGB，值越大对应的通道越亮；典型取值 0.6~1.4（默认暖色 1.1/0.92/0.75）
DISK_BASE_TINT = (1.1, 0.92, 0.75)
# DISK_RADIAL_BRIGHTNESS_POWER >0 会让亮度按 (1 - radial_t)^p 递减（常用 1~3，此处默认 12 便于放大对比）
DISK_RADIAL_BRIGHTNESS_POWER = 3
# 半径亮度增益的下限/上限，避免指数爆炸
DISK_RADIAL_BRIGHTNESS_MIN = 0.2
DISK_RADIAL_BRIGHTNESS_MAX = 16.0

# ============================================================================
# 公共模块：相机
# ============================================================================

def build_camera(cam_pos, fov_deg, width, height):
    """
    构建相机参数

    参数:
        cam_pos: 相机位置 [x, y, z]
        fov_deg: 视野角度（度）
        width: 图像宽度（像素）
        height: 图像高度（像素）

    返回:
        (cam_pos, cam_right, cam_up, cam_forward, pixel_width, pixel_height)
    """
    cam_pos = np.array(cam_pos, dtype=np.float64)
    cam_forward = -cam_pos / np.linalg.norm(cam_pos)

    world_up = np.array([0.0, 0.0, 1.0])
    cam_right = np.cross(cam_forward, world_up)
    rn = np.linalg.norm(cam_right)
    if rn < 1e-6:
        cam_right = np.array([1.0, 0.0, 0.0])
    else:
        cam_right /= rn
    cam_up = np.cross(cam_right, cam_forward)
    cam_up /= np.linalg.norm(cam_up)

    fov_rad = np.radians(fov_deg)
    aspect = width / height
    image_plane_height = 2.0 * np.tan(fov_rad / 2)
    image_plane_width = image_plane_height * aspect

    pixel_width = image_plane_width / width
    pixel_height = image_plane_height / height

    return cam_pos, cam_right, cam_up, cam_forward, pixel_width, pixel_height


def make_all_rays(width, height, cam_pos, cam_right, cam_up, cam_forward, pixel_width, pixel_height):
    """
    为所有像素生成光线（位置、方向、角动量平方）

    参数:
        width: 图像宽度
        height: 图像高度
        cam_pos: 相机位置
        cam_right: 相机右向量
        cam_up: 相机上向量
        cam_forward: 相机前向量（指向目标）
        pixel_width: 像素宽度（世界坐标）
        pixel_height: 像素高度（世界坐标）

    返回:
        (positions, velocities, L2, pixel_coords)
        - positions: (N, 3) 光线起始位置
        - velocities: (N, 3) 光线方向单位向量
        - L2: (N,) 角动量平方（守恒量）
        - pixel_coords: (N, 2) 像素坐标 (x, y)
    """
    py, px = np.mgrid[0:height, 0:width]
    px = px.ravel().astype(np.float64)
    py = py.ravel().astype(np.float64)

    # 计算图像平面左上角位置
    image_plane_center = cam_pos + cam_forward * 1.0
    top_left = (image_plane_center
                - cam_right * (pixel_width * width / 2)
                + cam_up * (pixel_height * height / 2))

    # 计算每个像素在图像平面上的位置
    pixel_pos_x = top_left[0] + (px + 0.5) * pixel_width * cam_right[0] - (py + 0.5) * pixel_height * cam_up[0]
    pixel_pos_y = top_left[1] + (px + 0.5) * pixel_width * cam_right[1] - (py + 0.5) * pixel_height * cam_up[1]
    pixel_pos_z = top_left[2] + (px + 0.5) * pixel_width * cam_right[2] - (py + 0.5) * pixel_height * cam_up[2]

    pixel_pos = np.column_stack([pixel_pos_x, pixel_pos_y, pixel_pos_z])

    # 光线方向：从相机指向像素
    dirs = pixel_pos - cam_pos
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs /= norms

    N = len(px)
    positions = np.tile(cam_pos, (N, 1))  # 所有光线从相机位置出发
    velocities = dirs  # 单位方向向量

    # 计算角动量平方 L² = |v × x|²（守恒量）
    cross = np.cross(velocities, positions)
    L2 = np.sum(cross * cross, axis=1)

    pixel_coords = np.column_stack([px.astype(int), py.astype(int)])
    return positions, velocities, L2, pixel_coords


# ============================================================================
# 公共模块：天空盒
# ============================================================================

def generate_skybox(tex_w=2048, tex_h=1024, seed=42, n_stars=6000):
    """
    程序化生成天空盒纹理（等距柱状投影）

    参数:
        tex_w: 纹理宽度
        tex_h: 纹理高度
        seed: 随机种子
        n_stars: 恒星数量

    返回:
        texture: (tex_h, tex_w, 3) float32 RGB 纹理，值域 [0, 1]
    """
    rng = np.random.default_rng(seed)
    texture = np.full((tex_h, tex_w, 3), 0.005, dtype=np.float32)  # 深蓝底色

    # 星云：低频噪声上采样
    neb_h, neb_w = tex_h // 16, tex_w // 16
    nebula_small = rng.random((neb_h, neb_w, 3)).astype(np.float32) * 0.08
    nebula = np.array(Image.fromarray(
        (nebula_small * 255).astype(np.uint8)
    ).resize((tex_w, tex_h), Image.Resampling.BILINEAR)) / 255.0 * 0.06
    texture += nebula

    # 恒星：高斯 blob
    z = rng.uniform(-1, 1, n_stars)
    phi_s = rng.uniform(0, 2 * np.pi, n_stars)
    theta_s = np.arccos(np.clip(z, -1, 1))

    cx = (phi_s / (2 * np.pi) * tex_w).astype(np.float32)
    cy = (theta_s / np.pi * tex_h).astype(np.float32)

    brightness = rng.uniform(0.3, 1.0, n_stars).astype(np.float32)
    sigma = rng.uniform(0.6, 1.5, n_stars).astype(np.float32)

    # 颜色分布：55% 白色、25% 蓝色、20% 暖色
    temp = rng.uniform(0, 1, n_stars)
    colors = np.ones((n_stars, 3), dtype=np.float32)
    colors[temp < 0.25] = [0.6, 0.7, 1.0]  # 蓝色
    colors[(temp >= 0.25) & (temp < 0.45)] = [1.0, 0.9, 0.7]  # 暖色

    # 高斯 blob 渲染
    R = 4
    offsets = np.arange(-R, R + 1, dtype=np.float32)
    dy_grid, dx_grid = np.meshgrid(offsets, offsets, indexing='ij')
    dy_flat = dy_grid.ravel()
    dx_flat = dx_grid.ravel()
    n_patch = len(dy_flat)

    px = (cx[:, None] + dx_flat[None, :]).astype(int) % tex_w
    py_raw = (cy[:, None] + dy_flat[None, :]).astype(int)

    d2 = dx_flat[None, :] ** 2 + dy_flat[None, :] ** 2
    vals = brightness[:, None] * np.exp(-d2 / (2 * sigma[:, None] ** 2))

    valid = (py_raw >= 0) & (py_raw < tex_h)
    flat_y = py_raw[valid]
    flat_x = px[valid]
    flat_vals = vals[valid]
    flat_colors = np.repeat(colors, n_patch, axis=0)[valid.ravel()]
    contributions = flat_colors * flat_vals[:, None]

    np.add.at(texture, (flat_y, flat_x), contributions)
    return np.clip(texture, 0, 1)


def load_or_generate_skybox(skybox_path, tex_w=2048, tex_h=1024, n_stars=6000):
    """
    加载或生成天空盒纹理

    参数:
        skybox_path: 纹理文件路径，如果为 None 或文件不存在则程序生成
        tex_w, tex_h: 程序生成时的纹理尺寸
        n_stars: 程序生成时的恒星数量

    返回:
        (texture, tex_h, tex_w)
    """
    if skybox_path and os.path.isfile(skybox_path):
        print(f"Loading skybox: {skybox_path}")
        img = Image.open(skybox_path).convert("RGB")
        texture = np.array(img, dtype=np.float32) / 255.0
        tex_h, tex_w = texture.shape[:2]
    else:
        if skybox_path:
            print(f"Texture not found: {skybox_path}, generating procedural skybox...")
        else:
            print("Generating procedural skybox...")
        texture = generate_skybox(tex_w=tex_w, tex_h=tex_h, n_stars=n_stars)

    return texture, tex_h, tex_w


def sample_skybox_bilinear(texture, directions):
    """
    双线性插值采样天空盒

    参数:
        texture: (tex_h, tex_w, 3) 天空盒纹理
        directions: (N, 3) 光线方向向量

    返回:
        (N, 3) RGB 颜色
    """
    tex_h, tex_w = texture.shape[:2]
    dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

    theta = np.arccos(np.clip(dz, -1, 1))
    phi = np.arctan2(dy, dx)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    u = phi / (2 * np.pi) * tex_w
    v = theta / np.pi * tex_h

    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    fu = (u - u0).astype(np.float32)
    fv = (v - v0).astype(np.float32)

    u0 = u0 % tex_w
    u1 = (u0 + 1) % tex_w
    v0 = np.clip(v0, 0, tex_h - 1)
    v1 = np.clip(v0 + 1, 0, tex_h - 1)

    c00 = texture[v0, u0]
    c10 = texture[v0, u1]
    c01 = texture[v1, u0]
    c11 = texture[v1, u1]

    fu = fu[:, None]
    fv = fv[:, None]

    return (c00 * (1 - fu) * (1 - fv) +
            c10 * fu * (1 - fv) +
            c01 * (1 - fu) * fv +
            c11 * fu * fv)


# ============================================================================
# 公共模块：图像保存
# ============================================================================

def save_image(image, path):
    """保存图像为 PNG 文件"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8, "RGB").save(path)
    print(f"Saved: {path}")


# ============================================================================
# 公共模块：吸积盘
# ============================================================================

# 默认吸积盘参数（对齐 JaeHyunLee94）
R_DISK_INNER_DEFAULT = 2.0 * RS
R_DISK_OUTER_DEFAULT = 3.5 * RS


def compute_edge_alpha(height, inner_soft=0.1, outer_soft=0.3):
    """计算边缘软化的 alpha 通道"""
    v = np.linspace(0, 1, height).astype(np.float32)
    alpha = np.ones_like(v)
    inner_mask = v < inner_soft
    outer_mask = v > (1 - outer_soft)
    alpha[inner_mask] = (v[inner_mask] / inner_soft) ** 3.0
    alpha[outer_mask] = ((1 - v[outer_mask]) / outer_soft) ** 2
    return alpha


def load_disk_texture(path):
    """加载吸积盘纹理，返回 (h, w, 4) float32 数组（RGBA，边缘软化 alpha）"""
    if path and os.path.isfile(path):
        print(f"Loading disk texture: {path}")
        img = Image.open(path).convert("RGB")
        rgb = np.array(img, dtype=np.float32) / 255.0
        h, w = rgb.shape[:2]
        alpha = compute_edge_alpha(h)[:, None].astype(np.float32)
        alpha = np.broadcast_to(alpha, (h, w)).copy()
        alpha = alpha[:, :, None]
        return np.concatenate([rgb, alpha], axis=2)
    return None


def sample_disk_texture(texture, x, y, r_inner, r_outer):
    """
    双线性插值采样吸积盘纹理（NumPy 批量版）
    返回 (N, 4) RGBA

    UV 映射（对齐 JaeHyunLee94）：
      u = phi / (2*pi)           — 方位角 → 纹理宽度
      v = (r - r_inner) / (r_outer - r_inner) — 径向 → 纹理高度
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    tex_h, tex_w = texture.shape[:2]
    u = phi / (2 * np.pi) * tex_w
    v = (r - r_inner) / (r_outer - r_inner) * tex_h

    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    fu = (u - u0)[:, None]
    fv = (v - v0)[:, None]

    u0_w = u0 % tex_w
    u1_w = (u0 + 1) % tex_w
    v0_h = np.clip(v0, 0, tex_h - 1)
    v1_h = np.clip(v0 + 1, 0, tex_h - 1)

    c00 = texture[v0_h, u0_w]
    c10 = texture[v0_h, u1_w]
    c01 = texture[v1_h, u0_w]
    c11 = texture[v1_h, u1_w]

    return (c00 * (1 - fu) * (1 - fv) +
            c10 * fu * (1 - fv) +
            c01 * (1 - fu) * fv +
            c11 * fu * fv)


def _tileable_noise(shape, rng, freq_u=6, freq_v=6):
    """用多条弧线生成云雾效果，保证 phi 方向无缝。"""
    h, w = shape

    cloud = np.zeros((h, w), dtype=np.float32)
    n_arcs = rng.integers(30, 60)

    for _ in range(n_arcs):
        arc_phi = rng.uniform(0, 2 * np.pi)
        arc_r = np.sqrt(rng.uniform(0.0, 1.0))
        arc_phi_width = rng.uniform(0.15, 0.5)
        arc_r_width = rng.uniform(0.03, 0.08)
        arc_intensity = rng.uniform(0.03, 0.12)

        kappa = 1.0 / (arc_phi_width ** 2) * 0.6

        phi = np.linspace(0, 2 * np.pi, w, endpoint=False)
        r_norm = np.linspace(0, 1, h)
        phi_grid, r_grid = np.meshgrid(phi, r_norm)

        r_diff = r_grid - arc_r
        arc_val = np.exp(kappa * (np.cos(phi_grid - arc_phi) - 1))
        arc_val *= np.exp(-0.5 * (r_diff / arc_r_width) ** 2)
        arc_val *= arc_intensity

        cloud += arc_val

    cloud = np.clip(cloud, 0, 1)
    return cloud


def _fbm_noise(shape, rng, octaves=4, persistence=0.5, base_scale=1, wrap_u=False):
    """分形布朗运动噪声（多层叠加）。wrap_u=True 时用 tileable 噪声替代。"""
    if wrap_u:
        result = np.zeros(shape, dtype=np.float32)
        for i in range(octaves):
            freq = int(base_scale * (2 ** i))
            tile_noise = _tileable_noise(shape, rng, freq_u=max(2, freq), freq_v=max(1, freq // 2))
            result += tile_noise * (persistence ** i)
        result /= np.max(result) + 1e-6
        return result
    result = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    for i in range(octaves):
        scale = base_scale * (2 ** i)
        sh = max(shape[0] // scale, 2)
        sw = max(shape[1] // scale, 2)
        small = rng.random((sh, sw)).astype(np.float32)
        pil = Image.fromarray((small * 255).astype(np.uint8))
        up = np.array(pil.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)) / 255.0
        result += up * amplitude
        total_amp += amplitude
        amplitude *= persistence
    return result / total_amp



def _blend_azimuthal_seam(tex, seam_width=64):
    """
    将纹理在 u=0/u=2π 方向做平滑过渡，避免拼接时出现明显缝隙。
    """
    if seam_width <= 0:
        return tex
    if seam_width * 2 >= tex.shape[1]:
        return tex
    tex_blended = tex.copy()
    left = tex[:, :seam_width, :].copy()
    right = tex[:, -seam_width:, :].copy()
    for i in range(seam_width):
        t = (i + 1) / (seam_width + 1)
        tex_blended[:, i, :] = (1 - t) * left[:, i, :] + t * right[:, i, :]
        tex_blended[:, -seam_width + i, :] = (1 - t) * right[:, i, :] + t * left[:, i, :]
    return tex_blended


def generate_disk_texture(n_phi=1024, n_r=512, seed=42, r_inner=2.0, r_outer=3.5):
    """
    直接在极坐标下生成吸积盘纹理，避免笛卡尔到极坐标的映射接缝问题。
    - n_phi: 角度方向分辨率（对应 0-2π）
    - n_r: 径向方向分辨率（对应 r_inner 到 r_outer）
    返回 (n_r, n_phi, 4) float32，第 4 通道为 alpha（面密度）
    """
    rng = np.random.default_rng(seed)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r_norm = np.linspace(0, 1, n_r)
    phi_grid, r_norm_grid = np.meshgrid(phi, r_norm)

    r_vals = r_inner + (r_outer - r_inner) * r_norm_grid

    # ----- 温度剖面（Novikov-Thorne power law）-----
    T = (1 - np.sqrt(r_inner / np.maximum(r_vals, r_inner + 1e-3))) ** 0.25
    T /= np.max(T)

    temp_coarse = _fbm_noise((n_r, n_phi), rng, octaves=4, persistence=0.6, base_scale=8, wrap_u=True)
    temp_fine = _fbm_noise((n_r, n_phi), rng, octaves=5, persistence=0.45, base_scale=3, wrap_u=True)
    temp_noise = 0.6 * temp_coarse + 0.4 * temp_fine
    temperature_field = np.clip(T * (0.8 + 0.3 * temp_noise), 0, 1)

    # ----- 密度场 -----
    # 1) 大尺度螺旋臂（8-15条，2-3条从中心开始，其余从随机位置开始，每条2-5圈）
    n_arms = rng.integers(8, 16)
    n_from_center = rng.integers(2, 4)
    spiral = np.zeros((n_r, n_phi), dtype=np.float32)

    # 生成一条噪声用于调制悬臂宽度和强度
    arm_noise = _tileable_noise((n_r, n_phi), rng, freq_u=8, freq_v=4)

    used_angles = []
    for arm_idx in range(n_arms):
        if arm_idx < n_from_center:
            r_start = 0.0
            base_angle = arm_idx * 2 * np.pi / n_from_center
        else:
            r_start = rng.uniform(0.05, 0.5)
            base_angle = rng.uniform(0, 2 * np.pi)

        for existing in used_angles:
            if abs(base_angle - existing) < 0.4:
                base_angle = (base_angle + 0.5) % (2 * np.pi)
        used_angles.append(base_angle)

        rotations = rng.uniform(2.0, 5.0)
        base_width = rng.uniform(0.1, 0.2)
        intensity = rng.uniform(0.6, 1.0)

        r_length = rotations / 6.0 * (1.0 - r_start)
        r_length = min(r_length, 1.0 - r_start)
        r_end = r_start + r_length

        arm_angle = phi_grid - base_angle - r_norm_grid * rotations * 2 * np.pi

        # 宽度随噪声变化
        width_mod = 1.0 + 0.8 * (arm_noise - 0.5)
        width_mod = np.clip(width_mod, 0.3, 2.0)

        arm_kappa = 1.0 / (base_width ** 2) * 1.5
        arm_val = np.exp(arm_kappa * (np.cos(arm_angle) - 1) / width_mod)

        mask = (r_norm_grid >= r_start) & (r_norm_grid <= r_end)
        arm_val = np.where(mask, arm_val, 0)

        # 强度随噪声变化（断断续续）
        intensity_mod = 0.3 + 0.7 * (arm_noise ** 0.5)

        fade_in = np.clip((r_norm_grid - r_start) / 0.08, 0, 1)
        fade_out = np.clip((r_end - r_norm_grid) / 0.08, 0, 1)
        arm_val *= fade_in * fade_out * intensity * intensity_mod

        spiral += arm_val

    spiral = np.clip(spiral / (np.max(spiral) + 1e-6), 0, 1)

    # 2) 高频云雾（不用 FBM，直接高频）
    turbulence = _tileable_noise((n_r, n_phi), rng, freq_u=24, freq_v=12)

    # 3) 角方向弧形结构（破碎结构）- 沿 phi 方向的弧
    arc_count = rng.integers(10, 25)
    arcs = np.zeros((n_r, n_phi), dtype=np.float32)
    for _ in range(arc_count):
        arc_phi_start = rng.uniform(0, 2 * np.pi)
        arc_phi_length = rng.uniform(0.15, 0.4)
        arc_r = rng.uniform(0.1, 0.9)
        arc_r_width = rng.uniform(0.015, 0.04)
        arc_intensity = rng.uniform(0.4, 1.0)

        arc_kappa = 1.0 / (arc_phi_length ** 2) * 1.5

        r_diff = r_norm_grid - arc_r

        arc_val = np.exp(arc_kappa * (np.cos(phi_grid - arc_phi_start) - 1))
        arc_val *= np.exp(-0.5 * (r_diff / arc_r_width) ** 2)
        arc_val *= arc_intensity

        arcs += arc_val
    arcs = np.clip(arcs, 0, 1)

    # 4) 多个温度热点（弧形，高斯径向）
    hotspot_count = rng.integers(15, 35)
    hotspot = np.zeros((n_r, n_phi), dtype=np.float32)
    for _ in range(hotspot_count):
        h_phi = rng.uniform(0, 2 * np.pi)
        h_r = rng.uniform(0.1, 0.9)
        h_phi_width = rng.uniform(0.08, 0.2)
        h_r_width = rng.uniform(0.01, 0.03)
        h_intensity = rng.uniform(0.4, 1.0)

        h_kappa = 1.0 / (h_phi_width ** 2) * 1.5

        r_diff = r_norm_grid - h_r
        h_val = np.exp(h_kappa * (np.cos(phi_grid - h_phi) - 1)) * np.exp(-0.5 * (r_diff / h_r_width) ** 2)
        hotspot += h_intensity * h_val
    hotspot = np.clip(hotspot, 0, 1)

    # 5) 方位热点（低频正弦 + 噪声，自转流动感）
    az_freq = rng.integers(2, 5)
    shear = r_norm_grid ** 1.2 * rng.uniform(2.0, 4.0)
    az_wave = 0.5 + 0.5 * np.sin((phi_grid + shear) * az_freq)
    az_noise = _fbm_noise((n_r, n_phi), rng, octaves=3, persistence=0.5, base_scale=3, wrap_u=True)
    az_hotspot = np.clip(0.6 * az_wave + 0.4 * az_noise, 0, 1) ** 1.2

    # 组合密度
    density = 0.1 + 0.3 * spiral + 0.3 * turbulence + 0.2 * hotspot + 0.1 * arcs

    # 边缘软化（沿径向）
    edge = compute_edge_alpha(n_r)
    density *= edge[:, None]

    # 归一化
    density = np.clip(density / (np.percentile(density, 98) + 1e-6), 0, 1)

    # ----- 颜色（温度 -> 颜色映射）-----
    temp_aniso = np.clip(temperature_field * (0.9 + 0.25 * az_hotspot), 0, 1)
    hot_bias = np.clip(temp_aniso, 0, 1)
    r_ch = np.clip(hot_bias ** 0.38 * (0.95 + 0.25 * hotspot), 0, 1)
    g_ch = np.clip(hot_bias ** 0.55 * (0.7 + 0.15 * turbulence), 0, 1)
    b_ch = np.clip(hot_bias ** 0.85 * (0.38 + 0.18 * (1 - hotspot)), 0, 1)

    brightness_scale = 0.25
    tex = np.zeros((n_r, n_phi, 4), dtype=np.float32)
    tex[:, :, 0] = np.clip(r_ch * density * 6.0 * brightness_scale, 0, 1)
    tex[:, :, 1] = np.clip(g_ch * density * 6.0 * brightness_scale, 0, 1)
    tex[:, :, 2] = np.clip(b_ch * density * 6.0 * brightness_scale, 0, 1)
    tex[:, :, 3] = np.clip(density * brightness_scale, 0, 1)

    return tex




# ============================================================================
# Taichi 渲染器（类封装，支持反复调用）
# ============================================================================

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
                 lens_flare=False):
        import taichi as ti
        self.ti = ti
        self.width = width
        self.height = height
        self.step_size = step_size
        self.r_max = r_max
        self.r_disk_inner = r_disk_inner
        self.r_disk_outer = r_disk_outer
        self.disk_tilt = disk_tilt
        self.lens_flare = lens_flare

        ti.init(arch=ti.cpu if device == "cpu" else ti.gpu, offline_cache=False)

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

        self._compile_kernels()

    def _compile_kernels(self):
        ti = self.ti
        width, height = self.width, self.height
        tex_w, tex_h = self.tex_w, self.tex_h
        dtex_w, dtex_h = self.dtex_w, self.dtex_h
        texture_field = self.texture_field
        disk_texture_field = self.disk_texture_field

        g_cap = ti.cast(G_FACTOR_CAP, ti.f32)
        lum_power = ti.cast(G_LUMINOSITY_POWER, ti.f32)
        gain = ti.cast(G_BRIGHTNESS_GAIN, ti.f32)
        alpha_gain = ti.cast(DISK_ALPHA_GAIN, ti.f32)

        @ti.func
        def apply_g_factor(base_color, hit_pos, hit_r, ray_dir_to_cam, cam_pos,
                           r_inner, r_outer, tilt_rad):
            """
            Taichi 版本的 g 因子调制（参照 starless Redshift 推导）
            """
            rs_f = ti.cast(RS, ti.f32)
            r_obs = cam_pos.norm()
            r_em = hit_pos.norm()
            r_cyl = ti.sqrt(hit_pos[0] ** 2 + hit_pos[1] ** 2)
            r_safe = ti.max(r_cyl, rs_f + 1e-3)

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

            shift = ti.max(ti.min(g - 1.0, 2.0), -1.2)
            pos_shift = ti.max(shift, 0.0)
            neg_shift = ti.max(-shift, 0.0)

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
            # 暂不调整多普勒亮度

            # neg_shift>0 (蓝移): 偏红; pos_shift>0 (红移): 偏蓝
            red_boost = 1.0 + 0.35 * ti.pow(neg_shift, 0.75)
            r_scale = ti.min(ti.max((1.0 + 0.6 * neg_shift - 0.25 * pos_shift) * red_boost, 0.25), 2.6)
            g_scale = ti.min(ti.max(1.0 - 0.1 * pos_shift - 0.05 * neg_shift, 0.5), 1.25)
            b_scale = ti.min(ti.max(1.05 + 0.3 * pos_shift - 0.15 * neg_shift, 0.3), 2.7)

            shifted = ti.Vector([
                base_color[0] * r_scale,
                base_color[1] * g_scale,
                base_color[2] * b_scale,
            ])
            tint = ti.Vector([DISK_BASE_TINT[0], DISK_BASE_TINT[1], DISK_BASE_TINT[2]])
            return ti.math.clamp(shifted * tint * brightness, 0.0, 10.0)

        @ti.func
        def compute_acceleration(pos, L2):
            r2 = pos.dot(pos)
            r = ti.sqrt(r2)
            r5 = r2 * r2 * r
            return -1.5 * L2 / r5 * pos

        @ti.func
        def sample_skybox(d):
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
        def sample_disk(hit_x, hit_y, r_inner, r_outer, t_offset):
            r = ti.sqrt(hit_x ** 2 + hit_y ** 2)
            phi = ti.atan2(hit_y, hit_x)
            omega = ti.sqrt(0.5 / (r + 0.01))
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

        @ti.kernel
        def render_kernel(image_field: ti.template(), disk_layer_field: ti.template(),
                          cam_pos_field: ti.template(), cam_right_field: ti.template(),
                          cam_up_field: ti.template(), cam_forward_field: ti.template(),
                          pixel_width_field: ti.template(),
                          pixel_height_field: ti.template(), r_escape_field: ti.template(),
                          h_base: ti.f32, r_inner: ti.f32, r_outer: ti.f32, t_offset: ti.f32,
                          disk_tilt: ti.f32):
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

                escaped = False
                escape_dir = ti.Vector([0.0, 0.0, 0.0])
                event_horizon_hit = False
                accum_disk = ti.Vector([0.0, 0.0, 0.0])
                disk_alpha_total = 0.0
                step_count = 0
                affine = 0.0

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

                    k1p = h * dir_
                    k1d = h * compute_acceleration(pos, L2_val)
                    k2p = h * (dir_ + 0.5 * k1d)
                    k2d = h * compute_acceleration(pos + 0.5 * k1p, L2_val)
                    k3p = h * (dir_ + 0.5 * k2d)
                    k3d = h * compute_acceleration(pos + 0.5 * k2p, L2_val)
                    k4p = h * (dir_ + k3d)
                    k4d = h * compute_acceleration(pos + k3p, L2_val)

                    new_pos = pos + (k1p + 2 * k2p + 2 * k3p + k4p) / 6
                    new_dir = dir_ + (k1d + 2 * k2d + 2 * k3d + k4d) / 6

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

                    new_z = new_pos[2]
                    new_y = new_pos[1]

                    # 吸积盘检测：穿过倾斜平面 z = y * tan(tilt)
                    # 平面方程: z - y * tan_t = 0
                    tan_t = ti.tan(tilt_rad)
                    f_old = old_z - old_y * tan_t
                    f_new = new_z - new_y * tan_t
                    if f_old * f_new < 0:
                        t_frac = f_old / (f_old - f_new + 1e-8)
                        hit_x = old_pos[0] + t_frac * (new_pos[0] - old_pos[0])
                        hit_y = old_pos[1] + t_frac * (new_pos[1] - old_pos[1])
                        hit_r = ti.sqrt(hit_x ** 2 + hit_y ** 2)
                        if r_outer >= hit_r >= r_inner:
                            disk_rgba = sample_disk(hit_x, hit_y, r_inner, r_outer, t_offset)
                            disk_col = ti.Vector([disk_rgba[0], disk_rgba[1], disk_rgba[2]])
                            base_alpha = ti.min(disk_rgba[3], 0.999)
                            disk_alpha = 1.0 - ti.pow(1.0 - base_alpha, alpha_gain)

                            hit_z = hit_y * ti.tan(tilt_rad)
                            hit_pos_vec = ti.Vector([hit_x, hit_y, hit_z])
                            ray_to_cam = -dir_
                            col_shifted = apply_g_factor(
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
                    bg_color = sample_skybox(escape_dir)

                bg_color = bg_color * (1.0 - disk_alpha_total)

                image_field[i, j] = bg_color
                disk_layer_field[i, j] = ti.math.clamp(accum_disk, 0.0, 1.0)

        self._render_kernel = render_kernel

        @ti.kernel
        def bloom_kernel(image_field: ti.template(), bright_field: ti.template(),
                         blur_field: ti.template(), threshold: ti.f32, intensity: ti.f32,
                         kernel_radius: ti.i32, sigma_scale: ti.f32):
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

        self._bloom_kernel = bloom_kernel

        @ti.kernel
        def lens_flare_kernel(image_field: ti.template(),
                              disk_center_x: ti.f32, disk_center_y: ti.f32,
                              screen_center_x: ti.f32, screen_center_y: ti.f32,
                              intensity: ti.f32):
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
                    gsize = ti.cast(20 + g * 15, ti.f32)
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
                ring_r = 80.0
                ring_w = 8.0
                ring_alpha = 0.0
                if ti.abs(rdist - ring_r) < ring_w:
                    ring_alpha = (1.0 - ti.abs(rdist - ring_r) / ring_w) * 0.15
                if ring_alpha > 0:
                    flare += ti.Vector([0.6, 0.7, 1.0]) * ring_alpha

                image_field[i, j] = ti.math.clamp(image_field[i, j] + flare * intensity, 0.0, 1.0)

        self._lens_flare_kernel = lens_flare_kernel

    def render(self, cam_pos, fov, frame=0):
        """
        渲染单帧图像。

        参数:
            cam_pos: 相机位置 [x, y, z]
            fov: 视野角度
            frame: 帧编号（用于吸积盘自转动画）

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
        t_offset = float(frame) * 0.1
        disk_tilt = float(self.disk_tilt)

        self._render_kernel(
            self.image_field, self.disk_layer_field, self.cam_pos_field, self.cam_right_field,
            self.cam_up_field, self.cam_forward_field, self.pixel_width_field,
            self.pixel_height_field, self.r_escape_field, h_base, r_inner, r_outer, t_offset,
            disk_tilt
        )

        # 对吸积盘层做 bloom（卷积范围和 sigma 按分辨率比例缩放）
        kernel_radius = int(self.width * 0.02)
        sigma_scale = (self.width / 640.0) ** 2
        self._bloom_kernel(self.disk_layer_field, self.bright_field, self.blur_field, 0, 0.4, kernel_radius, sigma_scale)

        # 合并：背景 + 吸积盘 + bloom
        img = self.image_field.to_numpy()
        disk = self.disk_layer_field.to_numpy()
        disk_bloom = self.blur_field.to_numpy()
        final = np.clip(img + disk + disk_bloom, 0, 1)
        
        # Lens flare（CPU 实现）
        if self.lens_flare:
            final = self._apply_lens_flare(final, disk)
        return final.transpose(1, 0, 2)
    
    def _apply_lens_flare(self, final, disk):
        """应用 lens flare 效果，final 和 disk 都是 (width, height, 3)"""
        w, h, _ = final.shape
        
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
            ghost_size = 25 + g * 30
            
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
            ring_r = 60 + ring_idx * 40
            ring_w = 6 + ring_idx * 3
            
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
        hex_r = 100
        
        dx = x_coords - hex_x
        dy = y_coords - hex_y
        angle = np.arctan2(dy, dx)
        dist = np.sqrt(dx**2 + dy**2)
        
        # 六边形边缘检测
        hex_edge = np.abs(np.mod(angle, np.pi/3) - np.pi/6)
        hex_factor = np.clip(1 - hex_edge / 0.2, 0, 1)
        ring_dist = np.abs(dist - hex_r)
        ring_alpha = np.clip(1 - ring_dist / 15, 0, 1) ** 2 * hex_factor * 0.3 * intensity
        
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


def render_taichi(width, height, cam_pos, fov, step_size, skybox_path=None,
                  n_stars=6000, tex_w=2048, tex_h=1024, r_max=10.0, device="cpu",
                  disk_texture_path=None, r_disk_inner=R_DISK_INNER_DEFAULT,
                  r_disk_outer=R_DISK_OUTER_DEFAULT, disk_tilt=0.0,
                  lens_flare=False):
    """
    使用 Taichi 渲染单帧图像（兼容旧接口）。
    """
    skybox, tex_h, tex_w = load_or_generate_skybox(skybox_path, tex_w, tex_h, n_stars)
    disk_tex = load_disk_texture(disk_texture_path)
    if disk_tex is None:
        disk_tex = generate_disk_texture()

    renderer = TaichiRenderer(
        width, height, skybox, disk_tex,
        step_size=step_size, r_max=r_max, device=device,
        r_disk_inner=r_disk_inner, r_disk_outer=r_disk_outer,
        disk_tilt=disk_tilt,
        lens_flare=lens_flare
    )

    t0 = time.time()
    print(f"Taichi: {width}x{height}, cam_pos={list(cam_pos)}, fov={fov}°, step_size={step_size}")
    img = renderer.render(cam_pos, fov, frame=0)
    print(f"Done in {time.time() - t0:.1f}s")

    return img


def render_video(renderer, width, height, n_frames, fps, output_path,
                 fov, static_cam_pos, orbit=False, resume=False):
    # orbit 时使用 POV 的距离绕原点旋转
    orbit_radius = float(np.linalg.norm(static_cam_pos))
    """
    渲染视频（多帧并合成视频）。

    参数:
        renderer: TaichiRenderer 实例
        width, height: 图像尺寸
        n_frames: 帧数
        fps: 帧率
        output_path: 输出视频路径
        fov: 视野角度
        static_cam_pos: 静态模式下的相机位置
        orbit: 是否围绕原点旋转
        orbit_radius: 轨道半径
        orbit_z: 轨道高度
        resume: 是否尝试从断点恢复
    """
    import imageio.v3 as iio
    import json
    from PIL import Image

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    temp_dir_name = ".frames_" + hashlib.md5(output_path.encode()).hexdigest()[:16]
    temp_dir = os.path.join(os.path.dirname(output_path), temp_dir_name)
    progress_file = os.path.join(temp_dir, "progress.json")

    params = {
        "n_frames": n_frames,
        "fov": fov,
        "orbit": orbit,
    }

    completed = set()
    if resume and os.path.isdir(temp_dir) and os.path.isfile(progress_file):
        with open(progress_file, "r") as f:
            saved = json.load(f)
        saved_params = saved.get("params", {})
        if saved_params != params:
            print(f"Warning: parameters changed, starting over")
            import shutil
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
        else:
            completed = set(saved.get("completed", []))
            print(f"Resuming: {len(completed)}/{n_frames} frames already rendered")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    total_t0 = time.time()
    angle_step = 360.0 / n_frames
    rendered_this_session = 0

    for frame in range(n_frames):
        if frame in completed:
            continue

        if orbit:
            angle_deg = frame * angle_step
            angle_rad = np.radians(angle_deg)
            orbit_z = static_cam_pos[2]
            cx = orbit_radius * np.cos(angle_rad)
            cy = orbit_radius * np.sin(angle_rad)
            cam_pos = [cx, cy, orbit_z]
            status_str = f"{angle_deg:.1f}°"
        else:
            cam_pos = static_cam_pos
            status_str = "static"

        t0 = time.time()
        img = renderer.render(cam_pos, fov, frame=frame)
        elapsed = time.time() - t0
        rendered_this_session += 1

        frame_path = os.path.join(temp_dir, f"frame_{frame:04d}.png")
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_uint8, "RGB").save(frame_path)

        completed.add(frame)
        if rendered_this_session % 10 == 0 or frame == n_frames - 1:
            with open(progress_file, "w") as f:
                json.dump({"params": params, "completed": list(completed)}, f)

        if rendered_this_session % 100 == 0 or frame == n_frames - 1:
            eta = (time.time() - total_t0) / rendered_this_session * (n_frames - len(completed))
            print(f"  frame {frame}/{n_frames} ({status_str}) {elapsed:.1f}s, done {len(completed)}/{n_frames}, ETA {eta/60:.0f}min")

    if rendered_this_session > 0:
        print(f"\nSession rendered {rendered_this_session} frames in {(time.time() - total_t0)/60:.1f} min")

    if len(completed) < n_frames:
        print(f"Warning: only {len(completed)}/{n_frames} frames completed. Run again to resume.")
        return

    total_elapsed = time.time() - total_t0
    print(f"\nAll frames rendered in {total_elapsed/60:.1f} min")

    print(f"Assembling video: {output_path} ({fps} fps, {n_frames/fps:.0f}s)...")
    writer = iio.imopen(output_path, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=fps)

    for frame in range(n_frames):
        frame_path = os.path.join(temp_dir, f"frame_{frame:04d}.png")
        img = iio.imread(frame_path)
        writer.write_frame(img)
        os.remove(frame_path)

    writer.close()
    if os.path.exists(progress_file):
        os.remove(progress_file)
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Video saved: {output_path}")


# ============================================================================
# 主入口
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Schwarzschild 黑洞光线追踪渲染器")
    parser.add_argument("--pov", type=float, nargs=3, default=[6, 0, 0.5],
                        metavar=("X", "Y", "Z"),
                        help="相机位置 (default: 6 0 0.5)")
    parser.add_argument("--fov", type=float, default=90,
                        help="视野角度 0-180° (default: 90)")
    parser.add_argument("--resolution", "-r", type=str, default="fhd",
                        choices=["4k", "fhd", "hd", "sd"],
                        help="分辨率: 4k/fhd/hd/sd (default: fhd)")
    parser.add_argument("--texture", "-t", type=str, default=None,
                        help="天空盒纹理路径")
    parser.add_argument("--output", "-o", type=str, default="output/blackhole.png",
                        help="输出路径 (default: output/blackhole.png)")
    parser.add_argument("--step_size", "-s", type=float, default=0.1,
                        help="积分步长 (default: 0.1)")
    parser.add_argument("--r_max", type=float, default=10,
                        help="逃逸半径 (default: 10)")
    parser.add_argument("--n_stars", type=int, default=6000,
                        help="程序天空盒恒星数 (default: 6000)")
    parser.add_argument("--disk_texture", type=str, default=None,
                        help="吸积盘纹理路径 (default: 程序生成)")
    parser.add_argument("--ar1", type=float, default=R_DISK_INNER_DEFAULT,
                        help=f"吸积盘内半径 (default: {R_DISK_INNER_DEFAULT})")
    parser.add_argument("--ar2", type=float, default=R_DISK_OUTER_DEFAULT,
                        help=f"吸积盘外半径 (default: {R_DISK_OUTER_DEFAULT})")
    parser.add_argument("--disk_tilt", type=float, default=0.0,
                        help="吸积盘倾角 (度, default: 0)")
    parser.add_argument("--lens_flare", action="store_true",
                        help="启用 lens flare 效果 (default: 关闭)")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                        choices=["cpu", "gpu"],
                        help="Taichi 设备: cpu 或 gpu (default: cpu)")
    parser.add_argument("--video", action="store_true",
                        help="视频模式：渲染多帧并合成视频")
    parser.add_argument("--orbit", action="store_true",
                        help="视频模式：相机围绕原点旋转（需配合 --video）")
    parser.add_argument("--n_frames", type=int, default=3600,
                        help="视频帧数 (default: 3600, 仅 --video 有效)")
    parser.add_argument("--fps", type=int, default=36,
                        help="视频帧率 (default: 36, 仅 --video 有效)")
    parser.add_argument("--resume", action="store_true",
                        help="视频模式：尝试从断点恢复（默认从头开始）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    resolutions = {"4k": (3840, 2160), "fhd": (1920, 1080), "hd": (1280, 720), "sd": (640, 360)}
    width, height = resolutions[args.resolution]
    fov = args.fov % 180

    if args.video:
        skybox, _, _ = load_or_generate_skybox(args.texture, 2048, 1024, args.n_stars)
        disk_tex = load_disk_texture(args.disk_texture)
        if disk_tex is None:
            disk_tex = generate_disk_texture()

        renderer = TaichiRenderer(
            width, height, skybox, disk_tex,
            step_size=args.step_size, r_max=args.r_max, device=args.device,
            r_disk_inner=args.ar1, r_disk_outer=args.ar2,
            disk_tilt=args.disk_tilt,
            lens_flare=args.lens_flare
        )

        print(f"Rendering video: {args.n_frames} frames at {width}x{height}")
        print(f"  orbit={args.orbit}")
        print(f"  fov={fov}°, step_size={args.step_size}, fps={args.fps}, disk_tilt={args.disk_tilt}°")

        render_video(
            renderer, width, height,
            n_frames=args.n_frames, fps=args.fps, output_path=args.output,
            fov=fov, static_cam_pos=args.pov,
            orbit=args.orbit,
            resume=args.resume
        )
    else:
        img = render_taichi(
            width=width,
            height=height,
            cam_pos=args.pov,
            fov=fov,
            step_size=args.step_size,
            skybox_path=args.texture,
            n_stars=args.n_stars,
            r_max=args.r_max,
            device=args.device,
            disk_texture_path=args.disk_texture,
            r_disk_inner=args.ar1,
            r_disk_outer=args.ar2,
            disk_tilt=args.disk_tilt,
            lens_flare=args.lens_flare,
        )
        save_image(img, args.output)
