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

RS = 1.0  # 史瓦西半径 (几何单位制 G=c=1, 对应质量 M=1/2)


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


def compute_edge_alpha(height, inner_soft=0.1, outer_soft=0.2):
    """计算边缘软化的 alpha 通道"""
    v = np.linspace(0, 1, height).astype(np.float32)
    alpha = np.ones_like(v)
    inner_mask = v < inner_soft
    outer_mask = v > (1 - outer_soft)
    alpha[inner_mask] = (v[inner_mask] / inner_soft) ** 3.0
    alpha[outer_mask] = ((1 - v[outer_mask]) / outer_soft) ** 2.0
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


def _fbm_noise(shape, rng, octaves=4, persistence=0.5, base_scale=1):
    """分形布朗运动噪声（多层叠加）"""
    result = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    for i in range(octaves):
        scale = base_scale * (2 ** i)
        sh = max(shape[0] // scale, 2)
        sw = max(shape[1] // scale, 2)
        small = rng.random((sh, sw)).astype(np.float32)
        up = np.array(Image.fromarray((small * 255).astype(np.uint8)).resize(
            (shape[1], shape[0]), Image.Resampling.BILINEAR)) / 255.0
        result += up * amplitude
        total_amp += amplitude
        amplitude *= persistence
    return result / total_amp


def generate_disk_texture(tex_w=1024, tex_h=512, seed=42):
    """
    程序化生成吸积盘纹理（絮状云雾 + 螺旋臂 + 径向丝状结构 + 间隙）
    返回 (tex_h, tex_w, 4) float32，第 4 通道为 alpha（密度）
    """
    rng = np.random.default_rng(seed)

    u = np.linspace(0, 1, tex_w, endpoint=False)  # 角度方向
    v = np.linspace(0, 1, tex_h)                   # 径向方向（0=inner, 1=outer）
    uu, vv = np.meshgrid(u, v)

    # 温度剖面
    r_vals = np.linspace(2.0, 3.5, tex_h)
    T = (1 - np.sqrt(2.0 / r_vals)) ** 0.25 / r_vals ** 0.75
    T = T / np.max(T)

    # --- 密度场（决定哪里有物质，哪里是间隙）---

    # 1. 大尺度螺旋臂（2-3 条）— 作为亮度调制而非开关
    n_arms = rng.integers(2, 4)
    spiral = np.zeros((tex_h, tex_w), dtype=np.float32)
    for arm in range(n_arms):
        phase = arm * 2 * np.pi / n_arms + rng.uniform(0, np.pi)
        tightness = rng.uniform(1.5, 3.0)
        width = rng.uniform(0.2, 0.35)
        angle = uu * 2 * np.pi + vv * tightness * 2 * np.pi + phase
        arm_val = np.exp(-0.5 * ((np.sin(angle) / width) ** 2))
        spiral += arm_val
    spiral = np.clip(spiral, 0, 1)

    # 2. FBM 云雾噪声（絮状结构）— 柔和
    cloud = _fbm_noise((tex_h, tex_w), rng, octaves=3, persistence=0.5, base_scale=8)

    # 3. 径向丝状结构（细条纹）— 禁用
    filaments = np.ones((tex_h, tex_w), dtype=np.float32)

    # 4. 角度方向的宽条纹 — 禁用
    streaks = np.ones((tex_h, tex_w), dtype=np.float32)

    # 5. 间隙 — 已禁用
    gaps = np.ones((tex_h, tex_w), dtype=np.float32)

    # 组合密度场
    density = 0.6 + cloud * 0.15 + spiral * 0.1
    density *= filaments * streaks * gaps

    # 边缘软化
    edge = compute_edge_alpha(tex_h)
    density *= edge[:, None]

    # 归一化到 [0, 1]
    density = np.clip(density / (np.percentile(density, 98) + 1e-6), 0, 1)

    # --- 颜色（基于温度）---
    temp = T[:, None]
    r_ch = np.clip(temp ** 0.35 * 1.1, 0, 1)
    g_ch = np.clip(temp ** 0.5 * 0.75, 0, 1)
    b_ch = np.clip(temp ** 0.8 * 0.5, 0, 1)

    # 加入颜色变化（云雾区域偏暖）
    color_var = _fbm_noise((tex_h, tex_w), np.random.default_rng(seed + 1),
                           octaves=3, persistence=0.5, base_scale=8)
    r_ch = r_ch + color_var * 0.1
    g_ch = g_ch - color_var * 0.05
    b_ch = b_ch - color_var * 0.05

    # 组合 RGBA
    tex = np.zeros((tex_h, tex_w, 4), dtype=np.float32)
    tex[:, :, 0] = np.clip(r_ch * density * 8.0, 0, 1)
    tex[:, :, 1] = np.clip(g_ch * density * 8.0, 0, 1)
    tex[:, :, 2] = np.clip(b_ch * density * 8.0, 0, 1)
    tex[:, :, 3] = np.clip(density, 0, 1)  # alpha = 密度

    return tex


# ============================================================================
# NumPy 渲染器（compact 算法）
# ============================================================================

def compute_acceleration_numpy(pos, L2):
    """
    计算加速度（NumPy 版本）

    加速度公式：a = -1.5 * L² * pos / r⁵
    这是史瓦西度规的笛卡尔等效形式
    """
    r2 = np.sum(pos * pos, axis=1)
    r = np.sqrt(r2)
    r5 = r2 * r2 * r
    factor = -1.5 * L2 / r5
    return factor[:, None] * pos


def rk4_step_numpy(pos, vel, L2, dt):
    """
    单步 RK4 积分（NumPy 版本）

    参数:
        pos: (N, 3) 位置
        vel: (N, 3) 速度
        L2: (N,) 角动量平方
        dt: 标量或 (N,) 自适应步长

    返回:
        (new_pos, new_vel)
    """
    # 支持标量和向量 dt
    if np.ndim(dt) > 0:
        dt = dt[:, None]  # (N, 1) for broadcasting

    a1 = compute_acceleration_numpy(pos, L2)
    p2 = pos + 0.5 * dt * vel
    v2 = vel + 0.5 * dt * a1
    a2 = compute_acceleration_numpy(p2, L2)
    p3 = pos + 0.5 * dt * v2
    v3 = vel + 0.5 * dt * a2
    a3 = compute_acceleration_numpy(p3, L2)
    p4 = pos + dt * v3
    v4 = vel + dt * a3
    a4 = compute_acceleration_numpy(p4, L2)

    new_pos = pos + (dt / 6.0) * (vel + 2 * v2 + 2 * v3 + v4)
    new_vel = vel + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
    return new_pos, new_vel


def render_numpy(width, height, cam_pos, fov, step_size, skybox_path=None,
                 n_stars=6000, tex_w=2048, tex_h=1024, r_max=10.0,
                 disk_texture_path=None, r_disk_inner=R_DISK_INNER_DEFAULT,
                 r_disk_outer=R_DISK_OUTER_DEFAULT):
    """
    使用 NumPy 渲染（compact 向量化算法）

    参数:
        width, height: 图像尺寸
        cam_pos: 相机位置
        fov: 视野角度
        step_size: 积分步长
        skybox_path: 天空盒纹理路径
        n_stars: 程序天空盒恒星数
        tex_w, tex_h: 纹理尺寸
        r_max: 逃逸半径
        disk_texture_path: 吸积盘纹理路径
        r_disk_inner: 吸积盘内半径
        r_disk_outer: 吸积盘外半径

    返回:
        image: (height, width, 3) RGB 图像
    """
    t0 = time.time()
    cam_pos_arr = np.array(cam_pos, dtype=np.float64)
    distance = np.linalg.norm(cam_pos_arr)

    # 构建相机
    cam_pos_arr, cam_right, cam_up, cam_forward, pixel_width, pixel_height = build_camera(
        cam_pos_arr, fov, width, height
    )

    # 生成光线
    positions, velocities, L2, pixels = make_all_rays(
        width, height, cam_pos_arr, cam_right, cam_up, cam_forward, pixel_width, pixel_height
    )

    # 加载天空盒
    skybox, tex_h, tex_w = load_or_generate_skybox(skybox_path, tex_w, tex_h, n_stars)

    # 加载吸积盘纹理
    disk_tex = load_disk_texture(disk_texture_path)
    if disk_tex is None:
        disk_tex = generate_disk_texture()
    has_disk = True

    N = len(positions)
    result_status = np.zeros(N, dtype=int)  # 0=active, 1=escaped, 2=captured
    escape_dirs = np.zeros((N, 3))
    accumulated_disk = np.zeros((N, 3), dtype=np.float32)  # 累积吸积盘颜色
    disk_transmit = np.ones(N, dtype=np.float32)  # 剩余透射率

    r_capture = RS
    r_escape = max(r_max, distance * 2)
    dt_base = step_size
    max_factor = 10.0
    max_steps = int(r_escape * 40 / dt_base)

    # compact 算法：批量处理活跃光线
    active_idx = np.arange(N)
    act_pos = positions.copy()
    act_vel = velocities.copy()
    act_L2 = L2.copy()

    print(f"NumPy compact: {N} rays, max {max_steps} steps, dt_base={dt_base} (adaptive)")

    for step in range(max_steps):
        n_active = len(active_idx)
        if n_active == 0:
            print(f"  All rays terminated at step {step}")
            break

        # 保存旧位置用于吸积盘检测
        old_pos = act_pos.copy()

        # 自适应步长：dt = dt_base * min(r / rs, max_factor)
        r = np.sqrt(np.sum(act_pos * act_pos, axis=1))
        dt = dt_base * np.minimum(r / RS, max_factor)

        # RK4 积分
        act_pos, act_vel = rk4_step_numpy(act_pos, act_vel, act_L2, dt)

        # 吸积盘体积渲染：|z| < h(r) 区域内累积颜色
        if has_disk:
            new_z = act_pos[:, 2]
            r_cyl = np.sqrt(act_pos[:, 0]**2 + act_pos[:, 1]**2)
            in_radial = (r_cyl >= r_disk_inner) & (r_cyl <= r_disk_outer)
            # h(r) = 0.15 * (r/r_inner)^1.2 (flared disk)
            disk_h = np.where(in_radial, 0.15 * np.power(r_cyl / r_disk_inner, 1.2), 0)
            in_volume = in_radial & (np.abs(new_z) < disk_h)
            if np.any(in_volume):
                vol_idx = np.where(in_volume)[0]
                vol_global = active_idx[vol_idx]
                vx = act_pos[vol_idx, 0]
                vy = act_pos[vol_idx, 1]
                vz = act_pos[vol_idx, 2]
                vr = r_cyl[vol_idx]
                vh = disk_h[vol_idx]

                rgba = sample_disk_texture(disk_tex, vx, vy, r_disk_inner, r_disk_outer)
                tex_alpha = rgba[:, 3]
                colors = rgba[:, :3]

                # 密度随 |z| 衰减（高斯剖面）
                z_frac = vz / vh
                density = np.exp(-2.0 * z_frac * z_frac) * tex_alpha

                # 多普勒
                v_orb = np.sqrt(0.5 / vr)
                phi = np.arctan2(vy, vx)
                vdx = -v_orb * np.sin(phi)
                vdy = v_orb * np.cos(phi)
                ray_vel = act_vel[vol_idx]
                ray_norms = np.linalg.norm(ray_vel, axis=1, keepdims=True)
                ray_dir = ray_vel / np.maximum(ray_norms, 1e-12)
                vdot = vdx * ray_dir[:, 0] + vdy * ray_dir[:, 1]
                doppler = 1.0 / np.maximum(1.0 - vdot, 0.1)
                boost = np.clip(1.0 + np.log(doppler) * 1.5, 0.1, None)

                shift = doppler - 1.0
                shift = shift[:, None]
                r_shift = 1.0 - shift * 0.75
                g_shift = 1.0 - np.abs(shift) * 0.1
                b_shift = 1.0 + shift * 0.7
                colors = colors * np.concatenate([r_shift, g_shift, b_shift], axis=1)
                colors = colors * boost[:, None]

                # 体积累积：step_alpha ∝ density * step_size
                dt_vol = dt[vol_idx] if np.ndim(dt) > 0 else dt
                step_alpha = np.clip(density * dt_vol * 4.0, 0, 0.5)
                transmit = disk_transmit[vol_global]
                accumulated_disk[vol_global] += colors * (transmit * step_alpha)[:, None]
                disk_transmit[vol_global] *= (1.0 - step_alpha)

        # 分类光线
        r = np.sqrt(np.sum(act_pos * act_pos, axis=1))
        captured = r < r_capture
        escaped = r > r_escape
        terminated = captured | escaped

        # 记录逃逸光线方向
        if np.any(escaped):
            esc_local = np.where(escaped)[0]
            esc_global = active_idx[esc_local]
            result_status[esc_global] = 1
            esc_vel = act_vel[esc_local]
            norms = np.linalg.norm(esc_vel, axis=1, keepdims=True)
            escape_dirs[esc_global] = esc_vel / np.maximum(norms, 1e-12)

        # 记录捕获光线
        if np.any(captured):
            cap_global = active_idx[captured]
            result_status[cap_global] = 2

        # 移除终止光线
        if np.any(terminated):
            keep = ~terminated
            active_idx = active_idx[keep]
            act_pos = act_pos[keep]
            act_vel = act_vel[keep]
            act_L2 = act_L2[keep]

        if step % 500 == 0:
            n_esc = np.sum(result_status == 1)
            n_cap = np.sum(result_status == 2)
            print(f"  step {step}: active={len(active_idx)}, escaped={n_esc}, captured={n_cap}")

    # 剩余活跃光线视为逃逸
    if len(active_idx) > 0:
        result_status[active_idx] = 1
        norms = np.linalg.norm(act_vel, axis=1, keepdims=True)
        escape_dirs[active_idx] = act_vel / np.maximum(norms, 1e-12)

    # 着色：背景 + 累积吸积盘颜色（加法混合）
    print("Shading...")
    image = np.zeros((height, width, 3), dtype=np.float32)

    escaped_mask = result_status == 1
    if np.any(escaped_mask):
        esc_idx = np.where(escaped_mask)[0]
        colors = sample_skybox_bilinear(skybox, escape_dirs[esc_idx])
        esc_pixels = pixels[esc_idx]
        image[esc_pixels[:, 1], esc_pixels[:, 0]] = colors + accumulated_disk[esc_idx]

    # 被捕获的光线：黑色 + 累积吸积盘颜色
    captured_mask = result_status == 2
    if np.any(captured_mask):
        cap_idx = np.where(captured_mask)[0]
        cap_pixels = pixels[cap_idx]
        image[cap_pixels[:, 1], cap_pixels[:, 0]] = accumulated_disk[cap_idx]

    image = np.clip(image, 0, 1)

    elapsed = time.time() - t0
    n_esc = np.sum(result_status == 1)
    n_cap = np.sum(result_status == 2)
    print(f"Done: {n_esc} escaped, {n_cap} captured, {elapsed:.1f}s")
    return image


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
                 disk_tilt=0.0):
        import taichi as ti
        self.ti = ti
        self.width = width
        self.height = height
        self.step_size = step_size
        self.r_max = r_max
        self.r_disk_inner = r_disk_inner
        self.r_disk_outer = r_disk_outer
        self.disk_tilt = disk_tilt

        ti.init(arch=ti.cpu if device == "cpu" else ti.gpu)

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
        tex_w, tex_h = self.tex_w, self.tex_h
        dtex_w, dtex_h = self.dtex_w, self.dtex_h
        texture_field = self.texture_field
        disk_texture_field = self.disk_texture_field

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

            center = cp + cf * 1.0
            tl = center - cr * (pw * width / 2) + cu * (ph * height / 2)

            max_fac = ti.cast(10.0, ti.f32)
            r_cap = ti.cast(RS, ti.f32)
            r_esc = r_escape_field[None]
            max_iter = ti.cast(r_esc * 40 / h_base, ti.i32)

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
                step_count = 0

                while step_count < max_iter:
                    old_pos = pos
                    old_z = pos[2]
                    old_y = pos[1]
                    r_cur = pos.norm()
                    h = h_base * ti.min(r_cur / r_cap, max_fac)

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

                    if r < r_cap:
                        event_horizon_hit = True
                        break
                    elif r > r_esc:
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
                    f_new = new_z - new_y * tan_t
                    if f_old * f_new < 0:
                        t_frac = f_old / (f_old - f_new + 1e-8)
                        hit_x = old_pos[0] + t_frac * (new_pos[0] - old_pos[0])
                        hit_y = old_pos[1] + t_frac * (new_pos[1] - old_pos[1])
                        hit_r = ti.sqrt(hit_x ** 2 + hit_y ** 2)
                        if r_outer >= hit_r >= r_inner:
                            disk_rgba = sample_disk(hit_x, hit_y, r_inner, r_outer, t_offset)
                            disk_col = ti.Vector([disk_rgba[0], disk_rgba[1], disk_rgba[2]])
                            disk_alpha = disk_rgba[3]

                            # 多普勒效应
                            v_orb = ti.sqrt(0.5 / hit_r)
                            phi = ti.atan2(hit_y, hit_x)
                            vdx = -v_orb * ti.sin(phi)
                            vdy = v_orb * ti.cos(phi)
                            ray_d = dir_.normalized()
                            vdot = vdx * ray_d[0] + vdy * ray_d[1]
                            doppler = 1.0 / ti.max(1.0 - vdot, 0.1)
                            brightness = ti.min((doppler * 0.7) ** 3, 8)

                            shift = doppler - 1.0
                            r_col = disk_col[0] * (1.0 - shift * 0.6)
                            g_col = disk_col[1] * (1.0 - ti.abs(shift) * 0.3)
                            b_col = disk_col[2] * (1.0 + shift * 0.4)
                            col_shifted = ti.Vector([ti.max(r_col, 0.0), ti.max(g_col, 0.0), ti.max(b_col, 0.0)])

                            accum_disk += col_shifted * brightness * disk_alpha

                    pos = new_pos
                    dir_ = new_dir
                    step_count += 1

                # 分离式渲染：背景和吸积盘分开存储
                bg_color = ti.Vector([0.0, 0.0, 0.0])
                if event_horizon_hit:
                    bg_color = ti.Vector([0.0, 0.0, 0.0])
                elif escaped:
                    bg_color = sample_skybox(escape_dir)

                image_field[i, j] = bg_color
                disk_layer_field[i, j] = ti.math.clamp(accum_disk, 0.0, 1.0)

        self._render_kernel = render_kernel

        @ti.kernel
        def bloom_kernel(image_field: ti.template(), bright_field: ti.template(),
                         blur_field: ti.template(), threshold: ti.f32, intensity: ti.f32):
            w = ti.cast(image_field.shape[0], ti.i32)
            h = ti.cast(image_field.shape[1], ti.i32)

            for i, j in image_field:
                col = image_field[i, j]
                lum = col[0] * 0.2126 + col[1] * 0.7152 + col[2] * 0.0722
                if lum > threshold:
                    bright_field[i, j] = col
                else:
                    bright_field[i, j] = ti.Vector([0.0, 0.0, 0.0])

            for i, j in blur_field:
                sum_col = ti.Vector([0.0, 0.0, 0.0])
                weight_sum = 0.0
                for dy in range(-12, 13):
                    for dx in range(-12, 13):
                        ni = i + dy
                        nj = j + dx
                        if 0 <= ni < w and 0 <= nj < h:
                            dist_sq = ti.cast(dx * dx + dy * dy, ti.f32)
                            weight = ti.exp(-dist_sq / 64.0)
                            sum_col += bright_field[ni, nj] * weight
                            weight_sum += weight
                if weight_sum > 0.0:
                    blur_field[i, j] = sum_col / weight_sum
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

        # 对吸积盘层做 bloom
        self._bloom_kernel(self.disk_layer_field, self.bright_field, self.blur_field, 0, 0.4)

        # 合并：背景 + 吸积盘 + bloom
        img = self.image_field.to_numpy()
        disk = self.disk_layer_field.to_numpy()
        disk_bloom = self.blur_field.to_numpy()
        final = np.clip(img + disk + disk_bloom, 0, 1)
        return final.transpose(1, 0, 2)


def render_taichi(width, height, cam_pos, fov, step_size, skybox_path=None,
                  n_stars=6000, tex_w=2048, tex_h=1024, r_max=10.0, device="cpu",
                  disk_texture_path=None, r_disk_inner=R_DISK_INNER_DEFAULT,
                  r_disk_outer=R_DISK_OUTER_DEFAULT, disk_tilt=0.0):
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
        disk_tilt=disk_tilt
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
    os.remove(progress_file)
    os.rmdir(temp_dir)
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
    parser.add_argument("--framework", "-f", type=str, default="taichi",
                        choices=["numpy", "taichi"],
                        help="渲染框架: numpy 或 taichi (default: taichi)")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                        choices=["cpu", "gpu"],
                        help="Taichi 设备: cpu 或 gpu (default: cpu, 仅 taichi 框架有效)")
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
        if args.framework != "taichi":
            print("Error: Video mode only supports taichi framework")
            exit(1)

        skybox, _, _ = load_or_generate_skybox(args.texture, 2048, 1024, args.n_stars)
        disk_tex = load_disk_texture(args.disk_texture)
        if disk_tex is None:
            disk_tex = generate_disk_texture()

        renderer = TaichiRenderer(
            width, height, skybox, disk_tex,
            step_size=args.step_size, r_max=args.r_max, device=args.device,
            r_disk_inner=args.ar1, r_disk_outer=args.ar2,
            disk_tilt=args.disk_tilt
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
        if args.framework == "taichi":
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
            )
        else:
            img = render_numpy(
                width=width,
                height=height,
                cam_pos=args.pov,
                fov=fov,
                step_size=args.step_size,
                skybox_path=args.texture,
                n_stars=args.n_stars,
                r_max=args.r_max,
                disk_texture_path=args.disk_texture,
                r_disk_inner=args.ar1,
                r_disk_outer=args.ar2,
            )

        save_image(img, args.output)
