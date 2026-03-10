#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# type: ignore[attr-defined, misc, valid-type]
"""
Schwarzschild 黑洞光线追踪渲染器

基于广义相对论的史瓦西度规，使用笛卡尔等效形式的光线方程：
    d²x/dλ² = -1.5 * L² * x / r⁵

基于 Taichi 框架的并行渲染器（支持 CPU/GPU）

Reference: https://github.com/JaeHyunLee94/BlackHoleRendering
"""

import numpy as np
from PIL import Image
import os
import time
import argparse
import hashlib
from dataclasses import dataclass
from tqdm import tqdm
from typing import Tuple, List, Optional
import math

import taichi as ti
import json
import shutil
import imageio.v3 as iio
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# 公共常量
# ============================================================================

# 核心常量（按需调参，可参考注释范围）
RS = 1.0
EPS = 1e-6

# —— g 因子着色相关 —— 影响吸积盘自身的亮度/颜色，背景天空不受这些参数影响。
# g 因子亮度压缩的软上限，推荐 0.5~6（默认 3.0），值越小盘面整体越暗
G_FACTOR_CAP = 1.5
# g 的幂次，决定亮度随 g 变化的敏感度，建议 1.5~3（默认 2.2）
G_LUMINOSITY_POWER = 1.5
# 亮度缩放系数，常用 0.2~0.6（默认 0.38），越大盘面全局越亮
G_BRIGHTNESS_GAIN = 0.38

# —— 吸积盘透明度与色温 —— 决定盘层遮挡背景与整体暖色偏移。
# DISK_COLOR_TEMPERATURE: 吸积盘基准色温（单位：开尔文 K）
#   典型取值：1000K(橙红) ~ 6500K(白) ~ 10000K+(白偏蓝)
#   默认 4500K 暖白色
DISK_COLOR_TEMPERATURE = 6000
# DISK_ALPHA_GAIN > 1 会让盘体更实心，推荐 1~20（默认 1.2）
DISK_ALPHA_GAIN = 6
# DISK_RADIAL_BRIGHTNESS_POWER >0 会让亮度按 (1 - radial_t)^p 递减（常用 1~3）
DISK_RADIAL_BRIGHTNESS_POWER = 1.2
# 半径亮度增益的下限/上限，避免指数爆炸
DISK_RADIAL_BRIGHTNESS_MIN = 0.2
DISK_RADIAL_BRIGHTNESS_MAX = 8

# —— 天空盒程序化生成 —— 控制恒星数量、亮度范围和银河弥漫光强度。
# 恒星最低亮度，推荐 0.03~0.15（默认 0.08），越大暗星越明显
SKY_STAR_BRIGHTNESS_MIN = 0.03
# 恒星最高亮度，推荐 0.8~1.0（默认 1.0）
SKY_STAR_BRIGHTNESS_MAX = 1.0
# 亮度增益倍数，推荐 1.0~3.0（默认 1.8），整体提亮星点
SKY_STAR_BRIGHTNESS_GAIN = 1.8
# 颜色饱和度，推荐 0.0~1.0（默认 0.3），0=纯白 1=全黑体色
SKY_STAR_COLOR_SATURATION = 0.3
# 恒星高斯 blob 最小半径（像素），推荐 0.3~0.8（默认 0.5）
SKY_STAR_SIZE_MIN = 0.5
# 恒星高斯 blob 最大半径（像素），推荐 1.0~2.5（默认 1.7）
SKY_STAR_SIZE_MAX = 1.7
# 银河弥漫光强度，推荐 0.01~0.15（默认 0.10）
SKY_MILKY_WAY_GLOW = 0.10
# 银心额外增亮强度，推荐 0.01~0.10（默认 0.08）
SKY_GALACTIC_CENTER_GLOW = 0.08
DISK_GENERATION_SCALE_CHOICES = (1, 2, 4)
ENABLE_DISK_SPIRAL_ARMS = False


def _validate_disk_generation_scale(generation_scale: int) -> int:
    if generation_scale not in DISK_GENERATION_SCALE_CHOICES:
        raise ValueError(
            f"disk_generation_scale must be one of {DISK_GENERATION_SCALE_CHOICES}, got {generation_scale}"
        )
    return generation_scale

# ============================================================================
# 公共模块：相机
# ============================================================================

def build_camera(cam_pos: np.ndarray, fov_deg: float, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
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




# ============================================================================
# 公共模块：天空盒
# ============================================================================

def _blackbody_rgb(T: np.ndarray) -> np.ndarray:
    """色温(K) -> RGB，基于 Tanner Helland 近似"""
    t = T / 100.0
    r = np.where(t <= 66, 1.0,
                 np.clip(1.292936 * np.power(np.maximum(t - 60, 1e-6),
                         -0.1332047592), 0, 1))
    g = np.where(t <= 66,
                 np.clip(0.390082 * np.log(np.maximum(t, 1e-6)) - 0.631841, 0, 1),
                 np.clip(1.129891 * np.power(np.maximum(t - 60, 1e-6),
                         -0.0755148492), 0, 1))
    b = np.where(t >= 66, 1.0,
            np.where(t <= 19, 0.0,
                 np.clip(0.543207 * np.log(np.maximum(t - 10, 1e-6))
                         - 1.19625, 0, 1)))
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def generate_skybox(tex_w: int = 2048, tex_h: int = 1024, seed: int = 42, n_stars: int = 6000) -> np.ndarray:
    """
    程序化生成天空盒纹理（等距柱状投影）

    特性：银道面密度增强、幂律亮度分布、黑体色温连续映射、
    银河弥漫光、水平方向无缝 wrap。

    参数:
        tex_w: 纹理宽度
        tex_h: 纹理高度
        seed: 随机种子
        n_stars: 恒星数量

    返回:
        texture: (tex_h, tex_w, 3) float32 RGB 纹理，值域 [0, 1]
    """
    rng = np.random.default_rng(seed)
    texture = np.full((tex_h, tex_w, 3), 0.003, dtype=np.float32)

    # 星云：低频噪声上采样
    neb_h, neb_w = tex_h // 16, tex_w // 16
    nebula_small = rng.random((neb_h, neb_w, 3)).astype(np.float32) * 0.06
    nebula = np.array(Image.fromarray(
        (nebula_small * 255).astype(np.uint8)
    ).resize((tex_w, tex_h), Image.Resampling.BILINEAR)) / 255.0 * 0.04
    texture += nebula

    # --- 银道面参数 ---
    gal_incl = np.radians(62.87)       # 银道面对赤道面倾角
    gal_ra_center = np.radians(266.4)  # 银心 RA
    gal_dec_center = np.radians(-28.9) # 银心 Dec

    # --- 恒星位置：拒绝采样实现银道面密度增强 ---
    stars_phi = []
    stars_theta = []
    n_generated = 0
    batch = n_stars * 3
    while n_generated < n_stars:
        z = rng.uniform(-1, 1, batch)
        phi = rng.uniform(0, 2 * np.pi, batch)
        theta = np.arccos(np.clip(z, -1, 1))
        dec = np.pi / 2 - theta

        # 银纬 b
        sin_b = (np.sin(dec) * np.cos(gal_incl)
                 - np.cos(dec) * np.sin(gal_incl)
                 * np.sin(phi - gal_ra_center))
        b = np.arcsin(np.clip(sin_b, -1, 1))

        # 银道面高斯增强 + 银心方向额外增强
        prob = 0.15 + 0.85 * np.exp(-0.5 * (b / np.radians(8)) ** 2)
        cos_dist = (np.sin(dec) * np.sin(gal_dec_center)
                    + np.cos(dec) * np.cos(gal_dec_center)
                    * np.cos(phi - gal_ra_center))
        ang_dist = np.arccos(np.clip(cos_dist, -1, 1))
        prob += 0.3 * np.exp(-0.5 * (ang_dist / np.radians(20)) ** 2)
        prob = prob / prob.max()

        accept = rng.random(batch) < prob
        need = n_stars - n_generated
        stars_phi.extend(phi[accept][:need])
        stars_theta.extend(theta[accept][:need])
        n_generated = len(stars_phi)

    phi_s = np.array(stars_phi[:n_stars])
    theta_s = np.array(stars_theta[:n_stars])

    cx = (phi_s / (2 * np.pi) * tex_w).astype(np.float32)
    cy = (theta_s / np.pi * tex_h).astype(np.float32)

    # --- Salpeter IMF 采样：dN/dM ∝ M^(-2.35) ---
    # 质量范围 [0.08, 50] 太阳质量，逆变换采样
    # 分配随机距离后按视星等截断，模拟观测选择效应
    alpha = 2.35
    m_lo, m_hi = 0.08, 50.0
    oversample = n_stars * 30
    u_mass = rng.random(oversample)
    mass_all = (m_lo ** (1 - alpha) + u_mass
                * (m_hi ** (1 - alpha) - m_lo ** (1 - alpha))
                ) ** (1 / (1 - alpha))

    # 主序星质量-光度关系：L ∝ M^a（Duric 2004）
    lum_exp = np.where(mass_all < 0.43, 2.3,
              np.where(mass_all < 2.0, 4.0,
              np.where(mass_all < 55.0, 3.5, 1.0)))
    luminosity_all = np.power(mass_all, lum_exp)

    # 绝对星等
    abs_mag = -2.5 * np.log10(luminosity_all + 1e-30) + 4.83  # 太阳 M=4.83

    # 随机距离（pc），银河系恒星典型分布
    dist_all = rng.exponential(scale=200.0, size=oversample)
    dist_all = np.clip(dist_all, 1.0, 5000.0)

    # 视星等 = 绝对星等 + 5*log10(d/10)
    app_mag = abs_mag + 5.0 * np.log10(dist_all / 10.0)

    # 视星等截断：肉眼极限 ~6.5，望远镜可到 ~10
    mag_cutoff = 8.0
    visible = app_mag <= mag_cutoff
    vis_idx = np.where(visible)[0]
    if len(vis_idx) >= n_stars:
        idx = rng.choice(vis_idx, size=n_stars, replace=False)
    else:
        # 不够则取最亮的
        idx = np.argsort(app_mag)[:n_stars]
    mass = mass_all[idx]
    app_mag_sel = app_mag[idx]

    # 视星等 → 亮度（对数压缩到可见范围）
    mag_norm = (app_mag_sel - app_mag_sel.min()) / (
        app_mag_sel.max() - app_mag_sel.min() + 1e-30)
    brightness = (SKY_STAR_BRIGHTNESS_MAX
                  - (SKY_STAR_BRIGHTNESS_MAX - SKY_STAR_BRIGHTNESS_MIN)
                  * mag_norm).astype(np.float32)  # 亮星 mag 小 → brightness 大
    brightness = np.clip(brightness * SKY_STAR_BRIGHTNESS_GAIN, 0, 1)
    sigma = (SKY_STAR_SIZE_MIN
             + (SKY_STAR_SIZE_MAX - SKY_STAR_SIZE_MIN)
             * brightness).astype(np.float32)

    # --- 主序星质量-温度关系 + Planck 黑体 RGB ---
    # T_eff ≈ 5778 * M^0.57 K（主序星经验关系）
    temp_K = 5778.0 * np.power(mass, 0.57)
    temp_K = np.clip(temp_K, 2000, 50000)
    colors = _blackbody_rgb(temp_K)
    # 降低饱和度：向白色混合，模拟肉眼观感
    white = np.ones_like(colors)
    colors = SKY_STAR_COLOR_SATURATION * colors + (1 - SKY_STAR_COLOR_SATURATION) * white

    # --- 高斯 blob 渲染（水平 wrap）---
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

    # --- 银河弥漫光（含旋臂结构）---
    v_grid = np.linspace(0, np.pi, tex_h)
    u_grid = np.linspace(0, 2 * np.pi, tex_w)
    uu, vv = np.meshgrid(u_grid, v_grid)
    dec_grid = np.pi / 2 - vv

    # 赤道坐标 → 银道坐标
    sin_b_grid = (np.sin(dec_grid) * np.cos(gal_incl)
                  - np.cos(dec_grid) * np.sin(gal_incl)
                  * np.sin(uu - gal_ra_center))
    b_grid = np.arcsin(np.clip(sin_b_grid, -1, 1))

    cos_b = np.cos(b_grid)
    sin_l_cos_b = (np.cos(dec_grid) * np.cos(gal_incl)
                   * np.sin(uu - gal_ra_center)
                   + np.sin(dec_grid) * np.sin(gal_incl))
    cos_l_cos_b = np.cos(dec_grid) * np.cos(uu - gal_ra_center)
    l_grid = np.arctan2(sin_l_cos_b, cos_l_cos_b)  # 银经 [-π, π]

    # 银道面基础辉光
    milky_way = SKY_MILKY_WAY_GLOW * np.exp(-0.5 * (b_grid / np.radians(6)) ** 2)

    # 银心增亮（l≈0, b≈0）
    center_dist2 = l_grid ** 2 + b_grid ** 2
    milky_way += SKY_GALACTIC_CENTER_GLOW * np.exp(
        -0.5 * center_dist2 / np.radians(15) ** 2)

    # 旋臂调制：4 条主旋臂在银经方向的投影
    # 从太阳视角看，旋臂在银经上近似等间距分布，用正弦调制模拟明暗交替
    arm_pattern = 0.4 + 0.6 * (0.5 + 0.5 * np.cos(4 * l_grid + np.radians(30)))
    # 旋臂只在银道面附近有效
    arm_mask = np.exp(-0.5 * (b_grid / np.radians(8)) ** 2)
    milky_way *= (1.0 - arm_mask) + arm_mask * arm_pattern

    texture += milky_way[:, :, None] * np.array([1.0, 0.95, 0.85])

    return np.clip(texture, 0, 1)


def load_or_generate_skybox(skybox_path: Optional[str], tex_w: int = 2048, tex_h: int = 1024, n_stars: int = 6000) -> Tuple[np.ndarray, int, int]:
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


def sample_skybox_bilinear(texture: np.ndarray, directions: np.ndarray) -> np.ndarray:
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

def save_image(image: np.ndarray, path: str) -> None:
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
R_DISK_OUTER_DEFAULT = 15.0 * RS


def compute_edge_alpha(height: int, inner_soft: float = 0.1, outer_soft: float = 0.3) -> np.ndarray:
    """计算边缘软化的 alpha 通道"""
    v = np.linspace(0, 1, height).astype(np.float32)
    alpha = np.ones_like(v)
    inner_mask = v < inner_soft
    outer_mask = v > (1 - outer_soft)
    alpha[inner_mask] = (v[inner_mask] / inner_soft) ** 3.0
    alpha[outer_mask] = ((1 - v[outer_mask]) / outer_soft) ** 2
    return alpha


def load_disk_texture(path: Optional[str]) -> Optional[np.ndarray]:
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


@dataclass(frozen=True)
class DiskTextureRotatingState:
    n_phi: int
    n_r: int
    seed: int
    generation_scale: int
    r_inner: float
    r_outer: float
    enable_rt: bool
    color_temp: float
    omega_rows: np.ndarray
    edge: np.ndarray
    temp_base: np.ndarray
    spiral: np.ndarray
    spiral_temp: np.ndarray
    turbulence: np.ndarray
    turb_temp: np.ndarray
    arcs: np.ndarray
    arcs_temp: np.ndarray
    rt_spikes: np.ndarray
    rt_temp: np.ndarray
    hotspot: np.ndarray
    hotspot_temp: np.ndarray
    az_hotspot: np.ndarray
    disturb_mod: np.ndarray


# ============================================================================
# 实体生命周期系统
# ============================================================================

@dataclass
class EntityInstance:
    """Single entity instance in the lifecycle system.

    Stores pre-computed sparse density/temperature contributions and Keplerian
    rotation parameters for one filament, hotspot, or RT spike.

    Args:
        row_indices: affected radial row indices, shape (n_affected,).
            Typically 2-8 rows for filaments, 40-100 for hotspots, 10-40 for RT spikes.
        phi_density: density contribution at affected rows, shape (n_affected, n_phi).
            Pre-computed at birth, rotated via np.roll each frame.
        phi_temp: temperature contribution at affected rows, shape (n_affected, n_phi).
            Same rotation as phi_density.
        omega: Keplerian angular velocity at entity center (rad/s).
            Used to compute per-frame azimuthal shift.
        birth_time: wall-clock time when entity was spawned (seconds).
        lifetime: total alive duration excluding fade periods (seconds).
        fade_in: fade-in duration (seconds). Entity alpha ramps 0→1.
        fade_out: fade-out duration (seconds). Entity alpha ramps 1→0.

    Physical Meaning:
        Represents a localized structure in the accretion disk with a finite
        lifespan. Undergoes Keplerian rotation at its radial position, fading
        in at birth and fading out at death to avoid visual pops.

    Simplifications:
        Uses a single omega for all rows (ignores differential rotation within
        the entity's narrow radial extent).
    """
    row_indices: np.ndarray
    phi_density: np.ndarray
    phi_temp: np.ndarray
    omega: float
    birth_time: float
    lifetime: float
    fade_in: float
    fade_out: float
    fade_noise: np.ndarray  # (n_phi,) smooth noise in [0,1] for dissolve effect

    @property
    def total_duration(self) -> float:
        """Total time from birth to fully faded out."""
        return self.fade_in + self.lifetime + self.fade_out

    def is_dead(self, now: float) -> bool:
        """Whether the entity has fully faded out and should be recycled."""
        return (now - self.birth_time) >= self.total_duration

    def fade_factor(self, now: float) -> float:
        """Compute current fade alpha based on age.

        Returns:
            alpha in [0, 1]: 0 during pre-birth or post-death,
            linear ramp during fade-in/out, 1.0 during alive phase.

        Formula:
            age = now - birth_time
            if age < fade_in:       alpha = age / fade_in
            elif age < fade_in + lifetime: alpha = 1.0
            elif age < total_duration:     alpha = 1.0 - (age - fade_in - lifetime) / fade_out
            else:                          alpha = 0.0
        """
        age = now - self.birth_time
        if age < 0:
            return 0.0
        if age < self.fade_in:
            return age / self.fade_in if self.fade_in > 0 else 1.0
        age_after_fade_in = age - self.fade_in
        if age_after_fade_in < self.lifetime:
            return 1.0
        age_in_fade_out = age_after_fade_in - self.lifetime
        if age_in_fade_out < self.fade_out:
            return 1.0 - age_in_fade_out / self.fade_out if self.fade_out > 0 else 0.0
        return 0.0


class EntityFactory:
    """Manages lifecycle of entity instances — spawning, aging, and recycling.

    Maintains a pool of alive entities, spawning new ones at a controlled rate
    to maintain a target count. Dead entities are automatically removed.

    Args:
        spawn_fn: callable(rng, n_r, n_phi, r_norm_all, omega_all) -> (row_indices, phi_density, phi_temp, omega).
            The single-instance generation function (e.g. _spawn_single_filament).
        target_count: desired number of alive entities at steady state.
        lifetime_range: (min_seconds, max_seconds) for entity lifetime.
        fade_in: fade-in duration in seconds.
        fade_out: fade-out duration in seconds.
        n_r: radial resolution.
        n_phi: azimuthal resolution.
        r_norm_all: normalized radial positions, shape (n_r,).
        omega_all: Keplerian angular velocity per row, shape (n_r,).
        seed: random seed for reproducibility.

    Physical Meaning:
        Models the continuous birth and death of transient structures in the
        accretion disk. The target_count and lifetime_range determine the
        visual density and turnover rate of structures.
    """

    def __init__(self, spawn_fn, target_count: int,
                 lifetime_range: Tuple[float, float],
                 fade_in: float, fade_out: float,
                 n_r: int, n_phi: int,
                 r_norm_all: np.ndarray, omega_all: np.ndarray,
                 seed: int = 0):
        self.spawn_fn = spawn_fn
        self.target_count = target_count
        self.lifetime_range = lifetime_range
        self.fade_in = fade_in
        self.fade_out = fade_out
        self.n_r = n_r
        self.n_phi = n_phi
        self.r_norm_all = r_norm_all
        self.omega_all = omega_all
        self.rng = np.random.default_rng(seed)
        self.entities: List[EntityInstance] = []
        self._spawn_debt = 0.0

    def _spawn_one(self, now: float) -> EntityInstance:
        """Spawn a single entity at the current time."""
        row_indices, phi_density, phi_temp, omega = self.spawn_fn(
            self.rng, self.n_r, self.n_phi, self.r_norm_all, self.omega_all)
        lifetime = float(self.rng.uniform(*self.lifetime_range))
        return EntityInstance(
            row_indices=row_indices,
            phi_density=phi_density,
            phi_temp=phi_temp,
            omega=omega,
            birth_time=now,
            lifetime=lifetime,
            fade_in=self.fade_in,
            fade_out=self.fade_out,
            fade_noise=self._make_fade_noise(),
        )

    def _make_fade_noise(self) -> np.ndarray:
        """Generate smooth 1D dissolve noise along phi, range [0, 1].

        Uses 2-3 sinusoidal components for low-frequency spatial variation,
        ensuring the dissolve front is smooth (cloud-like), not pixelated.
        """
        phi = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)
        freq1 = int(self.rng.integers(3, 8))
        freq2 = int(self.rng.integers(8, 16))
        p1 = float(self.rng.uniform(0, 2 * np.pi))
        p2 = float(self.rng.uniform(0, 2 * np.pi))
        noise = (0.6 * np.sin(phi * freq1 + p1)
                 + 0.4 * np.sin(phi * freq2 + p2))
        noise = np.clip(noise * 0.5 + 0.5, 0, 1)
        return noise.astype(np.float32)

    def seed_initial(self, now: float) -> None:
        """Pre-populate with target_count entities at staggered ages.

        Distributes entities uniformly across their lifecycle so that the
        visual result is immediately at steady state, avoiding a "cold start"
        where all entities fade in simultaneously.

        Args:
            now: current wall-clock time in seconds
        """
        for i in range(self.target_count):
            entity = self._spawn_one(now)
            max_age = entity.fade_in + entity.lifetime
            stagger = max_age * (i / max(self.target_count, 1))
            entity.birth_time = now - stagger
            self.entities.append(entity)

    def tick(self, now: float, dt: float) -> None:
        """Advance the factory by one frame: remove dead, spawn replacements.

        Args:
            now: current wall-clock time in seconds
            dt: time elapsed since last frame (seconds)
        """
        self.entities = [e for e in self.entities if not e.is_dead(now)]

        deficit = self.target_count - len(self.entities)
        if deficit <= 0:
            return

        avg_lifetime = sum(self.lifetime_range) / 2.0
        spawn_rate = self.target_count / avg_lifetime
        self._spawn_debt += spawn_rate * dt
        n_spawn = min(int(self._spawn_debt), deficit)
        self._spawn_debt -= n_spawn

        for _ in range(n_spawn):
            self.entities.append(self._spawn_one(now))

    @property
    def alive_entities(self) -> List['EntityInstance']:
        """Return list of currently alive (not fully dead) entities."""
        return self.entities


def _generate_temperature_base(rng: np.random.Generator, n_r: int, n_phi: int,
                               r_norm_grid: np.ndarray) -> np.ndarray:
    """生成吸积盘基础温度场（不含动态旋转）。"""
    radial_decay = np.clip(1.0 - r_norm_grid, 0, 1) ** 1.3
    temp_coarse = _fbm_noise((n_r, n_phi), rng, octaves=4, persistence=0.6, base_scale=8, wrap_u=True)
    temp_fine = _fbm_noise((n_r, n_phi), rng, octaves=5, persistence=0.45, base_scale=3, wrap_u=True)
    temp_noise = 0.6 * temp_coarse + 0.4 * temp_fine
    temp_base = np.clip(radial_decay * (0.85 + 0.15 * temp_noise), 0, 1)
    temp_base *= 0.25
    return temp_base.astype(np.float32)


def _generate_disturbance_mod(rng: np.random.Generator, n_r: int, n_phi: int,
                              kep_shift_pixels: np.ndarray, r_norm_grid: np.ndarray,
                              t_offset: float = 0.0, omega_grid: np.ndarray = None,
                              generation_scale: int = 2) -> np.ndarray:
    """生成湍流扰动调制场。"""
    scale_factor = _validate_disk_generation_scale(generation_scale)
    low_n_r = n_r // scale_factor
    low_n_phi = n_phi // scale_factor

    low_r_norm_grid = r_norm_grid[::scale_factor, ::scale_factor]
    kep_shift_pixels_low = (kep_shift_pixels // scale_factor).astype(np.int32)[:low_n_r, :]

    disturb_coarse = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=8, freq_v=4)
    disturb_mid = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=32, freq_v=16)
    disturb_fine = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=100, freq_v=50)
    disturb_extra = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=250, freq_v=125)

    for layer in [disturb_coarse, disturb_mid, disturb_fine, disturb_extra]:
        for ri in range(low_n_r):
            layer[ri, :] = np.roll(layer[ri, :], kep_shift_pixels_low[ri, 0])

    rotation_pixels_low = None
    if t_offset != 0.0 and omega_grid is not None:
        omega_grid_low = omega_grid[::scale_factor, ::scale_factor]
        rotation_pixels_low = (t_offset * omega_grid_low / (2 * np.pi) * low_n_phi).astype(int)
        for layer in [disturb_coarse, disturb_mid, disturb_fine, disturb_extra]:
            for ri in range(low_n_r):
                layer[ri, :] = np.roll(layer[ri, :], -rotation_pixels_low[ri, 0])

    disturb_pixel = _periodic_pixel_noise((low_n_r, low_n_phi), rng)
    if rotation_pixels_low is not None:
        for ri in range(low_n_r):
            disturb_pixel[ri, :] = np.roll(disturb_pixel[ri, :], -rotation_pixels_low[ri, 0])

    disturb_mod_low = (0.05 * disturb_coarse + 0.15 * disturb_mid + 0.30 * disturb_fine
                       + 0.30 * disturb_extra + 0.20 * disturb_pixel)
    disturb_mod_low = np.clip(disturb_mod_low * 1.4, 0.05, 1.0)

    radial_preserve = 0.6 + 0.4 * low_r_norm_grid
    disturb_mod_low = np.clip(disturb_mod_low * radial_preserve, 0.1, 1.0)

    upscale_kernel = np.ones((scale_factor, scale_factor), dtype=np.float32)
    return np.kron(disturb_mod_low, upscale_kernel)[:n_r, :n_phi].astype(np.float32)


def _compose_disk_texture_from_fields(temp_base: np.ndarray, temp_struct: np.ndarray,
                                      density: np.ndarray, az_hotspot: np.ndarray,
                                      edge: np.ndarray, color_temp: float) -> np.ndarray:
    """从温度/密度场合成最终 RGBA 纹理。"""
    density = density * edge[:, None]
    density = np.clip(density / (np.percentile(density, 98) + 1e-6), 0, 1)

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

    t_factor = (color_temp - 4500) / (6500 - 2700)
    T_min = 2000 + t_factor * 1000
    T_max = 9000 + t_factor * 3000

    temp_aniso = np.clip(temperature_field * (0.9 + 0.25 * az_hotspot), 0, 1)
    T_K = T_min + temp_aniso * (T_max - T_min)
    bb_color = _blackbody_rgb(T_K)
    bb_color[:, :, 2] = np.minimum(bb_color[:, :, 2], bb_color[:, :, 0])

    luminosity = np.clip(np.sqrt(temp_aniso), 0, 1)

    tex = np.zeros((temp_base.shape[0], temp_base.shape[1], 4), dtype=np.float32)
    tex[:, :, 0] = np.clip(bb_color[:, :, 0] * luminosity, 0, 1)
    tex[:, :, 1] = np.clip(bb_color[:, :, 1] * luminosity, 0, 1)
    tex[:, :, 2] = np.clip(bb_color[:, :, 2] * luminosity, 0, 1)
    tex[:, :, 3] = np.clip(density, 0, 1)
    return tex


def _roll_rows(field: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """按行循环平移二维/三维场。"""
    rolled = np.empty_like(field)
    if field.ndim == 2:
        for ri, shift in enumerate(shifts):
            rolled[ri, :] = np.roll(field[ri, :], -int(shift))
        return rolled
    if field.ndim == 3:
        for ri, shift in enumerate(shifts):
            rolled[ri, :, :] = np.roll(field[ri, :, :], -int(shift), axis=0)
        return rolled
    raise ValueError(f"Unsupported field ndim: {field.ndim}")


def _compute_rotation_pixels(omega_rows: np.ndarray, t_offset: float, n_phi: int) -> np.ndarray:
    return (t_offset * omega_rows / (2 * np.pi) * n_phi).astype(np.int32)


def _compute_upscaled_rotation_pixels(omega_rows: np.ndarray, t_offset: float, n_phi: int,
                                      scale_factor: int = 2) -> np.ndarray:
    scale_factor = _validate_disk_generation_scale(scale_factor)
    low_n_phi = n_phi // scale_factor
    low_omega_rows = omega_rows[::scale_factor]
    low_shifts = (t_offset * low_omega_rows / (2 * np.pi) * low_n_phi).astype(np.int32)
    return np.repeat(low_shifts * scale_factor, scale_factor)[:omega_rows.shape[0]]


def build_disk_texture_rotating_state(n_phi: int = 1024, n_r: int = 512, seed: int = 42,
                                      r_inner: float = 2.0, r_outer: float = 3.5,
                                      enable_rt: bool = True, color_temp: float = None,
                                      generation_scale: int = 2) -> DiskTextureRotatingState:
    """预计算 `parametric` 旋转纹理的静态状态。"""
    generation_scale = _validate_disk_generation_scale(generation_scale)

    if color_temp is None:
        color_temp = DISK_COLOR_TEMPERATURE

    rng = np.random.default_rng(seed)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r_norm = np.linspace(0, 1, n_r)
    phi_grid_base, r_norm_grid = np.meshgrid(phi, r_norm)

    r_vals = r_inner + (r_outer - r_inner) * r_norm_grid
    disk_area = (r_outer ** 2 - r_inner ** 2) / 10.0
    omega_grid = np.sqrt(0.5 / (r_vals ** 3 + 1e-6))

    temp_base = _generate_temperature_base(rng, n_r, n_phi, r_norm_grid)
    spiral, spiral_temp = _generate_spiral_arms(
        rng, n_r, n_phi, phi_grid_base, r_norm_grid, 0.0, None, generation_scale=generation_scale
    )
    turbulence, kep_shift_pixels, turb_temp = _generate_turbulence(
        rng, n_r, n_phi, r_norm_grid, 0.0, None, generation_scale=generation_scale
    )
    arcs, arcs_temp = _generate_filaments(
        rng, n_r, n_phi, phi_grid_base, r_norm_grid, disk_area, 0.0, None, generation_scale=generation_scale
    )
    rt_spikes, rt_temp = _generate_rt_spikes(
        rng, n_r, n_phi, phi_grid_base, r_norm_grid, disk_area, enable_rt, 0.0, None, generation_scale=generation_scale
    )
    hotspot, hotspot_temp = _generate_hotspots(rng, n_r, n_phi, phi_grid_base, r_norm_grid, disk_area, 0.0, None)
    az_hotspot = _generate_azimuthal_hotspot(
        rng, n_r, n_phi, phi_grid_base, r_norm_grid, 0.0, None, generation_scale=generation_scale
    )
    disturb_mod = _generate_disturbance_mod(
        rng, n_r, n_phi, kep_shift_pixels, r_norm_grid, 0.0, None, generation_scale=generation_scale
    )

    return DiskTextureRotatingState(
        n_phi=n_phi,
        n_r=n_r,
        seed=seed,
        generation_scale=generation_scale,
        r_inner=r_inner,
        r_outer=r_outer,
        enable_rt=enable_rt,
        color_temp=float(color_temp),
        omega_rows=omega_grid[:, 0].astype(np.float32),
        edge=compute_edge_alpha(n_r).astype(np.float32),
        temp_base=temp_base.astype(np.float32),
        spiral=spiral.astype(np.float32),
        spiral_temp=spiral_temp.astype(np.float32),
        turbulence=turbulence.astype(np.float32),
        turb_temp=turb_temp.astype(np.float32),
        arcs=arcs.astype(np.float32),
        arcs_temp=arcs_temp.astype(np.float32),
        rt_spikes=rt_spikes.astype(np.float32),
        rt_temp=rt_temp.astype(np.float32),
        hotspot=hotspot.astype(np.float32),
        hotspot_temp=hotspot_temp.astype(np.float32),
        az_hotspot=az_hotspot.astype(np.float32),
        disturb_mod=disturb_mod.astype(np.float32),
    )


def _generate_disk_texture_rotating_from_state(state: DiskTextureRotatingState,
                                               t_offset: float = 0.0,
                                               color_temp: float = None) -> np.ndarray:
    """基于预计算状态生成某一时刻的旋转纹理。"""
    if color_temp is None:
        color_temp = state.color_temp

    full_res_rot = _compute_rotation_pixels(state.omega_rows, t_offset, state.n_phi)
    low_res_rot = _compute_upscaled_rotation_pixels(
        state.omega_rows, t_offset, state.n_phi, scale_factor=state.generation_scale
    )

    temp_base = _roll_rows(state.temp_base, full_res_rot)
    spiral = _roll_rows(state.spiral, low_res_rot)
    spiral_temp = _roll_rows(state.spiral_temp, low_res_rot)
    turbulence = _roll_rows(state.turbulence, low_res_rot)
    turb_temp = _roll_rows(state.turb_temp, low_res_rot)
    arcs = _roll_rows(state.arcs, low_res_rot)
    arcs_temp = _roll_rows(state.arcs_temp, low_res_rot)
    rt_spikes = _roll_rows(state.rt_spikes, low_res_rot)
    rt_temp = _roll_rows(state.rt_temp, low_res_rot)
    hotspot = _roll_rows(state.hotspot, full_res_rot)
    hotspot_temp = _roll_rows(state.hotspot_temp, full_res_rot)
    az_hotspot = _roll_rows(state.az_hotspot, low_res_rot)
    disturb_mod = _roll_rows(state.disturb_mod, low_res_rot)

    temp_struct = spiral_temp + turb_temp + arcs_temp + rt_temp + hotspot_temp
    rt_weight = 0.20 if state.enable_rt else 0.0
    density = 0.15 + 0.10 * spiral + 0.30 * turbulence + 0.20 * hotspot + 0.30 * arcs + rt_weight * rt_spikes

    density = density * disturb_mod
    temp_struct = temp_struct * disturb_mod

    return _compose_disk_texture_from_fields(temp_base, temp_struct, density, az_hotspot, state.edge, color_temp)




def _tileable_noise(shape: Tuple[int, int], rng: np.random.Generator, freq_u: int = 6, freq_v: int = 6) -> np.ndarray:
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


def _periodic_pixel_noise(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """生成像素级白噪声，保证 phi 方向周期性（首尾相接）。

    用于湍流的 pixel_noise 层，提供高频颗粒感，同时保证纹理无缝。
    """
    h, w = shape
    noise = rng.random((h, w)).astype(np.float32)
    noise[:, -1] = noise[:, 0]  # 强制周期性：phi=0 和 phi=2π 相同
    return noise * 2 - 1  # 返回 [-1, 1] 范围


def _fbm_noise(shape: Tuple[int, int], rng: np.random.Generator, octaves: int = 4, persistence: float = 0.5, base_scale: int = 1, wrap_u: bool = False) -> np.ndarray:
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



def _blend_azimuthal_seam(tex: np.ndarray, seam_width: int = 64) -> np.ndarray:
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


def generate_disk_mipmaps(base_tex: np.ndarray, levels: int = 4) -> np.ndarray:
    """生成吸积盘纹理的 mipmap 金字塔"""
    mips = [base_tex.copy()]
    for _ in range(levels):
        h, w = mips[-1].shape[:2]
        if h < 2 or w < 2:
            break
        new_h, new_w = h // 2, w // 2
        down = np.zeros((new_h, new_w, 4), dtype=np.float32)
        down = (mips[-1][0::2, 0::2] + mips[-1][1::2, 0::2] +
                mips[-1][0::2, 1::2] + mips[-1][1::2, 1::2]) / 4.0
        mips.append(down.astype(np.float32))
    return mips


def compute_disk_texture_resolution(width: int, height: int, cam_pos: List[float], fov: float, r_inner: float, r_outer: float, rs: float = 1.0) -> Tuple[int, int]:
    """
    根据相机参数计算吸积盘纹理分辨率。
    n_phi: 基于视角覆盖的角分辨率，每个像素约 1 个 phi 样本
    n_r: 基于径向覆盖的分辨率，每个径向单位约 0.5 个样本
    """
    camera_distance = math.sqrt(cam_pos[0]**2 + cam_pos[1]**2 + cam_pos[2]**2)

    disk_angular_radius = math.atan(r_outer / camera_distance)
    disk_angular_extent = 2 * disk_angular_radius
    screen_fraction = fov * math.pi / 180.0

    n_phi = int(width * (disk_angular_extent / screen_fraction))
    n_r = int(height * (disk_angular_radius / screen_fraction) * 0.5)

    n_phi = max(256, n_phi)
    n_r = max(128, n_r)

    n_phi = n_phi + (16 - n_phi % 16) % 16
    n_r = n_r + (16 - n_r % 16) % 16

    return n_phi, n_r


def load_cached_disk_texture(width: Optional[int] = None, height: Optional[int] = None, cam_pos: Optional[List[float]] = None, fov: Optional[float] = None,
                               seed: int = 42, r_inner: float = 2.0, r_outer: float = 3.5, force: bool = False,
                               generation_scale: int = 2) -> np.ndarray:
    """
    加载或生成吸积盘纹理（带缓存）。
    - width, height, cam_pos, fov: 用于计算纹理分辨率
    - seed: 随机种子
    - r_inner, r_outer: 吸积盘内外半径
    - force: 强制重新生成，忽略缓存
    返回 (n_r, n_phi, 4) float32
    """
    generation_scale = _validate_disk_generation_scale(generation_scale)

    if width and height and cam_pos and fov:
        n_phi, n_r = compute_disk_texture_resolution(width, height, cam_pos, fov, r_inner, r_outer)
    else:
        n_phi, n_r = 1024, 512

    cache_dir = "output/.disk_texture_cache"
    cache_key = f"disk_{r_inner:.2f}_{r_outer:.2f}_{seed}_{n_phi}x{n_r}_scale{generation_scale}.npy"
    cache_path = os.path.join(cache_dir, cache_key)

    if not force and os.path.exists(cache_path):
        print(f"Loading cached disk texture: {cache_key}")
        return np.load(cache_path)

    print(f"Generating disk texture: r_inner={r_inner}, r_outer={r_outer}, seed={seed}, n_phi={n_phi}, n_r={n_r}")
    tex = generate_disk_texture(
        n_phi=n_phi, n_r=n_r, seed=seed, r_inner=r_inner, r_outer=r_outer,
        generation_scale=generation_scale,
    )

    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, tex)
    print(f"Cached to: {cache_path}")
    return tex



def _generate_spiral_arms(rng: np.random.Generator, n_r: int, n_phi: int, phi_grid: np.ndarray, r_norm_grid: np.ndarray,
                           t_offset: float = 0.0, omega_grid: np.ndarray = None,
                           generation_scale: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成螺旋臂密度和温度贡献
    返回：(spiral, temp_contribution)

    优化：使用 2x 低分辨率生成 + upscale，获得约 5x 加速比
    """
    if not ENABLE_DISK_SPIRAL_ARMS:
        zeros = np.zeros((n_r, n_phi), dtype=np.float32)
        return zeros, zeros

    # ===== 性能优化：2x 低分辨率生成 + upscale =====
    scale_factor = _validate_disk_generation_scale(generation_scale)
    low_n_r = n_r // scale_factor
    low_n_phi = n_phi // scale_factor

    # 从传入的网格降级采样（保留旋转信息）
    low_phi_grid = phi_grid[::scale_factor, ::scale_factor]
    low_r_norm_grid = r_norm_grid[::scale_factor, ::scale_factor]

    n_arms = rng.integers(2, 5)
    n_from_center = rng.integers(2, 4)

    # 在低分辨率下生成
    low_spiral = np.zeros((low_n_r, low_n_phi), dtype=np.float32)
    low_temp_contribution = np.zeros((low_n_r, low_n_phi), dtype=np.float32)

    used_angles = []
    for arm_idx in tqdm(range(n_arms), desc="Spiral arms (2x lowres)", leave=False):
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

        rotations = rng.uniform(2.5, 5.0)
        base_width = rng.uniform(0.2, 0.4)
        arm_delta_T = rng.uniform(0.1, 0.3)

        r_length = rotations / 6.0 * (1.0 - r_start)
        r_length = min(r_length, 1.0 - r_start)

        # 每条螺旋臂由 4-8 个 sub-arm 段组成，每段之间有明显的间隙
        sub_arm_count = rng.integers(4, 9)
        sub_arm_fill = rng.uniform(0.4, 0.6)  # sub-arm 占总长度的比例（40-60%）
        sub_arm_lengths = rng.uniform(0.08, 0.20, sub_arm_count)
        sub_arm_lengths = sub_arm_lengths / sub_arm_lengths.sum() * r_length * sub_arm_fill

        # sub-arm 的起始径向位置 - 大间隙让分段更明显
        sub_r_starts = np.zeros(sub_arm_count)
        for j in range(1, sub_arm_count):
            gap = rng.uniform(0.08, 0.15)  # 大间隙
            sub_r_starts[j] = sub_r_starts[j-1] + sub_arm_lengths[j-1] + gap
        sub_r_starts += r_start

        # sub-arm 的宽度和强度变化 - 增加对比度
        sub_widths = base_width * rng.uniform(0.3, 2.5, sub_arm_count)
        sub_widths = np.clip(sub_widths, 0.06, 1.2)
        sub_intensities = rng.uniform(0.1, 0.7, sub_arm_count)

        # 预先生成 arm_noise（在 sub-arm 循环外，避免重复计算）
        arm_noise = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=3, freq_v=2)

        for j in range(sub_arm_count):
            sr = sub_r_starts[j]
            sr_len = sub_arm_lengths[j]
            sr_width = sub_widths[j]
            sr_int = sub_intensities[j]
            sr_end = sr + sr_len

            # 螺旋臂角度公式
            arm_angle = low_phi_grid - base_angle + low_r_norm_grid * rotations * 2 * np.pi

            # 宽度调制
            width_mod = 0.2 + 1.5 * arm_noise
            width_mod = np.clip(width_mod, 0.15, 3.0)

            arm_kappa = 1.0 / (sr_width ** 2) * 1.5
            arm_val = np.exp(arm_kappa * (np.cos(arm_angle) - 1) * width_mod)

            # 径向 mask - 使用硬边界，减少 fade 效果
            mask = (low_r_norm_grid >= sr) & (low_r_norm_grid <= sr_end)
            arm_val = np.where(mask, arm_val, 0)

            # 强度调制 - 降低断裂效果，让 sub-arm 更连续
            intensity_mod = 0.1 + 0.9 * (arm_noise ** 0.15)

            # 轻微的边缘软化（比之前小）
            fade_edge = 0.02
            fade_in = np.clip((low_r_norm_grid - sr) / fade_edge, 0, 1)
            fade_out = np.clip((sr_end - low_r_norm_grid) / fade_edge, 0, 1)
            arm_val *= fade_in * fade_out * sr_int * intensity_mod

            low_spiral += arm_val
            low_temp_contribution += arm_val * arm_delta_T

    low_spiral = np.clip(low_spiral / (np.max(low_spiral) + 1e-6), 0, 1)

    # 使用 np.kron 进行 upscale
    upscale_kernel = np.ones((scale_factor, scale_factor), dtype=np.float32)
    spiral = np.kron(low_spiral, upscale_kernel)
    temp_contribution = np.kron(low_temp_contribution, upscale_kernel)

    # 裁剪到目标尺寸（防止整除时尺寸不匹配）
    spiral = spiral[:n_r, :n_phi]
    temp_contribution = temp_contribution[:n_r, :n_phi]

    return spiral, temp_contribution


def _generate_turbulence(rng: np.random.Generator, n_r: int, n_phi: int, r_norm_grid: np.ndarray,
                          t_offset: float = 0.0, omega_grid: np.ndarray = None,
                          generation_scale: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成云雾/湍流密度和温度贡献
    返回：(turbulence, kep_shift_pixels, temp_contribution)

    优化：使用 2x 低分辨率生成 + upscale，获得约 2-3x 加速比
    """
    # ===== 性能优化：2x 低分辨率生成 + upscale =====
    scale_factor = _validate_disk_generation_scale(generation_scale)
    low_n_r = n_r // scale_factor
    low_n_phi = n_phi // scale_factor

    # 从传入的 r_norm_grid 降级采样
    low_r_norm_grid = r_norm_grid[::scale_factor, ::scale_factor]

    shear_strength = rng.uniform(3.0, 6.0)
    # 低分辨率下的开普勒剪切
    kep_shear_low = shear_strength * (1.0 / (low_r_norm_grid + 0.3) ** 1.5 - 0.8)
    kep_shear_low = np.clip(kep_shear_low, 0, shear_strength * 8)
    kep_shift_pixels_low = (kep_shear_low / (2 * np.pi) * low_n_phi).astype(int)
    max_shift_low = low_n_phi // 4
    kep_shift_pixels_low = np.clip(kep_shift_pixels_low, -max_shift_low, max_shift_low)

    # 在低分辨率下生成 5 层噪声
    turbulence_coarse = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=8, freq_v=4)
    turbulence_mid = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=24, freq_v=12)
    turbulence_fine = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=80, freq_v=40)
    turbulence_extra = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=200, freq_v=100)
    turbulence_ultra = _tileable_noise((low_n_r, low_n_phi), rng, freq_u=400, freq_v=200)

    # 应用开普勒剪切滚动（低分辨率）
    for layer in [turbulence_coarse, turbulence_mid, turbulence_fine, turbulence_extra, turbulence_ultra]:
        for ri in range(low_n_r):
            layer[ri, :] = np.roll(layer[ri, :], kep_shift_pixels_low[ri, 0])

    # 动态旋转支持（低分辨率）
    rotation_pixels_low = None
    if t_offset != 0.0 and omega_grid is not None:
        # 降级采样 omega_grid 到低分辨率
        omega_grid_low = omega_grid[::scale_factor, ::scale_factor]
        rotation_pixels_low = (t_offset * omega_grid_low / (2 * np.pi) * low_n_phi).astype(int)
        for layer in [turbulence_coarse, turbulence_mid, turbulence_fine, turbulence_extra, turbulence_ultra]:
            for ri in range(low_n_r):
                layer[ri, :] = np.roll(layer[ri, :], -rotation_pixels_low[ri, 0])

    # 像素级高频噪声（低分辨率）
    pixel_noise = _periodic_pixel_noise((low_n_r, low_n_phi), rng)

    # 对 pixel_noise 应用开普勒旋转（t_offset != 0 时）
    if rotation_pixels_low is not None:
        for ri in range(low_n_r):
            pixel_noise[ri, :] = np.roll(pixel_noise[ri, :], -rotation_pixels_low[ri, 0])

    # 湍流权重：多层噪声叠加
    turbulence_low = (0.08 * turbulence_coarse + 0.15 * turbulence_mid
                      + 0.25 * turbulence_fine + 0.22 * turbulence_extra
                      + 0.18 * turbulence_ultra + 0.12 * np.clip(pixel_noise, 0, 1))

    # upscale
    upscale_kernel = np.ones((scale_factor, scale_factor), dtype=np.float32)
    turbulence = np.kron(turbulence_low, upscale_kernel)[:n_r, :n_phi]

    temp_contribution = 0.05 * np.clip(turbulence, 0, 1)

    # 高分辨率 kep_shift_pixels（用于后续处理）
    kep_shear = shear_strength * (1.0 / (r_norm_grid + 0.3) ** 1.5 - 0.8)
    kep_shear = np.clip(kep_shear, 0, shear_strength * 8)
    kep_shift_pixels = (kep_shear / (2 * np.pi) * n_phi).astype(int)
    max_shift = n_phi // 4
    kep_shift_pixels = np.clip(kep_shift_pixels, -max_shift, max_shift)

    return turbulence, kep_shift_pixels, temp_contribution


def _generate_filaments(rng: np.random.Generator, n_r: int, n_phi: int, phi_grid: np.ndarray, r_norm_grid: np.ndarray, disk_area: float,
                       t_offset: float = 0.0, omega_grid: np.ndarray = None,
                       generation_scale: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成细丝（filaments）密度和温度贡献
    物理意义：吸积盘中的丝状结构，可能是磁重联或剪切流形成的细长条纹
    特征：沿角度方向延伸（长条状），径向很窄，由多个 sub-filament 接续而成

    数量说明：真实吸积盘中细丝结构约 20-50 条，这里用 30-60 条保证可见性

    优化：使用 2x 低分辨率生成 + upscale，获得约 5x 加速比
    """
    # ===== 性能优化：2x 低分辨率生成 + upscale =====
    # 在低分辨率下生成 filaments，然后 upscale 到目标分辨率
    # 测试表明：2x 低分辨率 + upscale 可获得 5x 加速，MSE 仅 0.044
    scale_factor = _validate_disk_generation_scale(generation_scale)
    low_n_r = n_r // scale_factor
    low_n_phi = n_phi // scale_factor

    # 从传入的网格降级采样（保留旋转信息）
    low_phi_grid = phi_grid[::scale_factor, ::scale_factor]
    low_r_norm_grid = r_norm_grid[::scale_factor, ::scale_factor]

    # 细丝数量：150-300 条（每条由多个 sub-filament 组成）
    arc_count = int(rng.uniform(150, 300))
    # 每条细丝的 sub-filament 数量：2-4 个
    sub_filament_counts = rng.integers(2, 5, arc_count)

    arc_phi_starts = rng.uniform(0, 2 * np.pi, arc_count)
    r_positions = rng.uniform(0.05, 0.95, arc_count)
    arc_rs = 0.05 + r_positions ** 0.6 * 0.9
    # 细丝径向宽度：0.002-0.008（适中宽度）
    arc_r_widths = rng.uniform(0.002, 0.008, arc_count)
    # 细丝总角度长度：0.5-1.2（约 180°-430°）
    arc_lengths = rng.uniform(0.5, 1.2, arc_count)

    arc_intensities = rng.uniform(0.7, 1.0, arc_count)  # 提高细丝强度
    arc_delta_Ts = 0.3 + 0.6 * rng.power(0.3, arc_count)  # 提高温度贡献范围：0.3-0.9

    print(f"Generating {arc_count} filaments with sub-segments (2x lowres + upscale)...")

    # 在低分辨率下生成
    low_arcs = np.zeros((low_n_r, low_n_phi), dtype=np.float32)
    low_temp_contribution = np.zeros((low_n_r, low_n_phi), dtype=np.float32)

    # 逐条生成细丝，每条细丝由多个 sub-filament 接续而成
    for i in tqdm(range(arc_count), desc="Filaments", leave=False):
        # 细丝基础参数
        base_phi = arc_phi_starts[i]
        base_r = arc_rs[i]
        base_width = arc_r_widths[i]
        total_length = arc_lengths[i]
        intensity = arc_intensities[i]
        delta_T = arc_delta_Ts[i]

        # 生成 sub-filament 参数
        sub_count = sub_filament_counts[i]
        sub_fill = rng.uniform(0.35, 0.55)  # sub-filament 占总长度的比例
        sub_lengths = rng.uniform(0.08, 0.20, sub_count)
        # 归一化 sub 长度，使其总和等于 total_length * sub_fill
        sub_lengths = sub_lengths / sub_lengths.sum() * total_length * sub_fill

        # sub-filament 的起始角度（沿细丝方向分布）- 大间隙
        sub_starts = np.zeros(sub_count)
        sub_starts[0] = base_phi
        for j in range(1, sub_count):
            gap = rng.uniform(0.08, 0.20)  # 大间隙
            sub_starts[j] = sub_starts[j-1] + sub_lengths[j-1] + gap

        # sub-filament 的宽度和强度变化 - 增加对比度
        sub_widths = base_width * rng.uniform(0.3, 3.0, sub_count)
        sub_widths = np.clip(sub_widths, 0.001, 0.025)
        sub_intensities = intensity * rng.uniform(0.15, 1.0, sub_count)

        # 生成每个 sub-filament
        for j in range(sub_count):
            sub_phi = sub_starts[j]
            sub_len = sub_lengths[j]
            sub_w = sub_widths[j]
            sub_int = sub_intensities[j]

            # 角度剖面
            phi_range = sub_len / (base_r + 0.01)
            phi_half_width = np.maximum(phi_range * 0.7, 0.2)
            kappa = 1.5 / (phi_half_width ** 2)

            sub_val = np.exp(kappa * (np.cos(low_phi_grid - sub_phi) - 1))

            # 径向剖面
            r_diff = low_r_norm_grid - base_r
            r_prof = np.exp(-0.5 * (r_diff / sub_w) ** 2)

            low_arcs += sub_val * r_prof * sub_int
            low_temp_contribution += sub_val * r_prof * sub_int * delta_T * 0.7

    # 使用 np.kron 进行 upscale
    upscale_kernel = np.ones((scale_factor, scale_factor), dtype=np.float32)
    arcs = np.kron(low_arcs, upscale_kernel)
    temp_contribution = np.kron(low_temp_contribution, upscale_kernel)

    # 裁剪到目标尺寸（防止整除时尺寸不匹配）
    arcs = arcs[:n_r, :n_phi]
    temp_contribution = temp_contribution[:n_r, :n_phi]

    arcs = np.clip(arcs, 0, 1)
    temp_contribution = np.clip(temp_contribution, 0, arcs * 0.5)
    return arcs, temp_contribution


def _generate_rt_spikes(rng: np.random.Generator, n_r: int, n_phi: int, phi_grid: np.ndarray, r_norm_grid: np.ndarray, disk_area: float, enable_rt: bool,
                       t_offset: float = 0.0, omega_grid: np.ndarray = None,
                       generation_scale: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 Rayleigh-Taylor 不稳定性密度和温度贡献
    返回：(rt_spikes, temp_contribution)

    优化：使用 2x 低分辨率生成 + upscale，获得约 4-5x 加速比
    """
    # 如果禁用 RT，返回零数组
    if not enable_rt:
        return np.zeros((n_r, n_phi), dtype=np.float32), np.zeros((n_r, n_phi), dtype=np.float32)

    # ===== 性能优化：2x 低分辨率生成 + upscale =====
    scale_factor = _validate_disk_generation_scale(generation_scale)
    low_n_r = n_r // scale_factor
    low_n_phi = n_phi // scale_factor

    # 从传入的网格降级采样（保留旋转信息）
    low_phi_grid = phi_grid[::scale_factor, ::scale_factor]
    low_r_norm_grid = r_norm_grid[::scale_factor, ::scale_factor]

    # RT 不稳定性主要出现在内圈，数量增加
    rt_count = int(rng.uniform(15, 30) * disk_area * 0.8)

    rt_phis = rng.uniform(0, 2 * np.pi, rt_count)
    # RT 位置偏内圈，更多集中在 r_norm < 0.3 区域 - 使用幂次分布偏向内圈
    rt_r_bases = np.power(rng.uniform(0.01, 0.15, rt_count), 1.5)  # 幂次分布偏向内圈
    rt_phi_widths = rng.uniform(0.08, 0.20, rt_count)  # 更窄，更集中
    rt_r_lengths = rng.uniform(0.08, 0.20, rt_count)  # 更长
    rt_intensities = rng.uniform(0.8, 1.0, rt_count)  # 提高强度
    rt_delta_Ts = rng.uniform(0.5, 1.2, rt_count)  # 提高温度贡献

    # 在低分辨率下生成
    rt_spikes = np.zeros((low_n_r, low_n_phi), dtype=np.float32)
    temp_contribution = np.zeros((low_n_r, low_n_phi), dtype=np.float32)

    for i in range(rt_count):
        rt_phi_kappa = 1.0 / (rt_phi_widths[i] ** 2) * 1.5
        rt_val = np.exp(rt_phi_kappa * (np.cos(low_phi_grid - rt_phis[i]) - 1))

        rt_r_diff = low_r_norm_grid - rt_r_bases[i]
        r_fade_out = np.clip(rt_r_lengths[i] * 2 - rt_r_diff, 0, 1)
        r_fade_in = np.clip((low_r_norm_grid - rt_r_bases[i]) / (rt_r_lengths[i] * 0.3), 0, 1)
        rt_r_profile = np.exp(-0.5 * (rt_r_diff / (rt_r_lengths[i] * 0.4)) ** 2) * r_fade_out * r_fade_in

        rt_val *= rt_r_profile * rt_intensities[i]
        rt_spikes += rt_val
        temp_contribution += rt_val * rt_delta_Ts[i]

    rt_spikes = np.clip(rt_spikes, 0, 1)

    # 使用 np.kron 进行 upscale
    upscale_kernel = np.ones((scale_factor, scale_factor), dtype=np.float32)
    rt_spikes = np.kron(rt_spikes, upscale_kernel)[:n_r, :n_phi]
    temp_contribution = np.kron(temp_contribution, upscale_kernel)[:n_r, :n_phi]

    return rt_spikes, temp_contribution


def _generate_azimuthal_hotspot(rng: np.random.Generator, n_r: int, n_phi: int, phi_grid: np.ndarray, r_norm_grid: np.ndarray,
                                 t_offset: float = 0.0, omega_grid: np.ndarray = None,
                                 generation_scale: int = 2) -> np.ndarray:
    """
    生成方位热点（低频正弦 + 噪声，自转流动感）
    返回：az_hotspot

    优化：使用 2x 低分辨率生成 + upscale，获得约 2-3x 加速比
    """
    # ===== 性能优化：2x 低分辨率生成 + upscale =====
    scale_factor = _validate_disk_generation_scale(generation_scale)
    low_n_r = n_r // scale_factor
    low_n_phi = n_phi // scale_factor

    # 从传入的网格降级采样（保留旋转信息）
    low_phi_grid = phi_grid[::scale_factor, ::scale_factor]
    low_r_norm_grid = r_norm_grid[::scale_factor, ::scale_factor]

    az_freq = rng.integers(2, 5)
    shear = low_r_norm_grid ** 1.2 * rng.uniform(2.0, 4.0)
    az_wave = 0.5 + 0.5 * np.sin((low_phi_grid + shear) * az_freq)
    az_noise = _fbm_noise((low_n_r, low_n_phi), rng, octaves=3, persistence=0.5, base_scale=3, wrap_u=True)

    # 对 az_noise 应用开普勒旋转（t_offset != 0 时）
    if t_offset != 0.0 and omega_grid is not None:
        omega_grid_low = omega_grid[::scale_factor, ::scale_factor]
        rotation_pixels_low = (t_offset * omega_grid_low / (2 * np.pi) * low_n_phi).astype(int)
        for ri in range(low_n_r):
            az_noise[ri, :] = np.roll(az_noise[ri, :], -rotation_pixels_low[ri, 0])

    az_hotspot_low = az_wave * az_noise

    # upscale
    upscale_kernel = np.ones((scale_factor, scale_factor), dtype=np.float32)
    az_hotspot = np.kron(az_hotspot_low, upscale_kernel)[:n_r, :n_phi]
    return az_hotspot


def _apply_disturbance(rng: np.random.Generator, n_r: int, n_phi: int, density: np.ndarray,
                        temp_struct: np.ndarray, kep_shift_pixels: np.ndarray,
                        r_norm_grid: np.ndarray, t_offset: float = 0.0,
                        omega_grid: np.ndarray = None, generation_scale: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply turbulence disturbance to density and temperature fields.
    Returns: (density, temp_struct)

    Args:
        t_offset: 时间偏移，用于动态旋转
        omega_grid: 开普勒角速度网格，用于计算旋转量

    优化：使用 2x 低分辨率生成 + upscale，获得约 1.5-2x 加速比
    """
    disturb_mod = _generate_disturbance_mod(
        rng, n_r, n_phi, kep_shift_pixels, r_norm_grid, t_offset, omega_grid,
        generation_scale=generation_scale,
    )
    density = density * disturb_mod
    temp_struct = temp_struct * disturb_mod
    return density, temp_struct


def _generate_hotspots(rng: np.random.Generator, n_r: int, n_phi: int, phi_grid: np.ndarray, r_norm_grid: np.ndarray, disk_area: float,
                      t_offset: float = 0.0, omega_grid: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成温度热点密度和温度贡献
    物理意义：吸积盘中的局部高温区域（如磁重联、激波碰撞形成的亮斑）
    特征：近似圆形或椭圆形的斑点，径向和角度宽度相近

    数量说明：真实吸积盘中观测到的热点约数个 - 数十个，这里用 20-40 个保证可见性
    """
    # 热点数量：20-40 个（物理合理范围）
    hotspot_count = int(rng.uniform(20, 40))
    hotspot_delta_Ts = 0.5 + 2.5 * rng.power(0.4, hotspot_count)

    h_phis = rng.uniform(0, 2 * np.pi, hotspot_count)
    r_rands = rng.uniform(0, 1, hotspot_count)
    h_rs = 0.1 + r_rands ** 0.6 * 0.85
    # 热点角度宽度：0.08-0.20（约 30°-70°），较宽形成斑点
    h_phi_widths = rng.uniform(0.08, 0.20, hotspot_count)
    # 热点径向宽度：0.02-0.05，与角度宽度相近，形成近似圆形的斑点
    h_r_widths = 0.02 + rng.uniform(0, 0.03, hotspot_count)
    h_intensities = 0.3 + (1 - h_rs) * 0.6 + rng.uniform(0, 0.1, hotspot_count)

    print(f"Generating {hotspot_count} hotspots...")
    hotspot = np.zeros((n_r, n_phi), dtype=np.float32)
    batch_size = 400

    for batch_start in tqdm(range(0, hotspot_count, batch_size), desc="Hotspots", leave=False):
        batch_end = min(batch_start + batch_size, hotspot_count)

        h_ps = h_phis[batch_start:batch_end, None, None]
        hs = h_rs[batch_start:batch_end, None, None]
        hp_ws = h_phi_widths[batch_start:batch_end, None, None]
        hr_ws = h_r_widths[batch_start:batch_end, None, None]
        h_ints = h_intensities[batch_start:batch_end, None, None]

        kappa = 1.0 / (hp_ws ** 2) * 1.5
        h_batch = np.exp(kappa * (np.cos(phi_grid[None, :, :] - h_ps) - 1.0))
        r_diff = r_norm_grid[None, :, :] - hs
        h_batch *= np.exp(-0.5 * (r_diff / hr_ws) ** 2)
        h_batch *= h_ints

        hotspot += np.sum(h_batch, axis=0)

    hotspot = np.clip(hotspot, 0, 1)
    temp_contribution = 0.12 * hotspot
    return hotspot, temp_contribution


# ============================================================================
# 单实例生成函数（供实体生命周期系统使用）
# ============================================================================

def _spawn_single_filament(rng: np.random.Generator, n_r: int, n_phi: int,
                           r_norm_all: np.ndarray, omega_all: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate a single filament instance for entity lifecycle system.

    Creates one filament composed of 2-4 sub-segments, computing density and
    temperature contributions only on affected rows (sparse representation).
    Parameters match _generate_filaments inner loop statistics.

    Args:
        rng: numpy random generator
        n_r: total number of radial rows
        n_phi: number of azimuthal columns
        r_norm_all: normalized radial position for each row, shape (n_r,)
        omega_all: Keplerian angular velocity for each row, shape (n_r,)

    Returns:
        (row_indices, phi_density, phi_temp, omega):
        - row_indices: int array of affected row indices, shape (n_affected,)
        - phi_density: density contribution, shape (n_affected, n_phi)
        - phi_temp: temperature contribution, shape (n_affected, n_phi)
        - omega: Keplerian angular velocity at entity center (rad/s)

    Physical Meaning:
        Represents a magnetic filament — a thin, arc-shaped structure formed by
        magnetic reconnection or shear flow in the accretion disk. Each filament
        is narrow radially (2-5 rows) but extends azimuthally (~180-430 degrees).
    """
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    base_phi = float(rng.uniform(0, 2 * np.pi))
    r_pos = float(rng.uniform(0.05, 0.95))
    base_r = 0.05 + r_pos ** 0.6 * 0.9
    base_width = float(rng.uniform(0.002, 0.008))
    total_length = float(rng.uniform(0.5, 1.2))
    intensity = float(rng.uniform(0.7, 1.0))
    delta_T = 0.3 + 0.6 * float(rng.power(0.3))

    # 受影响行（5 sigma 截断）
    r_min = base_r - 5 * base_width
    r_max = base_r + 5 * base_width
    row_mask = (r_norm_all >= r_min) & (r_norm_all <= r_max)
    row_indices = np.where(row_mask)[0]

    if len(row_indices) == 0:
        center_idx = int(np.argmin(np.abs(r_norm_all - base_r)))
        row_indices = np.array([center_idx])

    r_subset = r_norm_all[row_indices]
    n_rows = len(row_indices)

    phi_density = np.zeros((n_rows, n_phi), dtype=np.float32)
    phi_temp = np.zeros((n_rows, n_phi), dtype=np.float32)

    sub_count = int(rng.integers(2, 5))
    sub_fill = float(rng.uniform(0.35, 0.55))
    sub_lengths = rng.uniform(0.08, 0.20, sub_count)
    sub_lengths = sub_lengths / sub_lengths.sum() * total_length * sub_fill

    sub_starts = np.zeros(sub_count)
    sub_starts[0] = base_phi
    for j in range(1, sub_count):
        gap = float(rng.uniform(0.08, 0.20))
        sub_starts[j] = sub_starts[j - 1] + sub_lengths[j - 1] + gap

    sub_widths = np.clip(base_width * rng.uniform(0.3, 3.0, sub_count), 0.001, 0.025)
    sub_intensities = intensity * rng.uniform(0.15, 1.0, sub_count)

    for j in range(sub_count):
        phi_range = sub_lengths[j] / (base_r + 0.01)
        phi_half_width = max(phi_range * 0.7, 0.2)
        kappa = 1.5 / (phi_half_width ** 2)
        phi_prof = np.exp(kappa * (np.cos(phi - sub_starts[j]) - 1))

        for k in range(n_rows):
            r_diff = r_subset[k] - base_r
            r_prof = np.exp(-0.5 * (r_diff / (sub_widths[j] + 1e-8)) ** 2)
            phi_density[k] += phi_prof * r_prof * sub_intensities[j]
            phi_temp[k] += phi_prof * r_prof * sub_intensities[j] * delta_T * 0.7

    phi_density = np.clip(phi_density, 0, 1)
    phi_temp = np.clip(phi_temp, 0, phi_density * 0.5)

    center_idx = int(np.argmin(np.abs(r_norm_all - base_r)))
    omega = float(omega_all[center_idx])

    return row_indices, phi_density, phi_temp, omega


def _spawn_single_hotspot(rng: np.random.Generator, n_r: int, n_phi: int,
                          r_norm_all: np.ndarray, omega_all: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate a single hotspot instance for entity lifecycle system.

    Creates one hotspot — an approximately circular bright patch. Parameters
    match _generate_hotspots statistics.

    Args:
        rng: numpy random generator
        n_r: total number of radial rows
        n_phi: number of azimuthal columns
        r_norm_all: normalized radial position for each row, shape (n_r,)
        omega_all: Keplerian angular velocity for each row, shape (n_r,)

    Returns:
        (row_indices, phi_density, phi_temp, omega):
        - row_indices: int array of affected row indices, shape (n_affected,)
        - phi_density: density contribution, shape (n_affected, n_phi)
        - phi_temp: temperature contribution, shape (n_affected, n_phi)
        - omega: Keplerian angular velocity at entity center (rad/s)

    Physical Meaning:
        Represents a localized high-temperature region in the accretion disk,
        caused by magnetic reconnection or shock collisions. Approximately
        circular in shape, with both radial and azimuthal Gaussian profiles.
    """
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    h_phi = float(rng.uniform(0, 2 * np.pi))
    r_rand = float(rng.uniform(0, 1))
    h_r = 0.1 + r_rand ** 0.6 * 0.85
    h_phi_width = float(rng.uniform(0.08, 0.20))
    h_r_width = 0.02 + float(rng.uniform(0, 0.03))
    h_intensity = 0.3 + (1 - h_r) * 0.6 + float(rng.uniform(0, 0.1))
    h_delta_T = 0.5 + 2.5 * float(rng.power(0.4))

    # 受影响行（3 sigma 截断）
    r_min = h_r - 3 * h_r_width
    r_max = h_r + 3 * h_r_width
    row_mask = (r_norm_all >= r_min) & (r_norm_all <= r_max)
    row_indices = np.where(row_mask)[0]

    if len(row_indices) == 0:
        center_idx = int(np.argmin(np.abs(r_norm_all - h_r)))
        row_indices = np.array([center_idx])

    r_subset = r_norm_all[row_indices]
    n_rows = len(row_indices)

    kappa = 1.5 / (h_phi_width ** 2)
    phi_prof = np.exp(kappa * (np.cos(phi - h_phi) - 1))

    phi_density = np.zeros((n_rows, n_phi), dtype=np.float32)
    phi_temp = np.zeros((n_rows, n_phi), dtype=np.float32)

    for k in range(n_rows):
        r_diff = r_subset[k] - h_r
        r_prof = np.exp(-0.5 * (r_diff / (h_r_width + 1e-8)) ** 2)
        phi_density[k] = phi_prof * r_prof * h_intensity
        phi_temp[k] = phi_density[k] * 0.12

    phi_density = np.clip(phi_density, 0, 1)
    phi_temp = np.clip(phi_temp, 0, 1)

    center_idx = int(np.argmin(np.abs(r_norm_all - h_r)))
    omega = float(omega_all[center_idx])

    return row_indices, phi_density, phi_temp, omega


def _spawn_single_rt_spike(rng: np.random.Generator, n_r: int, n_phi: int,
                           r_norm_all: np.ndarray, omega_all: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate a single RT spike instance for entity lifecycle system.

    Creates one Rayleigh-Taylor instability spike. Parameters match
    _generate_rt_spikes statistics. RT spikes are biased toward the inner disk.

    Args:
        rng: numpy random generator
        n_r: total number of radial rows
        n_phi: number of azimuthal columns
        r_norm_all: normalized radial position for each row, shape (n_r,)
        omega_all: Keplerian angular velocity for each row, shape (n_r,)

    Returns:
        (row_indices, phi_density, phi_temp, omega):
        - row_indices: int array of affected row indices, shape (n_affected,)
        - phi_density: density contribution, shape (n_affected, n_phi)
        - phi_temp: temperature contribution, shape (n_affected, n_phi)
        - omega: Keplerian angular velocity at entity center (rad/s)

    Physical Meaning:
        Represents a Rayleigh-Taylor instability — a radial finger-like structure
        near the inner disk edge, where denser outer material plunges inward.
        Biased toward small r_norm (inner disk) with power-law distribution.
    """
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    rt_phi = float(rng.uniform(0, 2 * np.pi))
    rt_r_base = float(np.power(rng.uniform(0.01, 0.15), 1.5))
    rt_phi_width = float(rng.uniform(0.08, 0.20))
    rt_r_length = float(rng.uniform(0.08, 0.20))
    rt_intensity = float(rng.uniform(0.8, 1.0))
    rt_delta_T = float(rng.uniform(0.5, 1.2))

    # 受影响行：从 r_base 向外延伸 2*r_length
    r_min = max(rt_r_base - 0.02, 0.0)
    r_max = rt_r_base + rt_r_length * 2.5
    row_mask = (r_norm_all >= r_min) & (r_norm_all <= r_max)
    row_indices = np.where(row_mask)[0]

    if len(row_indices) == 0:
        center_idx = int(np.argmin(np.abs(r_norm_all - rt_r_base)))
        row_indices = np.array([center_idx])

    r_subset = r_norm_all[row_indices]
    n_rows = len(row_indices)

    rt_phi_kappa = 1.5 / (rt_phi_width ** 2)
    phi_prof = np.exp(rt_phi_kappa * (np.cos(phi - rt_phi) - 1))

    phi_density = np.zeros((n_rows, n_phi), dtype=np.float32)
    phi_temp = np.zeros((n_rows, n_phi), dtype=np.float32)

    for k in range(n_rows):
        rt_r_diff = r_subset[k] - rt_r_base
        r_fade_out = np.clip(rt_r_length * 2 - rt_r_diff, 0, 1)
        r_fade_in = np.clip((r_subset[k] - rt_r_base) / (rt_r_length * 0.3 + 1e-8), 0, 1)
        rt_r_prof = (np.exp(-0.5 * (rt_r_diff / (rt_r_length * 0.4 + 1e-8)) ** 2)
                     * r_fade_out * r_fade_in)
        phi_density[k] = phi_prof * rt_r_prof * rt_intensity
        phi_temp[k] = phi_density[k] * rt_delta_T

    phi_density = np.clip(phi_density, 0, 1)

    center_r = rt_r_base + rt_r_length * 0.5
    center_idx = int(np.argmin(np.abs(r_norm_all - center_r)))
    omega = float(omega_all[center_idx])

    return row_indices, phi_density, phi_temp, omega


def generate_disk_texture(n_phi: int = 1024, n_r: int = 512, seed: int = 42,
                          r_inner: float = 2.0, r_outer: float = 3.5,
                          enable_rt: bool = True, color_temp: float = None,
                          generation_scale: int = 2) -> np.ndarray:
    """
    直接在极坐标下生成吸积盘纹理，避免笛卡尔到极坐标的映射接缝问题。

    Args:
        n_phi: 角度方向分辨率（对应 0-2π）
        n_r: 径向方向分辨率（对应 r_inner 到 r_outer）
        seed: 随机种子
        r_inner: 内半径
        r_outer: 外半径
        enable_rt: 是否启用 Rayleigh-Taylor 不稳定性
        color_temp: 色温（单位：K），控制整体颜色。默认 None 使用 DISK_COLOR_TEMPERATURE

    Returns:
        (n_r, n_phi, 4) float32，第 4 通道为 alpha（面密度）
    """
    # 使用全局色温参数或传入的色温
    if color_temp is None:
        color_temp = DISK_COLOR_TEMPERATURE

    rng = np.random.default_rng(seed)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r_norm = np.linspace(0, 1, n_r)
    phi_grid, r_norm_grid = np.meshgrid(phi, r_norm)

    r_vals = r_inner + (r_outer - r_inner) * r_norm_grid
    disk_area = (r_outer ** 2 - r_inner ** 2) / 10.0

    # ----- 温度基底（内热外冷 + 噪声扰动）-----
    # 先生成径向递减的基底，再叠加轻微噪声，最终数值控制在 0~0.45
    radial_decay = np.clip(1.0 - r_norm_grid, 0, 1) ** 1.3
    temp_coarse = _fbm_noise((n_r, n_phi), rng, octaves=4, persistence=0.6, base_scale=8, wrap_u=True)
    temp_fine = _fbm_noise((n_r, n_phi), rng, octaves=5, persistence=0.45, base_scale=3, wrap_u=True)
    temp_noise = 0.6 * temp_coarse + 0.4 * temp_fine
    temp_base = np.clip(radial_decay * (0.85 + 0.15 * temp_noise), 0, 1)
    temp_base *= 0.25

    # 各结构的温度贡献将在下面的循环中累积
    temp_struct = np.zeros((n_r, n_phi), dtype=np.float32)

    # ----- 密度场 -----
# 1) 螺旋臂
    generation_scale = _validate_disk_generation_scale(generation_scale)
    spiral, spiral_temp = _generate_spiral_arms(
        rng, n_r, n_phi, phi_grid, r_norm_grid, 0.0, None, generation_scale=generation_scale
    )
    temp_struct += spiral_temp


    # 2) 云雾
    turbulence, kep_shift_pixels, turb_temp = _generate_turbulence(
        rng, n_r, n_phi, r_norm_grid, 0.0, None, generation_scale=generation_scale
    )
    temp_struct += turb_temp

    # 3) Filaments
    arcs, arcs_temp = _generate_filaments(
        rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, 0.0, None, generation_scale=generation_scale
    )
    temp_struct += arcs_temp

    # 4) Rayleigh-Taylor 不稳定性
    rt_spikes, rt_temp = _generate_rt_spikes(
        rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, enable_rt, 0.0, None, generation_scale=generation_scale
    )
    temp_struct += rt_temp

    # 5) 温度热点
    hotspot, hotspot_temp = _generate_hotspots(rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, 0.0, None)
    temp_struct += hotspot_temp

    # 5) 方位热点
    az_hotspot = _generate_azimuthal_hotspot(
        rng, n_r, n_phi, phi_grid, r_norm_grid, 0.0, None, generation_scale=generation_scale
    )

    # 组合密度
    rt_weight = 0.20 if enable_rt else 0.0
    density = 0.15 + 0.10 * spiral + 0.30 * turbulence + 0.20 * hotspot + 0.30 * arcs + rt_weight * rt_spikes

# 湍流扰动 - 降低 disturbance 强度，保留更多 spiral arm 和 filament 的分段结构
    density, temp_struct = _apply_disturbance(
        rng, n_r, n_phi, density, temp_struct, kep_shift_pixels, r_norm_grid, 0.0, None,
        generation_scale=generation_scale,
    )

    # 边缘软化（沿径向）
    edge = compute_edge_alpha(n_r)
    density *= edge[:, None]

    # 归一化
    density = np.clip(density / (np.percentile(density, 98) + 1e-6), 0, 1)

    # ----- 合成温度场（取 max，非相加）-----
    # temp_struct 先按 95 分位缩放到 0~1，再与基底比较
    if np.any(temp_struct > 0):
        struct_scale = np.percentile(temp_struct[temp_struct > 0], 95)
        temp_struct_scaled = temp_struct / (struct_scale + 1e-6)
    else:
        temp_struct_scaled = temp_struct
    temp_struct_scaled = np.clip(temp_struct_scaled * 0.8, 0, 1.2)

    # 基底在每个半径上不得高于该半径的结构温度（使用 P70 作为典型上限）
    struct_max_per_r = np.max(temp_struct_scaled, axis=1)
    struct_p70_per_r = np.quantile(temp_struct_scaled, 0.7, axis=1)
    struct_ceiling = np.maximum(struct_p70_per_r, 0.05)
    temp_base = np.minimum(temp_base, struct_ceiling[:, None])
    temp_base = np.minimum(temp_base, struct_max_per_r[:, None])

    temperature_field = np.clip(np.maximum(temp_base, temp_struct_scaled), 0, 1)

    # ----- 颜色（温度 -> 黑体辐射 RGB）-----
    # 色温控制温度映射范围：
    # - 2700K: 整体温度降低，更多区域处于红橙温度 (1500K-6000K)
    # - 4500K: 中等温度范围 (2000K-9000K)
    # - 6500K: 整体温度升高，更多区域处于白色温度 (3000K-12000K)
    # 使用线性插值：以 4500K 为基准，色温变化时调整 T_min 和 T_max
    t_factor = (color_temp - 4500) / (6500 - 2700)  # -0.47 ~ 0.47
    T_min = 2000 + t_factor * 1000  # 1500K ~ 2500K
    T_max = 9000 + t_factor * 3000  # 6000K ~ 12000K

    temp_aniso = np.clip(temperature_field * (0.9 + 0.25 * az_hotspot), 0, 1)
    T_K = T_min + temp_aniso * (T_max - T_min)
    bb_color = _blackbody_rgb(T_K)  # (n_r, n_phi, 3)
    # 高温端钳制：确保 R >= B，避免蓝色偏移（真正的白热不偏蓝）
    bb_color[:, :, 2] = np.minimum(bb_color[:, :, 2], bb_color[:, :, 0])

    # RGB = 黑体色 × 亮度（温度驱动），alpha = 密度（不透明度）
    # 亮度用 sqrt 而非 T^2/T^4，保留低温区可见的红/橙色
    luminosity = np.clip(np.sqrt(temp_aniso), 0, 1)

    tex = np.zeros((n_r, n_phi, 4), dtype=np.float32)
    tex[:, :, 0] = np.clip(bb_color[:, :, 0] * luminosity, 0, 1)
    tex[:, :, 1] = np.clip(bb_color[:, :, 1] * luminosity, 0, 1)
    tex[:, :, 2] = np.clip(bb_color[:, :, 2] * luminosity, 0, 1)
    tex[:, :, 3] = np.clip(density, 0, 1)  # alpha 只由面密度决定

    return tex


def generate_disk_texture_rotating(n_phi: int = 1024, n_r: int = 512, seed: int = 42,
                                    r_inner: float = 2.0, r_outer: float = 3.5,
                                    enable_rt: bool = True, t_offset: float = 0.0,
                                    color_temp: float = None,
                                    state: Optional[DiskTextureRotatingState] = None,
                                    generation_scale: int = 2) -> np.ndarray:
    """
    生成吸积盘纹理，支持参数化旋转（用于动画渲染）

    关键特性：
    - 使用固定的随机种子生成结构参数
    - 对温度基底和各组件应用开普勒旋转
    - 保证不同 t_offset 下是同一结构的旋转

    Args:
        n_phi: 角度方向分辨率（对应 0-2π）
        n_r: 径向方向分辨率（对应 r_inner 到 r_outer）
        seed: 随机种子
        r_inner: 内半径
        r_outer: 外半径
        enable_rt: 是否启用 Rayleigh-Taylor 不稳定性
        t_offset: 旋转偏移量（用于动画）
        color_temp: 色温（单位：K），控制整体颜色。默认 None 使用 DISK_COLOR_TEMPERATURE

    Returns:
        (n_r, n_phi, 4) float32 纹理，RGBA
    """
    generation_scale = _validate_disk_generation_scale(generation_scale)

    if state is not None:
        if state.n_phi != n_phi or state.n_r != n_r:
            raise ValueError(
                f"State size mismatch: expected {state.n_r}x{state.n_phi}, got {n_r}x{n_phi}"
            )
        if state.generation_scale != generation_scale:
            raise ValueError(
                f"State generation_scale mismatch: expected {state.generation_scale}, got {generation_scale}"
            )
        return _generate_disk_texture_rotating_from_state(state, t_offset=t_offset, color_temp=color_temp)

    # 使用全局色温参数或传入的色温
    if color_temp is None:
        color_temp = DISK_COLOR_TEMPERATURE

    rng = np.random.default_rng(seed)

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r_norm = np.linspace(0, 1, n_r)
    phi_grid_base, r_norm_grid = np.meshgrid(phi, r_norm)

    r_vals = r_inner + (r_outer - r_inner) * r_norm_grid
    disk_area = (r_outer ** 2 - r_inner ** 2) / 10.0

    # 计算开普勒角速度
    omega_grid = np.sqrt(0.5 / (r_vals ** 3 + 1e-6))

    # 应用旋转后的 phi_grid（开普勒旋转：逆时针，角度增加）
    phi_grid = phi_grid_base + t_offset * omega_grid

    # ----- 温度基底 -----
    radial_decay = np.clip(1.0 - r_norm_grid, 0, 1) ** 1.3
    temp_coarse = _fbm_noise((n_r, n_phi), rng, octaves=4, persistence=0.6, base_scale=8, wrap_u=True)
    temp_fine = _fbm_noise((n_r, n_phi), rng, octaves=5, persistence=0.45, base_scale=3, wrap_u=True)

    # 对温度基底应用开普勒旋转
    if t_offset != 0.0:
        for ri in range(n_r):
            rotation_pixels = int(t_offset * omega_grid[ri, 0] / (2 * np.pi) * n_phi)
            temp_coarse[ri, :] = np.roll(temp_coarse[ri, :], -rotation_pixels)
            temp_fine[ri, :] = np.roll(temp_fine[ri, :], -rotation_pixels)

    temp_noise = 0.6 * temp_coarse + 0.4 * temp_fine
    temp_base = np.clip(radial_decay * (0.85 + 0.15 * temp_noise), 0, 1)
    temp_base *= 0.25

    temp_struct = np.zeros((n_r, n_phi), dtype=np.float32)

    # ----- 密度场 -----
    # 1) 螺旋臂
    spiral, spiral_temp = _generate_spiral_arms(
        rng, n_r, n_phi, phi_grid, r_norm_grid, t_offset, omega_grid, generation_scale=generation_scale
    )
    temp_struct += spiral_temp

    # 2) 云雾
    turbulence, kep_shift_pixels, turb_temp = _generate_turbulence(
        rng, n_r, n_phi, r_norm_grid, t_offset, omega_grid, generation_scale=generation_scale
    )
    temp_struct += turb_temp

    # 3) Filaments
    arcs, arcs_temp = _generate_filaments(
        rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, t_offset, omega_grid, generation_scale=generation_scale
    )
    temp_struct += arcs_temp

    # 4) RT 不稳定性
    rt_spikes, rt_temp = _generate_rt_spikes(
        rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, enable_rt, t_offset, omega_grid, generation_scale=generation_scale
    )
    temp_struct += rt_temp

    # 5) 温度热点
    hotspot, hotspot_temp = _generate_hotspots(rng, n_r, n_phi, phi_grid, r_norm_grid, disk_area, t_offset, omega_grid)
    temp_struct += hotspot_temp

    # 6) 方位热点
    az_hotspot = _generate_azimuthal_hotspot(
        rng, n_r, n_phi, phi_grid, r_norm_grid, t_offset, omega_grid, generation_scale=generation_scale
    )

    # 组合密度
    rt_weight = 0.20 if enable_rt else 0.0
    density = 0.15 + 0.10 * spiral + 0.30 * turbulence + 0.20 * hotspot + 0.30 * arcs + rt_weight * rt_spikes

    # 湍流扰动
    density, temp_struct = _apply_disturbance(
        rng, n_r, n_phi, density, temp_struct, kep_shift_pixels, r_norm_grid, t_offset, omega_grid,
        generation_scale=generation_scale,
    )

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

    # 颜色（温度 -> 黑体辐射 RGB）
    # 色温控制温度映射范围：
    # - 2700K: 整体温度降低，更多区域处于红橙温度 (1500K-6000K)
    # - 4500K: 中等温度范围 (2000K-9000K)
    # - 6500K: 整体温度升高，更多区域处于白色温度 (3000K-12000K)
    # 使用线性插值：以 4500K 为基准，色温变化时调整 T_min 和 T_max
    t_factor = (color_temp - 4500) / (6500 - 2700)  # -0.47 ~ 0.47
    T_min = 2000 + t_factor * 1000  # 1500K ~ 2500K
    T_max = 9000 + t_factor * 3000  # 6000K ~ 12000K

    temp_aniso = np.clip(temperature_field * (0.9 + 0.25 * az_hotspot), 0, 1)
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
                density = (0.15 + 0.10 * sp + 0.30 * turb + 0.20 * hs
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

        for key, d_idx, t_idx in component_map:
            factory = factories.get(key)
            if factory is None:
                continue
            for entity in factory.alive_entities:
                alpha = entity.fade_factor(now)
                if alpha <= 0:
                    continue
                age = now - entity.birth_time
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
        density = (0.15 + 0.10 * sp + 0.30 * turb + 0.20 * hs
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


def render_image(width: int, height: int, cam_pos: List[float], fov: float, step_size: float, skybox_path: Optional[str] = None,
                  n_stars: int = 6000, tex_w: int = 2048, tex_h: int = 1024, r_max: float = 10.0, device: str = "cpu",
                  disk_texture_path: Optional[str] = None, r_disk_inner: float = R_DISK_INNER_DEFAULT,
                  r_disk_outer: float = R_DISK_OUTER_DEFAULT, disk_tilt: float = 0.0,
                  lens_flare: bool = False, anti_alias: str = "disabled", aa_strength: float = 1.0,
                  disk_rotation_speed: float = 0.1, disk_generation_scale: int = 2,
                  force_regenerate_disk_texture: bool = False, ignore_taichi_cache: bool = False) -> np.ndarray:
    """
    使用 Taichi 渲染单帧图像。

    纹理通过实体生命周期系统在 t=0 生成，与 interactive 模式使用相同流程。
    若提供 disk_texture_path 则直接加载外部纹理（跳过生命周期系统）。
    """
    skybox, tex_h, tex_w = load_or_generate_skybox(skybox_path, tex_w, tex_h, n_stars)

    # 外部纹理优先；否则用占位纹理 + 生命周期系统生成
    disk_tex = load_disk_texture(disk_texture_path)
    use_lifecycle = disk_tex is None
    if use_lifecycle:
        n_phi, n_r = compute_disk_texture_resolution(
            width, height, cam_pos, fov, r_disk_inner, r_disk_outer)
        disk_tex = np.zeros((n_r, n_phi, 4), dtype=np.float32)

    renderer = TaichiRenderer(
        width, height, skybox, disk_tex,
        step_size=step_size, r_max=r_max, device=device,
        r_disk_inner=r_disk_inner, r_disk_outer=r_disk_outer,
        disk_tilt=disk_tilt,
        lens_flare=lens_flare,
        anti_alias=anti_alias,
        aa_strength=aa_strength,
        disk_rotation_speed=disk_rotation_speed,
        ignore_taichi_cache=ignore_taichi_cache
    )

    if use_lifecycle:
        factories = _init_lifecycle_system(renderer, n_r, n_phi, seed=42)
        _advance_lifecycle_frame(renderer, factories, t=0.0, dt=0.0,
                                 recompute_stats=True)

    t0 = time.time()
    print(f"Taichi: {width}x{height}, cam_pos={list(cam_pos)}, fov={fov}°, step_size={step_size}")
    img = renderer.render(cam_pos, fov, frame=0)
    print(f"Done in {time.time() - t0:.1f}s")

    return img


def _init_lifecycle_system(renderer: TaichiRenderer, n_r: int, n_phi: int,
                          seed: int = 42) -> dict:
    """Initialize the entity lifecycle system for disk texture generation.

    Sets up background layer (GPU noise) and entity factories (filaments,
    hotspots, RT spikes), pre-populates entities at staggered ages, generates
    the first frame, and computes initial normalization stats.

    Args:
        renderer: TaichiRenderer instance (must have r_disk_inner/r_disk_outer set)
        n_r: radial resolution of disk texture
        n_phi: azimuthal resolution of disk texture
        seed: random seed for reproducibility

    Returns:
        dict with keys 'filament', 'hotspot', 'rt_spike', each an EntityFactory
    """
    renderer.init_background_layer(n_r=n_r, n_phi=n_phi, seed=seed)

    r_norm_all = np.linspace(0, 1, n_r)
    r_vals = renderer.r_disk_inner + (renderer.r_disk_outer - renderer.r_disk_inner) * r_norm_all
    omega_all = np.sqrt(0.5 / (r_vals ** 3 + 1e-6)).astype(np.float32)

    factories = {
        'filament': EntityFactory(
            _spawn_single_filament, target_count=200,
            lifetime_range=(20.0, 40.0), fade_in=5.0, fade_out=5.0,
            n_r=n_r, n_phi=n_phi,
            r_norm_all=r_norm_all, omega_all=omega_all, seed=seed + 100),
        'hotspot': EntityFactory(
            _spawn_single_hotspot, target_count=30,
            lifetime_range=(15.0, 30.0), fade_in=4.0, fade_out=4.0,
            n_r=n_r, n_phi=n_phi,
            r_norm_all=r_norm_all, omega_all=omega_all, seed=seed + 200),
        'rt_spike': EntityFactory(
            _spawn_single_rt_spike, target_count=15,
            lifetime_range=(15.0, 30.0), fade_in=3.0, fade_out=3.0,
            n_r=n_r, n_phi=n_phi,
            r_norm_all=r_norm_all, omega_all=omega_all, seed=seed + 300),
    }
    for f in factories.values():
        f.seed_initial(now=0.0)

    renderer.generate_background(t=0.0)
    renderer.accumulate_entity_layer(factories, now=0.0)
    renderer.recompute_interactive_stats()
    renderer.compose_interactive_texture()

    return factories


def _advance_lifecycle_frame(renderer: TaichiRenderer, factories: dict,
                             t: float, dt: float,
                             recompute_stats: bool = False,
                             solo_idx: int = -1) -> None:
    """Advance the lifecycle system by one frame and compose disk texture.

    Args:
        renderer: TaichiRenderer instance with lifecycle system initialized
        factories: dict of EntityFactory instances
        t: current simulation time in seconds
        dt: time step since last frame (seconds)
        recompute_stats: whether to recompute normalization stats this frame
        solo_idx: -1 for all components, >= 0 to solo a single component
    """
    for f in factories.values():
        f.tick(now=t, dt=dt)
    renderer.generate_background(t=t)
    renderer.accumulate_entity_layer(factories, now=t)
    if recompute_stats:
        renderer.recompute_interactive_stats()
    renderer.compose_interactive_texture(solo_idx=solo_idx)


def render_interactive(renderer: TaichiRenderer, width: int, height: int,
                       fov: float, initial_cam_pos: List[float],
                       disk_rotation_speed: float = 0.05) -> None:
    """实时交互预览模式（实体生命周期系统）。

    使用 Taichi Legacy GUI 实时渲染黑洞场景，支持鼠标/键盘控制相机和渲染开关。
    吸积盘纹理由两层系统实时生成：
        - GPU 背景层：3D simplex noise 驱动的时间演化（宽 r 组件）
        - CPU 实体层：filament/hotspot/RT spike 的生命周期管理（窄 r 组件）

    相机控制（球坐标，始终朝向原点）:
        鼠标左键拖拽: 旋转视角 (φ, θ)
        滚轮上/下: 缩放距离
        ↑/↓: 调整 FOV

    渲染开关:
        D: 切换微分光线（抗锯齿基础，默认关）
        B: 切换 Bloom 泛光（默认关）
        L: 切换 Lens Flare（默认关）
        S: 保存当前帧截图
        ESC/Q: 退出

    Args:
        renderer: TaichiRenderer 实例
        width, height: 窗口分辨率
        fov: 初始视野角度
        initial_cam_pos: 初始相机位置 [x, y, z]
        disk_rotation_speed: 盘旋转速度
    """
    import taichi as ti

    cam_pos = np.array(initial_cam_pos, dtype=np.float64)
    r = float(np.linalg.norm(cam_pos))
    theta = float(np.arccos(np.clip(cam_pos[2] / r, -1, 1)))
    phi = float(np.arctan2(cam_pos[1], cam_pos[0]))

    toggle_diff = False
    toggle_bloom = True
    toggle_flare = False
    renderer.lens_flare = False

    # —— 初始化实体生命周期系统 ——
    factories = _init_lifecycle_system(renderer, renderer.dtex_h, renderer.dtex_w, seed=42)
    print("实体生命周期系统已启用 (filaments=200, hotspots=30, rt_spikes=15)")

    gui = ti.GUI('Black Hole Interactive', res=(width, height))
    wall_time = 0.0
    frame_count = 0
    total_frames = 0
    fps_timer = time.time()
    fps_display = 0.0
    last_frame_time = time.time()

    mouse_pressed = False
    mouse_last = (0.0, 0.0)
    solo_idx = -1

    _SOLO_NAMES = {
        0: "temp_base", 1: "spiral", 2: "spiral_temp",
        3: "turbulence", 4: "turb_temp",
        5: "filaments", 6: "filaments_temp",
        7: "rt_spikes", 8: "rt_temp",
        9: "hotspot", 10: "hotspot_temp",
        11: "az_hotspot", 12: "disturb_mod",
    }

    print(f"\n=== 交互模式 ({width}x{height}) ===")
    print(f"鼠标拖拽: 旋转 | 滚轮: 缩放 | ↑↓: FOV")
    print(f"D: 微分光线 | B: Bloom | L: Lens Flare | S: 截图 | ESC: 退出")
    print(f"1-8: solo 组件 | 0: 显示全部\n")

    while gui.running:
        # —— 事件处理 ——
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE or e.key == 'q':
                gui.running = False
            elif e.key == 'd':
                toggle_diff = not toggle_diff
                print(f"微分光线: {'开' if toggle_diff else '关'}")
            elif e.key == 'b':
                toggle_bloom = not toggle_bloom
                print(f"Bloom: {'开' if toggle_bloom else '关'}")
            elif e.key == 'l':
                toggle_flare = not toggle_flare
                renderer.lens_flare = toggle_flare
                print(f"Lens Flare: {'开' if toggle_flare else '关'}")
            elif e.key == '0':
                solo_idx = -1
                print("Solo: 全部组件")
            elif e.key == '1':
                solo_idx = 0
                print(f"Solo: {_SOLO_NAMES[0]} (idx 0)")
            elif e.key == '2':
                solo_idx = 1
                print(f"Solo: {_SOLO_NAMES[1]} (idx 1)")
            elif e.key == '3':
                solo_idx = 3
                print(f"Solo: {_SOLO_NAMES[3]} (idx 3)")
            elif e.key == '4':
                solo_idx = 11
                print(f"Solo: {_SOLO_NAMES[11]} (idx 11)")
            elif e.key == '5':
                solo_idx = 12
                print(f"Solo: {_SOLO_NAMES[12]} (idx 12)")
            elif e.key == '6':
                solo_idx = 5
                print(f"Solo: {_SOLO_NAMES[5]} (idx 5)")
            elif e.key == '7':
                solo_idx = 9
                print(f"Solo: {_SOLO_NAMES[9]} (idx 9)")
            elif e.key == '8':
                solo_idx = 7
                print(f"Solo: {_SOLO_NAMES[7]} (idx 7)")
            elif e.key == 's':
                screenshot_path = f"output/screenshot_{int(time.time())}.png"
                os.makedirs("output", exist_ok=True)
                img_save = renderer.render(cam_pos.tolist(), fov, frame=0)
                img_uint8 = (np.clip(img_save, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(img_uint8, "RGB").save(screenshot_path)
                print(f"截图已保存: {screenshot_path}")
            elif e.key == ti.GUI.UP:
                fov = max(10, fov - 5)
                print(f"FOV: {fov}°")
            elif e.key == ti.GUI.DOWN:
                fov = min(170, fov + 5)
                print(f"FOV: {fov}°")
            elif e.key == ti.GUI.LMB:
                mouse_pressed = True
                mouse_last = gui.get_cursor_pos()

        for e in gui.get_events(ti.GUI.RELEASE):
            if e.key == ti.GUI.LMB:
                mouse_pressed = False

        # 鼠标拖拽旋转
        if mouse_pressed and gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            dx = mx - mouse_last[0]
            dy = my - mouse_last[1]
            phi -= dx * 3.0
            theta = np.clip(theta - dy * 3.0, 0.05, np.pi - 0.05)
            mouse_last = (mx, my)

        # 滚轮缩放
        if gui.is_pressed(ti.GUI.UP):
            pass
        # Taichi Legacy GUI 不直接支持滚轮，用 +/- 键代替
        if gui.is_pressed('=') or gui.is_pressed('+'):
            r = max(2.0, r * 0.97)
        if gui.is_pressed('-'):
            r *= 1.03

        # 更新相机位置（球坐标 → 笛卡尔）
        cam_pos[0] = r * np.sin(theta) * np.cos(phi)
        cam_pos[1] = r * np.sin(theta) * np.sin(phi)
        cam_pos[2] = r * np.cos(theta)

        # —— 实体生命周期更新 ——
        now_real = time.time()
        dt = min(now_real - last_frame_time, 0.1)
        last_frame_time = now_real
        scaled_dt = dt * disk_rotation_speed * 20.0
        wall_time += scaled_dt

        total_frames += 1
        _advance_lifecycle_frame(renderer, factories, wall_time, scaled_dt,
                                 recompute_stats=(total_frames % 60 == 1),
                                 solo_idx=solo_idx)

        # —— 渲染到 GPU field ——
        renderer.render_to_field(
            cam_pos.tolist(), fov, frame=0,
            skip_differentials=not toggle_diff,
            skip_bloom=not toggle_bloom,
        )

        # —— 显示（直接从 GPU field，无 CPU 传输）——
        gui.set_image(renderer.final_field)

        # HUD 信息
        frame_count += 1
        fps_now = time.time()
        if fps_now - fps_timer >= 0.5:
            fps_display = frame_count / (fps_now - fps_timer)
            frame_count = 0
            fps_timer = fps_now

        n_entities = sum(len(f.entities) for f in factories.values())
        toggles = f"D:{'ON' if toggle_diff else 'off'} B:{'ON' if toggle_bloom else 'off'} L:{'ON' if toggle_flare else 'off'}"
        solo_txt = f" SOLO:{_SOLO_NAMES[solo_idx]}" if solo_idx >= 0 else ""
        gui.text(f"{fps_display:.0f} FPS | {toggles} | E:{n_entities}{solo_txt}", pos=(0.01, 0.97), color=0xFFFFFF)
        gui.text(f"r={r:.1f} fov={fov:.0f} t={wall_time:.1f}", pos=(0.01, 0.93), color=0xCCCCCC)
        gui.text("+/-: zoom | arrows: FOV | S: screenshot", pos=(0.01, 0.02), color=0x888888)

        gui.show()

    gui.close()
    print("交互模式退出")


def render_video(renderer: TaichiRenderer, width: int, height: int, n_frames: int, fps: int, output_path: str,
                 fov: float, static_cam_pos: List[float], orbit: bool = False, resume: bool = False,
                 disk_rotation_speed: float = 0.1, orbit_degrees: float = 360.0,
                 **_deprecated_kwargs) -> None:
    """
    渲染视频（多帧并合成视频），使用实体生命周期系统生成纹理。

    参数:
        renderer: TaichiRenderer 实例
        width, height: 图像尺寸
        n_frames: 帧数
        fps: 帧率
        output_path: 输出视频路径
        fov: 视野角度
        static_cam_pos: 静态模式下的相机位置
        orbit: 是否围绕原点旋转
        resume: 是否尝试从断点恢复
        disk_rotation_speed: 旋转速度系数
        orbit_degrees: 轨道模式下整段视频的总旋转角度（度）
    """
    orbit_radius = float(np.linalg.norm(static_cam_pos))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    temp_dir_name = ".frames_" + hashlib.md5(output_path.encode()).hexdigest()[:16]
    temp_dir = os.path.join(os.path.dirname(output_path), temp_dir_name)
    progress_file = os.path.join(temp_dir, "progress.json")

    params = {
        "n_frames": n_frames,
        "fov": fov,
        "orbit": orbit,
        "disk_rotation_speed": disk_rotation_speed,
        "orbit_degrees": orbit_degrees,
    }

    completed = set()
    if resume and os.path.isdir(temp_dir) and os.path.isfile(progress_file):
        with open(progress_file, "r") as f:
            saved = json.load(f)
        saved_params = saved.get("params", {})
        if saved_params != params:
            print(f"Warning: parameters changed, starting over")
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
        else:
            completed = set(saved.get("completed", []))
            print(f"Resuming: {len(completed)}/{n_frames} frames already rendered")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    total_t0 = time.time()
    angle_step = orbit_degrees / n_frames
    rendered_this_session = 0

    # —— 异步 PNG 保存 ——
    MAX_PENDING_PNGS = 4
    png_pool = ThreadPoolExecutor(max_workers=2)
    png_futures = []

    def _save_png(path, img_uint8):
        Image.fromarray(img_uint8, "RGB").save(path)

    # —— 初始化生命周期系统 ——
    n_r = renderer.dtex_h
    n_phi = renderer.dtex_w
    factories = _init_lifecycle_system(renderer, n_r, n_phi, seed=42)
    dt = disk_rotation_speed
    print(f"  生命周期系统已初始化 (n_r={n_r}, n_phi={n_phi})")

    # —— 断点恢复：快速重演模拟到 resume 点 ——
    if completed:
        max_completed = max(completed)
        print(f"  快速重演模拟到帧 {max_completed}...")
        replay_t0 = time.time()
        for f in range(max_completed + 1):
            t = f * dt
            _advance_lifecycle_frame(renderer, factories, t, dt)
        print(f"  重演完成: {time.time() - replay_t0:.1f}s")

    # —— 主渲染循环 ——
    for frame in range(n_frames):
        t = frame * dt

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

        if frame in completed:
            continue

        t0 = time.time()
        _advance_lifecycle_frame(renderer, factories, t, dt,
                                 recompute_stats=(frame % 60 == 0))
        img = renderer.render(cam_pos, fov, frame=0)
        elapsed = time.time() - t0
        rendered_this_session += 1

        frame_path = os.path.join(temp_dir, f"frame_{frame:04d}.png")
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        if len(png_futures) >= MAX_PENDING_PNGS:
            png_futures.pop(0).result()
        png_futures.append(png_pool.submit(_save_png, frame_path, img_uint8))

        completed.add(frame)
        if rendered_this_session % 10 == 0 or frame == n_frames - 1:
            with open(progress_file, "w") as f:
                json.dump({"params": params, "completed": list(completed)}, f)

        if rendered_this_session % 100 == 0 or frame == n_frames - 1:
            eta = (time.time() - total_t0) / rendered_this_session * (n_frames - len(completed))
            print(f"  frame {frame}/{n_frames} ({status_str}) {elapsed:.1f}s, done {len(completed)}/{n_frames}, ETA {eta/60:.0f}min")

    # 等待所有 PNG 写入完成
    for f in png_futures:
        f.result()
    png_pool.shutdown(wait=False)

    if rendered_this_session > 0:
        print(f"\nSession rendered {rendered_this_session} frames in {(time.time() - total_t0)/60:.1f} min")

    if len(completed) < n_frames:
        print(f"Warning: only {len(completed)}/{n_frames} frames completed. Run again to resume.")
        return

    total_elapsed = time.time() - total_t0
    print(f"\nAll frames rendered in {total_elapsed/60:.1f} min")

    print(f"Assembling video: {output_path} ({fps} fps, {n_frames/fps:.0f}s)...")
    # 注意：如果需要更高质量的视频（减少摩尔纹），可以：
    # 1. 使用 ffmpeg 直接编码：ffmpeg -framerate {fps} -i frame_%04d.png -c:v libx264 -crf 18 -preset slow output.mp4
    # 2. 或安装 imageio-ffmpeg 并使用更高质量的编码参数
    writer = iio.imopen(output_path, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=fps)

    for frame in range(n_frames):
        frame_path = os.path.join(temp_dir, f"frame_{frame:04d}.png")
        img = iio.imread(frame_path)
        writer.write_frame(img)
        ## 逐帧写入后删除临时文件以节省空间
        #os.remove(frame_path)

    #    os.remove(progress_file)
    print(f"\n提示：如果视频有摩尔纹，可手动用 ffmpeg 重新编码更高质量：")
    print(f"  ffmpeg -framerate {fps} -i {temp_dir}/frame_%04d.png -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p {output_path}")
    #shutil.rmtree(temp_dir)
    print(f"Video saved: {output_path}")


# ============================================================================
# 主入口
# ============================================================================

def parse_args() -> argparse.Namespace:
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
                        help="吸积盘纹理路径 (default: 程序生成，仅静态单帧模式支持)")
    parser.add_argument("--disk_generation_scale", type=int, default=2, choices=DISK_GENERATION_SCALE_CHOICES,
                        help="[已废弃] 生命周期系统不使用此参数 (default: 2)")
    parser.add_argument("--force_regenerate_disk_texture", action="store_true",
                        help="[已废弃] 生命周期系统每次实时生成 (default: 关闭)")
    parser.add_argument("--disk_inner_radius", "--ar1", dest="disk_inner_radius", type=float, default=R_DISK_INNER_DEFAULT,
                        help=f"吸积盘内半径 (default: {R_DISK_INNER_DEFAULT})")
    parser.add_argument("--disk_outer_radius", "--ar2", dest="disk_outer_radius", type=float, default=R_DISK_OUTER_DEFAULT,
                        help=f"吸积盘外半径 (default: {R_DISK_OUTER_DEFAULT})")
    parser.add_argument("--disk_tilt", type=float, default=0.0,
                        help="吸积盘倾角 (度, default: 0)")
    parser.add_argument("--lens_flare", action="store_true",
                        help="启用 lens flare 效果 (default: 关闭)")
    parser.add_argument("--anti_alias", type=str, default="disabled",
                        choices=["disabled", "lod_radius"],
                        help="抗锯齿算法: disabled(关闭), lod_radius(基于半径的启发式LOD) (default: disabled)")
    parser.add_argument("--aa_strength", type=float, default=1.0,
                        help="抗锯齿强度，乘以 LOD 值。>1 更模糊(更强抗锯齿)，<1 更清晰。范围 0.5-2.0 (default: 1.0)")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                        choices=["cpu", "gpu"],
                        help="Taichi 设备: cpu 或 gpu (default: cpu)")
    parser.add_argument("--ignore_taichi_cache", action="store_true",
                        help="忽略 Taichi 离线缓存，强制重新编译 kernel (default: 关闭)")
    parser.add_argument("--video", action="store_true",
                        help="视频模式：渲染多帧并合成视频")
    parser.add_argument("--interactive", action="store_true",
                        help="交互模式：实时预览，鼠标拖拽旋转，按键切换渲染开关")
    parser.add_argument("--orbit", action="store_true",
                        help="视频模式：相机围绕原点旋转（需配合 --video）")
    parser.add_argument("--orbit_degrees", type=float, default=360.0,
                        help="轨道模式下整段视频的总旋转角度，支持负数反向旋转 (default: 360.0)")
    parser.add_argument("--n_frames", type=int, default=3600,
                        help="视频帧数 (default: 3600, 仅 --video 有效)")
    parser.add_argument("--fps", type=int, default=36,
                        help="视频帧率 (default: 36, 仅 --video 有效)")
    parser.add_argument("--resume", action="store_true",
                        help="视频模式：尝试从断点恢复（默认从头开始）")
    parser.add_argument("--disk_rotation_algorithm", type=str, default="baseline",
                        choices=["baseline", "parametric", "keyframes"],
                        help="[已废弃] 统一使用生命周期系统，此参数被忽略")
    parser.add_argument("--disk_rotation_speed", type=float, default=0.1,
                        help="吸积盘旋转速度系数 (default: 0.1)")
    parser.add_argument("--keyframes_count", type=int, default=10,
                        help="[已废弃] 统一使用生命周期系统，此参数被忽略")
    return parser.parse_args()


def validate_args(args) -> None:
    """Validate CLI arguments."""
    # FOV range check
    if not (0 < args.fov < 180):
        raise ValueError(f"FOV must be between 0 and 180 degrees, got {args.fov}")

    # Disk radius check
    if args.disk_inner_radius >= args.disk_outer_radius:
        raise ValueError(f"disk_inner_radius ({args.disk_inner_radius}) must be less than "
                        f"disk_outer_radius ({args.disk_outer_radius})")

    # Step size check
    if args.step_size <= 0:
        raise ValueError(f"step_size must be positive, got {args.step_size}")

    # AA strength range
    if not (0.5 <= args.aa_strength <= 2.0):
        raise ValueError(f"aa_strength must be between 0.5 and 2.0, got {args.aa_strength}")

    # Video parameters
    if args.n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {args.n_frames}")

    if args.fps <= 0:
        raise ValueError(f"fps must be positive, got {args.fps}")

    if not math.isfinite(args.orbit_degrees):
        raise ValueError(f"orbit_degrees must be finite, got {args.orbit_degrees}")

    if args.disk_texture and (args.video or args.interactive):
        raise ValueError("--disk_texture 仅支持静态单帧渲染，video/interactive 模式请使用生命周期系统")


if __name__ == "__main__":
    args = parse_args()
    validate_args(args)

    resolutions = {"4k": (3840, 2160), "fhd": (1920, 1080), "hd": (1280, 720), "sd": (640, 360)}
    width, height = resolutions[args.resolution]
    fov = args.fov % 180

    def _make_renderer_with_placeholder(device="cpu"):
        """Create renderer with placeholder disk texture for lifecycle system."""
        skybox, _, _ = load_or_generate_skybox(args.texture, 2048, 1024, args.n_stars)
        n_phi, n_r = compute_disk_texture_resolution(
            width, height, args.pov, fov,
            args.disk_inner_radius, args.disk_outer_radius)
        disk_tex = np.zeros((n_r, n_phi, 4), dtype=np.float32)
        return TaichiRenderer(
            width, height, skybox, disk_tex,
            step_size=args.step_size, r_max=args.r_max, device=device,
            r_disk_inner=args.disk_inner_radius, r_disk_outer=args.disk_outer_radius,
            disk_tilt=args.disk_tilt,
            lens_flare=args.lens_flare if not args.interactive else False,
            anti_alias=args.anti_alias if not args.interactive else "disabled",
            aa_strength=args.aa_strength,
            disk_rotation_speed=args.disk_rotation_speed,
            ignore_taichi_cache=args.ignore_taichi_cache
        )

    if args.interactive:
        renderer = _make_renderer_with_placeholder(device="gpu")
        render_interactive(
            renderer, width, height,
            fov=fov, initial_cam_pos=args.pov,
            disk_rotation_speed=args.disk_rotation_speed,
        )
    elif args.video:
        renderer = _make_renderer_with_placeholder(device=args.device)

        print(f"Rendering video: {args.n_frames} frames at {width}x{height}")
        print(f"  orbit={args.orbit}")
        if args.orbit:
            print(f"  orbit_degrees={args.orbit_degrees}°")
        print(f"  fov={fov}°, step_size={args.step_size}, fps={args.fps}, disk_tilt={args.disk_tilt}°")
        print(f"  disk_rotation_speed={args.disk_rotation_speed}")

        render_video(
            renderer, width, height,
            n_frames=args.n_frames, fps=args.fps, output_path=args.output,
            fov=fov, static_cam_pos=args.pov,
            orbit=args.orbit,
            resume=args.resume,
            disk_rotation_speed=args.disk_rotation_speed,
            orbit_degrees=args.orbit_degrees,
        )
    else:
        img = render_image(
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
            r_disk_inner=args.disk_inner_radius,
            r_disk_outer=args.disk_outer_radius,
            disk_tilt=args.disk_tilt,
            lens_flare=args.lens_flare,
            anti_alias=args.anti_alias,
            aa_strength=args.aa_strength,
            disk_generation_scale=args.disk_generation_scale,
            force_regenerate_disk_texture=args.force_regenerate_disk_texture,
            ignore_taichi_cache=args.ignore_taichi_cache,
        )
        save_image(img, args.output)
