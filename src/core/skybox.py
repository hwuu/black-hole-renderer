from typing import Tuple, List, Optional
import os
from PIL import Image
import numpy as np
from src.core.constants import SKY_GALACTIC_CENTER_GLOW, SKY_MILKY_WAY_GLOW, SKY_STAR_BRIGHTNESS_GAIN, SKY_STAR_BRIGHTNESS_MAX, SKY_STAR_BRIGHTNESS_MIN, SKY_STAR_COLOR_SATURATION, SKY_STAR_SIZE_MAX, SKY_STAR_SIZE_MIN


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
