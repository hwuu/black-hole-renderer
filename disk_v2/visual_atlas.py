"""Disk V2 视觉 atlas 预烘焙（V1 云雾 + Blender 径向扭曲 + Alpha Clip）。

把 V1 `_generate_turbulence` 的多尺度云雾烘焙到 `(r, φ)` 极坐标 atlas，
供 Taichi 体积积分双线性查表。Blender 字幕中的 radial/spherical gradient
驱动旋转扭曲在本模块以 `spiral_warp_strength` 近似实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .geometry import disk_radial_weight
from .params import DiskV2Params, DiskV2StructureParams


def _tileable_noise(
    shape: Tuple[int, int],
    rng: np.random.Generator,
    freq_u: int = 6,
    freq_v: int = 6,
) -> np.ndarray:
    """用多条弧线生成 phi 方向无缝云雾噪声（自 V1 `render._tileable_noise` 搬迁）。"""
    h, w = shape
    cloud = np.zeros((h, w), dtype=np.float32)
    n_arcs = int(rng.integers(30, 60))
    for _ in range(n_arcs):
        arc_phi = float(rng.uniform(0.0, 2.0 * np.pi))
        arc_r = float(np.sqrt(rng.uniform(0.0, 1.0)))
        arc_phi_width = float(rng.uniform(0.15, 0.5))
        arc_r_width = float(rng.uniform(0.03, 0.08))
        arc_intensity = float(rng.uniform(0.03, 0.12))
        kappa = 1.0 / (arc_phi_width ** 2) * 0.6
        phi = np.linspace(0.0, 2.0 * np.pi, w, endpoint=False)
        r_norm = np.linspace(0.0, 1.0, h)
        phi_grid, r_grid = np.meshgrid(phi, r_norm, indexing="xy")
        r_diff = r_grid - arc_r
        arc_val = np.exp(kappa * (np.cos(phi_grid - arc_phi) - 1.0))
        arc_val *= np.exp(-0.5 * (r_diff / arc_r_width) ** 2)
        arc_val *= arc_intensity
        cloud += arc_val.astype(np.float32)
    return np.clip(cloud, 0.0, 1.0)


def _periodic_pixel_noise(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """像素级白噪声，phi 方向周期（自 V1 搬迁）。"""
    h, w = shape
    noise = rng.random((h, w)).astype(np.float32)
    noise[:, -1] = noise[:, 0]
    return noise * 2.0 - 1.0


def _build_turbulence_layers(
    n_r: int,
    n_phi: int,
    r_norm_grid: np.ndarray,
    rng: np.random.Generator,
    generation_scale: int,
) -> np.ndarray:
    """生成 V1 风格多层云雾（5 层 tileable + pixel noise + Kepler shear roll）。

    Args:
        n_r: 径向分辨率。
        n_phi: 角向分辨率。
        r_norm_grid: 形状 `(n_r, n_phi)` 的归一化半径网格。
        rng: 随机数生成器。
        generation_scale: 低分辨率生成倍率（1/2/4）。

    Returns:
        形状 `(n_r, n_phi)` 的湍流场，值域 `[0, 1]`。
    """
    scale_factor = max(int(generation_scale), 1)
    if scale_factor not in (1, 2, 4):
        raise ValueError("generation_scale must be 1, 2, or 4")

    low_n_r = max(n_r // scale_factor, 2)
    low_n_phi = max(n_phi // scale_factor, 2)
    low_r_norm = r_norm_grid[::scale_factor, ::scale_factor][:low_n_r, :low_n_phi]

    shear_strength = float(rng.uniform(3.0, 6.0))
    kep_shear_low = shear_strength * (1.0 / (low_r_norm + 0.3) ** 1.5 - 0.8)
    kep_shear_low = np.clip(kep_shear_low, 0.0, shear_strength * 8.0)
    kep_shift_low = (kep_shear_low / (2.0 * np.pi) * low_n_phi).astype(np.int32)
    max_shift = low_n_phi // 4
    kep_shift_low = np.clip(kep_shift_low, -max_shift, max_shift)

    layers = [
        _tileable_noise((low_n_r, low_n_phi), rng, freq_u=8, freq_v=4),
        _tileable_noise((low_n_r, low_n_phi), rng, freq_u=24, freq_v=12),
        _tileable_noise((low_n_r, low_n_phi), rng, freq_u=80, freq_v=40),
        _tileable_noise((low_n_r, low_n_phi), rng, freq_u=200, freq_v=100),
        _tileable_noise((low_n_r, low_n_phi), rng, freq_u=400, freq_v=200),
    ]
    for layer in layers:
        for ri in range(low_n_r):
            layer[ri, :] = np.roll(layer[ri, :], int(kep_shift_low[ri, 0]))

    pixel_noise = np.clip(_periodic_pixel_noise((low_n_r, low_n_phi), rng), 0.0, 1.0)
    turbulence_low = (
        0.08 * layers[0]
        + 0.15 * layers[1]
        + 0.25 * layers[2]
        + 0.22 * layers[3]
        + 0.18 * layers[4]
        + 0.12 * pixel_noise
    )

    if scale_factor == 1:
        return np.clip(turbulence_low[:n_r, :n_phi], 0.0, 1.0).astype(np.float32)

    upscale = np.ones((scale_factor, scale_factor), dtype=np.float32)
    turbulence = np.kron(turbulence_low, upscale)[:n_r, :n_phi]
    return np.clip(turbulence, 0.0, 1.0).astype(np.float32)


def _apply_spiral_warp(
    field: np.ndarray,
    phi_grid: np.ndarray,
    r_norm_grid: np.ndarray,
    spiral_warp_strength: float,
) -> np.ndarray:
    """参考 Blender 字幕：径向 gradient 驱动 φ 方向旋转扭曲。

    Args:
        field: 源场 `(n_r, n_phi)`。
        phi_grid: 方位角网格（弧度）。
        r_norm_grid: 归一化半径 `[0, 1]`。
        spiral_warp_strength: 扭曲强度；0 表示不扭曲。

    Returns:
        扭曲采样后的场，形状与 `field` 相同。
    """
    if spiral_warp_strength <= 0.0:
        return field

    n_r, n_phi = field.shape
    out = np.zeros_like(field, dtype=np.float64)
    phi_axis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    for ri in range(n_r):
        r_norm = float(r_norm_grid[ri, 0])
        warp_offset = spiral_warp_strength * r_norm * 2.0 * np.pi
        source_phi = (phi_grid[ri, :] - warp_offset) % (2.0 * np.pi)
        idx_f = source_phi / (2.0 * np.pi) * n_phi
        idx0 = np.floor(idx_f).astype(np.int32) % n_phi
        idx1 = (idx0 + 1) % n_phi
        frac = idx_f - np.floor(idx_f)
        out[ri, :] = (1.0 - frac) * field[ri, idx0] + frac * field[ri, idx1]
    return out.astype(np.float32)


def _radial_disk_mask(r_grid: np.ndarray, params: DiskV2Params) -> np.ndarray:
    """盘内径向软 mask，参考 Blender spherical gradient 盘缘收口。"""
    return np.asarray(disk_radial_weight(r_grid, params), dtype=np.float64)


def _apply_alpha_clip(field: np.ndarray, threshold: float) -> np.ndarray:
    """Alpha Clip 软过渡：弱于 threshold 的区域渐隐，保留云雾细丝对比。"""
    span = max(1.0 - threshold, 1e-6)
    fade = np.clip((field - threshold) / span, 0.0, 1.0)
    return (field * fade).astype(np.float32)


@dataclass(frozen=True)
class VisualAtlas:
    """预烘焙视觉 atlas，供 Taichi 双线性采样。

    Attributes:
        emission_weight: 发射乘子场，形状 `(n_r, n_phi)`，盘外为 0。
        density_weight: 密度乘子场，形状 `(n_r, n_phi)`，盘外为 0。
        n_r, n_phi: atlas 分辨率。
        r_in, r_out: 采样半径范围。
    """

    emission_weight: np.ndarray
    density_weight: np.ndarray
    n_r: int
    n_phi: int
    r_in: float
    r_out: float


def _cinematic_radial_glow(r_norm_grid: np.ndarray) -> np.ndarray:
    """Blender ColorRamp 近似：内缘更亮，外缘弱压平（保留 atlas 云雾对比）。"""
    inner = np.exp(-1.2 * r_norm_grid)
    return (0.72 + 0.28 * inner).astype(np.float32)


def build_visual_atlas(
    params: DiskV2Params,
    structure_params: DiskV2StructureParams | None = None,
    seed: int = 42,
) -> VisualAtlas:
    """构建 V2 视觉 atlas（turbulence + spiral warp + alpha clip）。

    Args:
        params: `DiskV2Params`，提供 `r_in` / `r_out`。
        structure_params: 可选 `DiskV2StructureParams`；默认使用 dataclass 默认值。
        seed: 随机种子。

    Returns:
        `VisualAtlas` 对象，含 `emission_weight` 与 `density_weight`。

    Formula:
        ```
        turbulence = V1_multiscale_noise + Kepler_shear
        warped = spiral_warp(turbulence, spiral_warp_strength)
        clipped = alpha_clip(warped, alpha_clip_threshold)
        emission_weight = (1 + turb_strength * (2*clipped - 1)) * radial_glow * radial_mask
        density_weight = (1 + (emission_weight - 1) * density_scale) * radial_mask  # 弱于 emission
        ```
    """
    sp = structure_params or DiskV2StructureParams()
    n_r = int(sp.atlas_n_r)
    n_phi = int(sp.atlas_n_phi)
    rng = np.random.default_rng(seed)

    r_vals = np.linspace(params.r_in, params.r_out, n_r, dtype=np.float64)
    phi_vals = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False, dtype=np.float64)
    r_grid, phi_grid = np.meshgrid(r_vals, phi_vals, indexing="ij")
    span = max(params.r_out - params.r_in, 1e-6)
    r_norm_grid = np.clip((r_grid - params.r_in) / span, 0.0, 1.0)

    turbulence = _build_turbulence_layers(
        n_r, n_phi, r_norm_grid, rng, sp.atlas_generation_scale,
    )
    warped = _apply_spiral_warp(
        turbulence, phi_grid, r_norm_grid, sp.spiral_warp_strength,
    )
    clipped = _apply_alpha_clip(warped, sp.alpha_clip_threshold)

    radial_mask = _radial_disk_mask(r_grid, params)
    radial_glow = _cinematic_radial_glow(r_norm_grid)
    signed = 2.0 * clipped - 1.0
    emission_core = (1.0 + sp.turbulence_strength * signed) * radial_glow
    density_core = 1.0 + (emission_core - 1.0) * sp.density_atlas_scale

    emission = np.clip(emission_core * radial_mask, 0.0, None)
    density = np.clip(density_core * radial_mask, 0.0, None)
    emission = np.where(radial_mask > 0.0, emission, 0.0)
    density = np.where(radial_mask > 0.0, density, 0.0)

    return VisualAtlas(
        emission_weight=emission.astype(np.float32),
        density_weight=density.astype(np.float32),
        n_r=n_r,
        n_phi=n_phi,
        r_in=float(params.r_in),
        r_out=float(params.r_out),
    )


def sample_atlas_bilinear(
    atlas: np.ndarray,
    r: float | np.ndarray,
    phi: float | np.ndarray,
    r_in: float,
    r_out: float,
) -> float | np.ndarray:
    """NumPy 双线性采样 atlas（Taichi parity reference）。

    Args:
        atlas: 形状 `(n_r, n_phi)`。
        r, phi: 局部盘坐标（标量或数组）。
        r_in, r_out: 半径范围。

    Returns:
        采样值；`r` 在盘外时返回 0。
    """
    n_r, n_phi = atlas.shape
    r_arr = np.asarray(r, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)
    span = max(r_out - r_in, 1e-6)
    u = np.clip((r_arr - r_in) / span, 0.0, 1.0)
    v = (phi_arr % (2.0 * np.pi)) / (2.0 * np.pi)

    ri_f = u * (n_r - 1)
    pj_f = v * n_phi
    r0 = np.floor(ri_f).astype(np.int32)
    r1 = np.minimum(r0 + 1, n_r - 1)
    p0 = np.floor(pj_f).astype(np.int32) % n_phi
    p1 = (p0 + 1) % n_phi
    fr = ri_f - r0
    fp = pj_f - np.floor(pj_f)

    c00 = atlas[r0, p0]
    c10 = atlas[r1, p0]
    c01 = atlas[r0, p1]
    c11 = atlas[r1, p1]
    c0 = c00 * (1.0 - fr) + c10 * fr
    c1 = c01 * (1.0 - fr) + c11 * fr
    out = c0 * (1.0 - fp) + c1 * fp
    out = np.where((r_arr >= r_in) & (r_arr <= r_out), out, 0.0)
    if np.ndim(r) == 0 and np.ndim(phi) == 0:
        return float(out)
    return out
