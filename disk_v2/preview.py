"""Disk V2 静态参数校验预览（v2.1 Phase 4）。

这个模块**不**负责视觉验收，只用于：

- 把 V2 各场（密度、温度、F_clump、emission、palette 颜色）一次性算到一个
  小网格上，便于排错与单测快照。
- 提供 `render_disk_face_on_preview()` 与 `render_disk_edge_on_preview()` 两个
  入口，输出小尺寸 LDR RGB 数组，可在脚本里 imshow 或保存为 PNG。

视觉验收必须在 `render.py --disk_model v2` 主光追路径上做，本模块的输出仅用于
确认"V2 物理场和调色链路本身没坏"。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .palette import apply_palette, render_hdr_to_ldr
from .params import DiskV2PaletteParams, DiskV2Params, DiskV2StructureParams
from .physical_fields import density_field, temperature_field
from .structure_modulations import (
    _ClumpCenters,
    _sample_clump_centers,
    clump_modulation,
    hotspot_modulation,
    weak_mode_modulation,
)


def _build_emission_modulation(
    r_grid: np.ndarray,
    phi_grid: np.ndarray,
    z_grid: np.ndarray,
    params: DiskV2Params,
    structure_params: DiskV2StructureParams,
    seed: int,
    centers: Optional[_ClumpCenters],
) -> np.ndarray:
    """组合 F_struct_emission 但允许复用预生成的 clump centers。

    `disk_v2.structure_modulations.structure_modulation_emission` 在每次
    调用都重新生成 centers；preview 需要在多次切片上复用同一组 centers，
    避免不同切片看起来"团块位置在跳"。
    """
    mode = weak_mode_modulation(r_grid, phi_grid, params, structure_params)
    if centers is None:
        centers = _sample_clump_centers(params, structure_params, seed)
    clump = clump_modulation(
        r_grid, phi_grid, z=z_grid,
        params=params, structure_params=structure_params,
        centers=centers,
    )
    hotspot = hotspot_modulation(r_grid, phi_grid, params, structure_params, seed=seed + 1)
    return mode * clump * hotspot


def render_disk_face_on_preview(
    params: DiskV2Params,
    structure_params: Optional[DiskV2StructureParams] = None,
    palette_params: Optional[DiskV2PaletteParams] = None,
    resolution: int = 256,
    z: float = 0.0,
    seed: int = 42,
    centers: Optional[_ClumpCenters] = None,
) -> np.ndarray:
    """生成正视盘预览图（盘面正对相机）。

    Args:
        params: `DiskV2Params`。
        structure_params: 可选 `DiskV2StructureParams`，默认值由本函数构造。
        palette_params: 可选 `DiskV2PaletteParams`，默认 `physical` 模式。
        resolution: 输出图像边长，单位像素。
        z: 切片的垂向高度，默认 0（中面）。
        seed: 团块随机种子。
        centers: 可选预生成 `_ClumpCenters`，便于跨函数复用。

    Returns:
        形状 `(resolution, resolution, 3)` 的 LDR RGB 数组，每通道 `[0, 1]`。
        盘外像素为黑（HDR 强度为 0）。

    Notes:
        - 该函数只是把物理场 + 调色链路串起来的可执行示例，不替代主光追渲染。
        - 不做引力透镜、相对论 beaming 或体积积分；只在 `z` 切片上画 emission。
    """

    if structure_params is None:
        structure_params = DiskV2StructureParams()
    if palette_params is None:
        palette_params = DiskV2PaletteParams()

    # 屏幕坐标 → 局部盘坐标。
    extent = params.r_out * 1.05
    xs = np.linspace(-extent, extent, resolution)
    ys = np.linspace(-extent, extent, resolution)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")
    r_grid = np.sqrt(x_grid ** 2 + y_grid ** 2)
    phi_grid = np.arctan2(y_grid, x_grid)
    z_grid = np.full_like(r_grid, z)

    # 物理场。
    rho = density_field(r_grid, z_grid, params)
    T = temperature_field(r_grid, z_grid, params)

    # 发射率调制（含 centers 复用，保证多切片一致）。
    f_emission = _build_emission_modulation(
        r_grid, phi_grid, z_grid, params, structure_params, seed, centers,
    )

    # 简化的 emission：j = ρ^α · (T/T_peak_K)^β · F_emission。
    T_norm = T / max(params.T_peak_K, 1.0)
    j = (
        np.power(np.maximum(rho, 0.0), params.alpha_density)
        * np.power(np.maximum(T_norm, 0.0), params.beta_temperature)
        * f_emission
    )

    # 颜色：温度 → palette；强度 = emission。
    rgb_hdr = apply_palette(j, T, palette_params)
    rgb_ldr = render_hdr_to_ldr(rgb_hdr, palette_params)
    return rgb_ldr


def render_disk_edge_on_preview(
    params: DiskV2Params,
    structure_params: Optional[DiskV2StructureParams] = None,
    palette_params: Optional[DiskV2PaletteParams] = None,
    resolution: int = 256,
    seed: int = 42,
    centers: Optional[_ClumpCenters] = None,
) -> np.ndarray:
    """生成侧视盘预览图（盘面侧对相机），用最简单的"沿 y 方向 emission 累积"近似体积积分。

    Args:
        params: `DiskV2Params`。
        structure_params: 可选 `DiskV2StructureParams`，默认值由本函数构造。
        palette_params: 可选 `DiskV2PaletteParams`。
        resolution: 输出图像边长。
        seed: 团块随机种子。
        centers: 可选预生成 `_ClumpCenters`。

    Returns:
        形状 `(resolution, resolution, 3)` 的 LDR RGB 数组，每通道 `[0, 1]`。

    Notes:
        - 简化模型：相机看向 +y 方向，沿 y 方向对 emission 做一个 line integral。
          这不是真正的 V2 体积积分（没有 transmittance、没有引力透镜），
          只用于"看看盘有没有厚度感"。
        - 真正的视觉验收在 `render.py --disk_model v2` 上做。
    """

    if structure_params is None:
        structure_params = DiskV2StructureParams()
    if palette_params is None:
        palette_params = DiskV2PaletteParams()

    extent_x = params.r_out * 1.05
    extent_z = params.r_out * 0.25  # 盘厚不大，z 方向缩窄显示
    xs = np.linspace(-extent_x, extent_x, resolution)
    zs = np.linspace(-extent_z, extent_z, resolution)
    x_grid, z_grid = np.meshgrid(xs, zs, indexing="xy")

    # 沿 y 方向做 line integral：取 ys 离散点累积 emission。
    n_y = 32
    ys = np.linspace(-extent_x, extent_x, n_y)

    # 累积 emission 与温度加权颜色。
    rgb_hdr_accum = np.zeros((resolution, resolution, 3), dtype=np.float64)
    ds = (ys[-1] - ys[0]) / (n_y - 1)

    # 复用一组 clump centers，避免每个 y 切片重新采样。
    if centers is None:
        centers = _sample_clump_centers(params, structure_params, seed)

    for y in ys:
        r_grid = np.sqrt(x_grid ** 2 + y ** 2)
        phi_grid = np.arctan2(y, x_grid)
        rho = density_field(r_grid, z_grid, params)
        T = temperature_field(r_grid, z_grid, params)
        f_emission = _build_emission_modulation(
            r_grid, phi_grid, z_grid, params, structure_params, seed, centers,
        )
        T_norm = T / max(params.T_peak_K, 1.0)
        j = (
            np.power(np.maximum(rho, 0.0), params.alpha_density)
            * np.power(np.maximum(T_norm, 0.0), params.beta_temperature)
            * f_emission
        )
        rgb_hdr_accum += apply_palette(j, T, palette_params) * ds

    rgb_ldr = render_hdr_to_ldr(rgb_hdr_accum, palette_params)
    return rgb_ldr
