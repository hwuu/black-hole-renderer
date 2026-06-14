"""Disk V2 的 Taichi 实现（v2.1 Phase 4）。

本模块提供主渲染管线（`render.py`）所需的 Taichi 级别构件：

- `@ti.func` 形式的基础物理场（密度、温度、几何掩码）。
- `@ti.func` 形式的 F_clump 团块场采样。
- `@ti.func` 形式的 palette（黑体色 + cinematic）和 tonemap（Reinhard）。
- 顶层辅助类 `DiskV2Taichi`，负责把 Python 端的 `DiskV2Params/StructureParams/PaletteParams`
  + clump centers 推送到 Taichi field，并提供供 `@ti.kernel` 调用的工具。

`render.py` 在 `--disk_model v2` 路径下，会用本模块在体积积分内部
完成 "几何掩码 → 物理场采样 → 结构调制 → 发射率 → 颜色 → HDR 累积"。

设计要点：

- 接口与 `disk_v2.physical_fields / structure_modulations / palette` 的 NumPy
  实现严格 parity（见 `tests/unit/test_disk_v2_numpy_taichi_parity.py`）。
- Taichi 端只放数值核函数，不放 Python 控制流；clump centers 等"集合"
  在外部一次性构造，传入 field。
- 黑体色查表与 NumPy 实现共用同一个 Tanner Helland 分段公式。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import taichi as ti

from .params import DiskV2PaletteParams, DiskV2Params, DiskV2StructureParams
from .structure_modulations import (
    _ClumpCenters,
    _sample_clump_centers,
    _sample_hotspot_centers,
    _sample_shear_components,
    hotspot_modulation,
    shear_modulation,
)
from .palette import physical_temperature_outer_K


# Disk V2 物理常数（与 disk_v2.physical_fields 一致）。
_THIN_DISK_PEAK_OVER_R_IN: float = 49.0 / 36.0
_THIN_DISK_PEAK_VALUE_RAW: float = (
    _THIN_DISK_PEAK_OVER_R_IN ** (-0.75)
    * (1.0 - 1.0 / math.sqrt(_THIN_DISK_PEAK_OVER_R_IN)) ** 0.25
)
_THIN_DISK_NORM_FACTOR: float = 1.0 / _THIN_DISK_PEAK_VALUE_RAW


@ti.func
def _ti_smoothstep(edge0, edge1, x):
    """三次平滑插值，与 NumPy `disk_v2.geometry.smoothstep` 数学等价。

    Args:
        edge0: 平滑区起点（必须 < edge1）。
        edge1: 平滑区终点。
        x: 输入标量。

    Returns:
        `[0, 1]` 区间的标量。`x <= edge0` 返回 0；`x >= edge1` 返回 1；
        中间用三次多项式平滑过渡。

    Formula:
        ```
        t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
        out = t² (3 - 2t)
        ```
    """
    denom = edge1 - edge0
    t = (x - edge0) / denom
    t = ti.min(ti.max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.func
def disk_half_thickness_ti(r, h0, beta_h, r_in):
    """Taichi 版本的 `disk_half_thickness`。

    Args:
        r: 径向距离（已保证 ≥ 0）。
        h0: 厚度比例。
        beta_h: 厚度径向幂指数。
        r_in: 盘内半径。

    Returns:
        半厚度 `H(r) = h0 · max(r, r_in) · (max(r, r_in) / r_in)^beta_h`。
    """
    safe_r = ti.max(r, r_in)
    return h0 * safe_r * ti.pow(safe_r / r_in, beta_h)


@ti.func
def disk_radial_weight_ti(r, r_in, r_out, edge_softness):
    """Taichi 版本的 `disk_radial_weight`。

    Args:
        r: 径向距离。
        r_in: 盘内半径。
        r_out: 盘外半径。
        edge_softness: 边界平滑比例。

    Returns:
        `W_r(r) ∈ [0, 1]`。盘外（含精确边界）返回 0，盘内中部返回 1。
    """
    radial_span = r_out - r_in
    soft_width = ti.max(radial_span * edge_softness, 1e-12)
    inner = _ti_smoothstep(r_in, r_in + soft_width, r)
    outer = 1.0 - _ti_smoothstep(r_out - soft_width, r_out, r)
    w = inner * outer
    # 盘外严格为 0。
    if r <= r_in or r >= r_out:
        w = 0.0
    return w


@ti.func
def disk_vertical_weight_ti(r, z, h0, beta_h,
                            r_in, r_out):
    """Taichi 版本的 `disk_vertical_weight`。

    Args:
        r: 径向距离。
        z: 垂向高度。
        h0, beta_h, r_in, r_out: 几何参数。

    Returns:
        `W_z(r, z) ∈ [0, 1]`。`|z| >= H(r)` 或 `r` 在径向外时返回 0。
    """
    h = ti.max(disk_half_thickness_ti(r, h0, beta_h, r_in), 1e-12)
    xi = ti.abs(z) / h
    w = 1.0 - _ti_smoothstep(0.0, 1.0, xi)
    # 径向不在盘内时返回 0（与 NumPy 实现行为一致：用 radial_mask 闭区间判定）。
    if r < r_in or r > r_out:
        w = 0.0
    return w


@ti.func
def disk_volume_mask_ti(r, z, h0, beta_h,
                       r_in, r_out):
    """Taichi 版本的 `disk_volume_mask`。

    Args:
        r: 径向距离。
        z: 垂向高度。
        h0, beta_h, r_in, r_out: 几何参数。

    Returns:
        1 表示盘内，0 表示盘外。
    """
    h = disk_half_thickness_ti(r, h0, beta_h, r_in)
    inside = 0
    if r >= r_in and r <= r_out and ti.abs(z) <= h:
        inside = 1
    return inside


@ti.func
def midplane_density_ti(r, r_in, r_out,
                        rho_power, edge_softness):
    """Taichi 版本的 `midplane_density_field`。

    Args:
        r: 径向距离。
        r_in, r_out, rho_power, edge_softness: 物理场参数。

    Returns:
        中面密度，`r <= r_in` 时为 0。
    """
    result = 0.0
    if r > r_in:
        safe_r = ti.max(r, r_in)
        ratio = safe_r / r_in
        inner_term = ti.max(1.0 - ti.sqrt(r_in / safe_r), 0.0)
        w_r = disk_radial_weight_ti(r, r_in, r_out, edge_softness)
        result = ti.pow(ratio, -rho_power) * ti.sqrt(inner_term) * w_r
    return result


@ti.func
def midplane_temperature_ti(r, r_in, r_out,
                            T_peak_K, edge_softness):
    """Taichi 版本的 `midplane_temperature_field`。

    Args:
        r: 径向距离。
        r_in, r_out, T_peak_K, edge_softness: 温度场参数。

    Returns:
        中面温度（单位 K）。`r <= r_in` 时返回 0。
    """
    result = 0.0
    if r > r_in:
        safe_r = ti.max(r, r_in)
        ratio = safe_r / r_in
        inner_term = ti.max(1.0 - ti.sqrt(r_in / safe_r), 0.0)
        w_r = disk_radial_weight_ti(r, r_in, r_out, edge_softness)
        raw = ti.pow(ratio, -0.75) * ti.pow(inner_term, 0.25)
        result = T_peak_K * _THIN_DISK_NORM_FACTOR * raw * w_r
    return result


@ti.func
def density_field_ti(r, z, r_in, r_out,
                    rho_power, h0, beta_h,
                    edge_softness):
    """Taichi 版本的 `density_field`（带垂向高斯轮廓）。

    Args:
        r: 径向距离。
        z: 垂向高度。
        其余参数同 `midplane_density_ti` + `disk_half_thickness_ti`。

    Returns:
        二维密度 `ρ(r, z)`。盘外为 0。
    """
    result = 0.0
    if disk_volume_mask_ti(r, z, h0, beta_h, r_in, r_out) == 1:
        rho_m = midplane_density_ti(r, r_in, r_out, rho_power, edge_softness)
        h = ti.max(disk_half_thickness_ti(r, h0, beta_h, r_in), 1e-12)
        zh = z / h
        wz = disk_vertical_weight_ti(r, z, h0, beta_h, r_in, r_out)
        result = rho_m * ti.exp(-0.5 * zh * zh) * wz
    return result


@ti.func
def temperature_field_ti(r, z, r_in, r_out,
                        T_peak_K, h0, beta_h,
                        edge_softness):
    """Taichi 版本的 `temperature_field`。

    Args:
        r: 径向距离。
        z: 垂向高度。
        其余参数同 `midplane_temperature_ti` + 几何参数。

    Returns:
        温度（单位 K）。盘外为 0。
    """
    result = 0.0
    if disk_volume_mask_ti(r, z, h0, beta_h, r_in, r_out) == 1:
        t_m = midplane_temperature_ti(r, r_in, r_out, T_peak_K, edge_softness)
        h = ti.max(disk_half_thickness_ti(r, h0, beta_h, r_in), 1e-12)
        v_factor = ti.max(ti.min(1.0 - 0.25 * ti.abs(z) / h, 1.0), 0.0)
        wz = disk_vertical_weight_ti(r, z, h0, beta_h, r_in, r_out)
        result = t_m * v_factor * wz
    return result


@ti.func
def _wrap_pi(x):
    """把角度差包裹到 `[-π, π]`，用于团块的角向距离计算。

    Args:
        x: 任意实数（弧度）。

    Returns:
        `[-π, π]` 区间的标量。
    """
    return ti.atan2(ti.sin(x), ti.cos(x))


@ti.func
def _clump_kernel_value(d):
    """紧支撑锐利衰减核：`d` 是归一化距离，`d > 1` 时返回 0。

    Args:
        d: 归一化距离（无量纲）。

    Returns:
        核值，落在 `[0, 1]`。

    Formula:
        ```
        k = max(0, 1 - d)
        out = k² (3 - 2k)
        ```
    """
    k = ti.max(0.0, 1.0 - d)
    return k * k * (3.0 - 2.0 * k)


@ti.data_oriented
class DiskV2Taichi:
    """Disk V2 的 Taichi 端句柄。

    负责把 Python 端的参数 + clump centers 推送到 Taichi field，并暴露
    `sample_emission` / `sample_palette_color` / `tonemap_reinhard` 一类
    可在 `@ti.kernel` 内调用的 `@ti.func`。

    Args:
        params: `DiskV2Params`。
        structure_params: `DiskV2StructureParams`。
        palette_params: `DiskV2PaletteParams`。
        seed: 用于生成 clump centers 的随机种子。
        centers: 可选预生成的 `_ClumpCenters`（用于 parity 测试）。

    Notes:
        本类只在 `--disk_model v2` 路径上构造一次；构造时把所有标量参数
        缓存为 Python float（供 `@ti.func` 调用时按参数闭包传入），把
        clump centers 上传到 Taichi field。
    """

    def __init__(
        self,
        params: DiskV2Params,
        structure_params: DiskV2StructureParams,
        palette_params: DiskV2PaletteParams,
        seed: int = 42,
        centers: Optional[_ClumpCenters] = None,
    ) -> None:
        self.params = params
        self.structure_params = structure_params
        self.palette_params = palette_params
        self.seed = seed

        # 把 dataclass 的标量字段平铺为 self.<name>，便于 @ti.func 内访问。
        # Taichi 不接受 dataclass 作为 runtime 常量，必须用 Python float。
        self._r_in = float(params.r_in)
        self._r_out = float(params.r_out)
        self._h0 = float(params.h0)
        self._beta_h = float(params.beta_h)
        self._rho_power = float(params.rho_power)
        self._T_peak_K = float(params.T_peak_K)
        self._edge_softness = float(params.edge_softness)
        self._alpha_density = float(params.alpha_density)
        self._beta_temperature = float(params.beta_temperature)
        self._clump_strength = float(structure_params.clump_strength)
        self._clump_emission_weight = float(structure_params.clump_emission_weight)
        self._shear_strength = float(structure_params.shear_strength)
        self._mode1_strength = float(structure_params.mode1_strength)
        self._mode2_strength = float(structure_params.mode2_strength)
        self._hotspot_strength = float(structure_params.hotspot_strength)
        self._hotspot_phi_sigma = float(structure_params.hotspot_phi_sigma)
        self._hotspot_logr_sigma = float(structure_params.hotspot_logr_sigma)
        self._gamma = float(palette_params.gamma)
        self._cinematic_saturation = float(palette_params.cinematic_saturation)
        self._cinematic_warm_shift = float(palette_params.cinematic_warm_shift)
        self._visual_temp_outer_K = float(palette_params.visual_temp_outer_K)
        self._visual_temp_inner_K = float(palette_params.visual_temp_inner_K)
        t_outer_phys = physical_temperature_outer_K(self._T_peak_K)
        self._log_t_peak = float(math.log(max(self._T_peak_K, t_outer_phys + 1.0)))
        self._log_t_outer = float(math.log(max(t_outer_phys, 1.0)))
        self._visual_log_span = float(
            math.log(max(palette_params.visual_temp_inner_K, 1.0))
            - math.log(max(palette_params.visual_temp_outer_K, 1.0))
        )
        # 字符串模式不能进 @ti.func，必须在 Python 端做模式分发。
        self._is_cinematic = (palette_params.palette_mode == "cinematic")

        if centers is None:
            centers = _sample_clump_centers(params, structure_params, seed)
        self.centers = centers

        # 上传 clump centers 到 Taichi field。
        n = len(centers.r)
        self._clump_count = n
        self._clump_r = ti.field(dtype=ti.f32, shape=n)
        self._clump_phi = ti.field(dtype=ti.f32, shape=n)
        self._clump_z = ti.field(dtype=ti.f32, shape=n)
        self._clump_amp = ti.field(dtype=ti.f32, shape=n)
        self._clump_sigma_z = ti.field(dtype=ti.f32, shape=n)

        self._clump_r.from_numpy(centers.r.astype(np.float32))
        self._clump_phi.from_numpy(centers.phi.astype(np.float32))
        self._clump_z.from_numpy(centers.z.astype(np.float32))
        self._clump_amp.from_numpy(centers.amplitude.astype(np.float32))

        # 预计算每个团块所在中心 `r_k` 处的 σ_z（与 NumPy 实现一致）。
        from .geometry import disk_half_thickness  # 避免顶层循环依赖
        h_centers = np.asarray(disk_half_thickness(centers.r, params), dtype=np.float64)
        sigma_z = np.maximum(
            structure_params.clump_vertical_sigma_scale * h_centers, 1e-6
        ).astype(np.float32)
        self._clump_sigma_z.from_numpy(sigma_z)

        # σ_r 是全局常量（不随团块变化）。
        self._sigma_r = float(structure_params.clump_radial_sigma_scale * params.r_in)
        self._sigma_phi = float(structure_params.clump_phi_sigma)

        # --- 剪切纹理分量（与 NumPy seed 规则一致） ---
        shear_parts = _sample_shear_components(structure_params, seed)
        n_shear = len(shear_parts.amplitude)
        self._shear_count = n_shear
        self._shear_phi_freq = ti.field(dtype=ti.i32, shape=n_shear)
        self._shear_log_r_freq = ti.field(dtype=ti.i32, shape=n_shear)
        self._shear_phase = ti.field(dtype=ti.f32, shape=n_shear)
        self._shear_amplitude = ti.field(dtype=ti.f32, shape=n_shear)
        self._shear_phi_freq.from_numpy(shear_parts.phi_frequency.astype(np.int32))
        self._shear_log_r_freq.from_numpy(shear_parts.log_r_frequency.astype(np.int32))
        self._shear_phase.from_numpy(shear_parts.phase.astype(np.float32))
        self._shear_amplitude.from_numpy(shear_parts.amplitude.astype(np.float32))
        r_probe = np.linspace(params.r_in + 1e-3, params.r_out - 1e-3, 32)
        phi_probe = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        rg, pg = np.meshgrid(r_probe, phi_probe, indexing="ij")
        shear_probe = np.asarray(
            shear_modulation(rg, pg, params, structure_params, seed=seed),
            dtype=np.float64,
        )
        self._shear_signed_scale = float(
            max(np.percentile(np.abs(shear_probe - 1.0), 99) / max(structure_params.shear_strength, 1e-6), 1e-3)
        )

        # --- 热斑中心 ---
        hotspot_centers = _sample_hotspot_centers(params, structure_params, seed + 1)
        n_hot = len(hotspot_centers.phi)
        self._hotspot_count = n_hot
        self._hotspot_phi = ti.field(dtype=ti.f32, shape=n_hot)
        self._hotspot_log_r = ti.field(dtype=ti.f32, shape=n_hot)
        self._hotspot_weight = ti.field(dtype=ti.f32, shape=n_hot)
        self._hotspot_phi.from_numpy(hotspot_centers.phi.astype(np.float32))
        self._hotspot_log_r.from_numpy(hotspot_centers.log_r.astype(np.float32))
        self._hotspot_weight.from_numpy(hotspot_centers.weight.astype(np.float32))
        hotspot_probe = np.asarray(
            hotspot_modulation(rg, pg, params, structure_params, seed=seed + 1),
            dtype=np.float64,
        )
        self._hotspot_signed_scale = float(
            max(np.percentile(np.abs(hotspot_probe - 1.0), 99) / max(structure_params.hotspot_strength, 1e-6), 1e-3)
        )

        # --- 视觉 atlas（V1 云雾预烘焙） ---
        self._use_visual_atlas = bool(structure_params.use_visual_atlas)
        if self._use_visual_atlas:
            from .visual_atlas import build_visual_atlas

            atlas = build_visual_atlas(params, structure_params, seed=seed)
            self._atlas_n_r = int(atlas.n_r)
            self._atlas_n_phi = int(atlas.n_phi)
            self._atlas_r_in = float(atlas.r_in)
            self._atlas_r_out = float(atlas.r_out)
            self._emission_atlas = ti.field(
                dtype=ti.f32, shape=(self._atlas_n_r, self._atlas_n_phi),
            )
            self._density_atlas = ti.field(
                dtype=ti.f32, shape=(self._atlas_n_r, self._atlas_n_phi),
            )
            self._emission_atlas.from_numpy(atlas.emission_weight.astype(np.float32))
            self._density_atlas.from_numpy(atlas.density_weight.astype(np.float32))
        else:
            self._atlas_n_r = 1
            self._atlas_n_phi = 1
            self._atlas_r_in = float(params.r_in)
            self._atlas_r_out = float(params.r_out)
            self._emission_atlas = ti.field(dtype=ti.f32, shape=(1, 1))
            self._density_atlas = ti.field(dtype=ti.f32, shape=(1, 1))
            self._emission_atlas.from_numpy(np.ones((1, 1), dtype=np.float32))
            self._density_atlas.from_numpy(np.ones((1, 1), dtype=np.float32))

    @ti.func
    def _sample_atlas_field(self, atlas_field, r, phi):
        """双线性采样 `(n_r, n_phi)` atlas；盘外返回 0。"""
        result = 0.0
        span = self._atlas_r_out - self._atlas_r_in
        if span > 1e-6 and r >= self._atlas_r_in and r <= self._atlas_r_out:
            u = (r - self._atlas_r_in) / span
            phi_w = phi
            while phi_w < 0.0:
                phi_w += 2.0 * ti.math.pi
            while phi_w >= 2.0 * ti.math.pi:
                phi_w -= 2.0 * ti.math.pi
            v = phi_w / (2.0 * ti.math.pi)
            n_r = ti.cast(self._atlas_n_r, ti.f32)
            n_phi = ti.cast(self._atlas_n_phi, ti.f32)
            ri_f = u * (n_r - 1.0)
            pj_f = v * n_phi
            r0 = ti.cast(ti.floor(ri_f), ti.i32)
            r1 = ti.min(r0 + 1, ti.cast(n_r, ti.i32) - 1)
            p0 = ti.cast(ti.floor(pj_f), ti.i32) % ti.cast(n_phi, ti.i32)
            p1 = (p0 + 1) % ti.cast(n_phi, ti.i32)
            fr = ri_f - ti.cast(r0, ti.f32)
            fp = pj_f - ti.floor(pj_f)
            c00 = atlas_field[r0, p0]
            c10 = atlas_field[r1, p0]
            c01 = atlas_field[r0, p1]
            c11 = atlas_field[r1, p1]
            c0 = c00 * (1.0 - fr) + c10 * fr
            c1 = c01 * (1.0 - fr) + c11 * fr
            result = c0 * (1.0 - fp) + c1 * fp
        return result

    @ti.func
    def sample_emission_atlas_ti(self, r, phi):
        """采样发射 atlas 乘子。"""
        if ti.static(self._use_visual_atlas):
            return self._sample_atlas_field(self._emission_atlas, r, phi)
        return 1.0

    @ti.func
    def sample_density_atlas_ti(self, r, phi):
        """采样密度 atlas 乘子。"""
        if ti.static(self._use_visual_atlas):
            return self._sample_atlas_field(self._density_atlas, r, phi)
        return 1.0

    @ti.func
    def sample_atlas_color_mod_ti(self, r, phi):
        """Atlas 亮度→RGB 调制（V1 纹理 luminosity 近似，增强盘面细节对比）。

        Returns:
            围绕 ~1 波动的乘子，bright filament 处 > 1，云雾暗区 < 1。
        """
        if ti.static(self._use_visual_atlas):
            ew = self.sample_emission_atlas_ti(r, phi)
            return ti.pow(ti.max(ew, 0.0), 0.62)
        return 1.0

    @ti.func
    def clump_signed(self, r, phi, z):
        """采样团块场的 signed 量（未乘 `clump_strength`、未加 1）。

        Args:
            r: 径向距离。
            phi: 方位角（弧度）。
            z: 垂向高度。

        Returns:
            来自所有团块的 signed 贡献之和，范围由 amplitude 决定。
            `_clip_3sigma` 在 Taichi 路径里不做（依赖全场 std，无法逐点算），
            而是依赖外部对结果再做有界裁剪。
        """
        sigma_r = self._sigma_r
        sigma_phi = self._sigma_phi
        accum = 0.0
        for k in range(self._clump_count):
            r_k = self._clump_r[k]
            phi_k = self._clump_phi[k]
            z_k = self._clump_z[k]
            amp_k = self._clump_amp[k]
            sigma_z = self._clump_sigma_z[k]

            dr = (r - r_k) / sigma_r
            dp_raw = _wrap_pi(phi - phi_k)
            # 与 NumPy 实现一致：d_phi 先按 sigma_r/r_k 标定，再乘 sigma_r/(sigma_phi*r_k)。
            d_phi = dp_raw * r_k / sigma_r
            d_phi = d_phi * (sigma_r / ti.max(sigma_phi * r_k, 1e-6))
            dz = (z - z_k) / sigma_z

            d2 = dr * dr + d_phi * d_phi + dz * dz
            d = ti.sqrt(d2)
            kernel = _clump_kernel_value(d)
            accum += amp_k * kernel
        return accum

    @ti.func
    def clump_modulation_ti(self, r, phi, z):
        """Taichi 版本的 `clump_modulation`，盘外为 1。

        Args:
            r, phi, z: 局部盘坐标。

        Returns:
            `F_clump(r, φ, z)`。盘外返回 1。
        """
        result = 1.0
        w_r = disk_radial_weight_ti(r, self._r_in, self._r_out, self._edge_softness)
        if w_r > 0.0:
            signed = self.clump_signed(r, phi, z)
            # 把 signed 限到 [-1, 1]（NumPy 端用 3σ 截断，这里用直接 clamp）。
            # 这是 parity 测试容差的主要来源。
            signed = ti.max(ti.min(signed, 1.0), -1.0)
            result = 1.0 + self._clump_strength * signed
        return result

    @ti.func
    def clump_modulation_emission_ti(self, r, phi, z):
        """发射率路径上的团块调制（降低视觉权重）。

        Returns:
            `1 + clump_emission_weight · (F_clump - 1)`。盘外为 1。
        """
        f_full = self.clump_modulation_ti(r, phi, z)
        return 1.0 + self._clump_emission_weight * (f_full - 1.0)

    @ti.func
    def mode_modulation_ti(self, r, phi):
        """Taichi 版弱模态调制 `F_mode`。"""
        result = 1.0
        w_r = disk_radial_weight_ti(r, self._r_in, self._r_out, self._edge_softness)
        if w_r > 0.0:
            log_r = ti.log(ti.max(r, self._r_in) / self._r_in)
            raw = (
                self._mode1_strength * ti.cos(phi + 0.35 * log_r)
                + self._mode2_strength * ti.cos(2.0 * phi - 0.65 * log_r)
            )
            result = 1.0 + raw
        return result

    @ti.func
    def shear_modulation_ti(self, r, phi):
        """Taichi 版剪切纹理调制 `F_shear`（逐点 clamp 近似 3σ）。"""
        result = 1.0
        w_r = disk_radial_weight_ti(r, self._r_in, self._r_out, self._edge_softness)
        if w_r > 0.0:
            log_r = ti.log(ti.max(r, self._r_in) / self._r_in)
            raw = 0.0
            for k in range(self._shear_count):
                pf = ti.cast(self._shear_phi_freq[k], ti.f32)
                lrf = ti.cast(self._shear_log_r_freq[k], ti.f32)
                ph = self._shear_phase[k]
                amp = self._shear_amplitude[k]
                raw += amp * ti.cos(pf * phi + lrf * log_r + ph)
                raw += 0.6 * amp * ti.sin(
                    (pf + 1.0) * phi - (lrf + 0.5) * log_r + 0.7 * ph
                )
            signed = ti.max(ti.min(raw / self._shear_signed_scale, 1.0), -1.0)
            result = 1.0 + self._shear_strength * signed
        return result

    @ti.func
    def hotspot_modulation_ti(self, r, phi):
        """Taichi 版热斑调制 `F_hotspot`。"""
        result = 1.0
        w_r = disk_radial_weight_ti(r, self._r_in, self._r_out, self._edge_softness)
        if w_r > 0.0:
            log_r = ti.log(ti.max(r, self._r_in) / self._r_in)
            raw = 0.0
            halo_phi_scale = 1.8
            halo_logr_scale = 1.8
            halo_weight_scale = 0.6
            for k in range(self._hotspot_count):
                dphi = _wrap_pi(phi - self._hotspot_phi[k])
                dlog = (log_r - self._hotspot_log_r[k]) / self._hotspot_logr_sigma
                core = ti.exp(
                    -0.5 * (dphi / self._hotspot_phi_sigma) ** 2 - 0.5 * dlog * dlog
                )
                halo = ti.exp(
                    -0.5 * (dphi / (halo_phi_scale * self._hotspot_phi_sigma)) ** 2
                    -0.5 * ((log_r - self._hotspot_log_r[k]) / (halo_logr_scale * self._hotspot_logr_sigma)) ** 2
                )
                raw += self._hotspot_weight[k] * (core - halo_weight_scale * halo)
            signed = ti.max(ti.min(raw / self._hotspot_signed_scale, 1.0), -1.0)
            result = 1.0 + self._hotspot_strength * signed
        return result

    @ti.func
    def sample_density(self, r, phi, z):
        """采样带结构调制的密度场。

        视觉 atlas 开启时：`ρ_envelope · density_atlas · F_clump_weak`。
        否则回退：`ρ_envelope · F_shear · F_clump`。
        """
        rho_e = density_field_ti(
            r, z, self._r_in, self._r_out, self._rho_power,
            self._h0, self._beta_h, self._edge_softness,
        )
        if ti.static(self._use_visual_atlas):
            f_atlas = self.sample_density_atlas_ti(r, phi)
            f_clump = self.clump_modulation_ti(r, phi, z)
            return rho_e * f_atlas * f_clump
        f_struct = self.shear_modulation_ti(r, phi) * self.clump_modulation_ti(r, phi, z)
        return rho_e * f_struct

    @ti.func
    def sample_temperature(self, r, z):
        """采样温度场（不含结构调制；用于颜色映射）。

        Args:
            r, z: 局部盘坐标。

        Returns:
            `T(r, z)`（单位 K）。盘外为 0。
        """
        return temperature_field_ti(
            r, z, self._r_in, self._r_out, self._T_peak_K,
            self._h0, self._beta_h, self._edge_softness,
        )

    @ti.func
    def sample_emission(self, r, phi, z):
        """采样发射率。

        视觉 atlas 开启时：
        `j = ρ_envelope^α · T^β · emission_atlas · F_mode · F_hotspot`。
        """
        rho_e = density_field_ti(
            r, z, self._r_in, self._r_out, self._rho_power,
            self._h0, self._beta_h, self._edge_softness,
        )
        t = temperature_field_ti(
            r, z, self._r_in, self._r_out, self._T_peak_K,
            self._h0, self._beta_h, self._edge_softness,
        )
        t_norm = t / ti.max(self._T_peak_K, 1.0)
        j_base = 0.0
        if ti.static(self._use_visual_atlas):
            ew = self.sample_emission_atlas_ti(r, phi)
            # 视觉模式：atlas 承载主亮度，物理场只做弱径向调制（避免外圈过暗）。
            rho_vis = ti.pow(ti.max(rho_e, 0.0), 0.25)
            t_vis = ti.pow(ti.max(t_norm, 0.0), 0.35)
            j_base = rho_vis * t_vis * ti.max(ew, 0.0)
        else:
            rho_tex = rho_e * self.shear_modulation_ti(r, phi)
            j_base = ti.pow(ti.max(rho_tex, 0.0), self._alpha_density) * ti.pow(
                ti.max(t_norm, 0.0), self._beta_temperature,
            )
        f_struct = self.mode_modulation_ti(r, phi) * self.hotspot_modulation_ti(r, phi)
        return j_base * f_struct

    @ti.func
    def sample_visual_disk_color_ti(self, r, phi, z):
        """视觉 atlas 模式盘面 RGB（V1 纹理思路：atlas 驱动色温 + 物理 T 弱混合）。

        Args:
            r, phi, z: 盘坐标。

        Returns:
            RGB 向量，每通道 `[0, 1]`。
        """
        ew = self.sample_emission_atlas_ti(r, phi)
        t_aniso = ti.pow(ti.min(ti.max(ew, 0.0), 1.0), 0.55)
        t_vis = self._visual_temp_outer_K + t_aniso * (
            self._visual_temp_inner_K - self._visual_temp_outer_K
        )
        rgb = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        if ti.static(self._is_cinematic):
            t = t_vis / 100.0
            t_safe = ti.max(t, 1e-6)
            t_m60 = ti.max(t - 60.0, 1e-6)
            t_m10 = ti.max(t - 10.0, 1e-6)
            r_c = 1.0
            if t > 66.0:
                r_c = ti.min(ti.max(1.292936 * ti.pow(t_m60, -0.1332047592), 0.0), 1.0)
            g_c = 0.0
            if t <= 66.0:
                g_c = ti.min(ti.max(0.390082 * ti.log(t_safe) - 0.631841, 0.0), 1.0)
            else:
                g_c = ti.min(ti.max(1.129891 * ti.pow(t_m60, -0.0755148492), 0.0), 1.0)
            b_c = 1.0
            if t < 66.0:
                if t <= 19.0:
                    b_c = 0.0
                else:
                    b_c = ti.min(ti.max(0.543207 * ti.log(t_m10) - 1.19625, 0.0), 1.0)
            sat = self._cinematic_saturation
            warm = self._cinematic_warm_shift
            luma = 0.2126 * r_c + 0.7152 * g_c + 0.0722 * b_c
            r_c = ti.min(ti.max(luma + sat * (r_c - luma), 0.0), 1.0)
            g_c = ti.min(ti.max(luma + sat * (g_c - luma), 0.0), 1.0)
            b_c = ti.min(ti.max(luma + sat * (b_c - luma), 0.0), 1.0)
            r_c = ti.min(r_c * (1.0 + warm), 1.0)
            b_c = ti.min(ti.max(b_c * (1.0 - warm), 0.0), 1.0)
            rgb = ti.Vector([r_c, g_c, b_c], dt=ti.f32)
            rgb[2] = ti.min(rgb[2], rgb[0])
        else:
            rgb = self.sample_palette_color(t_vis)
        filament = 0.28 + 0.72 * ti.pow(ti.max(ew, 0.0), 0.38)
        return rgb * filament

    @ti.func
    def sample_palette_color(self, T_K):
        """温度 → RGB（physical 直查 / cinematic 可见色温重映射）。

        Args:
            T_K: 温度（单位 K）。

        Returns:
            形状 (3,) 的 RGB 向量，每通道 `[0, 1]`。
        """
        rgb = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        if T_K > 0.0:
            t_source = T_K
            if ti.static(self._is_cinematic):
                safe_T = ti.max(T_K, ti.exp(self._log_t_outer))
                log_span = ti.max(self._log_t_peak - self._log_t_outer, 1e-6)
                t_norm = (ti.log(safe_T) - self._log_t_outer) / log_span
                t_norm = ti.min(ti.max(t_norm, 0.0), 1.0)
                t_source = self._visual_temp_outer_K + t_norm * (
                    self._visual_temp_inner_K - self._visual_temp_outer_K
                )
            t = t_source / 100.0
            t_safe = ti.max(t, 1e-6)
            t_m60 = ti.max(t - 60.0, 1e-6)
            t_m10 = ti.max(t - 10.0, 1e-6)

            r_c = 1.0
            if t > 66.0:
                r_c = ti.min(ti.max(1.292936 * ti.pow(t_m60, -0.1332047592), 0.0), 1.0)

            g_c = 0.0
            if t <= 66.0:
                g_c = ti.min(ti.max(0.390082 * ti.log(t_safe) - 0.631841, 0.0), 1.0)
            else:
                g_c = ti.min(ti.max(1.129891 * ti.pow(t_m60, -0.0755148492), 0.0), 1.0)

            b_c = 1.0
            if t < 66.0:
                if t <= 19.0:
                    b_c = 0.0
                else:
                    b_c = ti.min(ti.max(0.543207 * ti.log(t_m10) - 1.19625, 0.0), 1.0)

            # cinematic 模式：饱和度增强 + 暖色偏移。
            if ti.static(self._is_cinematic):
                sat = self._cinematic_saturation
                warm = self._cinematic_warm_shift
                luma = 0.2126 * r_c + 0.7152 * g_c + 0.0722 * b_c
                r_c = ti.min(ti.max(luma + sat * (r_c - luma), 0.0), 1.0)
                g_c = ti.min(ti.max(luma + sat * (g_c - luma), 0.0), 1.0)
                b_c = ti.min(ti.max(luma + sat * (b_c - luma), 0.0), 1.0)
                r_c = ti.min(ti.max(r_c * (1.0 + warm), 0.0), 1.0)
                g_c = ti.min(ti.max(g_c * 1.0, 0.0), 1.0)
                b_c = ti.min(ti.max(b_c * (1.0 - warm), 0.0), 1.0)

            rgb = ti.Vector([r_c, g_c, b_c], dt=ti.f32)
        return rgb

    @ti.func
    def tonemap_reinhard(self, rgb_hdr):
        """Reinhard 色调映射：`x → x / (1 + x)`，逐通道。

        Args:
            rgb_hdr: HDR RGB 向量。

        Returns:
            LDR RGB 向量，每通道 `[0, 1)`。
        """
        safe = ti.Vector([
            ti.max(rgb_hdr[0], 0.0),
            ti.max(rgb_hdr[1], 0.0),
            ti.max(rgb_hdr[2], 0.0),
        ], dt=ti.f32)
        return safe / (1.0 + safe)

    @ti.func
    def gamma_correct_ti(self, rgb_lin):
        """sRGB 伽马校正：`x → clip(x, 0, 1)^(1/gamma)`。

        Args:
            rgb_lin: 线性 RGB 向量。

        Returns:
            伽马校正后的 LDR RGB 向量，每通道 `[0, 1]`。
        """
        inv = 1.0 / self._gamma
        clipped = ti.Vector([
            ti.min(ti.max(rgb_lin[0], 0.0), 1.0),
            ti.min(ti.max(rgb_lin[1], 0.0), 1.0),
            ti.min(ti.max(rgb_lin[2], 0.0), 1.0),
        ], dt=ti.f32)
        return ti.Vector([
            ti.pow(clipped[0], inv),
            ti.pow(clipped[1], inv),
            ti.pow(clipped[2], inv),
        ], dt=ti.f32)
