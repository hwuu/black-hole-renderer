"""Disk V2 的 Taichi 实现（v2.1 Phase 4）。

本模块提供主渲染管线（`render.py`）所需的 Taichi 级别构件：

- `@ti.func` 形式的基础物理场（密度、温度、几何掩码）。
- `@ti.func` 形式的 F_clump 团块场采样。
- `@ti.func` 形式的 palette（CIE 黑体色度 / 亮度查找表）和 tonemap（Reinhard）。
- 顶层辅助类 `DiskV2Taichi`，负责把 Python 端的 `DiskV2Params/StructureParams/PaletteParams`
  + clump centers 推送到 Taichi field，并提供供 `@ti.kernel` 调用的工具。

`render.py` 在 `--disk_model v2` 路径下，会用本模块在体积积分内部
完成 "几何掩码 → 物理场采样 → 结构调制 → 发射率 → 颜色 → HDR 累积"。

设计要点：

- 接口与 `disk_v2.physical_fields / structure_modulations / palette` 的 NumPy
  实现严格 parity（见 `tests/unit/test_disk_v2_numpy_taichi_parity.py`）。
- Taichi 端只放数值核函数，不放 Python 控制流；clump centers 等"集合"
  在外部一次性构造，传入 field。
- 黑体色度 / 亮度与 NumPy 实现共用同一份 CIE 查找表（palette.BB_LUT / LNY_LUT）。
"""


import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import taichi as ti

from .palette import (
    _blackbody_luminance_exact,
    BB_LUT,
    LNY_LUT,
    _BB_LUT_N,
    _BB_T_MAX_K,
    _BB_T_MIN_K,
    _LNY_T_MAX_K,
    _LNY_T_MIN_K,
    white_balance_gain,
)
from .advection import RigidRingBands, make_fields, upload as adv_upload
from .noise_ti import cascade, fbm_gradient, gnoise, hashf as _hashf, softplus
from .params import (
    DiskV2PaletteParams,
    DiskV2Params,
    DiskV2StructureParams,
    DiskV2VolumeParams,
)
from .structure_modulations import (
    _ClumpCenters,
    _sample_clump_centers,
    _sample_hotspot_centers,
    _sample_shear_components,
    hotspot_modulation,
    shear_modulation,
)


# Disk V2 物理常数（与 disk_v2.physical_fields 一致）。
_THIN_DISK_PEAK_OVER_R_IN: float = 49.0 / 36.0
_THIN_DISK_PEAK_VALUE_RAW: float = (
    _THIN_DISK_PEAK_OVER_R_IN ** (-0.75)
    * (1.0 - 1.0 / math.sqrt(_THIN_DISK_PEAK_OVER_R_IN)) ** 0.25
)
_THIN_DISK_NORM_FACTOR: float = 1.0 / _THIN_DISK_PEAK_VALUE_RAW


@ti.func
def schwarzschild_gravitational_g_ti(r_em, r_obs, rs):
    """Schwarzschild 静态引力红移因子 `nu_obs / nu_em`。"""
    em = ti.max(r_em, rs + 1e-6)
    obs = ti.max(r_obs, rs + 1e-6)
    numerator = ti.sqrt(ti.max(1.0 - rs / em, 1e-12))
    denominator = ti.sqrt(ti.max(1.0 - rs / obs, 1e-12))
    return numerator / denominator


@ti.func
def schwarzschild_orbital_beta_ti(r, rs, eps, beta_cap):
    """本地静止观者测得的 Schwarzschild 赤道圆轨道速度（与 `relativity.orbital_beta_local` 一致）。

    Args:
        r: 发射半径（r_s）。
        rs: Schwarzschild 半径。
        eps: 分母 `1 − 2M/r` 下限。
        beta_cap: 速度上限。

    Returns:
        `β = sqrt(M/r) / sqrt(1 − 2M/r)`，范围 `[0, beta_cap]`；ISCO 处 0.5。
    """
    safe_r = ti.max(r, rs + 1e-6)
    mass = 0.5 * rs
    denom = ti.sqrt(ti.max(1.0 - 2.0 * mass / safe_r, eps))
    beta = ti.sqrt(mass / safe_r) / denom
    return ti.min(ti.max(beta, 0.0), beta_cap)


@ti.func
def local_photon_direction_ti(pos, k_coord, rs):
    """光子坐标方向 → 本地静止观者单位方向（与 `relativity.local_photon_direction` 一致）。

    Args:
        pos: 位置向量（r_s）。
        k_coord: 坐标方向（不必归一）。
        rs: Schwarzschild 半径。

    Returns:
        单位向量；`tanψ_local = sqrt(1 − r_s/r) · tanψ_coord`。
    """
    r = ti.max(pos.norm(), 1e-6)
    r_hat = pos / r
    k_rad = k_coord.dot(r_hat) * r_hat
    k_tan = (k_coord - k_rad) * ti.sqrt(ti.max(1.0 - rs / r, 1e-12))
    return (k_rad + k_tan).normalized()


@ti.func
def disk_g_factor_ti(pos, trace_dir, r_obs, rs):
    """盘面圆轨道发射体的频移 g（与 `relativity.disk_g_factor` 一致）。

    Args:
        pos: 盘局部坐标发射点（盘面 z = 0，逆时针旋转）。
        trace_dir: 反向追踪光线方向（光子真实方向为其反向）。
        r_obs: 静止观察者半径。
        rs: Schwarzschild 半径。

    Returns:
        `g = g_grav / (γ (1 − β cosθ_local))`。
    """
    big_r = ti.max(ti.sqrt(pos[0] * pos[0] + pos[1] * pos[1]), 1e-6)
    r3 = ti.max(pos.norm(), rs + 1e-6)
    beta = schwarzschild_orbital_beta_ti(ti.max(big_r, 3.0 * rs), rs, 1e-6, 0.99)
    v_hat = ti.Vector([-pos[1], pos[0], 0.0]) / big_r
    cos_th = v_hat.dot(local_photon_direction_ti(pos, -trace_dir, rs))
    gamma = 1.0 / ti.sqrt(1.0 - beta * beta)
    g_grav = ti.sqrt(ti.max(1.0 - rs / r3, 1e-12)) / ti.sqrt(ti.max(1.0 - rs / r_obs, 1e-12))
    return g_grav / (gamma * (1.0 - beta * cos_th))


@ti.func
def planck_band_boost_ti(t_em, g):
    """550 nm 波段频移亮度增强 `B_ν(gT)/B_ν(T)`（与 `relativity.planck_band_boost` 一致）。

    Args:
        t_em: 发射温度（K）。
        g: 频移因子。

    Returns:
        `(exp(x/T) − 1) / (exp(x/(gT)) − 1)`，x = 26160 K。
    """
    t = ti.max(t_em, 1.0)
    gg = ti.max(g, 1e-6)
    x_em = ti.min(26160.0 / t, 80.0)
    x_obs = ti.min(26160.0 / (t * gg), 80.0)
    return (ti.exp(x_em) - 1.0) / (ti.exp(x_obs) - 1.0)


@ti.func
def doppler_g_factor_ti(beta, cos_theta):
    """特殊相对论 Doppler 因子。"""
    beta_safe = ti.min(ti.max(beta, 0.0), 0.999999)
    cos_safe = ti.min(ti.max(cos_theta, -1.0), 1.0)
    gamma = 1.0 / ti.sqrt(ti.max(1.0 - beta_safe * beta_safe, 1e-12))
    return 1.0 / ti.max(gamma * (1.0 - beta_safe * cos_safe), 1e-12)


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
        外边界使用 `0.6 r_s` 的最小软化宽度，内边界保持比例软化。
    """
    radial_span = r_out - r_in
    base_soft_width = ti.max(radial_span * edge_softness, 1e-12)
    max_width = ti.max(radial_span * 0.49, 1e-12)
    inner_soft_width = ti.min(ti.max(base_soft_width, 0.3), max_width)
    outer_soft_width = ti.min(ti.max(base_soft_width, 0.6), max_width)
    inner = _ti_smoothstep(r_in, r_in + inner_soft_width, r)
    outer = 1.0 - _ti_smoothstep(r_out - outer_soft_width, r_out, r)
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
def raw_midplane_density_ti(r, r_in, rho_power):
    """Taichi 版本的 `raw_midplane_density_field`（不乘径向 support）。"""
    result = 0.0
    if r > r_in:
        safe_r = ti.max(r, r_in)
        ratio = safe_r / r_in
        inner_term = ti.max(1.0 - ti.sqrt(r_in / safe_r), 0.0)
        result = ti.pow(ratio, -rho_power) * ti.sqrt(inner_term)
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
def raw_midplane_temperature_ti(r, r_in, T_peak_K):
    """Taichi 版本的 `raw_midplane_temperature_field`（不乘径向 support）。"""
    result = 0.0
    if r > r_in:
        safe_r = ti.max(r, r_in)
        ratio = safe_r / r_in
        inner_term = ti.max(1.0 - ti.sqrt(r_in / safe_r), 0.0)
        raw = ti.pow(ratio, -0.75) * ti.pow(inner_term, 0.25)
        result = T_peak_K * _THIN_DISK_NORM_FACTOR * raw
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
        emission_opacity_scale: 用于 v2.2 effective emission 的 opacity 缩放。
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
        emission_opacity_scale: float = 1.0,
        seed: int = 42,
        centers: Optional[_ClumpCenters] = None,
        volume_params: Optional[DiskV2VolumeParams] = None,
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
        self._emission_opacity_scale = float(emission_opacity_scale)
        self._clump_strength = float(structure_params.clump_strength)
        self._clump_emission_weight = float(structure_params.clump_emission_weight)
        self._shear_strength = float(structure_params.shear_strength)
        self._mode1_strength = float(structure_params.mode1_strength)
        self._mode2_strength = float(structure_params.mode2_strength)
        self._hotspot_strength = float(structure_params.hotspot_strength)
        self._hotspot_phi_sigma = float(structure_params.hotspot_phi_sigma)
        self._hotspot_logr_sigma = float(structure_params.hotspot_logr_sigma)
        self._gamma = float(palette_params.gamma)
        # 字符串模式不能进 @ti.func，必须在 Python 端做模式分发。
        # v2.3：cinematic 已删除，颜色只走 CIE 黑体查找表。
        self._is_aces_tonemap = (palette_params.tonemap_mode == "aces")
        # CIE 黑体查找表（与 palette.py 共用同一份数据 → parity 恒成立）。
        self._bb_lut = ti.Vector.field(3, dtype=ti.f32, shape=_BB_LUT_N)
        self._lny_lut = ti.field(dtype=ti.f32, shape=_BB_LUT_N)
        self._bb_lut.from_numpy(BB_LUT.astype(np.float32))
        self._lny_lut.from_numpy(LNY_LUT.astype(np.float32))
        self._wb_gain = ti.Vector.field(3, dtype=ti.f32, shape=())
        self._wb_gain[None] = np.asarray(
            white_balance_gain(float(palette_params.white_balance_K)), dtype=np.float32
        ).tolist()

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

        # --- 体积密度场（v2.3 S5，对应参考实现预设 M）---
        self.volume_params = volume_params
        if volume_params is not None:
            self._init_volume_params(volume_params)

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



    def _init_volume_params(self, vp: DiskV2VolumeParams) -> None:
        """把 DiskV2VolumeParams 平铺为 self._xxx（Taichi @ti.func 限制）+ 标定。"""
        import taichi as ti

        # SS 结构
        self._hr_ref = float(vp.hr_ref)
        self._r_ref_vol = float(vp.r_ref)
        self._f_ref_ss = 1.0 - math.sqrt(self._r_in / self._r_ref_vol)
        self._surf_noise = float(vp.surf_noise)
        # 灰大气
        self._grey_mix = float(vp.grey_mix)
        self._grey_cap = float(vp.grey_cap)
        self._core_opac = float(vp.core_opac)
        self._core_floor = float(vp.core_floor)
        self._dt_i = float(vp.dt_i)
        self._surf_lo = float(vp.surf_lo)
        self._surf_k = float(vp.surf_k)
        # 烟雾
        self._smoke_i = float(vp.smoke_i)
        self._smoke_tr = float(vp.smoke_tr)
        self._smoke_s = float(vp.smoke_s)
        self._n_cl_half = int(vp.n_cl_half)
        self._cl_spacing = float(vp.cl_spacing)
        self._cl_width = float(vp.cl_width)
        self._cl_decay = float(vp.cl_decay)
        self._cl_amp_norm = 1.0 / sum(
            math.exp(-self._cl_decay * abs(k)) for k in range(-self._n_cl_half, self._n_cl_half + 1)
        )
        self._fr_c = float(vp.fr_c)
        self._nphi_c = int(vp.nphi_c)
        self._fz_c = float(vp.fz_c)
        self._sigma_c = float(vp.sigma_c)
        self._cloud_c0 = float(vp.cloud_c0)
        self._cloud_soft = float(vp.cloud_soft)
        self._smoke_on = self._smoke_i > 0.0
        # 低频
        self._lowf_sigma = float(vp.lowf_sigma)
        self._fr_l = float(vp.fr_l)
        self._nphi_l = int(vp.nphi_l)
        self._dln_l = float(vp.dln_l)
        self._k_rigid_l = float(vp.k_rigid_l)
        # 主云
        self._kr_i = float(vp.kr_i)
        self._nphi_i = int(vp.nphi_i)
        self._l0_i = float(vp.l0_i)
        self._con_i = float(vp.con_i)
        self._kt_i = float(vp.kt_i)
        self._nphi_t = int(vp.nphi_t)
        self._lt0_i = float(vp.lt0_i)
        # 尘埃
        self._dust_em = float(vp.dust_em)
        self._dust_s = float(vp.dust_s)
        self._dust_on = bool(vp.dust_on)
        self._dust_kepler = bool(vp.dust_kepler)
        # 刚体环
        self._dln_r = float(vp.dln_r)
        self._k_rigid_vol = float(vp.k_rigid)
        self._lnr0_r = math.log(self._r_in) - 2 * self._dln_r

        # 刚体环平流表（核心 / 尘埃 / 低频各一套；尘埃与低频用不同哈希流）
        self._adv_core = RigidRingBands(self._r_in, self._r_out, self._dln_r, self._k_rigid_vol)
        self._adv_core_f = make_fields(self._adv_core)
        self._adv_dust = RigidRingBands(
            self._r_in, self._r_out, self._dln_r, self._k_rigid_vol,
            phi_b_hash=(23, 7), ph0_hash=(19, 3),
        )
        self._adv_dust_f = make_fields(self._adv_dust)
        # 低频层用宽带
        self._adv_low = RigidRingBands(
            self._r_in, self._r_out, self._dln_l, self._k_rigid_l,
            phi_b_hash=(23, 1), ph0_hash=(29, 5),
        )
        self._adv_low_f = make_fields(self._adv_low)

        # Page–Thorne 温度 LUT
        self._pt_lut = ti.field(dtype=ti.f32, shape=_BB_LUT_N)
        from .physical_fields import build_page_thorne_lut
        build_page_thorne_lut(self._pt_lut, self._r_in, self._r_out, _BB_LUT_N)

        # T_peak（第 1 层：M、Mdot 推出）
        if vp.t_peak_override_K > 0:
            self._t_peak_vol = float(vp.t_peak_override_K)
        else:
            from .physical_fields import derive_t_peak
            self._t_peak_vol = derive_t_peak(vp.bh_mass_msun, vp.mdot_edd)
        # ln Y(T_peak)（Y(g·T)/Y(T_peak) 用）
        self._ln_y_peak = math.log(
            max(_blackbody_luminance_exact(self._t_peak_vol), 1e-300)
        )

        # κ 预设 1.0（density_I 的灰大气 tau_z 在标定期间引用 κ）
        self._kappa_vol = 1.0
        # 上传标定时刻的平流相位表（density_I / _flow_I 查表需要）
        self.update_advection(2000.0)
        # 噪声标定（⟨c⟩ 与 κ）
        self._calibrate_volume()

        # 光行时间 / 静止观者相机
        self._light_delay = bool(vp.light_delay)
        self._static_cam = bool(vp.static_cam)

    def _calibrate_volume(self) -> None:
        """标定 ⟨c⟩（主云级联均值）与 κ（吸收系数，使 r ≈ 6 处 face-on τ = TAU_I）。"""
        import taichi as ti

        # ⟨c⟩
        cbuf = ti.field(dtype=ti.f32, shape=512)

        @ti.kernel
        def _cmean(out: ti.template()):
            for i in out:
                r = self._r_in + 0.5 + _hashf(i, 3, 7) * (
                    ti.min(self._r_out, 20.0) - self._r_in - 0.5
                )
                phi = _hashf(i, 5, 11) * 2.0 * math.pi
                c, tn = self._flow_I(r, phi, 0.0, 2000.0)
                out[i] = c

        _cmean(cbuf)
        self._c_mean = max(float(cbuf.to_numpy().mean()), 1e-6)

        # κ
        kbuf = ti.field(dtype=ti.f32, shape=256)

        @ti.kernel
        def _column(out: ti.template()):
            for i in out:
                r = 5.5 + ti.cast(i % 16, ti.f32) / 16.0
                phi = ti.cast(i, ti.f32) * 0.61803 * 2.0 * math.pi
                zmax = 3.0 * self._ss_half_thickness(r) + 0.01
                col = 0.0
                for k in range(200):
                    z = -zmax + (ti.cast(k, ti.f32) + 0.5) / 200.0 * 2.0 * zmax
                    em_c, tf_c, ab_c, em_o, ab_o, em_s = self.density_I(
                        r, z, phi, 2000.0, 0.0)
                    col += (ab_c + ab_o) * 2.0 * zmax / 200.0
                out[i] = col

        _column(kbuf)
        col_mean = max(float(kbuf.to_numpy().mean()), 1e-12)
        self._kappa_vol = self.volume_params.tau_i / col_mean
        print(f"[S5] ⟨c⟩ = {self._c_mean:.3g}，吸收柱均值 = {col_mean:.4g} → κ = {self._kappa_vol:.4g}")

    def update_advection(self, t: float) -> None:
        """每帧上传刚体环相位表（核心 / 尘埃 / 低频）。"""
        adv_upload(self._adv_core_f, self._adv_core.phase_table(t))
        adv_upload(self._adv_dust_f, self._adv_dust.phase_table(t))
        adv_upload(self._adv_low_f, self._adv_low.phase_table(t))

    # ---- SS 结构 ----

    @ti.func
    def _ss_half_thickness(self, r):
        """SS 外区标高 H = HR_REF·r·(r/r_ref)^{1/8}·(f/f_ref)^{3/20}。"""
        fr = ti.max(1.0 - ti.sqrt(self._r_in / ti.max(r, self._r_in)), 1e-6)
        return self._hr_ref * r * ti.pow(r / self._r_ref_vol, 0.125) * ti.pow(fr / self._f_ref_ss, 0.15)

    @ti.func
    def _ss_surface_density(self, r):
        """SS 外区柱密度 Σ ∝ (r/r_ref)^{-3/4}·(f/f_ref)^{7/10}·外缘截断。"""
        fr = ti.max(1.0 - ti.sqrt(self._r_in / ti.max(r, self._r_in)), 1e-6)
        outer = 1.0 - _ti_smoothstep(0.72 * self._r_out, self._r_out, r)
        return ti.pow(r / self._r_ref_vol, -0.75) * ti.pow(fr / self._f_ref_ss, 0.7) * outer

    @ti.func
    def _erfc_pos(self, x):
        """erfc(x)，x ≥ 0（A&S 7.1.26，误差 < 1.5e-7）。"""
        t = 1.0 / (1.0 + 0.3275911 * x)
        y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        return y * ti.exp(-x * x)

    @ti.func
    def _page_thorne_temperature(self, r):
        """Page–Thorne 相对论温度 T(r)（查表，log r 线性插值）。"""
        u = (ti.log(ti.min(ti.max(r, self._r_in), self._r_out)) - ti.log(self._r_in)) / (
            ti.log(self._r_out) - ti.log(self._r_in)
        )
        f = u * (_BB_LUT_N - 1)
        i0 = ti.min(ti.cast(ti.floor(f), ti.i32), _BB_LUT_N - 2)
        w = f - ti.cast(i0, ti.f32)
        return self._t_peak_vol * (self._pt_lut[i0] * (1.0 - w) + self._pt_lut[i0 + 1] * w)

    # ---- 主云流动噪声 ----

    @ti.func
    def _flow_noise_core(self, ru, th, z, ox, oz, lev_cut, con):
        """主云 + 厚度扰动两路级联。"""
        c = cascade(self._kr_i * ru + ox, th / (2.0 * math.pi) * self._nphi_i,
                    self._kr_i * z + oz, self._nphi_i,
                    self._l0_i - lev_cut, self._l0_i + 2.0 - lev_cut, con)
        tn = cascade(self._kt_i * ru + ox + 17.0, th / (2.0 * math.pi) * self._nphi_t,
                     oz + 5.0, self._nphi_t, self._lt0_i, self._lt0_i + 2.0, self._con_i)
        return c, tn

    @ti.func
    def _flow_I(self, r, phi, z, t):
        """mode 3 刚体环：两带 × 两相位混合，各带以 Ω(r_b) 刚体旋转。"""
        r_rg = 2.0 * r
        lev_cut = 0.91 * ti.log(1.0 + 0.066 * ti.max(0.0, r_rg - 10.0))
        con = self._con_i - 80.0 * ti.log(1.0 + 0.006 * ti.max(0.0, r_rg - 10.0))
        lnr = ti.log(r)
        fb = (lnr - self._lnr0_r) / self._dln_r + 0.35 * gnoise(lnr * 4.0, 0.37, 11.3, 8)
        b0 = ti.floor(fb)
        fbf = fb - b0
        c = 0.0
        tn = 0.0
        for db in ti.static(range(2)):
            bi = ti.cast(b0, ti.i32) + db
            wb = ti.cos(0.5 * math.pi * fbf) ** 2
            if db == 1:
                wb = ti.sin(0.5 * math.pi * fbf) ** 2
            # 用相位表查表（长视频精度）
            idx = bi - self._adv_core_f.b_lo
            if 0 <= idx < self._adv_core_f.rot.shape[0]:
                rot = self._adv_core_f.rot[idx]
                phi_b = self._adv_core_f.phi_b[idx]
                phi_rigid = phi - rot - phi_b
                for p in ti.static(range(2)):
                    fr = self._adv_core_f.frac[idx][p]
                    cyc = self._adv_core_f.cyc[idx][p]
                    wp = ti.sin(math.pi * fr) ** 2
                    ox = _hashf(bi, cyc, 2 * p) * 97.0
                    oz = _hashf(bi, cyc, 2 * p + 1) * 97.0
                    cc, tt = self._flow_noise_core(r, phi_rigid, z, ox, oz, lev_cut, con)
                    c += wb * wp * cc
                    tn += wb * wp * tt
        return c, tn

    # ---- 尘埃 ----

    @ti.func
    def _dust_flow(self, r, phi, z, t):
        """尘埃噪声：与主云相同的刚体环流场，不同哈希流。"""
        lnr = ti.log(r)
        fb = (lnr - self._lnr0_r) / self._dln_r + 0.35 * gnoise(lnr * 4.0, 0.37, 11.3, 8)
        b0 = ti.floor(fb)
        fbf = fb - b0
        out = 0.0
        for db in ti.static(range(2)):
            bi = ti.cast(b0, ti.i32) + db
            wb = ti.cos(0.5 * math.pi * fbf) ** 2
            if db == 1:
                wb = ti.sin(0.5 * math.pi * fbf) ** 2
            idx = bi - self._adv_dust_f.b_lo
            if 0 <= idx < self._adv_dust_f.rot.shape[0]:
                rot = self._adv_dust_f.rot[idx]
                phi_b = self._adv_dust_f.phi_b[idx]
                phi_rigid = phi - rot - phi_b
                for p in ti.static(range(2)):
                    fr = self._adv_dust_f.frac[idx][p]
                    cyc = self._adv_dust_f.cyc[idx][p]
                    wp = ti.sin(math.pi * fr) ** 2
                    ox = _hashf(bi, cyc, 40 + 2 * p) * 97.0
                    oz = _hashf(bi, cyc, 41 + 2 * p) * 97.0
                    out += wb * wp * cascade(
                        2.0 * r + ox, phi_rigid / (2.0 * math.pi) * 9.0, 2.0 * z + oz, 9, 0.0, 6.0, 80.0)
        return out

    # ---- 低频调制 ----

    @ti.func
    def _turb_low(self, r, phi, t):
        """大尺度低频 lognormal 调制（宽带刚体环）。"""
        lnr = ti.log(r)
        fb = (lnr - ti.log(self._r_in)) / self._dln_l
        b0 = ti.floor(fb)
        fbf = fb - b0
        acc = 0.0
        wsq = 0.0
        for db in ti.static(range(2)):
            bi = ti.cast(b0, ti.i32) + db
            wb = ti.cos(0.5 * math.pi * fbf) ** 2
            if db == 1:
                wb = ti.sin(0.5 * math.pi * fbf) ** 2
            idx = bi - self._adv_low_f.b_lo
            if 0 <= idx < self._adv_low_f.rot.shape[0]:
                rot = self._adv_low_f.rot[idx]
                phi_b = self._adv_low_f.phi_b[idx]
                phi0 = phi - rot - phi_b
                for p in ti.static(range(2)):
                    fr = self._adv_low_f.frac[idx][p]
                    cyc = self._adv_low_f.cyc[idx][p]
                    wp = ti.sin(math.pi * fr) ** 2
                    ox = _hashf(bi, cyc, 7 + p) * 97.0
                    oz = _hashf(bi, cyc, 11 + p) * 97.0
                    n = fbm_gradient(lnr * self._fr_l + ox,
                                     phi0 / (2.0 * math.pi) * self._nphi_l,
                                     oz, self._nphi_l, 3, 0.5)
                    w = wb * wp
                    acc += w * n
                    wsq += w * w
        return acc / ti.sqrt(ti.max(wsq, 1e-6))

    # ---- 烟雾层 ----

    @ti.func
    def _eval_cloud(self, lnr, phi0, zeta, ox, oz, layer_off):
        """烟雾层 fBm（域扭曲），φ 周期 NPHI_C。"""
        x = lnr * self._fr_c + ox + 57.3 + layer_off
        y = phi0 / (2.0 * math.pi) * self._nphi_c
        z = zeta * self._fz_c + oz + 41.9 + 0.37 * layer_off
        wx = gnoise(x * 0.37 + 2.1, y * 0.5, z * 0.5 + 1.3, self._nphi_c // 2)
        return fbm_gradient(x + 0.9 * wx, y, z, self._nphi_c, 5, 0.5)

    @ti.func
    def _turb_pair_smoke(self, r, phi, zeta, t, loff):
        """烟雾层刚体环（与主云共用 adv_core 表，不同噪声函数）。"""
        lnr = ti.log(r)
        fb = (lnr - self._lnr0_r) / self._dln_r + 0.35 * gnoise(lnr * 4.0, 0.37, 11.3, 8)
        b0 = ti.floor(fb)
        fbf = fb - b0
        acc = 0.0
        wsq = 0.0
        for db in ti.static(range(2)):
            bi = ti.cast(b0, ti.i32) + db
            wb = ti.cos(0.5 * math.pi * fbf) ** 2
            if db == 1:
                wb = ti.sin(0.5 * math.pi * fbf) ** 2
            idx = bi - self._adv_core_f.b_lo
            if 0 <= idx < self._adv_core_f.rot.shape[0]:
                rot = self._adv_core_f.rot[idx]
                phi_b = self._adv_core_f.phi_b[idx]
                phi_rigid = phi - rot - phi_b
                for p in ti.static(range(2)):
                    fr = self._adv_core_f.frac[idx][p]
                    cyc = self._adv_core_f.cyc[idx][p]
                    wp = ti.sin(math.pi * fr) ** 2
                    ox = _hashf(bi, cyc, 2 * p) * 97.0
                    oz = _hashf(bi, cyc, 2 * p + 1) * 97.0
                    n = self._eval_cloud(lnr, phi_rigid, zeta, ox, oz, loff)
                    w = wb * wp
                    acc += w * n
                    wsq += w * w
        return acc / ti.sqrt(ti.max(wsq, 1e-6))

    # ---- 体积密度场 ----

    @ti.func
    def density_I(self, r, z, phi, t, dir_z):
        """体积密度场：返回 (核心发射, 核心温度倍率, 核心吸收, 其他发射, 其他吸收, 烟雾发射)。

        核心吸收在渲染核中再乘 CORE_OPAC；"其他" = 尘埃，温度取当地 T(r)；
        SMOKE_TR > 0 时烟雾发射单列，温度取 SMOKE_TR·T(r)。
        PHYS_STRUCT = 1：SS 外区 H(r)、Σ(r) + 竖直高斯。
        """
        em = 0.0
        ab = 0.0
        em_c = 0.0
        em_s = 0.0
        ab_c_out = 0.0
        tf_c = 1.0
        if r > self._r_in and r < self._r_out:
            # SS 结构
            h_geo = self._ss_half_thickness(r)
            h_cap = h_geo
            sig = self._ss_surface_density(r)
            zc = 3.0 * h_cap
            # 大尺度低频
            if ti.static(True):
                nl = self._turb_low(r, phi, t)
                sig *= ti.exp(self._lowf_sigma * nl - 0.5 * self._lowf_sigma * self._lowf_sigma)
            # 尘埃竖直包络
            xi = (r - self._r_in) / ti.min(self._r_out - self._r_in, 6.0)
            dust_bound = h_geo * ti.max(0.0, 1.0 - 5.0 * xi * xi)
            az = ti.abs(z)
            # 烟雾层
            if ti.static(self._smoke_on):
                if az < self._cl_spacing * (self._n_cl_half + 3) * r:
                    for kk in range(2 * self._n_cl_half + 1):
                        kf = ti.cast(kk - self._n_cl_half, ti.f32)
                        dz = (z - kf * self._cl_spacing * r) / (self._cl_width * r)
                        if ti.abs(dz) < 3.0:
                            amp = self._cl_amp_norm * ti.exp(-self._cl_decay * ti.abs(kf))
                            rho_c = self._smoke_i * sig * amp * ti.exp(-0.5 * dz * dz) / (
                                2.5066283 * self._cl_width * r)
                            nc = self._turb_pair_smoke(r, phi, dz, t, 131.7 * ti.cast(kk + 1, ti.f32))
                            cov = 1.0 / (1.0 + ti.exp(-(nc - self._cloud_c0) / self._cloud_soft))
                            ab_sm = rho_c * ti.exp(
                                self._sigma_c * nc - 0.5 * self._sigma_c * self._sigma_c) * cov
                            ab += ab_sm
                            if ti.static(True):
                                em_s += ab_sm
            # 核心
            if az < ti.max(zc, dust_bound):
                if az < zc:
                    c, tn = self._flow_I(r, phi, z, t)
                    softsat = 1.0 - 1.0 / (ti.max(tn, 0.0) + 1.0)
                    h_s = ti.max(h_cap * (1.0 - self._surf_noise + self._surf_noise * softsat), 1e-6)
                    zs = 3.0 * h_s
                    if az < zs:
                        # 等温静力平衡：ρ = Σ/(√(2π)·H_s)·exp(-z²/2H_s²)
                        rho_s = sig * ti.exp(-0.5 * (az / h_s) ** 2) / (2.5066283 * h_s)
                        # 温和密度起伏
                        cfac = self._core_floor + (1.0 - self._core_floor) * c / self._c_mean
                        ab_b = cfac * rho_s
                        ab_c_out = ab_b
                        em_c = ab_b * (self._surf_lo + self._surf_k * az / h_s)
                        # 灰大气温度倍率
                        if ti.static(True):
                            tau_z = self._kappa_vol * self._core_opac * cfac * sig * 0.5 * self._erfc_pos(
                                az / (1.4142136 * h_s))
                            tf_c = 1.0 + self._grey_mix * (
                                ti.min(ti.pow(0.75 * (tau_z + 2.0 / 3.0), 0.25), self._grey_cap) - 1.0)
                        # 温度起伏
                        if ti.static(True):
                            tf_c *= ti.min(ti.max(
                                1.0 + self._dt_i * (c / self._c_mean - 1.0), 0.7), 1.3)
            # 尘埃
            if ti.static(self._dust_on):
                if az < dust_bound:
                    di = ti.max(1.0 - (z / ti.max(dust_bound, 1e-6)) ** 2, 0.0)
                    if ti.static(self._dust_kepler):
                        dn = self._dust_flow(r, phi, z, t)
                        ab_d = self._dust_em * di * dn
                        ab += ab_d
                        em += ab_d * self._dust_s
        return em_c, tf_c, ab_c_out, em, ab, em_s

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
        """采样发射率（单位体积 emissivity per unit volume）。

        v2.2 默认（D3 修订）：
        `j(r, z) = support · opacity · rho_envelope(r, z) · (T(r, z)/T_peak)^4 · F_turb · F_mode · F_hotspot`

        其中 ρ_envelope 与 T 都是真 z 函数：

        - `rho_envelope(r, z) = rho_raw(r) · exp(-0.5 (z/H)²) · W_z(z)`
        - `T(r, z) = T_raw(r) · V_T(|z|/H) · W_z(z)`

        Notes:
            返回的是"单位体积发射率"，沿光线 `ds` 累积才得到屏幕 HDR 强度。
            face-on 视线下 `∫ j(r, z) dz` 等价于 NumPy
            `physical_baseline_volume_flux(r)`（D3 reference 量纲一致）。

            v2.2.2 前 sample_emission 在 z 方向是常数（盘内取中面值），
            违反 thin disk emissivity 的真实物理：emissivity ∝ ρ(r,z) · T(r,z)^4，
            两者都沿 z 高斯衰减。v2.2.3 改为真 z 函数：

            - 与 NumPy `physical_baseline_volume_flux` 的被积函数一致
            - edge-on / 大倾角下盘体厚度感、自遮挡、垂向 emission 衰减自然呈现
            - 仍保持 A1 的 `W_r` 只乘一次约定（`support = disk_radial_weight`）
        """
        rho_raw = raw_midplane_density_ti(r, self._r_in, self._rho_power)
        t_raw = raw_midplane_temperature_ti(r, self._r_in, self._T_peak_K)
        h = ti.max(disk_half_thickness_ti(r, self._h0, self._beta_h, self._r_in), 1e-12)
        zh = z / h
        # 垂向高斯密度衰减（与 NumPy `density_field` 一致）
        v_density = ti.exp(-0.5 * zh * zh)
        # 垂向温度衰减（与 NumPy `temperature_field` 一致）
        v_temp = ti.max(1.0 - 0.25 * ti.abs(zh), 0.0)
        # 几何垂向 support 关闭
        w_z = disk_vertical_weight_ti(
            r, z, self._h0, self._beta_h, self._r_in, self._r_out,
        )
        # ρ_envelope(r, z) = ρ_raw(r) · exp(-0.5 (z/H)²) · W_z(z)
        rho_envelope = ti.max(rho_raw, 0.0) * v_density * w_z
        # T(r, z) = T_raw(r) · V_T · W_z，归一化为 [T/T_peak]
        t_norm = t_raw * v_temp * w_z / ti.max(self._T_peak_K, 1.0)
        # 单位体积发射率（emissivity per unit volume）
        emissivity = self._emission_opacity_scale * rho_envelope * ti.pow(
            ti.max(t_norm, 0.0), 4.0,
        )
        # 径向 support 只乘一次（A1）
        support = disk_radial_weight_ti(r, self._r_in, self._r_out, self._edge_softness)
        j_base = support * emissivity
        if ti.static(self._use_visual_atlas):
            ew = self.sample_emission_atlas_ti(r, phi)
            # v2.2：atlas 只做有限扰动，不再决定主径向亮度。
            f_turb = 0.75 + 0.5 * ti.min(ti.max(ew, 0.0), 1.0)
            j_base *= f_turb
        else:
            j_base *= self.shear_modulation_ti(r, phi)
        f_struct = self.mode_modulation_ti(r, phi) * self.hotspot_modulation_ti(r, phi)
        return j_base * f_struct

    @ti.func
    def blackbody_color_ti(self, T_K):
        """温度 → 黑体色度（线性 sRGB D65，查 `palette.BB_LUT`，与 NumPy parity）。

        Args:
            T_K: 温度（K）。

        Returns:
            `(3,)` RGB 向量，每通道 ≥ 0、BT.709 亮度 = 1；`T_K ≤ 0` 返回 0。
        """
        rgb = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        if T_K > 0.0:
            u = (ti.log(ti.min(ti.max(T_K, _BB_T_MIN_K), _BB_T_MAX_K)) - ti.log(_BB_T_MIN_K)) / (
                ti.log(_BB_T_MAX_K) - ti.log(_BB_T_MIN_K)
            )
            f = u * (_BB_LUT_N - 1)
            i0 = ti.min(ti.cast(ti.floor(f), ti.i32), _BB_LUT_N - 2)
            w = f - ti.cast(i0, ti.f32)
            rgb = self._bb_lut[i0] * (1.0 - w) + self._bb_lut[i0 + 1] * w
        return rgb

    @ti.func
    def blackbody_luminance_ti(self, T_K):
        """温度 → 黑体可见光亮度 `Y(T)`（查 `palette.LNY_LUT`，与 NumPy parity）。

        Args:
            T_K: 温度（K）。

        Returns:
            `Y(T)` ≥ 0；`T_K ≤ 0` 返回 0。频移后的观测亮度即 `Y(g·T)`。
        """
        out = 0.0
        if T_K > 0.0:
            u = (ti.log(ti.min(ti.max(T_K, _LNY_T_MIN_K), _LNY_T_MAX_K)) - ti.log(_LNY_T_MIN_K)) / (
                ti.log(_LNY_T_MAX_K) - ti.log(_LNY_T_MIN_K)
            )
            f = u * (_BB_LUT_N - 1)
            i0 = ti.min(ti.cast(ti.floor(f), ti.i32), _BB_LUT_N - 2)
            w = f - ti.cast(i0, ti.f32)
            out = ti.exp(self._lny_lut[i0] * (1.0 - w) + self._lny_lut[i0 + 1] * w)
        return out

    @ti.func
    def white_balance_gain_ti(self, T_wb):
        """von Kries 白平衡增益（Python 端按 `white_balance_K` 预计算，kernel 只查表）。

        Args:
            T_wb: 白平衡色温（K）。仅作文档语义：Taichi kernel 闭包不接受
                运行时向量参数，增益取自构造 `DiskV2PaletteParams.white_balance_K`
                时上传的 field。

        Returns:
            `(3,)` RGB 增益，全为正；使该色温的黑体呈精确中性 (1,1,1)。
        """
        return self._wb_gain[None]

    @ti.func
    def sample_palette_color(self, T_K):
        """温度 → 黑体色度（v2.3：无二级映射，cinematic 已删除）。

        Args:
            T_K: 温度（单位 K）。

        Returns:
            形状 (3,) 的 RGB 向量，每通道 ≥ 0、亮度 = 1；`T_K ≤ 0` 返回 0。
        """
        return self.blackbody_color_ti(T_K)

    @ti.func
    def sample_observed_palette_color(self, T_K, g_factor):
        """观测色度：温度 `T·g_factor` 的黑体（I_ν/ν³ 洛伦兹不变）。

        Args:
            T_K: 发射位置物理温度（K）。
            g_factor: `nu_obs / nu_em`（应为 `g^doppler_color`，强度旋钮由调用方施加）。

        Returns:
            `(3,)` RGB 向量；`T_K ≤ 0` 返回 0。

        Notes:
            v2.3 之前 cinematic 模式把 `g·T_visible` 送入 Helland 公式并叠加
            饱和度 / 暖色 / 低温压暗；这些非物理调整已删除，频移后的色度就是
            温度 `g·T` 的黑体色度。
        """
        return self.blackbody_color_ti(T_K * g_factor)

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
    def tonemap_aces(self, rgb_hdr):
        """ACES Filmic tonemap（Narkowicz 2015 单变量近似）。

        Args:
            rgb_hdr: HDR RGB 向量。

        Returns:
            LDR RGB 向量，每通道 `[0, 1]`（已 clip）。

        Formula:
            `x → clip(x · (a·x + b) / (x · (c·x + d) + e), 0, 1)`
            `a=2.51, b=0.03, c=2.43, d=0.59, e=0.14`

        Notes:
            与 NumPy `tonemap_aces` 公式严格一致；parity 测试保护。
            相对 Reinhard：x=1 → 0.77（vs 0.5），x=10 → 0.95（vs 0.91）。
            高动态范围下保留更多中调和高光细节。
        """
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        result = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        for ch in ti.static(range(3)):
            x = ti.max(rgb_hdr[ch], 0.0)
            num = x * (a * x + b)
            den = ti.max(x * (c * x + d) + e, 1e-12)
            y = num / den
            result[ch] = ti.min(ti.max(y, 0.0), 1.0)
        return result

    @ti.func
    def tonemap_ti(self, rgb_hdr):
        """统一 tonemap 入口，按 `palette_params.tonemap_mode` 静态分发。

        Args:
            rgb_hdr: HDR RGB 向量。

        Returns:
            LDR RGB 向量。
        """
        result = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        if ti.static(self._is_aces_tonemap):
            result = self.tonemap_aces(rgb_hdr)
        else:
            result = self.tonemap_reinhard(rgb_hdr)
        return result

    @ti.func
    def apply_exposure_ti(self, rgb_hdr, exposure_scale):
        """对 HDR RGB 应用曝光缩放。"""
        return rgb_hdr * exposure_scale

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
