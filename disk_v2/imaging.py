"""Disk V2 成像方程参考实现（v2.2）。

本模块只定义 NumPy 侧的成像语义真源，不参与实时渲染：

- `tau_effective_midplane()`：把密度包络转换为有效光学厚度标度。
- `physical_baseline_flux()`：默认 thin-disk effective radiative transfer 的相对通量。
- `reference_exposure()`：cinematic 曝光的相机无关参考值。
- `observed_visible_temperature()`：g-factor 在可见色温显示链上的近似作用。

这些函数用于单元测试、文档和后续 Taichi 移植的 reference。
"""

from __future__ import annotations

import numpy as np

from ._array_utils import _restore_shape, _to_array
from .geometry import disk_half_thickness
from .params import DiskV2PaletteParams, DiskV2Params
from .geometry import disk_radial_weight
from .physical_fields import raw_midplane_density_field, raw_midplane_temperature_field


def tau_effective_midplane(
    r: float | np.ndarray,
    params: DiskV2Params,
    opacity_scale: float,
) -> float | np.ndarray:
    """计算中面有效光学厚度标度 `tau_eff(r)`。

    Args:
        r: 局部盘坐标中的径向距离，可为标量或数组。
        params: Disk V2 基础盘体参数。
        opacity_scale: 有效 opacity 缩放。当前是工程标定参数，不是完整频率相关
            opacity。

    Returns:
        与 `r` 同形状的非负标量或数组。量纲为当前无量纲渲染单位下的有效光学厚度
        标度；当整体显著小于 1 时，该模型应解释为 optically-thin effective opacity，
        而不是真实 photosphere。

    Formula:
        ```
        tau_eff(r) = opacity_scale · rho_mid(r) · H(r)
        ```

    Physical Meaning:
        这是把密度包络、几何厚度和一个简化 opacity 系数折算为光学厚度的启发式
        参考量。它主要服务于“内圈更不透明、外圈更稀薄”的视觉和物理约束。

    Simplifications:
        不包含 Kramers opacity、频率依赖、垂向 τ 积分或 τ≈1 面求解。
    """
    r_arr = _to_array(r)
    rho_mid = _to_array(raw_midplane_density_field(r_arr, params))
    thickness = _to_array(disk_half_thickness(r_arr, params))
    tau = np.maximum(float(opacity_scale), 0.0) * rho_mid * thickness
    return _restore_shape(tau.astype(np.float64), r)


def physical_baseline_flux(
    r: float | np.ndarray,
    params: DiskV2Params,
    opacity_scale: float,
) -> float | np.ndarray:
    """计算默认物理主链的相对面通量 `F_phys(r)`。

    Args:
        r: 局部盘坐标中的径向距离，可为标量或数组。当前 Step 1 reference 只定义
            中面剖面，后续体积路径可扩展为 `(r, z)`。
        params: Disk V2 基础盘体参数。
        opacity_scale: 有效 opacity 缩放。

    Returns:
        与 `r` 同形状的非负相对通量。该值是相机无关的物理场域 reference，
        可作为 cinematic 曝光标定输入。

    Formula:
        ```
        F_phys(r) = W_r(r) · tau_raw(r) · [T_raw_mid(r) / T_peak]^4
        tau_raw(r) = opacity_scale · rho_raw_mid(r) · H(r)
        ```

        `rho_raw_mid` 与 `T_raw_mid` 不包含 `W_r`，因此 support 只在最终通量中
        出现一次，避免内外边界变成 `W_r^5`。

    Physical Meaning:
        这是 thin-disk effective radiative transfer 的相对通量近似。`T^4` 保留
        薄盘内热外冷的主趋势，`tau_eff` 让低密度/低厚度区域更暗。

    Simplifications:
        不是完整 Stefan-Boltzmann CGS 通量；使用 `(T/T_peak)^4` 做相对归一化，
        避免 `T_peak_K≈1e7` 的绝对量直接进入 LDR 显示链。
    """
    r_arr = _to_array(r)
    tau = _to_array(tau_effective_midplane(r_arr, params, opacity_scale))
    t_mid = _to_array(raw_midplane_temperature_field(r_arr, params))
    support = _to_array(disk_radial_weight(r_arr, params))
    t_norm = np.maximum(t_mid / max(params.T_peak_K, 1.0), 0.0)
    flux = support * tau * np.power(t_norm, 4.0)
    return _restore_shape(flux.astype(np.float64), r)


def physical_baseline_volume_flux(
    r: float | np.ndarray,
    params: DiskV2Params,
    opacity_scale: float,
    *,
    n_z: int = 32,
) -> float | np.ndarray:
    """沿垂向积分默认物理主链，得到与体积渲染更接近的 reference 通量。

    Args:
        r: 局部盘坐标中的径向距离，可为标量或数组。
        params: Disk V2 基础盘体参数。
        opacity_scale: 有效 opacity 缩放。
        n_z: 垂向 Gauss-Legendre 积分点数。

    Returns:
        与 `r` 同形状的非负通量 reference。相对 `physical_baseline_flux()`，
        该函数包含垂向密度高斯、温度垂向衰减与 `W_z` 收口。

    Formula:
        ```
        F_vol(r) = W_r(r) · ∫ opacity_scale · rho_raw(r)
                   · exp[-0.5(z/H)^2]
                   · [T_raw(r) · V_T(z/H) · W_z(z/H) / T_peak]^4 dz
        ```

    Physical Meaning:
        这是 exposure reference，不是完整辐射转移。它让曝光标定和有限厚度
        积分的量级更接近，避免用中面点值去标定路径积分。
    """
    if n_z < 4:
        raise ValueError("n_z must be >= 4")

    r_arr = _to_array(r)
    support = _to_array(disk_radial_weight(r_arr, params))
    rho_raw = _to_array(raw_midplane_density_field(r_arr, params))
    t_raw = _to_array(raw_midplane_temperature_field(r_arr, params))
    h = _to_array(disk_half_thickness(r_arr, params))

    x, w = np.polynomial.legendre.leggauss(int(n_z))
    accum = np.zeros_like(r_arr, dtype=np.float64)
    for xi, wi in zip(x, w):
        abs_xi = abs(float(xi))
        # 与 NumPy `density_field` / `temperature_field` 一致：
        # ρ_envelope(r, z) = ρ_raw · exp(-0.5 (z/H)²) · W_z(z)
        # T(r, z)         = T_raw · V_T(z/H) · W_z(z)
        # 因此被积函数 = opacity · ρ_envelope · (T/T_peak)^4
        #            = opacity · ρ_raw · exp(-0.5 xi²) · W_z · (T_raw · V_T · W_z / T_peak)^4
        # v2.2.2 前 vertical_density 只乘 `exp(-0.5 xi²)`，漏了 W_z；
        # v2.2.3 修复，与 Taichi `sample_emission` 量纲一致。
        vertical_density = np.exp(-0.5 * float(xi) * float(xi))
        vertical_weight = 1.0 - (abs_xi * abs_xi * (3.0 - 2.0 * abs_xi))
        vertical_temp = np.clip(1.0 - 0.25 * abs_xi, 0.0, 1.0) * vertical_weight
        t_norm = np.maximum(t_raw * vertical_temp / max(params.T_peak_K, 1.0), 0.0)
        accum += (
            float(wi)
            * vertical_density
            * vertical_weight  # v2.2.3：补 W_z 让 ρ_envelope 完整
            * np.power(t_norm, 4.0)
        )

    flux = support * float(opacity_scale) * rho_raw * h * accum
    return _restore_shape(flux.astype(np.float64), r)


def reference_exposure(
    params: DiskV2Params,
    opacity_scale: float,
    *,
    target_ldr: float = 0.7,
    sample_count: int = 4096,
) -> float:
    """计算 cinematic 曝光参考值（相机无关）。

    Args:
        params: Disk V2 基础盘体参数。
        opacity_scale: 有效 opacity 缩放。
        target_ldr: 参考通量经 Reinhard tonemap 后希望落到的 LDR 亮度。
        sample_count: 径向剖面采样数。

    Returns:
        曝光缩放 `exposure`，使得 `reinhard(exposure · F_ref) ≈ target_ldr`。
        若 `F_ref` 极小则返回 `1.0`。

    Formula:
        ```
        F_ref = percentile_99(F_volume(r))
        target = x / (1 + x)  =>  x = target / (1 - target)
        exposure = x / F_ref
        ```

    Physical Meaning:
        曝光标定基于物理场域剖面，不依赖相机、FOV 或分辨率。face-on 相机仅用于
        后续校验显示效果，而不参与定义 `F_ref`。

    Simplifications:
        使用 p99 而不是 max，避免单个采样点或扰动尖峰主导曝光。
    """
    if not 0.0 < target_ldr < 1.0:
        raise ValueError("target_ldr must be in (0, 1)")
    if sample_count < 16:
        raise ValueError("sample_count must be >= 16")

    radii = np.linspace(params.r_in + 1e-6, params.r_out - 1e-6, int(sample_count))
    flux = np.asarray(physical_baseline_volume_flux(radii, params, opacity_scale), dtype=np.float64)
    finite = flux[np.isfinite(flux) & (flux > 0.0)]
    if finite.size == 0:
        return 1.0
    f_ref = float(np.percentile(finite, 99.0))
    if f_ref <= np.finfo(np.float64).eps:
        return 1.0
    target_hdr = float(target_ldr) / (1.0 - float(target_ldr))
    return target_hdr / f_ref


def observed_visible_temperature(
    T_visible_K: float | np.ndarray,
    g_factor: float | np.ndarray,
    params: DiskV2PaletteParams,
) -> float | np.ndarray:
    """把 g-factor 作用到 cinematic 可见色温显示链上。

    Args:
        T_visible_K: 已经由物理温度 log 映射得到的可见色温（K）。
        g_factor: 相对论频移因子 `nu_obs / nu_em`，可为标量或数组。
        params: 调色参数，提供可见色温上下界。

    Returns:
        与输入广播后同形状的可见观测色温，范围被限制在
        `[visual_temp_outer_K, visual_temp_inner_K]`；输入非正温度返回 0。

    Formula:
        ```
        T_visible_obs = clamp(g · T_visible_em, Tvis_min, Tvis_max)
        ```

    Physical Meaning:
        这是 band-limited cinematic 显示近似。真实 `T_phys≈1e7K` 不直接进入
        Tanner Helland/LDR 显示链；g-factor 只移动已经映射到可见区间的色温。

    Simplifications:
        不使用指数 Wien 通道缩放。若未来恢复 Wien 近似，也必须作用在
        `T_visible` 上，并用测试锁定蓝移/红移方向。
    """
    t_arr = _to_array(T_visible_K)
    g_arr = _to_array(g_factor)
    obs = t_arr * g_arr
    obs = np.clip(obs, params.visual_temp_outer_K, params.visual_temp_inner_K)
    obs = np.where(t_arr > 0.0, obs, 0.0)
    return _restore_shape(obs.astype(np.float64), T_visible_K if np.ndim(T_visible_K) >= np.ndim(g_arr) else g_factor)
