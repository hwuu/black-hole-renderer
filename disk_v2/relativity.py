"""Disk V2 Schwarzschild 相对论辅助公式（NumPy reference）。

本模块提供 g-factor 与 Keplerian 角速度的可测试参考实现。Taichi 端必须与这里的
符号约定保持一致。
"""

from __future__ import annotations

import math

import numpy as np

from ._array_utils import _restore_shape, _to_array


def schwarzschild_mass_from_rs(rs: float = 1.0) -> float:
    """由 Schwarzschild 半径得到几何单位质量 `M`。

    Args:
        rs: Schwarzschild 半径。项目无量纲单位默认 `rs = 1`。

    Returns:
        几何单位质量 `M = rs / 2`。
    """
    return 0.5 * float(rs)


def omega_kep(
    r: float | np.ndarray,
    rs: float = 1.0,
) -> float | np.ndarray:
    """计算 Schwarzschild/Newtonian 共用的 Keplerian 坐标角速度标度。

    Args:
        r: 径向坐标，单位 `r_s`。
        rs: Schwarzschild 半径。

    Returns:
        `Ω_K(r) = sqrt(M / r^3)`。返回形状跟随 `r`。

    Formula:
        ```
        M = rs / 2
        Ω_K = sqrt(M / r^3)
        ```
    """
    r_arr = _to_array(r)
    safe_r = np.maximum(r_arr, np.finfo(np.float64).eps)
    omega = np.sqrt(schwarzschild_mass_from_rs(rs) / np.power(safe_r, 3.0))
    return _restore_shape(omega.astype(np.float64), r)


def omega_norm(
    r: float | np.ndarray,
    r_in: float,
    rs: float = 1.0,
) -> float | np.ndarray:
    """计算归一化 Keplerian 角速度，用于纹理平流/eddy 时间标度。

    Args:
        r: 径向坐标，单位 `r_s`。
        r_in: 归一化参考半径。
        rs: Schwarzschild 半径。

    Returns:
        `Ω_K(r) / Ω_K(r_in)`，等价于 `(r/r_in)^(-3/2)`。
    """
    omega = _to_array(omega_kep(r, rs))
    omega_ref = float(omega_kep(max(float(r_in), np.finfo(np.float64).eps), rs))
    out = omega / max(omega_ref, np.finfo(np.float64).eps)
    return _restore_shape(out.astype(np.float64), r)


def gravitational_g_factor(
    r_em: float | np.ndarray,
    r_obs: float | np.ndarray,
    rs: float = 1.0,
) -> float | np.ndarray:
    """Schwarzschild 静态引力红移因子 `ν_obs / ν_em`。

    Args:
        r_em: 发射点径向坐标，单位 `r_s`。
        r_obs: 观察者径向坐标，单位 `r_s`。
        rs: Schwarzschild 半径。

    Returns:
        引力频移因子。远处观察近黑洞发射时小于 1。

    Formula:
        ```
        g_grav = sqrt(1 - rs / r_em) / sqrt(1 - rs / r_obs)
        ```
    """
    em = np.maximum(_to_array(r_em), float(rs) + 1e-6)
    obs = np.maximum(_to_array(r_obs), float(rs) + 1e-6)
    numerator = np.sqrt(np.maximum(1.0 - float(rs) / em, 1e-12))
    denominator = np.sqrt(np.maximum(1.0 - float(rs) / obs, 1e-12))
    out = numerator / denominator
    ref = r_em if np.ndim(r_em) >= np.ndim(r_obs) else r_obs
    return _restore_shape(out.astype(np.float64), ref)


def orbital_beta_local(
    r: float | np.ndarray,
    rs: float = 1.0,
    *,
    eps: float = 1e-6,
    beta_cap: float = 0.99,
) -> float | np.ndarray:
    """本地静止观者测得的 Schwarzschild 赤道圆轨道速度。

    Args:
        r: 发射半径，单位 `r_s`；标量或数组。物理上只在 `r ≥ 3 r_s`（ISCO）有稳定圆轨道。
        rs: Schwarzschild 半径。
        eps: 分母 `1 − 2M/r` 的下限，防止 `r → r_s` 时发散。
        beta_cap: 工程速度上限。

    Returns:
        `β = v/c`，形状跟随 `r`，范围 `[0, beta_cap]`。ISCO 处恰为 0.5。

    Formula:
        ```
        M = rs / 2
        β = sqrt(M / r) / sqrt(1 − 2M / r) = sqrt(M / (r − 2M))
        ```
        推导：静止观者测得 v = (r dφ/dτ_static) = r Ω / sqrt(1 − 2M/r)，Ω = sqrt(M/r³)。

    Physical Meaning:
        g-factor Doppler 部分所需的发射体相对本地静止观者的速度。

    Simplifications:
        假设盘物质严格做开普勒圆轨道（无径向速度、无压强修正）。
        v2.3 之前误用 `1 − 3M/r`，在 ISCO 处给出 0.577，高估 Doppler 强度。
    """
    r_arr = np.maximum(_to_array(r), float(rs) + 1e-6)
    mass = schwarzschild_mass_from_rs(rs)
    denom = np.sqrt(np.maximum(1.0 - 2.0 * mass / r_arr, float(eps)))
    beta = np.sqrt(mass / r_arr) / denom
    beta = np.clip(beta, 0.0, float(beta_cap))
    return _restore_shape(beta.astype(np.float64), r)


def doppler_g_factor(
    beta: float | np.ndarray,
    cos_theta: float | np.ndarray,
) -> float | np.ndarray:
    """特殊相对论 Doppler 因子。

    Args:
        beta: 局部速度 `v/c`。
        cos_theta: 速度方向与发射光线朝观察者方向的夹角余弦；朝向观察者运动时为正。

    Returns:
        `g_doppler = 1 / (gamma · (1 - beta · cos_theta))`。
    """
    beta_arr = np.clip(_to_array(beta), 0.0, 0.999999)
    cos_arr = np.clip(_to_array(cos_theta), -1.0, 1.0)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - beta_arr * beta_arr, 1e-12))
    out = 1.0 / np.maximum(gamma * (1.0 - beta_arr * cos_arr), 1e-12)
    ref = beta if np.ndim(beta) >= np.ndim(cos_theta) else cos_theta
    return _restore_shape(out.astype(np.float64), ref)


def total_g_factor(
    r_em: float | np.ndarray,
    r_obs: float | np.ndarray,
    cos_theta: float | np.ndarray,
    rs: float = 1.0,
    *,
    g_cap: float = 6.0,
) -> float | np.ndarray:
    """组合引力红移与 Doppler 因子，并应用工程上限。

    Args:
        r_em: 发射半径，单位 `r_s`。
        r_obs: 观察者半径，单位 `r_s`。
        cos_theta: 速度方向与发射光线朝观察者方向的夹角余弦。
        rs: Schwarzschild 半径。
        g_cap: 工程上限，避免极端单点过曝。

    Returns:
        `min(g_grav · g_doppler, g_cap)`。
    """
    beta = orbital_beta_local(r_em, rs)
    g = _to_array(gravitational_g_factor(r_em, r_obs, rs)) * _to_array(
        doppler_g_factor(beta, cos_theta)
    )
    g = np.minimum(g, float(g_cap))
    ref = r_em if np.ndim(r_em) >= np.ndim(cos_theta) else cos_theta
    return _restore_shape(g.astype(np.float64), ref)


def local_photon_direction(
    pos: np.ndarray,
    k_coord: np.ndarray,
    rs: float = 1.0,
) -> np.ndarray:
    """把光子在笛卡尔等效坐标下的方向变换为本地静止观者测得的方向。

    Args:
        pos: 光子所在位置，形状 `(..., 3)`，单位 `r_s`，原点为黑洞中心。
        k_coord: 同一点的坐标方向（不必归一），形状 `(..., 3)`，与 `pos` 广播。
        rs: Schwarzschild 半径。

    Returns:
        本地静止观者坐标系中的单位方向，形状 `(..., 3)`。

    Formula:
        ```
        r̂ = pos / |pos|
        k_rad = (k · r̂) r̂,  k_tan = k − k_rad
        k_loc ∝ k_rad + sqrt(1 − r_s/r) · k_tan
        ```
        等价于 `tanψ_local = sqrt(1 − r_s/r) · tanψ_coord`（ψ 为与径向夹角）：
        静止观者的径向固有长度 `dl = dr / sqrt(1 − r_s/r)`，切向 `r dφ` 不变。

    Physical Meaning:
        Doppler 公式中的 `cosθ` 必须在发射点的本地静止标架中计算；直接用坐标方向
        在近黑洞处最大有约 5% 的 g 误差。

    Simplifications:
        仅做方向（角度）变换，不涉及频率；`r` 取三维球半径。
    """
    p = _to_array(pos)
    k = _to_array(k_coord)
    r = np.linalg.norm(p, axis=-1, keepdims=True)
    r_hat = p / np.maximum(r, 1e-30)
    k_rad = np.sum(k * r_hat, axis=-1, keepdims=True) * r_hat
    k_tan = (k - k_rad) * np.sqrt(np.maximum(1.0 - float(rs) / np.maximum(r, 1e-30), 1e-12))
    k_loc = k_rad + k_tan
    return k_loc / np.maximum(np.linalg.norm(k_loc, axis=-1, keepdims=True), 1e-30)


def disk_g_factor(
    pos: np.ndarray,
    trace_dir: np.ndarray,
    r_obs: float,
    rs: float = 1.0,
) -> float | np.ndarray:
    """盘面圆轨道发射体 → 远处静止观者的频移因子 `g = ν_obs / ν_em`。

    Args:
        pos: 发射点，盘局部坐标（盘面为 z = 0，盘逆时针旋转，从 +z 看），形状 `(..., 3)`。
        trace_dir: 反向追踪光线在该点的坐标方向（从相机射出的方向），形状 `(..., 3)`。
            光子真实传播方向为 `−trace_dir`。
        r_obs: 观察者半径（静止观者），单位 `r_s`。
        rs: Schwarzschild 半径。

    Returns:
        `g`，形状为广播后的前导形状（标量输入返回 float）。逼近侧 > 1，远离侧 < 1。

    Formula:
        ```
        R = sqrt(x² + y²)（圆柱半径，决定轨道），r = |pos|（球半径，决定引力红移）
        v̂ = (−y, x, 0) / R,  β = sqrt(M / (R − 2M)),  γ = 1/sqrt(1 − β²)
        cosθ = v̂ · local_photon_direction(pos, −trace_dir)
        g = sqrt(1 − r_s/r) / sqrt(1 − r_s/r_obs) · 1 / (γ (1 − β cosθ))
        ```

    Physical Meaning:
        体积积分中每个采样点的相对论频移，同时决定 Doppler beaming 亮度与颜色偏移。

    Simplifications:
        盘内（含有限厚度处）物质都按其圆柱半径做赤道开普勒圆轨道；`R < 3 r_s` 处按 ISCO
        取 β，保持数值稳定（该区域不参与发射）。
    """
    p = _to_array(pos)
    k = -_to_array(trace_dir)
    big_r = np.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2)
    r3 = np.linalg.norm(p, axis=-1)
    beta = _to_array(orbital_beta_local(np.maximum(big_r, 3.0 * float(rs)), rs))
    v_hat = np.stack([-p[..., 1], p[..., 0], np.zeros_like(big_r)], axis=-1) / np.maximum(big_r, 1e-30)[..., None]
    cos_th = np.sum(v_hat * local_photon_direction(p, k, rs), axis=-1)
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    g_grav = np.sqrt(np.maximum(1.0 - float(rs) / r3, 1e-12)) / math.sqrt(max(1.0 - float(rs) / float(r_obs), 1e-12))
    g = g_grav / (gamma * (1.0 - beta * cos_th))
    return float(g) if np.ndim(g) == 0 else g


def exact_equatorial_g_factor(
    r: float,
    lz_over_e: float,
    r_obs: float,
    rs: float = 1.0,
) -> float:
    """赤道开普勒圆轨道 → 静止观者的严格 GR 频移（用于测试对照）。

    Args:
        r: 发射半径，单位 `r_s`，`r > 3M`。
        lz_over_e: 光子 `L_z / E`（带符号冲击参数，沿盘旋转方向为正）。
        r_obs: 静止观察者半径。
        rs: Schwarzschild 半径。

    Returns:
        `g = ν_obs / ν_em`（标量）。

    Formula:
        ```
        u^t = 1 / sqrt(1 − 3M/r),  Ω = sqrt(M / r³)
        ν_em ∝ u^t (E − Ω L_z),  ν_obs ∝ E / sqrt(1 − r_s/r_obs)
        g = sqrt(1 − 3M/r) / (1 − Ω · L_z/E) / sqrt(1 − r_s/r_obs)
        ```
    """
    mass = schwarzschild_mass_from_rs(rs)
    omega = math.sqrt(mass / r ** 3)
    return math.sqrt(1.0 - 3.0 * mass / r) / (1.0 - omega * lz_over_e) / math.sqrt(1.0 - float(rs) / r_obs)


PLANCK_X_550NM_K: float = 26160.0
"""`h c / (λ k_B)`，λ = 550 nm，单位 K。"""


def planck_band_boost(
    t_em: float | np.ndarray,
    g: float | np.ndarray,
    *,
    x_k: float = PLANCK_X_550NM_K,
) -> float | np.ndarray:
    """可见光波段（默认 550 nm）观测强度相对发射强度的频移增强因子。

    Args:
        t_em: 发射处温度，单位 K；标量或数组，> 0。
        g: 频移因子 `ν_obs / ν_em`；与 `t_em` 广播。
        x_k: `h c / (λ k_B)`，单位 K；决定参考波长。

    Returns:
        `B_ν(ν, g·T) / B_ν(ν, T)`，形状为广播结果；`g = 1` 时为 1，随 `g` 单调递增。

    Formula:
        ```
        I_ν,obs(ν) = g³ I_ν,em(ν / g) = B_ν(ν, g T)       # I_ν/ν³ 洛伦兹不变
        boost = (exp(x/T) − 1) / (exp(x/(g T)) − 1),  x = h c / (λ k_B)
        ```

    Physical Meaning:
        黑体在频移后仍是黑体（温度 g·T），因此固定波段的亮度变化由 Planck 函数决定。
        `T ≪ x` 时近似 `exp(x/T·(1 − 1/g))`，比 bolometric `g⁴` 更陡；`T ≫ x` 时 → g。

    Simplifications:
        单波长近似整个可见波段；指数参数钳制在 80 以内避免溢出。
    """
    t = np.maximum(_to_array(t_em), 1.0)
    gg = np.maximum(_to_array(g), 1e-6)
    x_em = np.minimum(x_k / t, 80.0)
    x_obs = np.minimum(x_k / (t * gg), 80.0)
    out = np.expm1(x_em) / np.expm1(x_obs)
    return float(out) if np.ndim(out) == 0 else out


def scaled_shift_factors(
    g: float | np.ndarray,
    s_lum: float,
    s_color: float,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """把物理 g 拆成亮度与颜色两路，各自乘以强度指数。

    Args:
        g: 物理频移因子；标量或数组。
        s_lum: 亮度频移强度指数（1 = 物理，0 = 关闭）。
        s_color: 颜色频移强度指数（1 = 物理，0 = 关闭，> 1 = 夸张）。

    Returns:
        `(g_lum, g_color) = (g^s_lum, g^s_color)`，形状跟随 `g`。

    Physical Meaning:
        显式的非物理艺术旋钮：`g_lum` 送入 `planck_band_boost`，`g_color` 用于观测色温 `T·g_color`。

    Simplifications:
        指数缩放同时作用于 Doppler 与引力红移两部分。
    """
    gg = np.maximum(_to_array(g), 1e-12)
    g_lum = np.power(gg, float(s_lum))
    g_col = np.power(gg, float(s_color))
    if np.ndim(g) == 0:
        return float(g_lum), float(g_col)
    return g_lum, g_col
