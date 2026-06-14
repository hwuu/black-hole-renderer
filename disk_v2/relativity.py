"""Disk V2 Schwarzschild 相对论辅助公式（NumPy reference）。

本模块提供 g-factor 与 Keplerian 角速度的可测试参考实现。Taichi 端必须与这里的
符号约定保持一致。
"""

from __future__ import annotations

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
    eps: float = 0.01,
    beta_cap: float = 0.99,
) -> float | np.ndarray:
    """局部静止观测者测得的 Schwarzschild 圆轨道速度近似。

    Args:
        r: 发射半径，单位 `r_s`。
        rs: Schwarzschild 半径。
        eps: ISCO 附近分母钳制，避免 `1 - 3M/r` 为 0。
        beta_cap: 工程速度上限。

    Returns:
        `β = v/c`，范围 `[0, beta_cap]`。

    Formula:
        ```
        M = rs / 2
        beta = sqrt(M / r) / sqrt(max(1 - 3M / r, eps))
        ```

    Simplifications:
        这是圆轨道局部速度的工程近似，只在 `r >= r_isco` 附近使用；ISCO 处
        用 `eps` 与 `beta_cap` 做数值保护。
    """
    r_arr = np.maximum(_to_array(r), float(rs) + 1e-6)
    mass = schwarzschild_mass_from_rs(rs)
    denom = np.sqrt(np.maximum(1.0 - 3.0 * mass / r_arr, float(eps)))
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
