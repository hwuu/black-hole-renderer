"""Disk V2 Schwarzschild 频移公式（NumPy 参考实现）。

Taichi 端 `taichi_impl.disk_g_factor_ti` 必须与这里的 `disk_g_factor` 逐值一致；
`exact_equatorial_g_factor` 给出赤道面圆轨道的严格 GR 解析频移，作为两者的对照。
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
