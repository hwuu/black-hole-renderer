"""Schwarzschild 光线测地线的 NumPy 积分（笛卡尔等效势），供 Manim 图解与单测使用。

与渲染核（`src/v2/taichi_render.py`）是同一个方程 `d²x/dλ² = −1.5·k·L²·x/r⁵`（质量 k·M），
但用 float64 固定步长积分、保留完整轨迹，用于画出光路（"帽子"的成因、光子环等）；
不做体积渲染。纯 NumPy、不 import Taichi——图解在制片层进程内运行
（`docs/design_film.md` §2）。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def critical_impact_parameter(gravity_scale: float = 1.0, rs: float = 1.0) -> float:
    """光子球临界冲击参数（= 无自旋黑洞的阴影半径）。

    Args:
        gravity_scale: 引力强度旋钮 k（黑洞质量缩放，见 `src/v2/scene_anim`）。
        rs: Schwarzschild 半径（基准质量）。

    Returns:
        `b_c = (3√3/2)·k·r_s`：|b| < b_c 的光线落入视界（阴影内），|b| > b_c 逃到无穷远。

    Formula:
        光子球半径 `r_ph = 1.5·k·r_s`，圆轨道光线的冲击参数 `b = r_ph / sqrt(1 − k·r_s/r_ph)`
        = `(3√3/2)·k·r_s`。

    Physical Meaning:
        远处观者看到的黑洞阴影半径：引力旋钮 k 越小阴影越小，k → 0 时没有阴影。
    """
    return 3.0 * math.sqrt(3.0) / 2.0 * float(gravity_scale) * float(rs)


def integrate_ray(
    pos0: np.ndarray,
    dir0: np.ndarray,
    gravity_scale: float = 1.0,
    *,
    rs: float = 1.0,
    step: float = 0.05,
    max_steps: int = 20000,
    r_escape: float = 100.0,
) -> Tuple[np.ndarray, str, Optional[np.ndarray]]:
    """RK4 积分一条光线，返回完整轨迹与结局。

    Args:
        pos0: 起点位置 `(3,)`（r_s）。
        dir0: 起点方向 `(3,)`（自动归一）。
        gravity_scale: 引力强度旋钮 k ∈ [0, 1]（黑洞质量缩放）；0 = 光走直线。
        rs: Schwarzschild 半径（基准质量）。
        step: 固定积分步长（r_s）。
        max_steps: 最大步数（防止光子环附近无限绕圈）。
        r_escape: 逃逸半径；起点距离大于它时须自行调大。

    Returns:
        `(positions, outcome, escape_dir)`：
        `positions` 形状 `(N, 3)`，含起点与终点；`outcome` 为 `"horizon"`（落入视界）、
        `"escaped"`（逃出 `r_escape`）或 `"max_steps"`（步数耗尽）；
        `escape_dir` 为逃逸方向的单位向量（`(3,)`），未逃逸时为 None。

    Formula:
        `d²x/dλ² = −1.5·k·L²·x / r⁵`，`L² = |x₀ × d̂₀|²`（角动量守恒）；
        吞光半径 `max(k, 10⁻³)·r_s`（与渲染核一致，下限防 k → 0 时发散）。

    Physical Meaning:
        图解中每条光线的弯曲路径：k = 1 是真实黑洞，k = 0 是"如果光不被引力掰弯"。

    Simplifications:
        与渲染核相比无自适应步长、无盘采样；float64 精度只为画图与测试，
        不用于成像（成像以 Taichi f32 为准）。
    """
    k = float(gravity_scale)
    if not (math.isfinite(k) and k >= 0.0):
        raise ValueError(f"gravity_scale 必须是有限非负数，得到 {gravity_scale}")
    pos = np.asarray(pos0, dtype=np.float64).copy()
    d = np.asarray(dir0, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm <= 0.0:
        raise ValueError("dir0 不能是零向量")
    d = d / norm
    r_cap = max(k, 1e-3) * float(rs)
    l_vec = np.cross(pos, d)
    l2 = float(l_vec @ l_vec)

    def accel(p: np.ndarray) -> np.ndarray:
        """方程右端：a = −1.5·k·L²·p/r⁵；k = 0 或 r = 0 时恒为零（无引力 / 奇点处不产生 NaN）。"""
        if k <= 0.0:
            return np.zeros(3)
        r2 = float(p @ p)
        if r2 <= 0.0:
            return np.zeros(3)
        r = math.sqrt(r2)
        return -1.5 * k * l2 / (r2 * r2 * r) * p

    positions = [pos.copy()]
    outcome = "max_steps"
    for _ in range(max_steps):
        prev = pos
        k1p = step * d
        k1d = step * accel(pos)
        k2p = step * (d + 0.5 * k1d)
        k2d = step * accel(pos + 0.5 * k1p)
        k3p = step * (d + 0.5 * k2d)
        k3d = step * accel(pos + 0.5 * k2p)
        k4p = step * (d + k3d)
        k4d = step * accel(pos + k3p)
        pos = pos + (k1p + 2 * k2p + 2 * k3p + k4p) / 6.0
        d = d + (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0
        positions.append(pos.copy())
        r = float(np.linalg.norm(pos))
        # 步长可能整步跨过吞光球：除步末位置外，还检查段 [prev, pos] 到原点的最近距离
        if r <= r_cap or _segment_min_distance(prev, pos) <= r_cap:
            outcome = "horizon"
            break
        if r >= r_escape:
            outcome = "escaped"
            break
    return np.array(positions), outcome, (d / np.linalg.norm(d) if outcome == "escaped" else None)


def _segment_min_distance(a: np.ndarray, b: np.ndarray) -> float:
    """线段 ab 到原点的最短距离。

    Args:
        a, b: 线段两端点 `(3,)`。

    Returns:
        非负距离；零长度线段返回 `|a|`。

    Formula:
        `t* = clamp(−a·s / (s·s), 0, 1)`，`s = b − a`；距离 = `|a + t*·s|`。
    """
    seg = b - a
    denom = float(seg @ seg)
    if denom <= 0.0:
        return float(np.linalg.norm(a))
    t = min(max(-float(a @ seg) / denom, 0.0), 1.0)
    return float(np.linalg.norm(a + t * seg))
