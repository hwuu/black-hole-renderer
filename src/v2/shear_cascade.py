"""对数极坐标剪切级联：主云结构场的几何构造与 NumPy 参考实现。

开普勒剪切把湍流团块沿轨道方向拉长，并使长轴略向拖尾方向倾斜。本模块在对数极坐标
`(ln r, φ)` 中为级联的每个八度构造一个各向异性噪声格：

- 八度 k 的 `ln r` 频率 `K_k = k0·3^k`（每单位 ln r 的格数）；由于格子在 `ln r` 中等距，
  物理尺寸随半径线性增长，各半径的形状（长宽比、倾角）相同。
- 长宽比从 `ar_big`（最大八度）几何插值到 `ar_small`（最小八度）：大尺度结构存活更久、被剪切
  拉得更长，小尺度结构接近各向同性。
- 拖尾倾角 `θ_k = tilt_k·atan(1/(AR_k − 1))`；`tilt_k = 1` 为由长宽比推出的物理估计，`tilt_k = 0`
  为长轴严格沿方位方向。

Taichi 实现见 `taichi_impl.DiskV2Taichi._casc_shear`，与本模块 `shear_cascade_np` 逐值一致。
"""

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass(frozen=True)
class ShearCascadeGeometry:
    """剪切级联各八度的噪声坐标系数（构造期一次算好，kernel 内作为常数使用）。

    Args:
        kr: 各八度 `ln r` 频率 `K_k`（格 / 单位 ln r），`k0·3^k`。
        period: 各八度方位周期 `P_k`（正整数，噪声第二坐标每转一圈前进 `P_k` 格，保证 φ 无缝）。
        u_scale: 各八度第一坐标缩放 `|N₀₀|`（无倾角时为 1）。
        shear: 各八度剪切系数 `s_k`；第二坐标 `v = φ/2π·P_k + spin·s_k·K_k·ln r`。
        aspect: 各八度目标长宽比 `AR_k`（≥ 1，方位长 / 径向宽，以各向同性格为单位）。
        tilt_rad: 各八度目标拖尾倾角 `θ_k`（rad，≥ 0）。

    Physical Meaning:
        描述"主云团块在每个尺度上被剪切成什么形状"：长宽比与倾角来自开普勒剪切，
        方位周期取整保证结构绕盘一圈无接缝。
    """

    kr: Tuple[float, ...]
    period: Tuple[int, ...]
    u_scale: Tuple[float, ...]
    shear: Tuple[float, ...]
    aspect: Tuple[float, ...]
    tilt_rad: Tuple[float, ...]


def shear_cascade_geometry(k0: float, n_oct: int, ar_big: float, ar_small: float,
                           tilt_k: float, disk_spin: float) -> ShearCascadeGeometry:
    """按目标长宽比与倾角解析构造各八度的噪声坐标系数。

    Args:
        k0: 最大八度的 `ln r` 频率（格 / 单位 ln r，> 0）。
        n_oct: 八度数（≥ 1），逐八度频率 ×3。
        ar_big / ar_small: 最大 / 最小八度的长宽比（≥ 1）；中间八度按几何级数插值。
        tilt_k: 拖尾倾角系数（≥ 0）：`θ_k = tilt_k·atan(1/(AR_k − 1))`。
        disk_spin: 盘旋转方向（+1 / −1），决定"拖尾"的方位方向。

    Returns:
        `ShearCascadeGeometry`，各字段为长度 `n_oct` 的元组。

    Formula:
        以各向同性格为单位，目标特征的两个轴为
        `e_L = (sin θ, −spin·cos θ)`（长轴，近方位方向）、`e_S = (cos θ, spin·sin θ)`（短轴），
        形状矩阵 `F = [AR·e_L, e_S]`（列向量，坐标顺序为 (ln r, φ) 方向）。
        噪声坐标 `N = G·F⁻¹`，G 为旋转，选取使 `N₀₁ = 0`（第一坐标只含 ln r，第二坐标保持 φ 周期）：
        `u = |N₀₀|·K·ln r`，`v = φ/2π·P + spin·s·K·ln r`，
        `P = round(|N₁₁|·2π·K)`，`s = sgn(N₁₁)·N₁₀·spin`。

    Physical Meaning:
        开普勒剪切 q = 3/2 下，存活时间为 τ 的团块长宽比约为 `1 + q·Ω·τ`，长轴拖尾倾角约
        `atan(1/(AR − 1))`；本函数把这一形状映射成可周期回绕的噪声坐标。

    Simplifications:
        - 长宽比按八度几何插值，而非由各尺度的真实存活时间推出；
        - `P` 取整会使实际方位尺寸与目标有至多半格的偏差（最大八度 P ≈ 4 时偏差最大）。
    """
    if k0 <= 0.0:
        raise ValueError("k0 must be positive")
    if n_oct < 1:
        raise ValueError("n_oct must be >= 1")
    if ar_big < 1.0 or ar_small < 1.0:
        raise ValueError("aspect ratios must be >= 1")
    if tilt_k < 0.0:
        raise ValueError("tilt_k must be >= 0")
    if disk_spin not in (1.0, -1.0):
        raise ValueError("disk_spin must be +1.0 or -1.0")
    ars = [ar_big * (ar_small / ar_big) ** (k / max(n_oct - 1, 1)) for k in range(n_oct)]
    krs = [float(k0) * 3.0 ** k for k in range(n_oct)]
    sp = float(disk_spin)
    per, ua, shear, tilts = [], [], [], []
    for kr, ar in zip(krs, ars):
        th = float(tilt_k) * math.atan(1.0 / max(ar - 1.0, 1e-6))
        e_l = (math.sin(th), -sp * math.cos(th))
        e_s = (math.cos(th), sp * math.sin(th))
        f = [[ar * e_l[0], e_s[0]], [ar * e_l[1], e_s[1]]]
        det = f[0][0] * f[1][1] - f[0][1] * f[1][0]
        fi = [[f[1][1] / det, -f[0][1] / det], [-f[1][0] / det, f[0][0] / det]]
        # 旋转 G 使 N 的 (0, 1) 元为 0
        ang = math.atan2(fi[0][1], fi[1][1])
        c_, s_ = math.cos(ang), math.sin(ang)
        n = [[c_ * fi[0][0] - s_ * fi[1][0], c_ * fi[0][1] - s_ * fi[1][1]],
             [s_ * fi[0][0] + c_ * fi[1][0], s_ * fi[0][1] + c_ * fi[1][1]]]
        per.append(max(1, int(round(abs(n[1][1]) * 2.0 * math.pi * kr))))
        sgn = 1.0 if n[1][1] > 0 else -1.0
        ua.append(float(abs(n[0][0])))
        shear.append(float(sgn * n[1][0] * sp))
        tilts.append(th)
    return ShearCascadeGeometry(tuple(krs), tuple(per), tuple(ua), tuple(shear), tuple(ars), tuple(tilts))


def shear_cascade_np(lnr: np.ndarray, phi: np.ndarray, zr: np.ndarray, ox: float, oz: float,
                     geom: ShearCascadeGeometry, con: float, oct_gain: float, disk_spin: float,
                     noise: Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]) -> np.ndarray:
    """剪切级联的 NumPy 参考实现（与 `DiskV2Taichi._casc_shear` 逐值一致）。

    Args:
        lnr: `ln r`（r 单位 r_s），标量或数组。
        phi: 带内流坐标 φ（rad，已扣除刚体环旋转），与 `lnr` 可广播。
        zr: 无量纲高度 `z / r`，与 `lnr` 可广播。
        ox / oz: 种子偏移（标量）。
        geom: `shear_cascade_geometry` 的返回值。
        con: 级联对比度（> 0），进入 softplus 前的放大倍数。
        oct_gain: 逐八度增益 γ（> 0），第 k 个八度幅度乘 γ^k。
        disk_spin: 盘旋转方向（+1 / −1），须与构造 `geom` 时一致。
        noise: 三维值噪声 `noise(x, y, z, period)`，y 方向周期为 `period`，输出 `[-1, 1]`、零均值。

    Returns:
        与输入广播形状相同的数组，`≥ 0`；均值约在 1 附近（由调用方按 ⟨c⟩ 归一）。

    Formula:
        `s = Σ_k ln(1 + 0.1·γ^k·n_k)`，`n_k = noise(u_k + ox, v_k, K_k·zr + oz, P_k)`，
        `u_k = |N₀₀|_k·K_k·ln r`，`v_k = φ/2π·P_k + spin·s_k·K_k·ln r`；
        输出 `softplus(con·s) = ln(1 + e^{con·s})`。

    Physical Meaning:
        乘性级联：每个尺度对密度做一次小幅相对扰动，累乘后形成对数正态式的团块与缝隙，
        各尺度的形状由剪切几何给出。

    Simplifications:
        竖直方向用各向同性坐标 `K_k·z/r`，不单独考虑竖直方向的剪切。
    """
    lnr = np.asarray(lnr, dtype=np.float64)
    s = 0.0
    for k in range(len(geom.kr)):
        u0 = geom.kr[k] * lnr
        u = geom.u_scale[k] * u0
        v = phi / (2.0 * math.pi) * geom.period[k] + disk_spin * geom.shear[k] * u0
        n = noise(u + ox, v, geom.kr[k] * zr + oz, geom.period[k])
        s = s + np.log(1.0 + 0.1 * n * oct_gain ** k)
    x = con * s
    return np.logaddexp(0.0, x)
