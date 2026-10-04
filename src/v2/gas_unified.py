"""统一气体模型竖直剖面的 NumPy 参考实现（供 `DiskV2Taichi.density_I` 逐值一致性测试）。

盘面（高斯核心）与大气（指数尾巴）是同一团气体：同一柱密度 Σ、同一湍流场 ĉ = c/⟨c⟩、
同一条灰大气温度规律与同一个 κ。模型见 `docs/plans/v2_unified_gas_plan.md` §3。

本模块只放给定局部量（Σ、H、H_s、ĉ）后的竖直剖面公式，不含噪声与平流。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_SQRT_2PI = 2.5066283


@dataclass(frozen=True)
class UnifiedGasLocal:
    """某一 (r, φ) 柱上的局部量（由噪声与 SS 结构给出，竖直方向不变）。

    Args:
        sig: 柱密度 Σ（无量纲，已乘大尺度 lognormal 调制，≥ 0）。
        h_geo: SS 标高 H（r_s，> 0）。
        h_s: 带表面起伏的核心标高 H_s（r_s，> 0）。
        cn: 归一化湍流密度 ĉ = c/⟨c⟩（≥ 0，均值约 1）。
        h_a: 大气标高 H_a = atm_height·r（r_s，> 0）。
    """

    sig: float
    h_geo: float
    h_s: float
    cn: float
    h_a: float


def atm_coverage(cn, c0: float, soft: float):
    """大气覆盖软阈值。

    Args:
        cn: 归一化湍流密度 ĉ（标量或数组，≥ 0）。
        c0: 阈值中心（以 ĉ 计）。
        soft: 阈值宽度（> 0）。

    Returns:
        覆盖率，形状随 `cn` 广播，值域 (0, 1)；ĉ ≫ c0 时 → 1，ĉ ≪ c0 时 → 0。

    Formula:
        `cov(ĉ) = 1 / (1 + exp(−(ĉ − c0)/soft))`

    Physical Meaning:
        下方湍流稀疏处托起的大气也稀疏，形成空隙；空隙与盘面结构相关。
    """
    return 1.0 / (1.0 + np.exp(-(np.asarray(cn, dtype=np.float64) - c0) / soft))


def scatter_albedo(rho_a, rho_mid, q: float):
    """散射反照率 ω(ρ)。

    Args:
        rho_a: 大气体积密度（无量纲，标量或数组，≥ 0）。
        rho_mid: 中面参考密度 Σ/(√(2π)·H)（> 0）。
        q: 中面密度处吸收与散射不透明度之比 κ_abs/κ_es（≥ 0）。

    Returns:
        ω，形状随输入广播，值域 (0, 1]；ρ → 0 时 → 1，随 ρ 单调减。

    Formula:
        `ω = 1 / (1 + q·ρ_a/ρ_mid)`

    Physical Meaning:
        Kramers 吸收 κ_abs ∝ ρ，电子散射 κ_es 与 ρ 无关；稀薄大气以散射为主。

    Simplifications:
        κ_abs 的 T^{−3.5} 依赖忽略。
    """
    return 1.0 / (1.0 + q * np.asarray(rho_a, dtype=np.float64) / max(rho_mid, 1e-12))


def unified_gas_profile(z, loc: UnifiedGasLocal, *, core_floor: float, atm_frac: float, atm_c0: float,
                        atm_soft: float, q: float, kappa: float, grey_mix: float, grey_cap: float,
                        dt_i: float, surf_lo: float = 1.0, surf_k: float = 0.0, fine_n=0.0,
                        fine_sigma: float = 0.0):
    """统一气体模型在一条竖直柱上的密度、温度倍率与散射权重（与 `density_I` 同序同义）。

    Args:
        z: 离中面高度（r_s，标量或数组，可正可负）。
        loc: 该柱的局部量 `UnifiedGasLocal`。
        core_floor: 核心密度起伏下限 f ∈ [0, 1]。
        atm_frac: 大气柱密度比 A（≥ 0）。
        atm_c0 / atm_soft: 大气覆盖软阈值中心与宽度。
        q: 吸收 / 散射不透明度比（见 `scatter_albedo`）。
        kappa: 不透明度 κ（已标定，> 0）。
        grey_mix / grey_cap: 灰大气强度与温度倍率上限。
        dt_i: 核心温度起伏系数。
        surf_lo / surf_k: 表面增亮源函数系数。
        fine_n: 大气小尺度起伏 n（零均值、单位方差；标量或与 `z` 同形状的数组）。
        fine_sigma: 小尺度起伏强度 σ_a（≥ 0）；0 = 无起伏（m = 1）。

    Returns:
        `(em_c, tf, ab_c, ab_a, em_a, sc_a)`，形状随 `z` 广播：核心发射权重（≥ 0）、温度倍率
        （值域 [0.7, 1.3·grey_cap]，围绕 1）、核心吸收（≥ 0）、大气消光（≥ 0）、大气热发射权重
        `(1 − ω)·ab_a ∈ [0, ab_a]`、大气散射权重 `ω·ab_a·(1 − e^{−τ_c}) ∈ [0, ab_a − em_a]`。
        核心在 |z| ≥ 3H_s 处为 0（截断）。

    Formula:
        ```
        cfac  = f + (1 − f)·ĉ
        ab_c  = cfac·Σ·exp(−z²/2H_s²)/(√(2π)·H_s)                      （|z| < 3H_s）
        ab_a  = A·Σ·ĉ·cov(ĉ)·m(n)·exp(−|z|/H_a)/(2H_a)，m(n) = exp(σ_a·n − σ_a²/2)
        em_a  = (1 − ω(ab_a))·ab_a
        sc_a  = (ab_a − em_a)·(1 − exp(−κ·cfac·Σ))                    （下方核心层的发射率）
        τ_z   = κ·[cfac·Σ·½·erfc(|z|/(√2·H_s)) + ½·A·Σ·ĉ·cov·exp(−|z|/H_a)]
        tf    = [1 + grey_mix·(min((¾(τ_z + ⅔))^{1/4}, grey_cap) − 1)]·clamp(1 + dt_i·(ĉ − 1), 0.7, 1.3)
        ```

    Physical Meaning:
        致密核心（等温静力平衡）与稀薄大气共用一条温度规律：温度只取决于上方光学深度 τ_z，
        从中面到大气顶连续下降，没有核心 / 大气的分界。

    Simplifications:
        平行平面近似；大气竖直剖面取指数（等温大气）而非高斯；散射入射光只取正下方核心层；
        τ_z 的大气项不含 m(n)（取局部平均柱）。
    """
    z = np.asarray(z, dtype=np.float64)
    az = np.abs(z)
    cfac = core_floor + (1.0 - core_floor) * loc.cn
    inside = az < 3.0 * loc.h_s
    ab_c = np.where(inside, cfac * loc.sig * np.exp(-0.5 * (az / loc.h_s) ** 2) / (_SQRT_2PI * loc.h_s), 0.0)
    em_c = ab_c * (surf_lo + surf_k * az / loc.h_s)
    col_a = atm_frac * loc.sig * loc.cn * float(atm_coverage(loc.cn, atm_c0, atm_soft))
    ab_a = col_a * np.exp(-az / loc.h_a) / (2.0 * loc.h_a)
    ab_a = ab_a * np.exp(fine_sigma * np.asarray(fine_n, dtype=np.float64) - 0.5 * fine_sigma * fine_sigma)
    rho_mid = loc.sig / (_SQRT_2PI * loc.h_geo)
    em_a = (1.0 - scatter_albedo(ab_a, rho_mid, q)) * ab_a
    sc_a = (ab_a - em_a) * (1.0 - math.exp(-kappa * cfac * loc.sig))
    erfc = np.vectorize(math.erfc, otypes=[np.float64])
    tau_z = kappa * (cfac * loc.sig * 0.5 * erfc(az / (math.sqrt(2.0) * loc.h_s)) + 0.5 * col_a * np.exp(-az / loc.h_a))
    tf = 1.0 + grey_mix * (np.minimum((0.75 * (tau_z + 2.0 / 3.0)) ** 0.25, grey_cap) - 1.0)
    tf = tf * min(max(1.0 + dt_i * (loc.cn - 1.0), 0.7), 1.3)
    return em_c, tf, ab_c, ab_a, em_a, sc_a


def unified_column(loc: UnifiedGasLocal, *, core_floor: float, atm_frac: float, atm_c0: float,
                   atm_soft: float, atm_extent: float) -> float:
    """竖直柱密度 ∫(ab_c + ab_a) dz 的解析值（含采样截断；无小尺度起伏，或其期望）。

    Args:
        loc: 该柱的局部量。
        core_floor / atm_frac / atm_c0 / atm_soft: 同 `unified_gas_profile`。
        atm_extent: 大气截断高度（以 H_a 计）。

    Returns:
        标量柱密度（无量纲，≥ 0）；乘 κ 即 face-on 竖直光学深度 τ⊥。

    Formula:
        `∫ρ dz = Σ·[cfac·erf(3/√2) + A·ĉ·cov(ĉ)·(1 − e^{−atm_extent})]`

    Physical Meaning:
        截断前（erf → 1、e^{−ext} → 0）即 Σ·(cfac + A·ĉ·cov)：核心与大气的柱密度之和。
    """
    cfac = core_floor + (1.0 - core_floor) * loc.cn
    cov = float(atm_coverage(loc.cn, atm_c0, atm_soft))
    return loc.sig * (cfac * math.erf(3.0 / math.sqrt(2.0)) + atm_frac * loc.cn * cov * (1.0 - math.exp(-atm_extent)))
