"""Disk V2 基础物理场（NumPy 参考实现）。

- Page–Thorne 相对论薄盘通量 / 温度剖面与 T_peak 推导（第 1 层：M、Ṁ → T_peak）。
- Shakura–Sunyaev 外区（气压主导、Kramers 不透明度）标高 H(r) 与柱密度 Σ(r)。
- 内边界力矩（Agol & Krolik 2000）与 ISCO 以内的坠落区（质量守恒 + Schwarzschild 坠落测地线）。

Taichi 端（`taichi_impl.DiskV2Taichi._ss_half_thickness` 等）与这里逐值一致。
"""

from __future__ import annotations

import math

import numpy as np


def page_thorne_flux(r, m=0.5, isco_stress=0.0):
    """Schwarzschild Page–Thorne 通量（几何单位，r 单位 r_s），可含 ISCO 处的内边界力矩。

    Args:
        r: 半径（r_s），标量或数组；r < 3 r_s 的值按 r = 3 r_s（ISCO）处理。
        m: 几何质量（r_s = 2M，默认 M = 0.5）。
        isco_stress: 内边界力矩系数 β（[0, 1)）：ISCO 处力矩 `W_in = β·Ṁ·L_in`；0 = 零力矩（标准 Page–Thorne）。

    Returns:
        相对通量 F(r)（任意单位，≥ 0；标量输入返回 NumPy 标量，数组输入返回同形状数组；β = 0 时在 ISCO 处为 0、
        峰值位于 r ≈ 4.8 r_s）。

    Formula:
        E = (1-2M/r)/sqrt(1-3M/r), L = sqrt(Mr)/sqrt(1-3M/r), Ω = sqrt(M/r³)
        F ∝ -Ω'(r) / (E-ΩL)² / r · [ ∫_{r_isco}^{r} (E-ΩL)L'(r') dr' + β·(E-ΩL)_isco·L_isco ]

    Physical Meaning:
        积分项是零力矩边界下由角动量守恒得到的耗散；方括号内的常数是 ISCO 处的力矩把角动量（与能量）
        继续向外输运带来的额外耗散。牛顿极限下方括号 ∝ √r − (1 − β)·√r_in，即内边界因子
        `f_β = 1 − (1 − β)·√(r_in/r)`。

    Simplifications:
        力矩以 Ṁ·L_in 为单位给定、不随时间变化（Agol & Krolik 2000 的稳态形式）；不含坠落区的辐射。
    """
    r = np.maximum(np.asarray(r, dtype=np.float64), 3.0 * 2 * m)
    rr = np.linspace(3.0 * 2 * m, max(float(r.max()), 3.0 * 2 * m + 1e-3), 20001)
    e = (1 - 2 * m / rr) / np.sqrt(1 - 3 * m / rr)
    l = np.sqrt(m * rr) / np.sqrt(1 - 3 * m / rr)
    om = np.sqrt(m / rr ** 3)
    dl = np.gradient(l, rr)
    dom = np.gradient(om, rr)
    integrand = (e - om * l) * dl
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(rr))])
    if isco_stress != 0.0:
        # 内边界力矩：积分常数 β·(E − ΩL)_isco·L_isco
        integ = integ + float(isco_stress) * (e[0] - om[0] * l[0]) * l[0]
    flux = -dom / (e - om * l) ** 2 / rr * integ
    return np.interp(r, rr, np.maximum(flux, 0.0))


def derive_t_peak(m_msun, mdot_edd):
    """由黑洞质量与吸积率推出 Page–Thorne 盘的峰值有效温度（K）。

    Args:
        m_msun: 黑洞质量（太阳质量）。
        mdot_edd: 吸积率（爱丁顿倍数）。

    Returns:
        T_peak（K）。公式：r_s = 2GM/c²，L_Edd = 4πGM m_p c/σ_T，
        Mdot = mdot·L_Edd/(ηc²)，η = 1-sqrt(8/9)，T_peak = (max F/σ_SB)^{1/4}。
    """
    g_, c_, msun, sig_sb, m_p, s_t = 6.674e-8, 2.998e10, 1.989e33, 5.6704e-5, 1.6726e-24, 6.6524e-25
    m = float(m_msun) * msun
    rs = 2 * g_ * m / c_ ** 2
    eta = 1 - math.sqrt(8 / 9)
    mdot = float(mdot_edd) * 4 * math.pi * g_ * m * m_p * c_ / s_t / (eta * c_ ** 2)
    f_geom = float(page_thorne_flux(np.linspace(3.0, 20.0, 4001)).max())
    return (mdot * c_ ** 2 / (4 * math.pi * rs ** 2) * f_geom / sig_sb) ** 0.25


def build_page_thorne_lut(lut_field, r_in, r_out, n, isco_stress=0.0):
    """把 Page–Thorne T(r)/T_peak 采样到 Taichi field（r ∈ [r_in, r_out] 等间距）。

    Args:
        lut_field: `ti.field(dtype=ti.f32, shape=n)`（或任何有 `from_numpy` 的对象）。
        r_in, r_out: 温度表覆盖的半径范围（r_s）。
        n: 表项数。
        isco_stress: 内边界力矩系数 β（[0, 1)），见 `page_thorne_flux`；0 = 零力矩。

    Returns:
        None；写入 n 个 float32 温度倍率 `T(r)/T_peak,0`（≥ 0）。T_peak,0 为零力矩盘的峰值温度，
        故 β = 0 时最大值为 1、β > 0 时内缘额外加热、最大值 ≥ 1。

    Formula:
        `lut_i = (F_β(r_i) / max_r F_0(r))^{1/4}`

    Physical Meaning:
        力矩是额外的能量来源，不改变由 M、Ṁ 推出的零力矩峰值温度 T_peak（`derive_t_peak`），只在内缘加热。
    """
    rs = np.linspace(r_in, r_out, n)
    tt0 = page_thorne_flux(rs) ** 0.25
    tt = tt0 if isco_stress == 0.0 else page_thorne_flux(rs, isco_stress=isco_stress) ** 0.25
    lut_field.from_numpy((tt / tt0.max()).astype(np.float32))


def inner_boundary_factor(r, r_in, isco_stress=0.0):
    """薄盘内边界因子 f_β（牛顿近似，SS 剖面与之配套）。

    Args:
        r: 半径（r_s），标量或数组；r < r_in 时按 r = r_in 计算。
        r_in: 盘内半径（r_s，> 0）。
        isco_stress: 内边界力矩系数 β（[0, 1)）；0 = 零力矩。

    Returns:
        与 r 同形状的数组，值域 [max(β, 1e-6), 1)：r = r_in 处为 β（β = 0 时截断到 1e-6），r → ∞ 时趋于 1。

    Formula:
        `f_β = max(1 − (1 − β)·√(r_in / max(r, r_in)), 1e-6)`

    Physical Meaning:
        粘滞力矩在内边界的取值：零力矩时 f → 0（ISCO 处无耗散、无质量堆积）；β > 0 时内边界仍有力矩，
        耗散与柱密度在 ISCO 处有限。
    """
    r_arr = np.asarray(r, dtype=np.float64)
    return np.maximum(1.0 - (1.0 - isco_stress) * np.sqrt(r_in / np.maximum(r_arr, r_in)), 1e-6)


def plunge_isco_velocity(r_in, plunge_width):
    """坠落区渐隐宽度 Δr 对应的 ISCO 处径向速度 v_I（单位 c）。

    Args:
        r_in: 盘内半径（r_s），按 ISCO 处理（Schwarzschild 为 3 r_s）。
        plunge_width: 坠落区渐隐宽度 Δr（r_s，0 < Δr < r_in − 1；本函数不校验，渲染路径由
            `DiskV2VolumeParams` 限制在 (0, 0.3]）。

    Returns:
        正标量 v_I（c）：Δr = 0.1 → ≈ 0.0021，Δr = 0.3 → ≈ 0.0123（r_in = 3）。

    Formula:
        `u_pl(r) = √(2 r_g / (3 r_I))·(r_I/r − 1)^{3/2}`（r_g = r_s/2 = 0.5，r_I = r_in）
        `v_I = u_pl(r_in − Δr)`，即 Δr 处坠落速度与 ISCO 初速相等（柱密度降到 ISCO 值约一半）。

    Physical Meaning:
        气体在 ISCO 处以有限径向速度 v_I 离开圆轨道，之后沿坠落测地线加速（Mummery & Balbus 2019）。
    """
    return math.sqrt(1.0 / (3.0 * r_in)) * (r_in / (r_in - plunge_width) - 1.0) ** 1.5


def plunge_surface_density_ratio(r, r_in, plunge_width):
    """坠落区柱密度比 Σ(r)/Σ(r_in)（质量守恒）。

    Args:
        r: 半径（r_s），标量或数组，> 0。
        r_in: 盘内半径 = ISCO（r_s）。
        plunge_width: 坠落区渐隐宽度 Δr（r_s），见 `plunge_isco_velocity`。

    Returns:
        与 r 同形状的正数组：r ≥ r_in 时为 1；r = r_in − Δr 处为 r_in/(2r)；r → 1.5 r_s 时降到约 0.4%（Δr = 0.05）
        至 7.1%（Δr = 0.3）。ISCO 内侧极窄一段（< 0.002 r_s）因几何因子 r_in/r 先于坠落速度增长，比值略高于 1
        （Δr = 0.1 时最大 1.000006，Δr = 0.3 时 1.0002），之后向内单调下降。

    Formula:
        `Σ/Σ_I = r_in·v_I / (r·(v_I + u_pl(r)))`，`u_pl` 见 `plunge_isco_velocity`

    Physical Meaning:
        稳态吸积 `Ṁ = 2π r Σ v_r` 处处相同；坠落区 v_r 由初速 v_I 加上测地线坠落速度，气体越落越快、越稀。

    Simplifications:
        径向速度取 v_I + u_pl 的线性叠加（保证 r_in 处连续，ISCO 内侧的微小增密即来自这一叠加）；
        忽略坠落区内的压力与磁应力。
    """
    r_arr = np.asarray(r, dtype=np.float64)
    v_i = plunge_isco_velocity(r_in, plunge_width)
    u = math.sqrt(1.0 / (3.0 * r_in)) * np.maximum(r_in / r_arr - 1.0, 0.0) ** 1.5
    return np.where(r_arr < r_in, r_in * v_i / (r_arr * (v_i + u)), 1.0)


def plunge_temperature(r, r_in, plunge_width):
    """坠落区温度倍率 T(r)/T(r_in)（绝热膨胀，标高不变）。

    Args:
        r: 半径（r_s），标量或数组，> 0。
        r_in: 盘内半径 = ISCO（r_s）。
        plunge_width: 坠落区渐隐宽度 Δr（r_s），见 `plunge_isco_velocity`。

    Returns:
        与 r 同形状的正数组：r ≥ r_in 时为 1，向内下降（ISCO 内侧极窄一段与柱密度比一样略高于 1，见
        `plunge_surface_density_ratio`）。

    Formula:
        `T/T_I = (ρ/ρ_I)^{γ−1} = (Σ/Σ_I)^{2/3}`（γ = 5/3，标高 H = H(r_in) 使 ρ ∝ Σ）

    Physical Meaning:
        坠落气体不再有粘滞耗散，按绝热规律随密度下降而冷却，越往内越暗。

    Simplifications:
        标高固定为 ISCO 处的值；不含辐射冷却与激波加热。
    """
    return plunge_surface_density_ratio(r, r_in, plunge_width) ** (2.0 / 3.0)


def ss_half_thickness(r, r_in, hr_ref, r_ref, isco_stress=0.0):
    """SS 外区标高 H(r)（NumPy 参考）。

    Args:
        r: 半径（r_s），标量或数组。
        r_in: 盘内半径（r_s）。
        hr_ref: r = r_ref 处的 H/r（已含盘厚缩放）。
        r_ref: 归一参考半径（r_s）。
        isco_stress: 内边界力矩系数 β（[0, 1)），见 `inner_boundary_factor`；β > 0 时坠落区（r < r_in）取 H(r_in)。

    Returns:
        与 r 同形状的正数组：H(r)（r_s）。

    Formula:
        `H = hr_ref·r·(r/r_ref)^{1/8}·(f_β/f_β,ref)^{3/20}`，`f_β,ref = f_β(r_ref)`；β > 0 时 r 取 max(r, r_in)

    Physical Meaning:
        气压主导、Kramers 不透明度的 SS 外区静力平衡标高。

    Simplifications:
        β > 0 时坠落区标高取 ISCO 处的值（气体来不及重新建立静力平衡，不随半径变化）；
        β = 0 时 r < r_in 仍按上式外推（采样下界为 r_in，不会被用到）。
    """
    r_arr = np.asarray(r, dtype=np.float64)
    rr = r_arr if isco_stress == 0.0 else np.maximum(r_arr, r_in)
    f_ref = float(inner_boundary_factor(r_ref, r_in, isco_stress))
    return hr_ref * rr * (rr / r_ref) ** 0.125 * (inner_boundary_factor(rr, r_in, isco_stress) / f_ref) ** 0.15


def ss_surface_density(r, r_in, r_out, r_ref, isco_stress=0.0, plunge_width=0.1):
    """SS 外区柱密度 Σ(r)（NumPy 参考），β > 0 时含坠落区。

    Args:
        r: 半径（r_s），标量或数组。
        r_in, r_out: 盘内外半径（r_s）。
        r_ref: 归一参考半径（r_s）。
        isco_stress: 内边界力矩系数 β（[0, 1)），见 `inner_boundary_factor`。
        plunge_width: 坠落区渐隐宽度 Δr（r_s），仅 β > 0 时生效，见 `plunge_surface_density_ratio`。

    Returns:
        与 r 同形状的非负数组：Σ(r)（无量纲形状，r = r_ref 附近 ≈ 1）；外缘 r_out 处为 0；
        β > 0 时 r_in 处有限、坠落区向内单调下降。

    Formula:
        `Σ = (r̃/r_ref)^{-3/4}·(f_β(r̃)/f_β,ref)^{7/10}·outer(r̃)·P(r)`，r̃ = max(r, r_in)（β = 0 时 r̃ = r、P = 1）
        `outer = 1 − sstep((r − 0.72 r_out)/(0.28 r_out))`，`P = plunge_surface_density_ratio`

    Physical Meaning:
        SS 外区柱密度剖面；零力矩时在 ISCO 处趋于 0，有限力矩时 ISCO 处有限，坠落区由质量守恒稀释。
    """
    r_arr = np.asarray(r, dtype=np.float64)
    rr = r_arr if isco_stress == 0.0 else np.maximum(r_arr, r_in)
    f_ref = float(inner_boundary_factor(r_ref, r_in, isco_stress))
    outer = 1.0 - np.clip((rr - 0.72 * r_out) / (r_out - 0.72 * r_out), 0, 1)
    outer = outer * outer * (3 - 2 * outer)
    sig = (rr / r_ref) ** (-0.75) * (inner_boundary_factor(rr, r_in, isco_stress) / f_ref) ** 0.7 * outer
    if isco_stress == 0.0:
        return sig
    return sig * plunge_surface_density_ratio(r_arr, r_in, plunge_width)
