"""Disk V2 基础物理场（NumPy 参考实现）。

- Page–Thorne 相对论薄盘通量 / 温度剖面与 T_peak 推导（第 1 层：M、Ṁ → T_peak）。
- Shakura–Sunyaev 外区（气压主导、Kramers 不透明度）标高 H(r) 与柱密度 Σ(r)。

Taichi 端（`taichi_impl.DiskV2Taichi._ss_half_thickness` 等）与这里逐值一致。
"""

from __future__ import annotations

import math

import numpy as np


def page_thorne_flux(r, m=0.5):
    """Schwarzschild Page–Thorne 通量（几何单位，r 单位 r_s）。

    Args:
        r: 半径（r_s），标量或数组，r > 3 r_s。
        m: 几何质量（r_s = 2M，默认 M = 0.5）。

    Returns:
        F(r) ∝ -Ω'/(E-ΩL)²/r · ∫(E-ΩL)L' dr'。

    Formula:
        E = (1-2M/r)/sqrt(1-3M/r), L = sqrt(Mr)/sqrt(1-3M/r), Ω = sqrt(M/r³)
        F ∝ -Ω'(r) / (E-ΩL)² / r · ∫_{r_in}^{r} (E-ΩL)L'(r') dr'
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


def build_page_thorne_lut(lut_field, r_in, r_out, n):
    """把 Page–Thorne T(r)/T_peak 采样到 Taichi field（r ∈ [r_in, r_out] 等间距）。

    Args:
        lut_field: `ti.field(dtype=ti.f32, shape=n)`。
        r_in, r_out: 温度表覆盖的半径范围（r_s）。
        n: 表项数。
    """
    rs = np.linspace(r_in, r_out, n)
    tt = page_thorne_flux(rs) ** 0.25
    lut_field.from_numpy((tt / tt.max()).astype(np.float32))


def ss_half_thickness(r, r_in, hr_ref, r_ref):
    """SS 外区标高 H(r)（NumPy 参考）。

    Args:
        r: 半径（r_s），标量或数组。
        r_in: 盘内半径（r_s）。
        hr_ref: r = r_ref 处的 H/r。
        r_ref: 归一参考半径（r_s）。

    Returns:
        H(r)（r_s），公式 H = hr_ref·r·(r/r_ref)^{1/8}·(f/f_ref)^{3/20}。
    """
    r_arr = np.asarray(r, dtype=np.float64)
    fr = np.maximum(1.0 - np.sqrt(r_in / np.maximum(r_arr, r_in)), 1e-6)
    f_ref = 1.0 - math.sqrt(r_in / r_ref)
    return hr_ref * r_arr * (r_arr / r_ref) ** 0.125 * (fr / f_ref) ** 0.15


def ss_surface_density(r, r_in, r_out, r_ref):
    """SS 外区柱密度 Σ(r)（NumPy 参考）。

    Args:
        r: 半径（r_s），标量或数组。
        r_in, r_out: 盘内外半径（r_s）。
        r_ref: 归一参考半径（r_s）。

    Returns:
        Σ(r)（无量纲形状），公式 ∝ (r/r_ref)^{-3/4}·(f/f_ref)^{7/10}·外缘截断。
    """
    r_arr = np.asarray(r, dtype=np.float64)
    fr = np.maximum(1.0 - np.sqrt(r_in / np.maximum(r_arr, r_in)), 1e-6)
    f_ref = 1.0 - math.sqrt(r_in / r_ref)
    outer = 1.0 - np.clip((r_arr - 0.72 * r_out) / (r_out - 0.72 * r_out), 0, 1)
    outer = outer * outer * (3 - 2 * outer)
    return (r_arr / r_ref) ** (-0.75) * (fr / f_ref) ** 0.7 * outer
