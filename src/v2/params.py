"""Disk V2 参数定义。

- `DiskV2Params`：第 1 层场景几何（盘内外半径）。
- `DiskV2VolumeParams`：体积密度场的第 1/2/3 层参数（对应参考实现预设 M）。

本模块只放参数对象，不放物理场或渲染函数。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass


# Schwarzschild 黑洞的最内稳定圆轨道（ISCO）半径，单位为 Schwarzschild 半径 r_s。
# 薄盘假设盘截断于 ISCO；若用户给的 r_in 比 ISCO 还小，则落入 plunging region，
# 盘公式已无物理意义，因此强制钳制并发出 warning。
SCHWARZSCHILD_ISCO_R_S: float = 3.0


@dataclass(frozen=True)
class DiskV2Params:
    """Disk V2 盘几何参数。

    Args:
        r_in: 盘内半径，单位 Schwarzschild 半径 `r_s`。物理上必须不小于 ISCO（`3 r_s`）；
            传入更小的值时 `__post_init__` 发出 warning 并钳制为 `SCHWARZSCHILD_ISCO_R_S`。
        r_out: 盘外半径（r_s），必须大于钳制后的 `r_in`。默认 30 与参考实现预设 M 一致。

    Args:
        disk_spin: 盘旋转方向，`+1` 为从盘法向 (+z) 看逆时针，`-1` 为顺时针；
            平流结构与多普勒频移使用同一符号，保证二者一致。

    Physical Meaning:
        只描述盘的径向范围与旋转方向；Page–Thorne 温度表、SS 结构与噪声带网格都在
        `[r_in, r_out]` 上构造。

    Simplifications:
        `r_in` 的 ISCO 钳制是工程约束（warn + 钳制而非 raise）：钳制后的盘仍可用。
    """

    r_in: float = SCHWARZSCHILD_ISCO_R_S
    r_out: float = 30.0
    disk_spin: float = 1.0

    def __post_init__(self) -> None:
        """校验半径并对 `r_in` 做 ISCO 钳制。

        Raises:
            ValueError: `r_out <= r_in`（钳制之后比较）。
        """
        if self.r_in < SCHWARZSCHILD_ISCO_R_S:
            warnings.warn(
                f"r_in={self.r_in} 小于 Schwarzschild ISCO ({SCHWARZSCHILD_ISCO_R_S} r_s)，"
                f"已自动钳制为 {SCHWARZSCHILD_ISCO_R_S}。",
                stacklevel=2,
            )
            # frozen dataclass 不能直接赋值；__post_init__ 内用 object.__setattr__ 修改。
            object.__setattr__(self, "r_in", SCHWARZSCHILD_ISCO_R_S)
        if self.r_out <= self.r_in:
            raise ValueError("r_out must be greater than r_in")
        if self.disk_spin not in (1.0, -1.0):
            raise ValueError("disk_spin must be +1.0 or -1.0")


@dataclass(frozen=True)
class DiskV2VolumeParams:
    """Disk V2 体积密度场参数（v2.3 S5，对应参考实现预设 M）。

    Args:
        bh_mass_msun: 黑洞质量（太阳质量）；与 mdot_edd 一起经 Page–Thorne 绝对通量推出 T_peak。
        mdot_edd: 吸积率（爱丁顿倍数，eta = 1 - sqrt(8/9)）。
        t_peak_override_K: 非 0 时直接覆盖峰值温度（K），跳过推导。
        hr_ref: r = r_ref 处 H/r（SS 理论对本场景 << 0.01；0.027 为视觉取值）。
        thickness_scale: 盘厚缩放：`hr_ref`、`cl_spacing`、`cl_width` 的有效值为定稿值乘以该系数，盘内步长
            同步缩放；柱密度与光学深度不变。1 = 预设 M 的视觉厚度（与参考实现对齐）；默认 1/9 使
            H/r ≈ 0.003（接近 Shakura–Sunyaev 物理量级），消除贴盘面视角的遮挡黑墙
            （见 docs/plans/v2_edge_on_plan.md）。
        r_ref: SS 剖面归一参考半径（r_s）。
        surf_noise: 表面起伏幅度：H_s = H·(1 - SURF_NOISE + SURF_NOISE·softsat(tn))。
        grey_mix: 灰大气强度：温度倍率 = 1 + GREY_MIX·(T_grey/T_eff - 1)；0 = 竖直均匀，1 = 完整灰大气。
        grey_cap: 灰大气温度倍率上限（tau ≈ 2 处；侧壁斜入的平行平面近似保护）。
        core_opac: 核心吸收倍率（相对 TAU_I 标定）；> 1 → 不透明核心。
        core_floor: > 0：核心密度 = FLOOR + (1 - FLOOR)·c/⟨c⟩（温和起伏、无空洞）。
        dt_i: 核心温度起伏：T ← T·(1 + DT_I·(c/⟨c⟩ - 1))。
        surf_lo / surf_k: 表面增亮源函数（M 预设为 1.0 / 0.0 = 竖直均匀，灰大气取代）。
        tau_i: r ≈ 6 处 face-on 竖直光学深度标定目标。
        smoke_i: 烟雾（盘风团块）总柱密度 / 核心柱密度。
        smoke_tr: 烟雾温度比：T_smoke = SMOKE_TR·T(r)；0 = 用亮度比例 smoke_s。
        smoke_s: SMOKE_TR = 0 时的烟雾亮度比例（旧 J/H 预设兼容）。
        n_cl_half / cl_spacing / cl_width / cl_decay: 烟雾层结构（k = -N..N，间距 / r，厚度 / r，衰减）。
        fr_c / nphi_c / fz_c / sigma_c / cloud_c0 / cloud_soft: 烟雾噪声参数。
        lowf_sigma: 大尺度低频 lognormal 调制强度。
        fr_l / nphi_l / dln_l / k_rigid_l: 低频层宽带刚体环参数。
        kr_i / nphi_i / l0_i / con_i: 主云乘性级联噪声参数（径向基频 / 方位周期 / 起始八度 / 对比度）。
        kt_i / nphi_t / lt0_i: 厚度扰动噪声参数。
        dust_em / dust_s / dust_kepler / dust_on: 内区尘埃参数。
        dln_r / k_rigid: 刚体环带宽与种子寿命（核心 + 烟雾 + 尘埃共用）。
        light_delay: 1 = 采样时间 = t - 光程（光行时间）。
        static_cam: 1 = 相机为静止观者本地标架。

    Physical Meaning:
        预设 M 的全部第 2/3 层参数；所有字段有物理默认值或视觉默认值（标注见方案 §2）。
    """

    bh_mass_msun: float = 1.0e8
    mdot_edd: float = 1.7e-6
    t_peak_override_K: float = 0.0
    hr_ref: float = 0.027
    r_ref: float = 10.0
    thickness_scale: float = 1.0 / 9.0
    surf_noise: float = 0.6
    grey_mix: float = 0.5
    grey_cap: float = 1.19
    core_opac: float = 2.0
    core_floor: float = 0.35
    dt_i: float = 0.05
    surf_lo: float = 1.0
    surf_k: float = 0.0
    tau_i: float = 1.5
    smoke_i: float = 0.8
    smoke_tr: float = 0.85
    smoke_s: float = 0.33
    n_cl_half: int = 3
    cl_spacing: float = 0.014
    cl_width: float = 0.011
    cl_decay: float = 0.35
    fr_c: float = 8.0
    nphi_c: int = 10
    fz_c: float = 0.9
    sigma_c: float = 1.2
    cloud_c0: float = 0.4
    cloud_soft: float = 0.35
    lowf_sigma: float = 0.9
    fr_l: float = 3.0
    nphi_l: int = 3
    dln_l: float = 0.5306  # ln(1.7)
    k_rigid_l: float = 6.0
    kr_i: float = 0.2
    nphi_i: int = 2
    l0_i: float = 3.0
    con_i: float = 50.0
    kt_i: float = 2.0
    nphi_t: int = 9
    lt0_i: float = 0.7
    dust_em: float = 0.02
    dust_s: float = 1.0
    dust_kepler: bool = True
    dust_on: bool = True
    dln_r: float = 0.1989  # ln(1.22)
    k_rigid: float = 4.0
    light_delay: bool = True
    static_cam: bool = True

    def __post_init__(self) -> None:
        if self.grey_mix < 0.0 or self.grey_mix > 1.0:
            raise ValueError("grey_mix must be in [0, 1]")
        if self.grey_cap < 1.0:
            raise ValueError("grey_cap must be >= 1")
        if self.core_opac <= 0.0:
            raise ValueError("core_opac must be positive")
        if not 0.0 <= self.core_floor <= 1.0:
            raise ValueError("core_floor must be in [0, 1]")
        if self.thickness_scale <= 0.0:
            raise ValueError("thickness_scale must be positive")
        if self.tau_i <= 0.0:
            raise ValueError("tau_i must be positive")
        if self.smoke_i < 0.0:
            raise ValueError("smoke_i must be >= 0")
        if not 0.0 < self.smoke_tr <= 1.0 and self.smoke_tr != 0.0:
            raise ValueError("smoke_tr must be in (0, 1] or 0 (use smoke_s)")
        if self.dln_r <= 0.0 or self.dln_l <= 0.0:
            raise ValueError("dln_r and dln_l must be positive")
        if self.k_rigid <= 0.0 or self.k_rigid_l <= 0.0:
            raise ValueError("k_rigid must be positive")
        if self.n_cl_half < 0:
            raise ValueError("n_cl_half must be >= 0")
        if self.nphi_i < 1 or self.nphi_c < 1 or self.nphi_t < 1 or self.nphi_l < 1:
            raise ValueError("nphi_* must be >= 1 (integer period)")
