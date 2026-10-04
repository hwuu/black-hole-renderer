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
    """Disk V2 体积密度场参数（统一气体模型；温度与刚体环参数沿用参考实现预设 M）。

    Args:
        bh_mass_msun: 黑洞质量（太阳质量）；与 mdot_edd 一起经 Page–Thorne 绝对通量推出 T_peak。
        mdot_edd: 吸积率（爱丁顿倍数，eta = 1 - sqrt(8/9)）。
        t_peak_override_K: 非 0 时直接覆盖峰值温度（K），跳过推导。
        hr_ref: r = r_ref 处 H/r（SS 理论对本场景 << 0.01；0.027 为视觉取值）。
        thickness_scale: 盘厚缩放：`hr_ref` 的有效值为定稿值乘以该系数，核心区盘内步长同步缩放；
            柱密度与光学深度不变。1 = 预设 M 的视觉厚度（与参考实现对齐）；默认 1/9 使
            H/r ≈ 0.003（接近 Shakura–Sunyaev 物理量级），消除贴盘面视角的遮挡黑墙
            （见 docs/plans/v2_edge_on_plan.md）。不作用于大气标高 `atm_height`。
        lum_temp_scale: 亮度温度倍率 s（> 0）：亮度按 `Y(s·g·T) / Y(s·T_peak)` 计算，色度仍按真实温度。
            1 = 物理（与参考实现对齐）；默认 1.25 为艺术夸张：外盘亮度提高（r = 20 处为峰值的 1.2%，
            s = 1 时 0.4%），贴盘面视角下外盘不再全黑；温度结构的亮度放大倍数（T_peak 处
            dlnY/dlnT）从 5.7 降到 4.6，纹理基本保留。
            多普勒亮度指数按 `palette.doppler_lum_compensation` 同步补偿，左右明暗不对称不变。
        r_ref: SS 剖面归一参考半径（r_s）。
        surf_noise: 表面起伏幅度：H_s = H·(1 - SURF_NOISE + SURF_NOISE·softsat(tn))。
        grey_mix: 灰大气强度：温度倍率 = 1 + GREY_MIX·(T_grey/T_eff - 1)；0 = 竖直均匀，1 = 完整灰大气。
        grey_cap: 灰大气温度倍率上限（tau ≈ 2 处；侧壁斜入的平行平面近似保护）。
        core_floor: 核心密度起伏下限 f ∈ [0, 1]：核心密度因子 `cfac = f + (1 − f)·c/⟨c⟩`（温和起伏、无空洞）。
            默认 0.15（稀处密度降到均值的 15%，盘面出现明显的稀疏区；预设 M 为 0.35）。
        dt_i: 核心温度起伏：T ← T·(1 + DT_I·(c/⟨c⟩ - 1))。
        surf_lo / surf_k: 表面增亮源函数（M 预设为 1.0 / 0.0 = 竖直均匀，灰大气取代）。
        tau_i: r ∈ [5.5, 6.5] 处 face-on 竖直光学深度（核心 + 大气总柱）的标定目标（> 0）。
            默认 1.84（预设 M 为 1.5 且核心另乘 core_opac = 2；统一模型取消该倍率后提高 tau_i 保持核心观感）。
        atm_frac: 大气柱密度比 A（≥ 0）：大气（指数尾巴）柱密度 / 核心柱密度的均值；大气与核心是同一团
            气体、共用同一湍流场 c。0 = 无大气（只有高斯核心）。默认 0.15（r = 22 处大气 τ⊥ ≈ 0.06）。
        atm_height: 大气标高 H_a / r（> 0，无量纲）：大气密度 ∝ exp(−|z|/H_a)。默认 0.01（约为核心标高的
            3 倍）；不随 `thickness_scale` 缩放。
        atm_extent: 大气截断高度（以 H_a 计，> 0）：|z| > atm_extent·H_a 处不采样（求解精度参数；5 时
            截掉的柱密度为 e^{−5} ≈ 0.7%）。
        atm_cov_c0 / atm_cov_soft: 大气覆盖软阈值的中心与宽度（以 ĉ = c/⟨c⟩ 计，宽度 > 0）：
            `cov(ĉ) = 1/(1 + exp(−(ĉ − c0)/soft))`，下方湍流稀疏处大气出现空隙。默认 1.0 / 0.3。
        atm_fine_sigma: 大气小尺度对数正态起伏强度 σ_a（≥ 0）：大气密度乘 `exp(σ_a·n − σ_a²/2)`，
            n 为零均值、单位方差的三维噪声（与主云同一套剪切级联几何，竖直方向按 H_a 计）。保均值：
            柱密度期望不变。使大气在高度方向逐渐与下方盘面去相关、出现独立的小团块。0 = 关闭（大气只
            跟随核心结构）。默认 0.5。
        atm_fine_fz: 小尺度起伏的竖直频率（> 0）：每个大气标高 H_a 内约 `0.9·atm_fine_fz` 个噪声格；
            越大，团块竖直越薄、随高度去相关越快。默认 2（为 1 时相机易落入整团高密度区，满幅橙雾）。
        abs_scatter_ratio: 中面密度处吸收与散射不透明度之比 q = κ_abs/κ_es（≥ 0）：散射反照率
            `ω = 1/(1 + q·ρ/ρ_mid)`（Kramers 吸收 ∝ ρ，电子散射与 ρ 无关）。0 = 纯散射。默认 20。
        scatter_j: 大气散射入射场系数 j（≥ 0）：`J = j·S_disk(r)`，取下方盘面（半个天空）的源函数。
            0 = 无散射光（大气只吸收与热发射）。默认 0.5。
        lowf_sigma: 大尺度低频 lognormal 调制强度 σ_L（≥ 0）：柱密度乘 `exp(σ_L·n_L − σ_L²/2)`（保均值）。
            默认 1.6（大尺度明暗与缝隙更强；预设 M 为 0.9）。
        fr_l / nphi_l / dln_l / k_rigid_l: 低频层宽带刚体环参数。
        kt_i / nphi_t / lt0_i / con_t: 厚度扰动噪声参数（径向基频 / 方位周期 / 起始八度 / 级联对比度，
            对比度 > 0）。
        az_stretch_t: 厚度扰动方位拉长系数 s（≥ 0）：刚体环带 b（中心半径 r_b）的方位基频周期为
            `n_φ(b) = max(n_t, round(n_t·r_b / (6·s)))`，方位特征尺寸不随半径线性增长。默认 1.5；
            0 = 周期固定为 `nphi_t`。
        core_oct_gain: 主云 / 厚度扰动级联的逐八度增益 γ（> 0）：第 k 个八度的扰动幅度乘 γ^k。
            γ = 0.69 ≈ 3^(-1/3) 对应 Kolmogorov 谱 E(k) ∝ k^(-5/3)（级联倍率 3）；默认 0.6（比 Kolmogorov
            略陡：小尺度条纹更弱，避免满盘锐利细纹造成的"大理石"质感）。1 = 八度等权（参考实现）。
            同时作用于主云剪切级联、厚度扰动与大气小尺度起伏。
        band_seam_fix: 刚体环带接缝修复（默认 True），两部分：
            (1) 带边界扰动随方位变化（参考实现的扰动只随 ln r 变化，边界是正圆，正视时可见同心环）；
            (2) 带间混合改为保方差：`c = m + Σw_k(c_k − m) / √Σw_k²`（m 为单图案均值，构造时标定），
            参考实现的 `Σw_k·c_k`（Σw_k = 1）在两带正中把起伏方差降到 50%（标准差 71%），形成带状明暗接缝。
            False = 参考实现。两部分均作用于主云 c 与厚度扰动 tn。
        core_contrast: 主云起伏系数 α（> 0）：保方差混合中 `c = m + α·Σw_k(c_k − m)/√Σw_k²`，
            统一缩放主云密度的明暗起伏，不改变特征形状与分布；只作用于主云 c，不作用于厚度扰动 tn。
            默认 0.8；1 = 不缩放。≠ 1 时要求 `band_seam_fix = True`（参考实现的混合没有 m 项，无从缩放）。
        shear_k0: 主云剪切级联（`src/v2/shear_cascade.py`）最大八度的 `ln r` 频率（格 / 单位 ln r，> 0）；
            第 k 个八度为 `k0·3^k`。各八度的噪声格在 `(ln r, φ)` 中按开普勒剪切取形，格子在 `ln r` 中
            等距，各半径形状相同。
            6.67 时 r ∈ [3, 30]（ln r 跨度 2.30）内最大八度约 15 格，径向尺寸约 0.15·r。
        shear_octaves: 剪切级联八度数（≥ 1）；默认 4（频率 k0 到 27·k0）。
        shear_ar_big / shear_ar_small: 最大 / 最小八度的长宽比（≥ 1，方位长 / 径向宽）。默认 10 / 2.5：
            大尺度结构存活久、被剪切拉长（AR ≈ 1 + q·Ω·τ，q = 3/2），小尺度接近各向同性。
        shear_con: 剪切级联对比度（> 0）：`softplus(con·Σ ln(1 + 0.1·γ^k·n_k))`；越大团块与缝隙越分明。
        shear_tilt_k: 拖尾倾角系数（≥ 0）：`θ_k = k·atan(1/(AR_k − 1))`。1 = 由长宽比推出的物理估计
            （大尺度约 6°、小尺度约 34°）；0 = 长轴严格沿方位方向。默认 0.01：静态倾角在视频中会造成
            "向内流动"的错觉（孔径问题），真实团块随时间继续卷绕、倾角趋于 0。
        dln_r / k_rigid: 刚体环带宽与种子寿命（主云 + 厚度扰动共用）。
        light_delay: 1 = 采样时间 = t - 光程（光行时间）。
        static_cam: 1 = 相机为静止观者本地标架。

    Physical Meaning:
        统一气体模型的全部结构 / 辐射参数：盘面（高斯核心）与大气（指数尾巴）是同一团气体，
        共用一个湍流场、一条灰大气温度规律与一个 κ（见 docs/plans/v2_unified_gas_plan.md）。

    Simplifications:
        无尘埃；大气无独立速度场（与核心同刚体环平流）；散射只取下方盘面的单次散射。
    """

    bh_mass_msun: float = 1.0e8
    mdot_edd: float = 1.7e-6
    t_peak_override_K: float = 0.0
    hr_ref: float = 0.027
    r_ref: float = 10.0
    thickness_scale: float = 1.0 / 9.0
    lum_temp_scale: float = 1.25
    surf_noise: float = 0.6
    grey_mix: float = 0.5
    grey_cap: float = 1.19
    core_floor: float = 0.15
    dt_i: float = 0.05
    surf_lo: float = 1.0
    surf_k: float = 0.0
    tau_i: float = 1.84
    atm_frac: float = 0.15
    atm_height: float = 0.01
    atm_extent: float = 5.0
    atm_cov_c0: float = 1.0
    atm_cov_soft: float = 0.3
    atm_fine_sigma: float = 0.5
    atm_fine_fz: float = 2.0
    abs_scatter_ratio: float = 20.0
    scatter_j: float = 0.5
    lowf_sigma: float = 1.6
    fr_l: float = 3.0
    nphi_l: int = 3
    dln_l: float = 0.5306  # ln(1.7)
    k_rigid_l: float = 6.0
    kt_i: float = 2.0
    nphi_t: int = 9
    lt0_i: float = 0.7
    con_t: float = 50.0
    az_stretch_t: float = 1.5
    core_oct_gain: float = 0.6
    band_seam_fix: bool = True
    core_contrast: float = 0.8
    shear_k0: float = 6.67
    shear_octaves: int = 4
    shear_ar_big: float = 10.0
    shear_ar_small: float = 2.5
    shear_con: float = 50.0
    shear_tilt_k: float = 0.01
    dln_r: float = 0.1989  # ln(1.22)
    k_rigid: float = 4.0
    light_delay: bool = True
    static_cam: bool = True

    def __post_init__(self) -> None:
        if self.grey_mix < 0.0 or self.grey_mix > 1.0:
            raise ValueError("grey_mix must be in [0, 1]")
        if self.grey_cap < 1.0:
            raise ValueError("grey_cap must be >= 1")
        if not 0.0 <= self.core_floor <= 1.0:
            raise ValueError("core_floor must be in [0, 1]")
        if self.thickness_scale <= 0.0:
            raise ValueError("thickness_scale must be positive")
        if self.lum_temp_scale <= 0.0:
            raise ValueError("lum_temp_scale must be positive")
        if self.az_stretch_t < 0.0:
            raise ValueError("az_stretch_t must be >= 0 (0 = fixed azimuthal period)")
        if self.con_t <= 0.0:
            raise ValueError("con_t must be positive")
        if self.core_oct_gain <= 0.0:
            raise ValueError("core_oct_gain must be positive")
        if self.core_contrast <= 0.0:
            raise ValueError("core_contrast must be positive")
        if self.core_contrast != 1.0 and not self.band_seam_fix:
            raise ValueError("core_contrast != 1 requires band_seam_fix = True")
        if self.tau_i <= 0.0:
            raise ValueError("tau_i must be positive")
        if self.lowf_sigma < 0.0:
            raise ValueError("lowf_sigma must be >= 0")
        if self.atm_frac < 0.0:
            raise ValueError("atm_frac must be >= 0")
        if self.atm_height <= 0.0 or self.atm_extent <= 0.0:
            raise ValueError("atm_height and atm_extent must be positive")
        if self.atm_cov_soft <= 0.0:
            raise ValueError("atm_cov_soft must be positive")
        if self.atm_fine_sigma < 0.0 or self.atm_fine_fz <= 0.0:
            raise ValueError("atm_fine_sigma must be >= 0 and atm_fine_fz must be positive")
        if self.abs_scatter_ratio < 0.0 or self.scatter_j < 0.0:
            raise ValueError("abs_scatter_ratio and scatter_j must be >= 0")
        if self.dln_r <= 0.0 or self.dln_l <= 0.0:
            raise ValueError("dln_r and dln_l must be positive")
        if self.k_rigid <= 0.0 or self.k_rigid_l <= 0.0:
            raise ValueError("k_rigid must be positive")
        if self.nphi_t < 1 or self.nphi_l < 1:
            raise ValueError("nphi_* must be >= 1 (integer period)")
        if self.shear_k0 <= 0.0:
            raise ValueError("shear_k0 must be positive")
        if self.shear_octaves < 1:
            raise ValueError("shear_octaves must be >= 1")
        if self.shear_ar_big < 1.0 or self.shear_ar_small < 1.0:
            raise ValueError("shear_ar_big and shear_ar_small must be >= 1")
        if self.shear_con <= 0.0:
            raise ValueError("shear_con must be positive")
        if self.shear_tilt_k < 0.0:
            raise ValueError("shear_tilt_k must be >= 0")
