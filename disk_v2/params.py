"""Disk V2 参数定义。

本模块统一存放 Disk V2 的参数对象，不放几何函数、物理场函数或结构调制函数。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace


# Schwarzschild 黑洞的最内稳定圆轨道（ISCO）半径，单位为 Schwarzschild 半径 r_s。
# 经典薄盘的内边界假设盘截断于 ISCO；若用户给的 r_in 比 ISCO 还小，
# 则盘公式落入 plunging region，已经无物理意义，因此强制钳制并发出 warning。
SCHWARZSCHILD_ISCO_R_S: float = 3.0


@dataclass(frozen=True)
class DiskV2Params:
    """Disk V2 基础盘体参数集合。

    Args:
        r_in: 盘体内半径，单位为 Schwarzschild 半径 `r_s`。物理上必须不小于 ISCO，
            即 `3 · r_s`；当传入更小的值时，`__post_init__` 会发出 warning
            并把值钳制为 `SCHWARZSCHILD_ISCO_R_S`。
        r_out: 盘体外半径，单位同 `r_in`，必须大于钳制后的 `r_in`。v2.1 把默认值
            从 10 提到 50，是为了让标准薄盘温度剖面在盘内有效区跨度达到 ~4.3 倍，
            而不是 v1.0 默认下的 ~1.9 倍。
        h0: 厚度比例系数，决定 `r ≈ r_in` 时的基础厚度。
        beta_h: 厚度随半径缓慢增长的幂律指数。
        rho_power: 中面密度 `ρ_mid(r)` 的径向衰减指数。v2.1 默认从 1.0 提到 1.5，
            进一步拉开内外密度对比。
        T_peak_K: 中面温度的物理峰值，单位为开氏度 `K`。v2.1 用它替换 v1.0 的
            无量纲 `temp_scale`。`physical_fields.py` 内部对剖面做归一化，使得
            `max_r T_mid(r) ≈ T_peak_K`。默认 `1.0e7` 对应 stellar-mass BH 的
            典型 X 射线吸积盘内缘温度。
        omega_scale: 角速度场 `Ω(r)` 的整体缩放系数。
        edge_softness: 边界平滑区宽度，占总径向跨度 `(r_out - r_in)` 的比例。
        alpha_density: 发射率公式 `j ∝ ρ^α · T^β` 中的密度指数 `α`。v2.1 新增。
        beta_temperature: 发射率公式 `j ∝ ρ^α · T^β` 中的温度指数 `β`。v2.1 新增。

    Physical Meaning:
        这些参数只描述 Disk V2 的"基础盘体"，即几何边界与基础物理场。
        它们不涉及时间平流、结构调制、辐射积分或颜色映射。

    Simplifications:
        当前实现刻意只保留少量强约束参数，避免在基础模型还未稳定时出现参数爆炸。
        `r_in` 的 ISCO 钳制只是工程约束，不是数学约束：经典薄盘在 `r < r_isco`
        本就没有意义，因此选择 warn + 钳制而不是 raise。

    Examples:
        >>> p = DiskV2Params()  # 默认参数
        >>> p.r_in, p.r_out, p.T_peak_K
        (3.0, 50.0, 10000000.0)
    """

    r_in: float = SCHWARZSCHILD_ISCO_R_S
    r_out: float = 50.0
    h0: float = 0.05
    beta_h: float = 0.05
    rho_power: float = 1.5
    T_peak_K: float = 1.0e7
    omega_scale: float = 1.0
    # v2.1：当 r_out 默认提到 50 后，0.1 的比例对应内边界软化区跨度 4.7 r_s，
    # 会把 SS 温度峰值（r ≈ 1.36 · r_in = 4.08）削掉约 20%。
    # 改为 0.02：软化区跨度 ≈ 0.94 r_s，让峰值落在 W_r ≈ 1 的区域。
    edge_softness: float = 0.02
    alpha_density: float = 1.0
    beta_temperature: float = 1.0

    def __post_init__(self) -> None:
        """校验参数的物理合法性与数值稳定性，并对 `r_in` 做 ISCO 钳制。

        Raises:
            ValueError: 当半径顺序、厚度比例、缩放系数或平滑参数落在非法范围时抛出。

        Notes:
            - `r_in < SCHWARZSCHILD_ISCO_R_S` 时仅发出 warning 并把值钳制到
              `SCHWARZSCHILD_ISCO_R_S`，不 raise。这与其他参数的"严格拒绝"策略
              不同，原因是 ISCO 钳制属于物理约定，钳制后的盘仍然可用。
            - `r_out` 的检查在 `r_in` 钳制之后进行，因此即便用户传入
              `(r_in=2.0, r_out=2.5)` 也会因 `r_out <= 钳制后 r_in = 3.0` 报错。
        """

        if self.r_in < SCHWARZSCHILD_ISCO_R_S:
            warnings.warn(
                f"r_in={self.r_in} 小于 Schwarzschild ISCO ({SCHWARZSCHILD_ISCO_R_S} r_s)，"
                f"已自动钳制为 {SCHWARZSCHILD_ISCO_R_S}。",
                stacklevel=2,
            )
            # frozen dataclass 不能直接赋值，使用 object.__setattr__ 绕过。
            # 这是 dataclasses 官方推荐的 __post_init__ 内修改 frozen 字段的方式。
            object.__setattr__(self, "r_in", SCHWARZSCHILD_ISCO_R_S)

        if self.r_in <= 0.0:
            raise ValueError("r_in must be positive")
        if self.r_out <= self.r_in:
            raise ValueError("r_out must be greater than r_in")
        if self.h0 <= 0.0:
            raise ValueError("h0 must be positive")
        if self.rho_power <= 0.0:
            raise ValueError("rho_power must be positive")
        if self.T_peak_K <= 0.0:
            raise ValueError("T_peak_K must be positive")
        if self.omega_scale <= 0.0:
            raise ValueError("omega_scale must be positive")
        if not 0.0 <= self.edge_softness < 0.5:
            raise ValueError("edge_softness must be in [0, 0.5)")
        if self.alpha_density < 0.0:
            raise ValueError("alpha_density must be non-negative")
        if self.beta_temperature < 0.0:
            raise ValueError("beta_temperature must be non-negative")


@dataclass(frozen=True)
class DiskV2StructureParams:
    """Disk V2 结构调制参数。

    Args:
        mode1_strength: `m = 1` 低频模态强度。
        mode2_strength: `m = 2` 低频模态强度。
        shear_strength: 剪切纹理的整体强度。视觉恢复后默认 `0.0`（关闭），
            避免与 `F_turbulence` atlas 叠加产生斑马纹；保留为高级实验参数。
        shear_components: 剪切纹理中随机傅里叶分量的数量。
        clump_strength: 团块调制强度。仅用于弱体积自遮挡，默认 `0.12`。
        clump_count: 显式团块中心的数量。
        clump_radial_sigma_scale: 团块径向尺度相对 `r_in` 的系数（v2.1 新增）。
            实际径向尺度为 `clump_radial_sigma_scale · r_in`。
        clump_vertical_sigma_scale: 团块垂向尺度相对 `H(r)` 的系数（v2.1 新增）。
            实际垂向尺度为 `clump_vertical_sigma_scale · H(r)`，让团块尺度随盘
            自然伸缩。
        clump_phi_sigma: 团块角向高斯宽度，单位为弧度（v2.1 新增）。
        clump_emission_weight: 团块调制在独立发射路径中的权重 `[0, 1]`。
            主光追发射已改用 `ρ_envelope · F_shear` 丝状纹理；团块仅经密度路径
            提供弱体积自遮挡。默认 `0.0` 避免盘面出现大块亮斑。
        hotspot_strength: 热斑调制的整体强度。v2.1 略增到 0.20。
        hotspot_count: 热斑数量。
        hotspot_phi_sigma: 热斑在方位角方向的宽度。
        hotspot_logr_sigma: 热斑在 `log(r / r_in)` 方向的宽度。
        hotspot_inner_bias: 热斑向内圈偏置的指数，值越大越偏向内圈。

    Physical Meaning:
        这些参数控制盘体表面与体内的细节层次：弱模态调制只提供轻微不对称性，
        剪切纹理和 visual atlas 只提供有界扰动，不能接管主发射、主色温或 alpha；
        团块调制（clump）仅提供弱体积自遮挡。
        所有调制都是围绕 `1` 波动的乘性因子，盘外返回中性值 `1`。

    Simplifications:
        当前实现不追求严格流体模拟，而是用可控、可测试、可复现的解析/随机场来近似。
        团块项采用显式点云团（每个团块一个核心位置 + 锐利衰减核），Worley/Voronoi
        噪声留作首版不达标时的回退方案。
    """

    mode1_strength: float = 0.03
    mode2_strength: float = 0.05
    shear_strength: float = 0.0
    shear_components: int = 16
    clump_strength: float = 0.12
    clump_count: int = 280
    clump_radial_sigma_scale: float = 0.09
    clump_vertical_sigma_scale: float = 0.35
    clump_phi_sigma: float = 0.10
    clump_emission_weight: float = 0.0
    hotspot_strength: float = 0.20
    hotspot_count: int = 8
    hotspot_phi_sigma: float = 0.18
    hotspot_logr_sigma: float = 0.12
    hotspot_inner_bias: float = 2.0
    # --- 视觉 atlas（V1 云雾 + Blender 径向扭曲） ---
    # use_visual_atlas=True 时主光追使用倾斜中面单次命中的 thin-layer 快速路径；
    # False 时走有限厚度体积积分。v2.2 起 atlas 只能作为 bounded turbulence 输入。
    use_visual_atlas: bool = True
    atlas_n_r: int = 512
    atlas_n_phi: int = 1024
    turbulence_strength: float = 0.35
    spiral_warp_strength: float = 1.8
    alpha_clip_threshold: float = 0.01
    density_atlas_scale: float = 0.55
    atlas_generation_scale: int = 2

    def __post_init__(self) -> None:
        """校验结构调制参数的合法范围。

        Raises:
            ValueError: 当强度、数量或尺度参数落在非法范围时抛出。

        Notes:
            为保持乘性调制因子 `1 + strength · signed_value` 严格为正，
            约束各调制项的强度上界。当前规则：

            - `mode1_strength + mode2_strength < 1`
            - `shear_strength < 1`
            - `clump_strength < 1`
            - `hotspot_strength < 1`

            `clump_strength` 可以接近 1，最终乘性调制仍然 `> 0`，
            但 `F_clump` 的实现需要保证 signed 值落在 `[-1, +1]`。
        """

        if self.mode1_strength < 0.0:
            raise ValueError("mode1_strength must be non-negative")
        if self.mode2_strength < 0.0:
            raise ValueError("mode2_strength must be non-negative")
        if self.mode1_strength + self.mode2_strength >= 1.0:
            raise ValueError("mode1_strength + mode2_strength must be less than 1")
        if self.shear_strength < 0.0:
            raise ValueError("shear_strength must be non-negative")
        if self.shear_strength >= 1.0:
            raise ValueError("shear_strength must be less than 1")
        if self.shear_components <= 0:
            raise ValueError("shear_components must be positive")
        if self.clump_strength < 0.0:
            raise ValueError("clump_strength must be non-negative")
        if self.clump_strength >= 1.0:
            raise ValueError("clump_strength must be less than 1")
        if self.clump_count <= 0:
            raise ValueError("clump_count must be positive")
        if self.clump_radial_sigma_scale <= 0.0:
            raise ValueError("clump_radial_sigma_scale must be positive")
        if self.clump_vertical_sigma_scale <= 0.0:
            raise ValueError("clump_vertical_sigma_scale must be positive")
        if self.clump_phi_sigma <= 0.0:
            raise ValueError("clump_phi_sigma must be positive")
        if not 0.0 <= self.clump_emission_weight <= 1.0:
            raise ValueError("clump_emission_weight must be in [0, 1]")
        if self.hotspot_strength < 0.0:
            raise ValueError("hotspot_strength must be non-negative")
        if self.hotspot_strength >= 1.0:
            raise ValueError("hotspot_strength must be less than 1")
        if self.hotspot_count <= 0:
            raise ValueError("hotspot_count must be positive")
        if self.hotspot_phi_sigma <= 0.0:
            raise ValueError("hotspot_phi_sigma must be positive")
        if self.hotspot_logr_sigma <= 0.0:
            raise ValueError("hotspot_logr_sigma must be positive")
        if self.hotspot_inner_bias <= 0.0:
            raise ValueError("hotspot_inner_bias must be positive")
        if self.atlas_n_r <= 1:
            raise ValueError("atlas_n_r must be > 1")
        if self.atlas_n_phi <= 1:
            raise ValueError("atlas_n_phi must be > 1")
        if self.turbulence_strength < 0.0:
            raise ValueError("turbulence_strength must be non-negative")
        if self.spiral_warp_strength < 0.0:
            raise ValueError("spiral_warp_strength must be non-negative")
        if not 0.0 <= self.alpha_clip_threshold < 1.0:
            raise ValueError("alpha_clip_threshold must be in [0, 1)")
        if not 0.0 < self.density_atlas_scale <= 1.0:
            raise ValueError("density_atlas_scale must be in (0, 1]")
        if self.atlas_generation_scale not in (1, 2, 4):
            raise ValueError("atlas_generation_scale must be 1, 2, or 4")


@dataclass(frozen=True)
class DiskV2PaletteParams:
    """Disk V2 颜色与显示链参数（v2.3 S2 精简）。

    Args:
        tonemap_mode: 色调映射算法。当前实现仅支持 `"reinhard"`；`"aces"` 预留
            （见 `__post_init__` 说明）。
        gamma: sRGB 伽马校正指数。色调映射后输出 LDR 用 `x^(1/gamma)`。
        white_balance_K: 相机白平衡色温（K）。von Kries 增益使该温度黑体呈精确
            中性 (1,1,1)；不保证其他温度黑体的 BT.709 亮度（von Kries 固有属性）。
            6600 K ≈ 中性白点（增益 ≈ 1）。
        opacity_scale: 有效 opacity 缩放。v2.2 reference 中用于
            `tau_effective(r) = opacity_scale · rho_mid(r) · H(r)`；当该值全盘
            显著小于 1 时，应解释为 optically-thin effective opacity，而不是真实
            photosphere。

    Physical Meaning:
        这一层不改变基础物理场，只定义显示链。v2.3 起颜色只有一条链：
        CIE 黑体色度 + 可见光亮度 Y(g·T) + 白平衡 + tonemap（见方案
        `docs/plans/v2_volumetric_video_plan.md` §2）。旧的 cinematic
        （log-T 可见色温映射、饱和度 / 暖色 / 低温压暗）已删除：其饱和度增强
        与 `visual_temp` 重映射属于非物理人工调整，且与 g-factor 频移链耦合
        后难以解释。

    Simplifications:
        - tonemap 只支持 Reinhard，结构上预留可扩展 ACES Filmic。
        - 白平衡只作用于最终 HDR RGB（S6 接线）。
    """

    tonemap_mode: str = "reinhard"
    gamma: float = 2.2
    white_balance_K: float = 6600.0
    opacity_scale: float = 0.5

    def __post_init__(self) -> None:
        """校验显示链参数的合法范围。

        Raises:
            ValueError: 当模式名未支持（含已删除的 cinematic）、伽马 / 不透明度 /
                白平衡温度非正时抛出。
        """

        if self.tonemap_mode not in ("reinhard", "aces"):
            raise ValueError(
                f"tonemap_mode must be 'reinhard' or 'aces', got {self.tonemap_mode!r}"
            )
        if self.tonemap_mode == "aces":
            # X1 已撤回（2026-06-14）：ACES Filmic 的低值响应曲线 (x→0 时斜率≈0.21)
            # 让"99% 黑底 + 1% 高亮"的黑洞场景被严重抬亮，背景灰雾。
            # `tonemap_aces` 函数本体保留在 palette.py / taichi_impl.py，未来若
            # 解决了 background pedestal 问题再启用。
            raise NotImplementedError(
                "tonemap_mode='aces' is reserved; ACES + ref_wp 让黑色背景被抬亮。"
                "use 'reinhard' (default)."
            )
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.opacity_scale <= 0.0:
            raise ValueError("opacity_scale must be positive")
        if self.white_balance_K <= 0.0:
            raise ValueError("white_balance_K must be positive")


@dataclass(frozen=True)
class DiskV2VolumeParams:
    """Disk V2 体积密度场参数（v2.3 S5，对应参考实现预设 M）。

    Args:
        bh_mass_msun: 黑洞质量（太阳质量）；与 mdot_edd 一起经 Page–Thorne 绝对通量推出 T_peak。
        mdot_edd: 吸积率（爱丁顿倍数，eta = 1 - sqrt(8/9)）。
        t_peak_override_K: 非 0 时直接覆盖峰值温度（K），跳过推导。
        hr_ref: r = r_ref 处 H/r（SS 理论对本场景 << 0.01；0.027 为视觉取值）。
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
