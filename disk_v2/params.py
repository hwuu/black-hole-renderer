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
        剪切纹理调制提供丝状/絮状细节，**视觉 atlas（F_turbulence）为主结构来源**，
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
    """Disk V2 调色与色调映射参数（v2.1 新增）。

    Args:
        palette_mode: 颜色映射模式。`"physical"` 直接用黑体色查表；`"cinematic"`
            在 physical 基础上增强饱和度，用于演示输出。
        tonemap_mode: 色调映射算法。当前实现仅支持 `"reinhard"`，预留 `"aces"`
            作为后续切换选项。
        gamma: sRGB 伽马校正指数。色调映射后输出 LDR 用 `x^(1/gamma)`。
        opacity_scale: 有限厚度积分的不透明度缩放，决定光学厚度 `α = opacity_scale · ρ`。
        cinematic_saturation: cinematic 模式下的饱和度增强系数。`1.0` 等价于
            physical 模式；典型取值 `1.2 ~ 1.6`。
        cinematic_warm_shift: cinematic 模式下的暖色偏移，对 R 通道做 `* (1 + warm_shift)`、
            对 B 通道做 `* (1 - warm_shift)` 的乘性调整。典型取值 `0.0 ~ 0.15`。
        visual_temp_outer_K: cinematic 模式下，物理温度归一化后映射到的可见色温下限（K）。
        visual_temp_inner_K: cinematic 模式下，物理温度归一化后映射到的可见色温上限（K）。

    Physical Meaning:
        这一层不影响物理场定义，只把物理量映射成像素颜色。
        cinematic 模式先把物理 Kelvin 重映射到可见色温区间，再查 Helland 黑体色，
        避免 `T_peak_K ~ 1e7` 直接白化。

    Simplifications:
        - tonemap 第一版只实现 Reinhard，结构上预留可扩展 ACES Filmic。
        - cinematic 模式用 log 温度归一化 + 可见色温 lerp，不引入 LUT。
    """

    palette_mode: str = "physical"
    tonemap_mode: str = "reinhard"
    gamma: float = 2.2
    opacity_scale: float = 0.5
    cinematic_saturation: float = 1.3
    cinematic_warm_shift: float = 0.08
    visual_temp_outer_K: float = 2500.0
    visual_temp_inner_K: float = 12000.0

    def __post_init__(self) -> None:
        """校验调色参数的合法范围。

        Raises:
            ValueError: 当模式名未支持、伽马或不透明度非正、cinematic 系数越界时抛出。
        """

        if self.palette_mode not in ("physical", "cinematic"):
            raise ValueError(
                f"palette_mode must be 'physical' or 'cinematic', got {self.palette_mode!r}"
            )
        if self.tonemap_mode not in ("reinhard", "aces"):
            raise ValueError(
                f"tonemap_mode must be 'reinhard' or 'aces', got {self.tonemap_mode!r}"
            )
        if self.tonemap_mode == "aces":
            # ACES 留作后续扩展，首版不允许直接使用。
            raise NotImplementedError(
                "tonemap_mode='aces' is reserved for future implementation; "
                "use 'reinhard' for v2.1."
            )
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.opacity_scale <= 0.0:
            raise ValueError("opacity_scale must be positive")
        if self.cinematic_saturation <= 0.0:
            raise ValueError("cinematic_saturation must be positive")
        if not -1.0 < self.cinematic_warm_shift < 1.0:
            raise ValueError("cinematic_warm_shift must be in (-1, 1)")
        if self.visual_temp_outer_K <= 0.0:
            raise ValueError("visual_temp_outer_K must be positive")
        if self.visual_temp_inner_K <= self.visual_temp_outer_K:
            raise ValueError("visual_temp_inner_K must be greater than visual_temp_outer_K")
