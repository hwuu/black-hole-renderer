"""Disk V2 参数定义。

本模块统一存放 Disk V2 的参数对象，不放几何函数、物理场函数或结构调制函数。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiskV2Params:
    """Disk V2 基础盘体参数集合。

    Args:
        r_in: 盘体内半径。
        r_out: 盘体外半径，必须大于 `r_in`。
        h0: 厚度比例系数，决定 `r ≈ r_in` 时的基础厚度。
        beta_h: 厚度随半径缓慢增长的幂律指数。
        rho_power: 中面密度 `ρ_mid(r)` 的径向衰减指数。
        temp_scale: 温度剖面的整体缩放系数。
        omega_scale: 角速度场 `Ω(r)` 的整体缩放系数。
        edge_softness: 边界平滑区宽度，占总径向跨度 `(r_out - r_in)` 的比例。

    Physical Meaning:
        这些参数只描述 Disk V2 的“基础盘体”，即几何边界与基础物理场。
        它们不涉及时间平流、结构调制、辐射积分或颜色映射。

    Simplifications:
        当前实现刻意只保留少量强约束参数，避免在基础模型还未稳定时出现参数爆炸。
    """

    r_in: float = 2.0
    r_out: float = 10.0
    h0: float = 0.05
    beta_h: float = 0.05
    rho_power: float = 1.0
    temp_scale: float = 1.0
    omega_scale: float = 1.0
    edge_softness: float = 0.1

    def __post_init__(self) -> None:
        """校验参数的物理合法性与数值稳定性。

        Raises:
            ValueError: 当半径顺序、厚度比例、缩放系数或平滑参数落在非法范围时抛出。

        Notes:
            这里不做自动修正，而是直接拒绝非法输入，避免错误参数被静默带入后续
            的密度场、温度场和积分阶段。
        """

        if self.r_in <= 0.0:
            raise ValueError("r_in must be positive")
        if self.r_out <= self.r_in:
            raise ValueError("r_out must be greater than r_in")
        if self.h0 <= 0.0:
            raise ValueError("h0 must be positive")
        if self.rho_power <= 0.0:
            raise ValueError("rho_power must be positive")
        if self.temp_scale <= 0.0:
            raise ValueError("temp_scale must be positive")
        if self.omega_scale <= 0.0:
            raise ValueError("omega_scale must be positive")
        if not 0.0 <= self.edge_softness < 0.5:
            raise ValueError("edge_softness must be in [0, 0.5)")


@dataclass(frozen=True)
class DiskV2StructureParams:
    """Disk V2 结构调制参数。

    Args:
        mode1_strength: `m = 1` 低频模态强度。
        mode2_strength: `m = 2` 低频模态强度。
        shear_strength: 剪切纹理的整体强度。
        shear_components: 剪切纹理中随机傅里叶分量的数量。
        hotspot_strength: 热斑调制的整体强度。
        hotspot_count: 热斑数量。
        hotspot_phi_sigma: 热斑在方位角方向的宽度。
        hotspot_logr_sigma: 热斑在 `log(r / r_in)` 方向的宽度。
        hotspot_inner_bias: 热斑向内圈偏置的指数，值越大越偏向内圈。

    Physical Meaning:
        这些参数用于控制盘体表面的静态细节层次：弱模态调制只提供轻微不对称性，
        剪切纹理调制提供主要结构，热斑调制只做稀疏点缀。
        其中热斑调制虽然在视觉上以局部热点为主，但在定义上仍属于围绕 `1` 波动的
        乘性调制，而不是纯粹的单侧增亮 boost。

    Simplifications:
        当前实现不追求严格流体模拟，而是用可控、可测试、可复现的解析/随机场来近似。
        为保证调制因子保持正值，当前参数范围要求各强度落在安全区间内。
    """

    mode1_strength: float = 0.03
    mode2_strength: float = 0.05
    shear_strength: float = 0.22
    shear_components: int = 8
    hotspot_strength: float = 0.16
    hotspot_count: int = 8
    hotspot_phi_sigma: float = 0.18
    hotspot_logr_sigma: float = 0.12
    hotspot_inner_bias: float = 2.0

    def __post_init__(self) -> None:
        """校验结构调制参数的合法范围。

        Raises:
            ValueError: 当强度、数量或尺度参数落在非法范围时抛出。

        Notes:
            这里不仅检查参数的非负性，也检查它们是否会破坏
            `1 + strength * signed_component > 0` 这一乘性调制约束。
            当前采用保守规则：

            - `mode1_strength + mode2_strength < 1`
            - `shear_strength < 1`
            - `hotspot_strength < 1`
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
