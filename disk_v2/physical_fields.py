"""Disk V2 基础物理场层。

本模块只定义不随时间变化的基础场：

- `Ω(r)`：角速度场
- `ρ(r, z)`：密度场
- `T(r, z)`：温度场（v2.1 带物理量纲，单位为 K）

这里不处理扰动场、平流、辐射积分或颜色映射。
"""

from __future__ import annotations

import math

import numpy as np

from ._array_utils import _restore_shape, _to_array
from .geometry import disk_half_thickness, disk_radial_weight, disk_vertical_weight, disk_volume_mask
from .params import DiskV2Params
from .relativity import omega_norm


# 标准薄盘启发式 T_raw(r) = (r/r_in)^(-3/4) · [1 - sqrt(r_in/r)]^(1/4)
# 的峰值出现在 r = r_peak_over_r_in · r_in 附近。
# 解 dT/dr = 0 得 r_peak / r_in ≈ 1.36（数值上 49/36 = 1.361...）。
# 这里给出代码中使用的常数值，供文档与单测共同引用，避免散落"魔数 1.36"。
_THIN_DISK_PEAK_OVER_R_IN: float = 49.0 / 36.0
_THIN_DISK_PEAK_VALUE_RAW: float = (
    _THIN_DISK_PEAK_OVER_R_IN ** (-0.75)
    * (1.0 - 1.0 / math.sqrt(_THIN_DISK_PEAK_OVER_R_IN)) ** 0.25
)
# 归一化系数：让乘上 norm_factor 后的 raw profile 在 r=r_peak 取值恰好为 1。
# 后续 midplane_temperature_field 再乘 T_peak_K 即可获得"峰值落在 T_peak_K"的剖面。
_THIN_DISK_NORM_FACTOR: float = 1.0 / _THIN_DISK_PEAK_VALUE_RAW


def angular_velocity_field(r: float | np.ndarray, params: DiskV2Params) -> float | np.ndarray:
    """计算开普勒型角速度场 `Ω(r)`。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        params: `DiskV2Params` 参数对象。

    Returns:
        与 `r` 同形状的标量或数组，表示各半径位置的角速度 `Ω(r)`。
        返回值始终为正；该函数本身不负责在盘外置零，而只给出基础开普勒角速度标度。

    Formula:
        ```
        Ω(r) = omega_scale · (r / r_in)^(-3/2)
        ```

    Physical Meaning:
        这是标准开普勒差动旋转的径向标度：内圈角速度高，外圈角速度低。
        它是后续统一平流 `φ_adv = φ - Ω(r)t` 和剪切结构形成的基础。

    Simplifications:
        为保持数值稳定，内部对 `r < r_in` 的输入使用 `r_in` 作为安全下界。
        该函数只表达径向标度，不引入更复杂的压力支撑、相对论修正或盘外截断。
    """

    r_arr = _to_array(r)
    safe_r = np.maximum(r_arr, params.r_in)
    omega = params.omega_scale * _to_array(omega_norm(safe_r, params.r_in))
    return _restore_shape(omega, r)


def _thin_disk_temperature_raw(r: float | np.ndarray, r_in: float) -> np.ndarray:
    """计算未乘归一化系数与边界权重的 raw thin-disk 温度剖面。

    Args:
        r: 径向坐标数组（内部已转为 ndarray）。
        r_in: 盘体内半径。

    Returns:
        与 `r` 同形状的数组，表示 `T_raw(r) = (r/r_in)^(-3/4) · [1 - sqrt(r_in/r)]^(1/4)`。
        对 `r < r_in` 的输入返回 0，避免内边界外侧出现负数或 NaN。

    Formula:
        ```
        T_raw(r) = (r / r_in)^(-3/4) · [1 - sqrt(r_in / r)]^(1/4)
        ```

    Physical Meaning:
        Shakura-Sunyaev 经典薄盘的零扭矩内边界温度剖面形式。
        峰值出现在 `r = (49/36) · r_in ≈ 1.36 · r_in`。

    Notes:
        - 该函数是 internal helper，对外接口是 `midplane_temperature_field`。
        - 单测 `test_v2_temperature_range_default` 直接用 raw 剖面做跨度检查，
          以避免完整 `T_mid(r_out)` 因 `W_r(r_out)=0` 而成 0、导致比值无穷。
    """

    r_arr = _to_array(r)
    safe_r = np.maximum(r_arr, r_in)
    inner_term = np.clip(1.0 - np.sqrt(r_in / safe_r), 0.0, None)
    raw = np.power(safe_r / r_in, -0.75) * np.power(inner_term, 0.25)
    raw = np.where(r_arr < r_in, 0.0, raw)
    return raw


def midplane_density_field(r: float | np.ndarray, params: DiskV2Params) -> float | np.ndarray:
    """计算中面密度剖面 `ρ_mid(r)`（v2.1 加入内边界压制项）。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        params: `DiskV2Params` 参数对象。

    Returns:
        与 `r` 同形状的标量或数组，表示中面 `z = 0` 处的密度剖面 `ρ_mid(r)`。
        返回值非负；由于乘上 `disk_radial_weight()`，在盘外与精确边界上收敛到 `0`。
        v2.1 起在 `r = r_in` 时也通过 `[1 - sqrt(r_in/r)]^(1/2)` 项自然取 0，
        让密度和温度场的内边界形态保持一致。

    Formula:
        ```
        ρ_mid(r) = (r / r_in)^(-rho_power) · [1 - sqrt(r_in / r)]^(1/2) · W_r(r)
        ```

    Physical Meaning:
        径向幂律表达"内密外疏"的总体趋势；内边界压制项 `[1 - sqrt(r_in/r)]^(1/2)`
        与温度场 `T_mid` 的内边界形态保持一致，避免出现"有密度无温度"导致的
        诡异暗带。乘上 `W_r(r)` 保证盘外与精确边界平滑收口。

    Simplifications:
        - 没有引入吸积率、黏滞参数或局部不稳定性模型。
        - 内边界压制项的幂指数取 `1/2`，对应"密度 ∝ T^2"这一启发式选择，
          与发射率项 `j ∝ ρ · T` 的整体能流结构匹配。
    """

    r_arr = _to_array(r)
    safe_r = np.maximum(r_arr, params.r_in)
    inner_term = np.clip(1.0 - np.sqrt(params.r_in / safe_r), 0.0, None)
    density_mid = (
        np.power(safe_r / params.r_in, -params.rho_power)
        * np.power(inner_term, 0.5)
        * disk_radial_weight(r_arr, params)
    )
    density_mid = np.where(r_arr <= params.r_in, 0.0, density_mid)
    return _restore_shape(density_mid, r)


def raw_midplane_density_field(r: float | np.ndarray, params: DiskV2Params) -> float | np.ndarray:
    """计算未乘径向 support 的中面密度剖面 `ρ_raw(r)`。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        params: `DiskV2Params` 参数对象。

    Returns:
        与 `r` 同形状的非负密度剖面。`r <= r_in` 时为 0；外边界外不做
        `W_r` 收口，调用方应单独乘 support。

    Formula:
        ```
        ρ_raw(r) = (r / r_in)^(-rho_power) · [1 - sqrt(r_in / r)]^(1/2)
        ```

    Physical Meaning:
        用于 v2.2 成像主链，避免 `ρ_mid` 与 `T_mid` 中已经包含的 `W_r`
        在 `F_phys` 中被重复乘成 `W_r^5`。

    Simplifications:
        不包含外边界 support；这是调用方的职责。
    """
    r_arr = _to_array(r)
    safe_r = np.maximum(r_arr, params.r_in)
    inner_term = np.clip(1.0 - np.sqrt(params.r_in / safe_r), 0.0, None)
    density_raw = np.power(safe_r / params.r_in, -params.rho_power) * np.power(
        inner_term, 0.5,
    )
    density_raw = np.where(r_arr <= params.r_in, 0.0, density_raw)
    return _restore_shape(density_raw, r)


def midplane_temperature_field(r: float | np.ndarray, params: DiskV2Params) -> float | np.ndarray:
    """计算中面温度剖面 `T_mid(r)`（v2.1 改为带量纲，单位为 K）。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        params: `DiskV2Params` 参数对象。

    Returns:
        与 `r` 同形状的标量或数组，表示中面 `z = 0` 处的温度剖面 `T_mid(r)`，
        单位为开氏度 `K`。返回值非负；在内边界内侧和盘外通过边界权重与显式裁剪
        收敛到 `0`。`max_r T_mid(r) ≤ params.T_peak_K`，二者之间的误差由
        `W_r` 在峰值附近的衰减决定：当 `edge_softness` 让内边界软化区不覆盖
        SS 峰值位置（`r ≈ 1.36 r_in`）时，误差通常 < 5%；否则可能高达 20% 以上。
        默认参数 `edge_softness=0.02`、`r_in=3, r_out=50` 满足前者。

    Formula:
        ```
        T_mid(r) = T_peak_K · norm_factor
                 · (r / r_in)^(-3/4)
                 · [1 - sqrt(r_in / r)]^(1/4)
                 · W_r(r)
        ```

        其中 `norm_factor` 是为了让未乘 `W_r` 的 raw 剖面峰值恰好等于 `T_peak_K`
        而引入的解析常数。具体值取 `1 / T_raw(r_peak)`，其中 `r_peak ≈ 1.36 r_in`。

    Physical Meaning:
        来自 Shakura-Sunyaev 经典薄盘启发式温度剖面。零扭矩内边界条件让
        `T_mid(r_in) = 0`，峰值出现在 `r ≈ 1.36 r_in`。v2.1 通过 `T_peak_K`
        让输出携带真实 K 量纲，后续 palette 层用黑体色查表能给出物理意义的颜色。

    Simplifications:
        - 这里只保留经典径向标度，不引入更复杂的吸积率、扭矩或辐射反馈模型。
        - `W_r(r)` 在外边界附近会让 `T_mid` 比真实 SS 剖面偏低；为获得严格的
          peak-to-outer 跨度，请测未乘 `W_r` 的 raw 剖面（见
          `_thin_disk_temperature_raw`）。
    """

    r_arr = _to_array(r)
    raw = _thin_disk_temperature_raw(r_arr, params.r_in)
    temperature_mid = (
        params.T_peak_K
        * _THIN_DISK_NORM_FACTOR
        * raw
        * disk_radial_weight(r_arr, params)
    )
    temperature_mid = np.where(r_arr <= params.r_in, 0.0, temperature_mid)
    return _restore_shape(temperature_mid, r)


def raw_midplane_temperature_field(r: float | np.ndarray, params: DiskV2Params) -> float | np.ndarray:
    """计算未乘径向 support 的中面温度剖面 `T_raw_mid(r)`。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        params: `DiskV2Params` 参数对象。

    Returns:
        与 `r` 同形状的温度（K）。`r <= r_in` 时为 0；外边界外不做 `W_r`
        收口，调用方应单独乘 support。

    Formula:
        ```
        T_raw_mid(r) = T_peak_K · norm_factor
                     · (r / r_in)^(-3/4)
                     · [1 - sqrt(r_in / r)]^(1/4)
        ```

    Physical Meaning:
        用于 v2.2 成像主链，使 support 只在最终 `F_phys` 中出现一次。

    Simplifications:
        不包含径向 support；这是调用方的职责。
    """
    r_arr = _to_array(r)
    raw = _thin_disk_temperature_raw(r_arr, params.r_in)
    temperature_raw = params.T_peak_K * _THIN_DISK_NORM_FACTOR * raw
    temperature_raw = np.where(r_arr <= params.r_in, 0.0, temperature_raw)
    return _restore_shape(temperature_raw, r)


def density_field(
    r: float | np.ndarray,
    z: float | np.ndarray,
    params: DiskV2Params,
) -> float | np.ndarray:
    """计算二维密度场 `ρ(r, z)`。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        z: 局部盘坐标中的垂向高度，可以是标量或数组。
        params: `DiskV2Params` 参数对象。

    Returns:
        与输入广播后同形状的标量或数组，表示点 `(r, z)` 处的密度值 `ρ(r, z)`。
        返回值非负；盘体外部由 `disk_volume_mask()` 明确置为 `0`。

    Formula:
        ```
        ρ(r, z) = ρ_mid(r) · exp[-0.5 · (z / H(r))²] · W_z(r, z)
        ```

    Physical Meaning:
        中面 `z = 0` 处密度最高，离开中面后密度沿 z 高斯衰减。
        额外乘上的 `W_z(r, z)` 让密度场在几何表面 `|z| = H(r)` 处与几何边界
        一致地收口到 0。

    Simplifications:
        - 点不在盘体内部时直接返回 0。
        - 用 `eps` 保护 `H(r)`，避免极小厚度带来的除零问题。
        - 当前保留高斯型垂向轮廓，再用几何层的 `W_z(r, z)` 关闭 support。
          垂向的"涨落"由 Phase 2 引入的 `F_clump` 三维调制提供，密度场本身
          只提供平滑包络。
    """

    r_arr = _to_array(r)
    z_arr = _to_array(z)
    thickness = np.maximum(_to_array(disk_half_thickness(r_arr, params)), np.finfo(np.float64).eps)
    density_mid = _to_array(midplane_density_field(r_arr, params))
    vertical_weight = _to_array(disk_vertical_weight(r_arr, z_arr, params))
    field = density_mid * np.exp(-0.5 * np.square(z_arr / thickness)) * vertical_weight
    field = np.where(disk_volume_mask(r_arr, z_arr, params), field, 0.0)
    return _restore_shape(field, r if np.ndim(r) >= np.ndim(z) else z)


def temperature_field(
    r: float | np.ndarray,
    z: float | np.ndarray,
    params: DiskV2Params,
) -> float | np.ndarray:
    """计算二维温度场 `T(r, z)`，单位为 K。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        z: 局部盘坐标中的垂向高度，可以是标量或数组。
        params: `DiskV2Params` 参数对象。

    Returns:
        与输入广播后同形状的标量或数组，表示点 `(r, z)` 处的温度值，单位 `K`。
        返回值非负；盘体外部由 `disk_volume_mask()` 明确置为 `0`。

    Formula:
        ```
        T(r, z) = T_mid(r) · clip(1 - 0.25 · |z| / H(r), 0, 1) · W_z(r, z)
        ```

    Physical Meaning:
        当前版本采用"中面温度 + 弱垂向衰减"的简化形式，用于表达中面更热、表层略冷。
        `W_z(r, z)` 让温度场在几何表面与硬边界保持一致。

    Simplifications:
        这里故意不引入完整垂向辐射转移；盘外温度直接返回 0。
        垂向温度衰减系数固定为 `0.25`，与 v1.0 保持一致；如需调节再独立参数化。
    """

    r_arr = _to_array(r)
    z_arr = _to_array(z)
    thickness = np.maximum(_to_array(disk_half_thickness(r_arr, params)), np.finfo(np.float64).eps)
    temp_mid = _to_array(midplane_temperature_field(r_arr, params))
    vertical_weight = _to_array(disk_vertical_weight(r_arr, z_arr, params))
    vertical_factor = np.clip(1.0 - 0.25 * np.abs(z_arr) / thickness, 0.0, 1.0)
    field = temp_mid * vertical_factor * vertical_weight
    field = np.where(disk_volume_mask(r_arr, z_arr, params), field, 0.0)
    return _restore_shape(field, r if np.ndim(r) >= np.ndim(z) else z)
