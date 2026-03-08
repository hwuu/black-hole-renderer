"""Disk V2 基础物理场层。

本模块只定义不随时间变化的基础场：

- `Ω(r)`：角速度场
- `ρ(r, z)`：密度场
- `T(r, z)`：温度场

这里不处理扰动场、平流、辐射积分或颜色映射。
"""

from __future__ import annotations

import numpy as np

from ._array_utils import _restore_shape, _to_array
from .geometry import disk_half_thickness, disk_radial_weight, disk_vertical_weight, disk_volume_mask
from .params import DiskV2Params


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
    omega = params.omega_scale * np.power(safe_r / params.r_in, -1.5)
    return _restore_shape(omega, r)


def midplane_density_field(r: float | np.ndarray, params: DiskV2Params) -> float | np.ndarray:
    """计算中面密度剖面 `ρ_mid(r)`。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        params: `DiskV2Params` 参数对象。

    Returns:
        与 `r` 同形状的标量或数组，表示中面 `z = 0` 处的密度剖面 `ρ_mid(r)`。
        返回值非负；由于乘上 `disk_radial_weight()`，在盘外与边界外侧会平滑收敛到 `0`。

    Formula:
        ```
        ρ_mid(r) = (r / r_in)^(-rho_power) · W_r(r)
        ```

    Physical Meaning:
        在不引入更复杂吸积率模型的前提下，用幂律衰减表达“内密外疏”，再乘上
        径向平滑权重 `W_r(r)` 让密度在盘边缘平滑收敛到 0。

    Simplifications:
        这里只保留单一幂律径向剖面，不引入更复杂的吸积率、黏滞参数或局部不稳定性模型。
    """

    r_arr = _to_array(r)
    safe_r = np.maximum(r_arr, params.r_in)
    density_mid = np.power(safe_r / params.r_in, -params.rho_power) * disk_radial_weight(r_arr, params)
    return _restore_shape(density_mid, r)


def midplane_temperature_field(r: float | np.ndarray, params: DiskV2Params) -> float | np.ndarray:
    """计算中面温度剖面 `T_mid(r)`。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        params: `DiskV2Params` 参数对象。

    Returns:
        与 `r` 同形状的标量或数组，表示中面 `z = 0` 处的温度剖面 `T_mid(r)`。
        返回值非负；在内边界内侧和盘外会通过边界权重与显式裁剪收敛到 `0`。

    Formula:
        ```
        T_mid(r) = temp_scale · (r / r_in)^(-3/4) · [1 - sqrt(r_in / r)]^(1/4) · W_r(r)
        ```

    Physical Meaning:
        该形式来自标准薄盘温度剖面的启发式近似。温度峰值应出现在 `r_in` 外侧，
        而不是恰好卡在内边界上，这比“最内圈直接最亮”的纯美术设定更接近经典盘模型。

    Simplifications:
        这里只保留经典径向标度和边界平滑权重，不引入更复杂的吸积率、扭矩或辐射反馈模型。
    """

    r_arr = _to_array(r)
    safe_r = np.maximum(r_arr, params.r_in)
    inner_term = np.clip(1.0 - np.sqrt(params.r_in / safe_r), 0.0, None)
    temperature_mid = (
        params.temp_scale
        * np.power(safe_r / params.r_in, -0.75)
        * np.power(inner_term, 0.25)
        * disk_radial_weight(r_arr, params)
    )
    temperature_mid = np.where(r_arr <= params.r_in, 0.0, temperature_mid)
    return _restore_shape(temperature_mid, r)


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
        中面 `z = 0` 处密度最高，离开中面后密度逐渐衰减。这是一种简洁、稳定、易测试的
        厚度表达。额外乘上的 `W_z(r, z)` 用于让密度场在几何表面 `|z| = H(r)` 处与
        几何边界一致地收口到 0，而不是在边界上仍保留明显非零值后再被硬截断。

    Simplifications:
        - 点不在盘体内部时直接返回 0；
        - 用 `eps` 保护 `H(r)`，避免极小厚度带来的除零问题；
        - 当前保留高斯型垂向主体轮廓，再用几何层的 `W_z(r, z)` 关闭 support。
    """

    r_arr = _to_array(r)
    z_arr = _to_array(z)
    thickness = np.maximum(_to_array(disk_half_thickness(r_arr, params)), np.finfo(np.float64).eps)
    density_mid = _to_array(midplane_density_field(r_arr, params))
    vertical_weight = _to_array(disk_vertical_weight(r_arr, z_arr, params))
    # 垂向采用关于中面对称的高斯分布。
    field = density_mid * np.exp(-0.5 * np.square(z_arr / thickness)) * vertical_weight
    # 盘体外部密度直接归零，避免把外部空间也当作发光介质。
    field = np.where(disk_volume_mask(r_arr, z_arr, params), field, 0.0)
    return _restore_shape(field, r if np.ndim(r) >= np.ndim(z) else z)


def temperature_field(
    r: float | np.ndarray,
    z: float | np.ndarray,
    params: DiskV2Params,
) -> float | np.ndarray:
    """计算二维温度场 `T(r, z)`。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        z: 局部盘坐标中的垂向高度，可以是标量或数组。
        params: `DiskV2Params` 参数对象。

    Returns:
        与输入广播后同形状的标量或数组，表示点 `(r, z)` 处的温度值 `T(r, z)`。
        返回值非负；盘体外部由 `disk_volume_mask()` 明确置为 `0`。

    Formula:
        ```
        T(r, z) = T_mid(r) · clip(1 - 0.25 · |z| / H(r), 0, 1) · W_z(r, z)
        ```

    Physical Meaning:
        当前版本采用“中面温度 + 弱垂向衰减”的简化形式，用于表达中面更热、表层略冷。
        额外的 `W_z(r, z)` 负责让温度场在几何表面与 `disk_volume_mask()` 的 hard boundary
        保持一致，避免在 `|z| = H(r)` 时仍残留明显非零温度。

    Simplifications:
        这里故意不引入完整垂向辐射转移；盘外温度直接返回 0。后续若接入更复杂辐射模型，
        这个函数可以被替换或升级。当前采用“弱垂向衰减 × 几何 support 关闭”的组合，
        重点是先保证几何边界与基础场定义一致。
    """

    r_arr = _to_array(r)
    z_arr = _to_array(z)
    thickness = np.maximum(_to_array(disk_half_thickness(r_arr, params)), np.finfo(np.float64).eps)
    temp_mid = _to_array(midplane_temperature_field(r_arr, params))
    vertical_weight = _to_array(disk_vertical_weight(r_arr, z_arr, params))
    # 先用线性衰减近似垂向冷却，后续可替换为更真实的垂向辐射模型。
    vertical_factor = np.clip(1.0 - 0.25 * np.abs(z_arr) / thickness, 0.0, 1.0)
    field = temp_mid * vertical_factor * vertical_weight
    # 盘外温度直接归零，保持基础场定义边界清晰。
    field = np.where(disk_volume_mask(r_arr, z_arr, params), field, 0.0)
    return _restore_shape(field, r if np.ndim(r) >= np.ndim(z) else z)
