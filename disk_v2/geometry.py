"""Disk V2 基础几何定义。

本模块只处理吸积盘的几何边界，不处理温度、密度、发射率或颜色。
所有量都采用项目当前使用的无量纲几何单位。
"""

from __future__ import annotations

import numpy as np

from ._array_utils import _restore_bool, _restore_shape, _to_array
from .params import DiskV2Params


def smoothstep(edge0: float, edge1: float, x: float | np.ndarray) -> float | np.ndarray:
    """计算三次平滑插值函数 `smoothstep`。

    Args:
        edge0: 平滑区起点。
        edge1: 平滑区终点，必须大于 `edge0`。
        x: 输入标量或数组。

    Returns:
        与 `x` 同形状的标量或数组，取值位于 `[0, 1]`。

    Formula:
        ```
        t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
        smoothstep = t² (3 - 2t)
        ```

    Physical Meaning:
        当 `x <= edge0` 时结果为 0；当 `x >= edge1` 时结果为 1；中间区域采用三次多项式
        平滑过渡，使一阶导数在两端为 0。这里它主要用于把吸积盘内外边界从“硬截断”
        改为“平滑窗函数”。

    Simplifications:
        这里使用的是标准三次 `smoothstep`，而不是更高阶的平滑核；
        对当前吸积盘边界过渡而言，这已经足够稳定且易于测试。
    """

    if edge1 <= edge0:
        raise ValueError("edge1 must be greater than edge0")
    x_arr = _to_array(x)
    t = np.clip((x_arr - edge0) / (edge1 - edge0), 0.0, 1.0)
    value = t * t * (3.0 - 2.0 * t)
    return _restore_shape(value, x)


def disk_half_thickness(r: float | np.ndarray, params: DiskV2Params) -> float | np.ndarray:
    """计算盘体半厚度 `H(r)`。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        params: `DiskV2Params` 参数对象。

    Returns:
        与 `r` 同形状的标量或数组，表示盘从中面 `z = 0` 到上表面的半厚度 `H(r)`。

    Formula:
        ```
        H(r) = h0 · r · (r / r_in)^beta_h
        ```

    Physical Meaning:
        `H(r)` 描述盘的垂向几何外形。当 `beta_h > 0` 时，外圈会比内圈略厚。
        当前版本只允许“薄盘到稍厚盘”的温和变化，不做夸张厚盘。

    Simplifications:
        对 `r < r_in` 的输入，内部使用 `r_in` 作为安全下界，避免幂函数在无意义区域内
        影响数值稳定性。真正的“是否在盘内”判断由 `disk_volume_mask()` 决定。
    """

    r_arr = _to_array(r)
    safe_r = np.maximum(r_arr, params.r_in)
    thickness = params.h0 * safe_r * np.power(safe_r / params.r_in, params.beta_h)
    return _restore_shape(thickness, r)


def disk_radial_mask(r: float | np.ndarray, params: DiskV2Params) -> bool | np.ndarray:
    """判断给定半径是否落在盘的径向有效域内。

    Args:
        r: 径向坐标，可以是标量或 `np.ndarray`。其物理含义是局部盘坐标中的
            半径 `r = sqrt(x'^2 + y'^2)`。
        params: `DiskV2Params` 参数对象，提供硬边界 `r_in` 与 `r_out`。

    Returns:
        与 `r` 同形状的布尔标量或布尔数组。

        - `True`：该半径位于盘的径向有效域内；
        - `False`：该半径位于盘外。

    Formula:
        ```
        M_r(r) = 1,  r_in <= r <= r_out
               = 0,  otherwise
        ```

    Physical Meaning:
        这是几何层的“硬边界 membership”判定，只回答“这个半径是否属于盘”。
        它不负责边界平滑，也不携带任何密度、温度等物理量信息。
        这里采用“闭区间 membership”约定：`r = r_in` 与 `r = r_out` 仍算作几何上属于盘，
        但这些边界点对应的基础物理场可继续通过 `disk_radial_weight()` 收口到 `0`。

    Simplifications:
        这里采用严格的硬边界判定，不引入容差带或概率 membership；
        边界平滑由 `disk_radial_weight()` 单独负责。
    """

    r_arr = _to_array(r)
    mask = (r_arr >= params.r_in) & (r_arr <= params.r_out)
    return _restore_bool(mask, r)


def disk_radial_weight(r: float | np.ndarray, params: DiskV2Params) -> float | np.ndarray:
    """计算盘体径向平滑权重 `W_r(r)`。

    数学形式：

    ```
    W_r(r) = W_in(r) · W_out(r)
    W_in(r)  = smoothstep(r_in, r_in + Δr, r)
    W_out(r) = 1 - smoothstep(r_out - Δr, r_out, r)
    ```

    其中：

    ```
    Δr = edge_softness · (r_out - r_in)
    ```

    含义说明：

    - 在盘体中间区域，`W_r(r) ≈ 1`；
    - 在内边界附近，`W_r(r)` 从 0 平滑过渡到 1；
    - 在外边界附近，`W_r(r)` 从 1 平滑过渡到 0；
    - 在盘体外部，`W_r(r) = 0`。

    该权重会被中面密度和中面温度共同复用，用于避免硬边界。

    Args:
        r: 径向坐标，可以是标量或 `np.ndarray`。其物理含义是局部盘坐标中的
            半径 `r = sqrt(x'^2 + y'^2)`。
        params: `DiskV2Params` 参数对象，提供 `r_in`、`r_out` 和 `edge_softness`。

    Returns:
        与 `r` 同形状的标量或数组，表示径向平滑权重 `W_r(r)`。

        - `0.0`：该半径位于盘外；
        - `(0.0, 1.0)`：该半径位于内外边界的平滑过渡区；
        - `1.0`：该半径位于盘体中部，不受边界衰减影响。

    Formula:
        ```
        W_r(r) = W_in(r) · W_out(r)
        W_in(r)  = smoothstep(r_in, r_in + Δr, r)
        W_out(r) = 1 - smoothstep(r_out - Δr, r_out, r)
        Δr = edge_softness · (r_out - r_in)
        ```

    Physical Meaning:
        该函数是 `disk_radial_mask()` 的软版本：
        `disk_radial_mask()` 给硬判定，`disk_radial_weight()` 给边界附近平滑过渡的权重。
        它把盘体“有边界”与“边界不要太硬”同时编码进一个权重 `W_r(r)` 中，供
        中面密度和中面温度共同复用。
        需要特别注意：当前约定下，`r = r_in` 和 `r = r_out` 虽然在 `disk_radial_mask()` 中
        仍属于盘内，但在 `disk_radial_weight()` 中会取到 `0`，因此基础物理场在精确径向边界上
        仍然收口为 `0`。

    Simplifications:
        当前的径向平滑只处理内外边界；垂向方向的平滑关闭由
        `disk_vertical_weight()` 单独负责，二者分开定义以保持语义清晰。
    """

    r_arr = _to_array(r)
    radial_span = params.r_out - params.r_in
    # 将总径向跨度的一部分作为平滑过渡宽度。
    soft_width = max(radial_span * params.edge_softness, np.finfo(np.float64).eps)
    # 内边界从 0 平滑爬升到 1，外边界从 1 平滑下降到 0。
    inner = smoothstep(params.r_in, params.r_in + soft_width, r_arr)
    outer = 1.0 - smoothstep(params.r_out - soft_width, params.r_out, r_arr)
    weight = inner * outer
    weight = np.where((r_arr <= params.r_in) | (r_arr >= params.r_out), 0.0, weight)
    return _restore_shape(weight, r)


def disk_vertical_weight(
    r: float | np.ndarray,
    z: float | np.ndarray,
    params: DiskV2Params,
) -> float | np.ndarray:
    """计算盘体垂向平滑权重 `W_z(r, z)`。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。它通过 `H(r)` 决定当前半径处的
            垂向包络尺度。
        z: 局部盘坐标中的垂向高度，可以是标量或数组。
        params: `DiskV2Params` 参数对象。

    Returns:
        与输入广播后同形状的标量或数组，表示垂向平滑权重 `W_z(r, z)`。

        - `1.0`：位于中面 `z = 0`；
        - `(0.0, 1.0)`：位于盘体内部、但接近上下表面；
        - `0.0`：位于几何表面 `|z| = H(r)`、盘体外部，或径向上已不在盘内。

    Formula:
        ```
        xi(r, z) = |z| / max(H(r), eps)
        W_z(r, z) = 1 - smoothstep(0, 1, xi)
        ```

    Physical Meaning:
        `disk_volume_mask()` 给出“是否在盘内”的硬判定，而 `disk_vertical_weight()` 给出
        与几何上表面一致收口的垂向软权重。它的核心目的不是定义新的物理场，而是把
        `|z| = H(r)` 这条几何边界转换为一个连续的 `[0, 1]` 权重，供基础物理场在靠近
        上下表面时平滑衰减到 0。
        这里采用与径向类似的分工：`disk_volume_mask()` 负责几何 membership，
        `disk_vertical_weight()` 负责让基础场在几何表面精确收口到 `0`。

    Simplifications:
        - 当前只对垂向方向做单调、关于中面对称的平滑关闭；
        - 不引入额外的表面层参数或更复杂的 photosphere 模型；
        - 径向上若已不在盘内，直接返回 0，而不是继续解释垂向权重。
    """

    r_arr = _to_array(r)
    z_arr = _to_array(z)
    thickness = np.maximum(_to_array(disk_half_thickness(r_arr, params)), np.finfo(np.float64).eps)
    radial_mask = _to_array(disk_radial_mask(r_arr, params)).astype(bool)
    xi = np.abs(z_arr) / thickness
    weight = 1.0 - _to_array(smoothstep(0.0, 1.0, xi))
    weight = np.where(radial_mask, weight, 0.0)
    return _restore_shape(weight, r if np.ndim(r) >= np.ndim(z) else z)


def disk_volume_mask(
    r: float | np.ndarray,
    z: float | np.ndarray,
    params: DiskV2Params,
) -> bool | np.ndarray:
    """判断给定点是否落在有限厚度盘体内部。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        z: 局部盘坐标中的垂向高度，可以是标量或数组。
        params: `DiskV2Params` 参数对象。

    Returns:
        布尔标量或布尔数组，表示对应点是否位于盘体内部。

        - `True`：点位于几何体积内部，或恰好落在几何边界上；
        - `False`：点严格位于几何包络之外。

    Formula:
        ```
        r_in <= r <= r_out
        |z| <= H(r)
        ```

    Physical Meaning:
        第一条限制点必须落在盘的径向范围内，第二条限制点必须落在由
        `disk_half_thickness(r)` 给出的垂向包络内。
        它是几何层从“径向 membership”扩展到“体积 membership”的版本。
        这里同样采用“闭区间 membership”约定：`|z| = H(r)` 的表面点在几何上仍算盘内，
        但基础物理场会通过 `disk_vertical_weight()` 在这些表面点收口到 `0`。

    Simplifications:
        当前采用硬体积边界：只要超出 `r` 或 `z` 的几何包络就直接判为盘外；
        该函数本身不返回平滑权重或半透明边缘；平滑关闭由 `disk_radial_weight()`
        与 `disk_vertical_weight()` 分别负责。
    """

    r_arr = _to_array(r)
    z_arr = _to_array(z)
    thickness = _to_array(disk_half_thickness(r_arr, params))
    radial_mask = _to_array(disk_radial_mask(r_arr, params)).astype(bool)
    mask = radial_mask & (np.abs(z_arr) <= thickness)
    return _restore_bool(mask, r if np.ndim(r) >= np.ndim(z) else z)
