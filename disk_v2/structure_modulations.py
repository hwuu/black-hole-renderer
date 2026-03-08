"""Disk V2 结构调制层。

本模块只定义 `t = 0` 时刻的静态结构调制，不处理：

- 时间平流 `φ_adv`
- 发射-吸收积分
- 颜色映射

当前实现的结构调制分成三层：

- `weak_mode_modulation()`：低频弱模态调制
- `shear_modulation()`：中频剪切纹理调制
- `hotspot_modulation()`：稀疏热斑调制

最终通过 `structure_modulation()` 合成为一个静态的总乘性调制因子。
"""

from __future__ import annotations

import numpy as np

from ._array_utils import _restore_shape, _to_array
from .geometry import disk_radial_weight
from .params import DiskV2Params, DiskV2StructureParams


def _normalize_signed(value: np.ndarray) -> np.ndarray:
    """把任意实值场归一化到 `[-1, 1]`。

    Args:
        value: 任意实值数组。

    Returns:
        与输入同形状的数组。若输入全 0，则返回全 0。

    Notes:
        该函数把任意实值模式压到统一的 `[-1, 1]` 幅度范围内，便于后续用
        强度参数直接控制调制幅度，而不让不同随机 realization 的绝对量级漂移。
    """

    max_abs = float(np.max(np.abs(value)))
    if max_abs <= np.finfo(np.float64).eps:
        return np.zeros_like(value)
    return value / max_abs


def _wrapped_delta_phi(phi: np.ndarray, phi_center: float) -> np.ndarray:
    """计算周期角度差 `Δφ`，并将结果包裹到 `[-π, π]`。

    Args:
        phi: 输入方位角数组，单位为弧度。
        phi_center: 参考方位角，单位为弧度。

    Returns:
        与 `phi` 同形状的数组，表示最短有符号角度差 `Δφ = phi - phi_center`。
        返回值落在 `[-π, π]`，便于后续构造周期连续的角向高斯或相位扰动。

    Formula:
        ```
        Δφ = atan2(sin(phi - phi_center), cos(phi - phi_center))
        ```

    Notes:
        直接做 `phi - phi_center` 会在 `2π` 周期边界处出现跳变；这里用包裹后的最短角度差
        保证热斑和角向结构在 `0` 与 `2π` 附近连续。
    """

    return np.arctan2(np.sin(phi - phi_center), np.cos(phi - phi_center))


def _log_radius(r: np.ndarray, params: DiskV2Params) -> np.ndarray:
    """计算对数半径坐标 `log(r / r_in)`。

    Args:
        r: 径向距离数组。
        params: `DiskV2Params` 参数对象。

    Returns:
        与 `r` 同形状的对数半径坐标。

    Formula:
        ```
        x_r = log(max(r, r_in) / r_in)
        ```

    Notes:
        `log(r / r_in)` 能更自然地表达盘内尺度变化，也更适合构造带有径向拉伸感的结构。
        对 `r < r_in` 的输入，内部使用 `r_in` 作为安全下界，避免对数在无意义区域发散。
    """

    safe_r = np.maximum(r, params.r_in)
    return np.log(safe_r / params.r_in)


def weak_mode_modulation(
    r: float | np.ndarray,
    phi: float | np.ndarray,
    params: DiskV2Params,
    structure_params: DiskV2StructureParams | None = None,
) -> float | np.ndarray:
    """计算低频弱模态调制。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        phi: 局部盘坐标中的方位角，可以是标量或数组。
        params: `DiskV2Params` 参数对象。
        structure_params: `DiskV2StructureParams` 参数对象；若为 `None` 则使用默认值。

    Returns:
        与输入广播后同形状的标量或数组，表示低频弱模态的乘性因子。
        盘外区域返回中性值 `1`，盘内区域围绕 `1` 作小幅波动。

    Formula:
        ```
        F_mode(r, φ) = 1
            + a1 · cos(φ + c1 · log(r / r_in))
            + a2 · cos(2φ - c2 · log(r / r_in))
        ```

        其中 `a1` 与 `a2` 分别对应 `mode1_strength` 和 `mode2_strength`，
        `c1`、`c2` 是当前实现里固定的径向相位耦合系数。

    Physical Meaning:
        该层只提供很弱的大尺度不对称性，避免盘面过于完美对称，但又不做成明显的稳定螺旋臂。

    Simplifications:
        这里直接使用解析余弦模态，而不是从流体不稳定性方程推导。
    """

    local_structure_params = structure_params or DiskV2StructureParams()
    r_arr = _to_array(r)
    phi_arr = _to_array(phi)
    log_r = _log_radius(r_arr, params)
    window = disk_radial_weight(r_arr, params)

    raw_mode = (
        local_structure_params.mode1_strength * np.cos(phi_arr + 0.35 * log_r)
        + local_structure_params.mode2_strength * np.cos(2.0 * phi_arr - 0.65 * log_r)
    )
    field = 1.0 + raw_mode
    field = np.where(window > 0.0, field, 1.0)
    return _restore_shape(field, r if np.ndim(r) >= np.ndim(phi) else phi)


def shear_modulation(
    r: float | np.ndarray,
    phi: float | np.ndarray,
    params: DiskV2Params,
    structure_params: DiskV2StructureParams | None = None,
    seed: int = 42,
) -> float | np.ndarray:
    """计算中频剪切纹理调制。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        phi: 局部盘坐标中的方位角，可以是标量或数组。
        params: `DiskV2Params` 参数对象。
        structure_params: `DiskV2StructureParams` 参数对象；若为 `None` 则使用默认值。
        seed: 随机种子，用于生成可复现的随机傅里叶分量。

    Returns:
        与输入广播后同形状的标量或数组，表示中频剪切纹理的乘性因子。
        盘外区域返回中性值 `1`，盘内区域围绕 `1` 作中等幅度波动。

    Formula:
        ```
        S(r, φ) = Σ_k A_k · cos(m_k φ + n_k log(r / r_in) + ψ_k)
        F_shear = 1 + shear_strength · normalize(S)
        ```

        其中 `A_k` 为随机幅值，`m_k` 为角向模态数，`n_k` 为径向相位耦合系数，
        `ψ_k` 为随机相位；这些量由 `seed` 决定，因此在相同 `seed` 下结果可复现。

    Physical Meaning:
        该层是静态结构调制的主力。通过将 `φ` 和 `log(r / r_in)` 耦合到同一相位中，
        可以生成带有径向拉伸感和剪切感的丝状/絮状结构。

    Simplifications:
        当前并不求解真实流体方程，而是用随机傅里叶叠加构造一个可控、可复现的近似剪切纹理。
    """

    local_structure_params = structure_params or DiskV2StructureParams()
    rng = np.random.default_rng(seed)
    r_arr = _to_array(r)
    phi_arr = _to_array(phi)
    log_r = _log_radius(r_arr, params)
    window = disk_radial_weight(r_arr, params)

    raw_shear = np.zeros(np.broadcast_shapes(r_arr.shape, phi_arr.shape), dtype=np.float64)
    log_r = np.broadcast_to(log_r, raw_shear.shape)
    phi_arr = np.broadcast_to(phi_arr, raw_shear.shape)

    # 叠加少量随机傅里叶分量，构造既有方位结构又有径向拉伸感的纹理。
    for component_idx in range(local_structure_params.shear_components):
        phi_frequency = int(rng.integers(2, 10))
        log_r_frequency = int(rng.integers(1, 6))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amplitude = 0.5 ** component_idx
        raw_shear += amplitude * np.cos(phi_frequency * phi_arr + log_r_frequency * log_r + phase)
        raw_shear += 0.6 * amplitude * np.sin(
            (phi_frequency + 1) * phi_arr - (log_r_frequency + 0.5) * log_r + 0.7 * phase
        )

    shear_signed = _normalize_signed(raw_shear)
    field = 1.0 + local_structure_params.shear_strength * shear_signed
    field = np.where(np.broadcast_to(window, raw_shear.shape) > 0.0, field, 1.0)
    return _restore_shape(field, r if np.ndim(r) >= np.ndim(phi) else phi)


def hotspot_modulation(
    r: float | np.ndarray,
    phi: float | np.ndarray,
    params: DiskV2Params,
    structure_params: DiskV2StructureParams | None = None,
    seed: int = 42,
) -> float | np.ndarray:
    """计算稀疏热斑调制。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        phi: 局部盘坐标中的方位角，可以是标量或数组。
        params: `DiskV2Params` 参数对象。
        structure_params: `DiskV2StructureParams` 参数对象；若为 `None` 则使用默认值。
        seed: 随机种子，用于生成可复现的热斑中心与强度。

    Returns:
        与输入广播后同形状的标量或数组，表示热斑调制的乘性因子。
        盘外区域返回中性值 `1`，盘内区域围绕 `1` 作以热点为主的局部正负起伏。

    Formula:
        ```
        G_core,k(r, φ) = exp[-0.5 (Δφ_k / σ_φ)^2 - 0.5 ((log(r/r_in) - μ_k) / σ_logr)^2]
        G_halo,k(r, φ) = exp[-0.5 (Δφ_k / (γ_φ σ_φ))^2 - 0.5 ((log(r/r_in) - μ_k) / (γ_r σ_logr))^2]
        S_hotspot = Σ_k w_k · [G_core,k - η · G_halo,k]
        F_hotspot = 1 + hotspot_strength · normalize(S_hotspot)
        ```

        其中 `Δφ_k` 是点 `(r, φ)` 到第 `k` 个热斑中心的包裹角距离，`μ_k` 是第 `k` 个热斑中心的
        对数半径位置，`σ_φ` 与 `σ_logr` 分别控制方位角和对数半径方向的热斑宽度，`w_k` 为热斑权重；
        `γ_φ > 1`、`γ_r > 1` 用于定义更宽的补偿 halo，`η` 控制 halo 的相对权重。

    Physical Meaning:
        热斑只用于打破盘面过度平滑和过度程序化的观感。其数量应少、分布偏内圈，
        只做稀疏点缀而不接管主结构。这里使用“亮核心 + 弱补偿 halo”的差分结构，
        让热斑在视觉上仍以局部热点为主，同时保持它是围绕 `1` 波动的真正乘性调制。

    Simplifications:
        热斑被简化为极坐标空间中的差分高斯斑点，不模拟寿命、合并或真实磁重联过程。
    """

    local_structure_params = structure_params or DiskV2StructureParams()
    rng = np.random.default_rng(seed)
    r_arr = _to_array(r)
    phi_arr = _to_array(phi)
    log_r = _log_radius(r_arr, params)
    window = disk_radial_weight(r_arr, params)

    raw_hotspot = np.zeros(np.broadcast_shapes(r_arr.shape, phi_arr.shape), dtype=np.float64)
    log_r = np.broadcast_to(log_r, raw_hotspot.shape)
    phi_arr = np.broadcast_to(phi_arr, raw_hotspot.shape)
    log_r_span = np.log(params.r_out / params.r_in)
    halo_phi_scale = 1.8
    halo_logr_scale = 1.8
    halo_weight_scale = 0.6

    # 通过 `u^bias` 让热斑中心在对数半径上偏向内圈。
    for _ in range(local_structure_params.hotspot_count):
        hotspot_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        hotspot_log_r = float((rng.uniform(0.0, 1.0) ** local_structure_params.hotspot_inner_bias) * log_r_span)
        hotspot_weight = float(rng.uniform(0.6, 1.0))

        delta_phi = _wrapped_delta_phi(phi_arr, hotspot_phase)
        delta_log_r = (log_r - hotspot_log_r) / local_structure_params.hotspot_logr_sigma
        hotspot_core = np.exp(
            -0.5 * (delta_phi / local_structure_params.hotspot_phi_sigma) ** 2
            -0.5 * delta_log_r ** 2
        )
        # 用更宽的 halo 提供局部补偿，使热斑既保留“中心提亮”的视觉特征，
        # 又能作为围绕 1 波动的 signed modulation，而不是纯 boost。
        hotspot_halo = np.exp(
            -0.5 * (delta_phi / (halo_phi_scale * local_structure_params.hotspot_phi_sigma)) ** 2
            -0.5 * ((log_r - hotspot_log_r) / (halo_logr_scale * local_structure_params.hotspot_logr_sigma)) ** 2
        )
        raw_hotspot += hotspot_weight * (hotspot_core - halo_weight_scale * hotspot_halo)

    hotspot_signed = _normalize_signed(raw_hotspot)
    field = 1.0 + local_structure_params.hotspot_strength * hotspot_signed
    field = np.where(np.broadcast_to(window, raw_hotspot.shape) > 0.0, field, 1.0)
    return _restore_shape(field, r if np.ndim(r) >= np.ndim(phi) else phi)


def structure_modulation(
    r: float | np.ndarray,
    phi: float | np.ndarray,
    params: DiskV2Params,
    structure_params: DiskV2StructureParams | None = None,
    seed: int = 42,
) -> float | np.ndarray:
    """合成静态结构调制 `F_struct(r, φ, t=0)`。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        phi: 局部盘坐标中的方位角，可以是标量或数组。
        params: `DiskV2Params` 参数对象。
        structure_params: `DiskV2StructureParams` 参数对象；若为 `None` 则使用默认值。
        seed: 随机种子，传给剪切纹理调制和热斑调制，保证结果可复现。

    Returns:
        与输入广播后同形状的标量或数组，表示静态结构调制的总乘性因子。
        盘外区域返回中性值 `1`，盘内区域为正值并围绕 `1` 波动。

    Formula:
        ```
        F_struct = F_mode · F_shear · F_hotspot
        ```

    Physical Meaning:
        该函数把弱模态调制、剪切纹理调制和稀疏热斑调制合成为一个静态结构调制，供后续发射率或预览层使用。

    Simplifications:
        当前实现仅构造 `t = 0` 的结构调制，不引入时间平流；后续动画阶段再统一切换到 `φ_adv`。
    """

    local_structure_params = structure_params or DiskV2StructureParams()
    mode_layer = weak_mode_modulation(r, phi, params, local_structure_params)
    shear_layer = shear_modulation(r, phi, params, local_structure_params, seed=seed)
    hotspot_layer = hotspot_modulation(r, phi, params, local_structure_params, seed=seed + 1)

    mode_arr = _to_array(mode_layer)
    shear_arr = _to_array(shear_layer)
    hotspot_arr = _to_array(hotspot_layer)
    combined = mode_arr * shear_arr * hotspot_arr
    combined = np.where(disk_radial_weight(_to_array(r), params) > 0.0, combined, 1.0)
    return _restore_shape(combined, r if np.ndim(r) >= np.ndim(phi) else phi)
