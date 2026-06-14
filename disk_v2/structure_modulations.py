"""Disk V2 结构调制层。

v2.1 把结构调制从二维 `(r, φ)` 升级到三维 `(r, φ, z)`，新增团块项 `F_clump`，
并区分两种合成方式：

- `structure_modulation_density`：用于密度场，主要由 `F_shear · F_clump` 提供。
- `structure_modulation_emission`：用于发射率，主要由 `F_clump · F_hotspot` 提供。
- `structure_modulation`：保留为向后兼容的总合成入口（`F_mode · F_shear · F_clump · F_hotspot`）。

各分量的角色：

- `weak_mode_modulation()`：低频弱模态，二维 `(r, φ)`，提供轻微大尺度不对称。
- `shear_modulation()`：中频剪切纹理，二维 `(r, φ)`，丝状/絮状结构。
- `clump_modulation()`：**v2.1 新增**，三维 `(r, φ, z)`，显式点云团，主结构来源。
- `hotspot_modulation()`：稀疏热斑，二维 `(r, φ)`，极少数局部高亮点。

向后兼容：
当不传 `z` 时，`weak_mode / shear / hotspot` 行为与 v1.0 完全一致；
`clump_modulation` 在不传 `z` 时退化为中面（`z = 0`）的切片。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._array_utils import _restore_shape, _to_array
from .geometry import disk_half_thickness, disk_radial_weight
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

        v2.1 改动：`shear_modulation` 不再调用这个函数（会压扁高频），
        但 `weak_mode_modulation` 与 `hotspot_modulation` 仍保留这一行为，
        因为它们的振幅天然较小，归一化没有副作用。
    """

    max_abs = float(np.max(np.abs(value)))
    if max_abs <= np.finfo(np.float64).eps:
        return np.zeros_like(value)
    return value / max_abs


def _clip_3sigma(value: np.ndarray) -> np.ndarray:
    """把任意实值场按其标准差做 3σ 截断并归一化到 `[-1, 1]`。

    Args:
        value: 任意实值数组。

    Returns:
        与输入同形状的数组，落在 `[-1, 1]`。

    Formula:
        ```
        sigma = std(value)
        normalized = clip(value / (3 sigma), -1, 1)
        ```

    Physical Meaning:
        相对于 `_normalize_signed`，这里**只截断稀有极值**，绝大多数样本
        保持原振幅的相对关系，**不压扁高频细节**。

    Notes:
        - 全 0 输入返回全 0。
        - 截断点取 3σ 是常用阈值；对高斯分布约 99.7% 样本不受影响。
    """

    sigma = float(np.std(value))
    if sigma <= np.finfo(np.float64).eps:
        return np.zeros_like(value)
    return np.clip(value / (3.0 * sigma), -1.0, 1.0)


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
    """计算低频弱模态调制（v1.0 行为，二维 `(r, φ)`）。

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
    """计算中频剪切纹理调制（v2.1：频谱衰减放慢、振幅 3σ 截断）。

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
        F_shear = 1 + shear_strength · clip3sigma(S)
        A_k ∝ 1 / sqrt(k + 1)
        ```

        其中 `A_k` 为衰减振幅、`m_k` 为角向模态数、`n_k` 为径向相位耦合系数、
        `ψ_k` 为随机相位；这些量由 `seed` 决定，因此相同 `seed` 下结果可复现。

    Physical Meaning:
        通过将 `φ` 和 `log(r / r_in)` 耦合到同一相位中，可以生成带有径向拉伸感
        和剪切感的丝状/絮状结构。v2.1 采用更慢的频谱衰减保留更多高频细节，
        并用 3σ 截断而不是 `_normalize_signed` 来避免压扁高频。

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
    phi_arr_b = np.broadcast_to(phi_arr, raw_shear.shape)

    for component_idx in range(local_structure_params.shear_components):
        phi_frequency = int(rng.integers(2, 14))
        log_r_frequency = int(rng.integers(1, 8))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        # v2.1：振幅衰减改为 1 / sqrt(k + 1)，保留更多高频细节。
        amplitude = 1.0 / np.sqrt(component_idx + 1.0)
        raw_shear += amplitude * np.cos(phi_frequency * phi_arr_b + log_r_frequency * log_r + phase)
        raw_shear += 0.6 * amplitude * np.sin(
            (phi_frequency + 1) * phi_arr_b - (log_r_frequency + 0.5) * log_r + 0.7 * phase
        )

    shear_signed = _clip_3sigma(raw_shear)
    field = 1.0 + local_structure_params.shear_strength * shear_signed
    field = np.where(np.broadcast_to(window, raw_shear.shape) > 0.0, field, 1.0)
    return _restore_shape(field, r if np.ndim(r) >= np.ndim(phi) else phi)


@dataclass(frozen=True)
class _ShearComponents:
    """剪切纹理随机傅里叶分量（供 Taichi 上传）。

    Args:
        phi_frequency: 角向频率，形状 `(N,)`。
        log_r_frequency: 径向 log 频率，形状 `(N,)`。
        phase: 随机相位，形状 `(N,)`。
        amplitude: 分量振幅，形状 `(N,)`。
    """

    phi_frequency: np.ndarray
    log_r_frequency: np.ndarray
    phase: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class _HotspotCenters:
    """热斑中心集合（供 Taichi 上传）。

    Args:
        phi: 热斑方位角（弧度），形状 `(N,)`。
        log_r: 热斑 log 半径坐标，形状 `(N,)`。
        weight: 热斑权重，形状 `(N,)`。
    """

    phi: np.ndarray
    log_r: np.ndarray
    weight: np.ndarray


def _sample_shear_components(
    structure_params: DiskV2StructureParams,
    seed: int,
) -> _ShearComponents:
    """按 NumPy `shear_modulation` 相同规则生成可复现傅里叶分量。

    Args:
        structure_params: 结构参数，提供 `shear_components`。
        seed: 随机种子。

    Returns:
        `_ShearComponents` 对象。
    """
    rng = np.random.default_rng(seed)
    n = structure_params.shear_components
    phi_frequency = np.zeros(n, dtype=np.int32)
    log_r_frequency = np.zeros(n, dtype=np.int32)
    phase = np.zeros(n, dtype=np.float64)
    amplitude = np.zeros(n, dtype=np.float64)
    for component_idx in range(n):
        phi_frequency[component_idx] = int(rng.integers(2, 14))
        log_r_frequency[component_idx] = int(rng.integers(1, 8))
        phase[component_idx] = float(rng.uniform(0.0, 2.0 * np.pi))
        amplitude[component_idx] = 1.0 / np.sqrt(component_idx + 1.0)
    return _ShearComponents(
        phi_frequency=phi_frequency,
        log_r_frequency=log_r_frequency,
        phase=phase,
        amplitude=amplitude,
    )


def _sample_hotspot_centers(
    params: DiskV2Params,
    structure_params: DiskV2StructureParams,
    seed: int,
) -> _HotspotCenters:
    """按 NumPy `hotspot_modulation` 相同规则生成热斑中心。

    Args:
        params: 盘体参数。
        structure_params: 结构参数，提供 `hotspot_count`。
        seed: 随机种子。

    Returns:
        `_HotspotCenters` 对象。
    """
    rng = np.random.default_rng(seed)
    n = structure_params.hotspot_count
    log_r_span = np.log(params.r_out / params.r_in)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    log_r = (rng.uniform(0.0, 1.0, size=n) ** structure_params.hotspot_inner_bias) * log_r_span
    weight = rng.uniform(0.6, 1.0, size=n)
    return _HotspotCenters(phi=phi.astype(np.float64), log_r=log_r.astype(np.float64), weight=weight.astype(np.float64))


@dataclass(frozen=True)
class _ClumpCenters:
    """显式团块中心的可复现集合。

    Args:
        r: 各团块中心的径向坐标，形状 `(N,)`。
        phi: 各团块中心的方位角坐标，单位弧度，形状 `(N,)`。
        z: 各团块中心的垂向坐标，形状 `(N,)`。
        amplitude: 各团块的振幅系数（围绕 1 波动的 signed 系数前的权重），
            形状 `(N,)`，取值 `[-1, 1]`。

    Notes:
        这个 dataclass 只是中心集合的容器；用 `_sample_clump_centers` 函数
        在固定 seed 下生成。中心分布偏内圈（径向 log 均匀），振幅符号让团块
        既有"亮于中性"也有"暗于中性"的成员，保持围绕 1 波动。
    """

    r: np.ndarray
    phi: np.ndarray
    z: np.ndarray
    amplitude: np.ndarray


def _sample_clump_centers(
    params: DiskV2Params,
    structure_params: DiskV2StructureParams,
    seed: int,
) -> _ClumpCenters:
    """在盘体内可复现地生成显式团块中心集合。

    Args:
        params: `DiskV2Params` 参数对象。
        structure_params: `DiskV2StructureParams` 参数对象，提供 `clump_count`。
        seed: 随机种子。

    Returns:
        `_ClumpCenters` 对象，含 `clump_count` 个团块的 `(r, φ, z, amplitude)`。

    Notes:
        分布约定：

        - 径向：在 `log(r/r_in)` 空间均匀采样，等价于实空间偏内圈。
        - 角向：在 `[0, 2π)` 均匀采样。
        - 垂向：在 `[-H(r), +H(r)]` 均匀采样，让团块尺度随盘自然伸缩。
        - 振幅：在 `[-1, +1]` 均匀采样，让团块"亮""暗"各半。
    """

    rng = np.random.default_rng(seed)
    n = structure_params.clump_count
    # log-均匀分布：u ~ U(0, log(r_out / r_in)) → r = r_in * exp(u)。
    log_r_span = np.log(params.r_out / params.r_in)
    u = rng.uniform(0.0, log_r_span, size=n)
    r_centers = params.r_in * np.exp(u)
    phi_centers = rng.uniform(0.0, 2.0 * np.pi, size=n)
    # 垂向：相对厚度均匀。
    h_centers = _to_array(disk_half_thickness(r_centers, params))
    z_rel = rng.uniform(-1.0, 1.0, size=n)
    z_centers = z_rel * h_centers
    amplitudes = rng.uniform(-1.0, 1.0, size=n)
    return _ClumpCenters(r=r_centers, phi=phi_centers, z=z_centers, amplitude=amplitudes)


def clump_modulation(
    r: float | np.ndarray,
    phi: float | np.ndarray,
    z: float | np.ndarray | None = None,
    params: DiskV2Params | None = None,
    structure_params: DiskV2StructureParams | None = None,
    seed: int = 42,
    centers: Optional[_ClumpCenters] = None,
) -> float | np.ndarray:
    """计算团块调制 `F_clump`（v2.1 新增，三维 `(r, φ, z)`）。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        phi: 局部盘坐标中的方位角，可以是标量或数组。
        z: 局部盘坐标中的垂向高度，可以是标量或数组。如为 `None` 则视为 `z = 0`，
            返回中面切片，向后兼容二维调用。
        params: `DiskV2Params` 参数对象。必须传入。
        structure_params: `DiskV2StructureParams` 参数对象；若为 `None` 则使用默认值。
        seed: 随机种子，用于生成可复现的团块中心。
        centers: 可选预生成的 `_ClumpCenters`。在批量采样时复用，避免重复
            随机。如未传则用 `seed` 现场生成。

    Returns:
        与输入广播后同形状的标量或数组，表示团块调制的乘性因子。
        盘外区域返回中性值 `1`，盘内区域围绕 `1` 作中到大幅度波动。

    Formula:
        ```
        d_k² = ((r - r_k) / σ_r)² + (Δφ_k / σ_φ)² · r_k² + ((z - z_k) / σ_z)²

        K_k(r, φ, z) = max(0, 1 - sqrt(d_k²))
        K_k_smooth = K_k² · (3 - 2 K_k)             # smoothstep-style 锐利衰减

        S_clump = Σ_k amplitude_k · K_k_smooth
        F_clump = 1 + clump_strength · clamp(S_clump, -1, 1)
        ```

        其中：

        - `σ_r = clump_radial_sigma_scale · r_in`，团块径向尺度。
        - `σ_φ = clump_phi_sigma`，团块角向尺度（弧度）。在距离公式里乘 `r_k`
          做近似弧长，让远端团块在实空间中的角向尺度比内端大。
        - `σ_z = clump_vertical_sigma_scale · H(r_k)`，团块垂向尺度随盘厚度伸缩。
        - `Δφ_k = wrap(φ - φ_k)`，包裹到 `[-π, π]` 的最短角度差。

    Physical Meaning:
        模拟盘体内大量发光等离子体团：每个团块有清晰的核（亮于中性）或暗核
        （暗于中性），紧支撑（团块外贡献严格为 0），振幅前 50% 团块亮、
        后 50% 团块暗（统计上）。这是 v2.1 引入的**主结构来源**，区别于
        `F_shear` 的连续丝状结构。

    Simplifications:
        - 团块用解析锐利衰减核，不模拟流体方程。
        - σ_φ 在距离公式里乘 `r_k` 是近似的弧长换算，没有用纯弧长 σ。
        - 团块位置一次性采样、不随时间变化（动态平流由后续 `φ_adv` 入口提供）。
    """

    if params is None:
        raise ValueError("clump_modulation requires params")
    local_structure_params = structure_params or DiskV2StructureParams()
    if centers is None:
        centers = _sample_clump_centers(params, local_structure_params, seed)

    r_arr = _to_array(r)
    phi_arr = _to_array(phi)
    if z is None:
        z_arr = np.zeros_like(r_arr)
    else:
        z_arr = _to_array(z)
    window = disk_radial_weight(r_arr, params)

    shape = np.broadcast_shapes(r_arr.shape, phi_arr.shape, z_arr.shape)
    r_b = np.broadcast_to(r_arr, shape)
    phi_b = np.broadcast_to(phi_arr, shape)
    z_b = np.broadcast_to(z_arr, shape)

    sigma_r = local_structure_params.clump_radial_sigma_scale * params.r_in
    sigma_phi = local_structure_params.clump_phi_sigma

    accum = np.zeros(shape, dtype=np.float64)

    for k in range(len(centers.r)):
        r_k = float(centers.r[k])
        phi_k = float(centers.phi[k])
        z_k = float(centers.z[k])
        amp_k = float(centers.amplitude[k])

        # 垂向尺度按中心位置的盘半厚度伸缩。
        h_k = float(_to_array(disk_half_thickness(r_k, params)))
        sigma_z = max(local_structure_params.clump_vertical_sigma_scale * h_k, 1e-6)

        dr = (r_b - r_k) / sigma_r
        d_phi = _wrapped_delta_phi(phi_b, phi_k) * r_k / sigma_r
        # 角向距离与径向距离用相同 σ_r 做归一化，让团块整体呈"长度尺度 ~ σ_r"。
        # 同时再乘 sigma_phi 因子允许角向尺度独立调整。
        d_phi = d_phi * (sigma_r / max(sigma_phi * r_k, 1e-6))
        dz = (z_b - z_k) / sigma_z

        d2 = dr * dr + d_phi * d_phi + dz * dz
        d = np.sqrt(d2)
        # 紧支撑锐利衰减核：核内 d∈[0,1] 用 smoothstep 形式，核外严格为 0。
        kernel = np.clip(1.0 - d, 0.0, 1.0)
        kernel = kernel * kernel * (3.0 - 2.0 * kernel)
        accum += amp_k * kernel

    # 与 Taichi 路径保持逐点一致：不依赖全场 std，直接把 signed 累积裁剪到 [-1, 1]。
    # 这样 `clump_modulation` 可作为真正的 NumPy reference 做 parity 测试。
    signed = np.clip(accum, -1.0, 1.0)
    field = 1.0 + local_structure_params.clump_strength * signed
    field = np.where(np.broadcast_to(window, shape) > 0.0, field, 1.0)
    return _restore_shape(field, r if np.ndim(r) >= max(np.ndim(phi), np.ndim(z_arr)) else (phi if np.ndim(phi) >= np.ndim(z_arr) else z))


def hotspot_modulation(
    r: float | np.ndarray,
    phi: float | np.ndarray,
    params: DiskV2Params,
    structure_params: DiskV2StructureParams | None = None,
    seed: int = 42,
) -> float | np.ndarray:
    """计算稀疏热斑调制（v1.0 行为）。

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

    Physical Meaning:
        热斑只用于打破盘面过度平滑和过度程序化的观感。数量少、分布偏内圈。

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
    phi_arr_b = np.broadcast_to(phi_arr, raw_hotspot.shape)
    log_r_span = np.log(params.r_out / params.r_in)
    halo_phi_scale = 1.8
    halo_logr_scale = 1.8
    halo_weight_scale = 0.6

    for _ in range(local_structure_params.hotspot_count):
        hotspot_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        hotspot_log_r = float((rng.uniform(0.0, 1.0) ** local_structure_params.hotspot_inner_bias) * log_r_span)
        hotspot_weight = float(rng.uniform(0.6, 1.0))

        delta_phi = _wrapped_delta_phi(phi_arr_b, hotspot_phase)
        delta_log_r = (log_r - hotspot_log_r) / local_structure_params.hotspot_logr_sigma
        hotspot_core = np.exp(
            -0.5 * (delta_phi / local_structure_params.hotspot_phi_sigma) ** 2
            -0.5 * delta_log_r ** 2
        )
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
    z: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """合成静态结构调制 `F_struct(r, φ, [z], t=0)`（向后兼容）。

    Args:
        r: 局部盘坐标中的径向距离，可以是标量或数组。
        phi: 局部盘坐标中的方位角，可以是标量或数组。
        params: `DiskV2Params` 参数对象。
        structure_params: `DiskV2StructureParams` 参数对象；若为 `None` 则使用默认值。
        seed: 随机种子，传给剪切纹理调制和热斑调制，保证结果可复现。
        z: 局部盘坐标中的垂向高度。当传入时合成会包含 `F_clump`；当为 `None`
            时退化为 v1.0 二维合成（不含 clump）。

    Returns:
        与输入广播后同形状的标量或数组，表示静态结构调制的总乘性因子。
        盘外区域返回中性值 `1`，盘内区域为正值并围绕 `1` 波动。

    Formula:
        ```
        # 二维（z=None）：v1.0 行为
        F_struct = F_mode · F_shear · F_hotspot

        # 三维（带 z）：v2.1
        F_struct = F_mode · F_shear · F_clump(z) · F_hotspot
        ```

    Physical Meaning:
        该函数把弱模态、剪切、（可选）团块、热斑合成为一个静态结构调制。
        新代码推荐用 `structure_modulation_density` 与 `structure_modulation_emission`，
        本入口仅作为向后兼容存在。
    """

    local_structure_params = structure_params or DiskV2StructureParams()
    mode_layer = weak_mode_modulation(r, phi, params, local_structure_params)
    shear_layer = shear_modulation(r, phi, params, local_structure_params, seed=seed)
    hotspot_layer = hotspot_modulation(r, phi, params, local_structure_params, seed=seed + 1)

    mode_arr = _to_array(mode_layer)
    shear_arr = _to_array(shear_layer)
    hotspot_arr = _to_array(hotspot_layer)
    combined = mode_arr * shear_arr * hotspot_arr

    if z is not None:
        clump_layer = clump_modulation(
            r, phi, z=z, params=params, structure_params=local_structure_params, seed=seed + 2
        )
        clump_arr = _to_array(clump_layer)
        combined = combined * clump_arr

    combined = np.where(disk_radial_weight(_to_array(r), params) > 0.0, combined, 1.0)
    return _restore_shape(combined, r if np.ndim(r) >= np.ndim(phi) else phi)


def structure_modulation_density(
    r: float | np.ndarray,
    phi: float | np.ndarray,
    z: float | np.ndarray,
    params: DiskV2Params,
    structure_params: DiskV2StructureParams | None = None,
    seed: int = 42,
) -> float | np.ndarray:
    """合成作用于密度的结构调制 `F_struct_density`（v2.1 新增）。

    Args:
        r: 径向距离，可以是标量或数组。
        phi: 方位角，可以是标量或数组。
        z: 垂向高度，可以是标量或数组。
        params: `DiskV2Params`。
        structure_params: `DiskV2StructureParams`；若 `None` 则用默认。
        seed: 随机种子。

    Returns:
        与输入广播后同形状的标量或数组。盘内为正、围绕 1 波动；盘外为 1。

    Formula:
        ```
        F_struct_density = F_shear · F_clump
        ```

    Physical Meaning:
        密度涨落由剪切丝状结构和团块联合贡献。
        热斑和低频模态对密度作用很弱，本合成不包含。

    Simplifications:
        - 不引入垂向密度涨落以外的物理建模（如压力支撑、辐射压）。
        - 各分量都是乘性、围绕 1 波动；最终结果严格为正。
    """

    local_structure_params = structure_params or DiskV2StructureParams()
    shear_layer = shear_modulation(r, phi, params, local_structure_params, seed=seed)
    clump_layer = clump_modulation(
        r, phi, z=z, params=params, structure_params=local_structure_params, seed=seed + 2
    )
    combined = _to_array(shear_layer) * _to_array(clump_layer)
    combined = np.where(disk_radial_weight(_to_array(r), params) > 0.0, combined, 1.0)
    return _restore_shape(combined, r if np.ndim(r) >= max(np.ndim(phi), np.ndim(z)) else (phi if np.ndim(phi) >= np.ndim(z) else z))


def structure_modulation_emission(
    r: float | np.ndarray,
    phi: float | np.ndarray,
    z: float | np.ndarray,
    params: DiskV2Params,
    structure_params: DiskV2StructureParams | None = None,
    seed: int = 42,
) -> float | np.ndarray:
    """合成作用于发射率的结构调制 `F_struct_emission`（v2.1 新增）。

    Args:
        r: 径向距离，可以是标量或数组。
        phi: 方位角，可以是标量或数组。
        z: 垂向高度，可以是标量或数组。
        params: `DiskV2Params`。
        structure_params: `DiskV2StructureParams`；若 `None` 则用默认。
        seed: 随机种子。

    Returns:
        与输入广播后同形状的标量或数组。盘内为正、围绕 1 波动；盘外为 1。

    Formula:
        ```
        F_struct_emission = F_mode · F_clump · F_hotspot
        ```

    Physical Meaning:
        发射率涨落由"低频模态 + 团块 + 热斑"提供，对应"大尺度不对称 +
        发光团块 + 极亮点"的视觉层次。

    Simplifications:
        - F_shear 不进入发射率合成，避免与密度调制重复双倍变化。
        - 这一选择是工程性判断，便于密度与发射率在视觉上扮演不同角色。
    """

    local_structure_params = structure_params or DiskV2StructureParams()
    mode_layer = weak_mode_modulation(r, phi, params, local_structure_params)
    clump_layer = clump_modulation(
        r, phi, z=z, params=params, structure_params=local_structure_params, seed=seed + 2
    )
    hotspot_layer = hotspot_modulation(r, phi, params, local_structure_params, seed=seed + 1)

    combined = _to_array(mode_layer) * _to_array(clump_layer) * _to_array(hotspot_layer)
    combined = np.where(disk_radial_weight(_to_array(r), params) > 0.0, combined, 1.0)
    return _restore_shape(combined, r if np.ndim(r) >= max(np.ndim(phi), np.ndim(z)) else (phi if np.ndim(phi) >= np.ndim(z) else z))
