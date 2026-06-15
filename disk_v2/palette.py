"""Disk V2 调色与色调映射层（v2.1 新增）。

本模块只处理"物理量 → 像素颜色"的映射，不涉及物理场定义本身：

- `blackbody_color(T_K)`：温度 → RGB（基于 Tanner Helland 近似）。
- `cinematic_color(T_K, params)`：在 `blackbody_color` 基础上做饱和度增强 + 暖色偏移。
- `tonemap(rgb_hdr, params)`：HDR → LDR 色调映射。
- `gamma_correct(rgb_linear, params)`：sRGB 伽马校正。
- `apply_palette(rgb_hdr, T_K, params)`：把基础 HDR 强度乘上 palette 颜色后输出 HDR 颜色。

Notes:
    所有函数都是纯 NumPy 实现，作为参考实现。Phase 4 的 Taichi 实现
    会以这些函数的逐元素行为为基准做 parity 测试。
"""

from __future__ import annotations

import numpy as np

from ._array_utils import _restore_shape, _to_array
from .params import DiskV2PaletteParams


def _blackbody_rgb_array(t_over_100: np.ndarray) -> np.ndarray:
    """Tanner Helland 黑体色温查表（向量化版本，单位 = T / 100）。

    Args:
        t_over_100: 温度除以 100 后的标量或数组。

    Returns:
        形状为 `(..., 3)` 的数组，最后一维是 `(R, G, B)`，每个通道在 `[0, 1]`。

    Notes:
        与 `render.py:136` 的 `_blackbody_rgb` 数学等价；这里用稳定的逐通道
        分段表达，便于 Phase 4 Taichi parity 对照。
    """

    t = t_over_100
    t_safe = np.maximum(t, 1e-6)
    t_minus_60 = np.maximum(t - 60.0, 1e-6)
    t_minus_10 = np.maximum(t - 10.0, 1e-6)

    r = np.where(
        t <= 66.0,
        1.0,
        np.clip(1.292936 * np.power(t_minus_60, -0.1332047592), 0.0, 1.0),
    )
    g = np.where(
        t <= 66.0,
        np.clip(0.390082 * np.log(t_safe) - 0.631841, 0.0, 1.0),
        np.clip(1.129891 * np.power(t_minus_60, -0.0755148492), 0.0, 1.0),
    )
    b = np.where(
        t >= 66.0,
        1.0,
        np.where(
            t <= 19.0,
            0.0,
            np.clip(0.543207 * np.log(t_minus_10) - 1.19625, 0.0, 1.0),
        ),
    )
    return np.stack([r, g, b], axis=-1)


def blackbody_color(
    T_K: float | np.ndarray,
) -> np.ndarray:
    """温度 → 黑体色 RGB（physical 模式）。

    Args:
        T_K: 温度，单位开氏度 K。可以是标量或任意形状数组。

    Returns:
        最后一维大小为 3 的 RGB 数组，每个通道在 `[0, 1]`。
        当输入为标量时返回形状 `(3,)` 的数组。

    Formula:
        基于 Tanner Helland 近似：
        ```
        t = T / 100
        R(t), G(t), B(t) 由分段解析公式给出
        ```

    Physical Meaning:
        把绝对温度映射为可视化的 RGB。高温（≥ 6600 K）偏蓝白，
        低温（< 4000 K）偏红橙。`1e7 K` 输入会落在公式表达上限，
        所有通道趋近 1（紫白）；`3000 K` 输入则强偏红。

    Simplifications:
        - 这是经验拟合，不是严格普朗克黑体辐射积分。
        - 温度为 0 或负时返回全 0 RGB（不抛错，便于在盘外/边界处使用）。
    """

    T_arr = _to_array(T_K)
    safe_T = np.maximum(T_arr, 1.0)  # 避免 log/pow 在 0/负温度发散
    t = safe_T / 100.0
    rgb = _blackbody_rgb_array(t)
    # 温度 ≤ 0 时整体置 0。
    mask_pos = (T_arr > 0.0)
    rgb = rgb * mask_pos[..., None]
    return rgb.astype(np.float64)


def _rgb_saturation_boost(rgb: np.ndarray, saturation: float) -> np.ndarray:
    """对 RGB 做饱和度增强，围绕亮度 luma 保持平均亮度不变。

    Args:
        rgb: 形状 `(..., 3)` 的数组，每个通道 `[0, 1]`。
        saturation: 饱和度倍率。`1.0` 等同原图；`>1` 增强；`<1` 降饱和。

    Returns:
        与输入同形状的数组，仍在 `[0, 1]`（超出部分会被 clip）。

    Formula:
        ```
        luma = 0.2126 R + 0.7152 G + 0.0722 B    # BT.709
        out = luma + saturation · (rgb - luma)
        ```

    Notes:
        在 cinematic palette 中用于让"亮蓝"和"暗红"区域颜色更显眼。
    """

    luma = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    luma = luma[..., None]
    out = luma + saturation * (rgb - luma)
    return np.clip(out, 0.0, 1.0)


def physical_temperature_outer_K(T_peak_K: float, T_outer_over_peak: float = 1.0 / 4.32) -> float:
    """估算盘外缘 raw 物理温度，用于 cinematic 可见色温归一化。

    Args:
        T_peak_K: 中面温度峰值（K）。
        T_outer_over_peak: 外缘 raw 温度与峰值之比。默认 `1/4.32` 对应
            `r_in=3, r_out=50` 的标准薄盘剖面。

    Returns:
        外缘参考温度（K），恒为正。
    """
    return max(float(T_peak_K) * float(T_outer_over_peak), 1.0)


def cinematic_visual_temperature(
    T_K: float | np.ndarray,
    T_peak_K: float,
    params: DiskV2PaletteParams,
    *,
    T_outer_K: float | None = None,
) -> np.ndarray:
    """把物理 Kelvin 重映射到 cinematic 可见色温区间。

    Args:
        T_K: 物理温度（K），标量或数组。
        T_peak_K: 盘内 raw 温度峰值（K）。
        params: `DiskV2PaletteParams`，提供 `visual_temp_outer_K` / `visual_temp_inner_K`。
        T_outer_K: 可选外缘物理温度；默认由 `physical_temperature_outer_K` 估算。

    Returns:
        与输入广播后同形状的可见色温（K）。

    Formula:
        ```
        t_norm = clamp((log T - log T_outer) / (log T_peak - log T_outer), 0, 1)
        T_vis = T_outer_vis + t_norm · (T_inner_vis - T_outer_vis)
        ```

    Physical Meaning:
        物理温度 `1e7 K` 远超 Helland 公式有效可见范围；cinematic 模式先把
        盘内相对温度映射到可见区间再查黑体色，保证内外圈连续渐变。

    Simplifications:
        - 使用 log 温度归一化，而非径向坐标百分位。
        - 外缘物理温度默认按 `T_peak/4.32` 估算，可由调用方覆盖。
    """
    T_arr = _to_array(T_K)
    t_outer_phys = float(
        T_outer_K if T_outer_K is not None else physical_temperature_outer_K(T_peak_K)
    )
    t_peak = max(float(T_peak_K), t_outer_phys + 1.0)
    mask_pos = T_arr > 0.0
    safe_T = np.maximum(T_arr, t_outer_phys)
    log_span = np.log(t_peak) - np.log(t_outer_phys)
    t_norm = np.clip(
        (np.log(safe_T) - np.log(t_outer_phys)) / max(log_span, 1e-12),
        0.0,
        1.0,
    )
    t_vis = params.visual_temp_outer_K + t_norm * (
        params.visual_temp_inner_K - params.visual_temp_outer_K
    )
    t_vis = np.where(mask_pos, t_vis, 0.0)
    return _restore_shape(t_vis.astype(np.float64), T_K)


def cinematic_color(
    T_K: float | np.ndarray,
    params: DiskV2PaletteParams,
    *,
    T_peak_K: float | None = None,
    T_outer_K: float | None = None,
) -> np.ndarray:
    """温度 → 颜色（cinematic 模式）。

    Args:
        T_K: 温度，单位 K。
        params: `DiskV2PaletteParams`，提供 cinematic 调色与可见色温映射参数。
        T_peak_K: 物理温度峰值；cinematic 重映射需要。未传时默认 `1e7`。
        T_outer_K: 可选外缘物理温度，用于 log 归一化。

    Returns:
        最后一维大小为 3 的 RGB 数组，每个通道在 `[0, 1]`。

    Formula:
        ```
        T_vis = cinematic_visual_temperature(T_K, T_peak_K, params)
        # log-T 归一化系数（0=外缘冷低温，1=内峰高温）
        t_norm = (T_vis - visual_temp_outer_K) / (visual_temp_inner_K - visual_temp_outer_K)
        # V1 风格亮度系数：低温偏暗、高温偏亮
        value = value_low_T + t_norm · (value_high_T - value_low_T)

        base = blackbody_color(T_vis)
        saturated = saturation_boost(base, cinematic_saturation)
        warmed = saturated · [1 + warm_shift, 1, 1 - warm_shift]
        # V1 风格的"低温偏暗、高温偏亮"
        out = clip(warmed · value, 0, 1)
        ```

    Physical Meaning:
        在可见色温映射后的黑体色基础上：

        1. 增强饱和度（让红色更红、蓝色更蓝）
        2. 可选暖色偏移（默认 0；用户可显式启用）
        3. **V1 着色规格**：温度依赖的亮度系数。低温区（红橙）整体压暗，
           高温区（白）整体提亮，对齐 V1 直观感受"温度高 = 又白又亮"。

    Simplifications:
        - value 用线性插值近似 Stefan-Boltzmann T^4 在归一化空间的相对系数；
          1.71 倍跨度足够视觉可辨。
        - 用解析饱和度增强 + 通道增益，没有引入 LUT。
    """
    peak = float(T_peak_K if T_peak_K is not None else 1.0e7)
    t_vis = cinematic_visual_temperature(T_K, peak, params, T_outer_K=T_outer_K)
    rgb = blackbody_color(t_vis)
    rgb = _rgb_saturation_boost(rgb, params.cinematic_saturation)
    warm = np.array(
        [1.0 + params.cinematic_warm_shift, 1.0, 1.0 - params.cinematic_warm_shift],
        dtype=np.float64,
    )
    rgb = np.clip(rgb * warm, 0.0, 1.0)
    # V1 风格亮度系数：t_norm 从 t_vis 归一到 [0, 1] 计算
    visual_outer = float(params.visual_temp_outer_K)
    visual_inner = float(params.visual_temp_inner_K)
    visual_span = max(visual_inner - visual_outer, 1e-6)
    t_norm = np.clip((np.asarray(t_vis) - visual_outer) / visual_span, 0.0, 1.0)
    value = (
        params.cinematic_value_low_T
        + t_norm * (params.cinematic_value_high_T - params.cinematic_value_low_T)
    )
    # value 可以 > 1（高温偏亮），所以 clip 上界在外面控制
    rgb = np.clip(rgb * value[..., None], 0.0, 1.0)
    return rgb


def palette_color(
    T_K: float | np.ndarray,
    params: DiskV2PaletteParams,
    *,
    T_peak_K: float | None = None,
    T_outer_K: float | None = None,
) -> np.ndarray:
    """温度 → 颜色 RGB，按 `params.palette_mode` 选 `physical` 或 `cinematic`。

    Args:
        T_K: 温度，单位 K。
        params: `DiskV2PaletteParams`。
        T_peak_K: cinematic 模式需要的物理峰值温度（K）。
        T_outer_K: cinematic 模式可选外缘物理温度（K）。

    Returns:
        最后一维大小为 3 的 RGB 数组，每个通道在 `[0, 1]`。
    """

    if params.palette_mode == "physical":
        return blackbody_color(T_K)
    elif params.palette_mode == "cinematic":
        return cinematic_color(T_K, params, T_peak_K=T_peak_K, T_outer_K=T_outer_K)
    else:
        # 应该已经被 __post_init__ 拦截，这里再做一次防御性检查。
        raise ValueError(f"unsupported palette_mode: {params.palette_mode!r}")


def tonemap_reinhard(rgb_hdr: np.ndarray) -> np.ndarray:
    """Reinhard 简化 tonemap：`x → x / (1 + x)`。

    Args:
        rgb_hdr: 非负 HDR RGB 数组。

    Returns:
        与输入同形状的数组，落在 `[0, 1)`。

    Notes:
        - x=0 → 0；x=1 → 0.5（中调）；x=∞ → 1
        - 视觉上高光区会显著饱和（HDR p99 / white_point ≈ 1 → LDR 0.5，
          已经偏白；高于此快速饱和到 0.97~1.0）
    """
    safe = np.maximum(rgb_hdr, 0.0)
    return safe / (1.0 + safe)


# ACES Filmic (Krzysztof Narkowicz 2015) 系数。
# 拟合自 ACES RRT + ODT，逐分量近似。
# 参考: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
_ACES_A: float = 2.51
_ACES_B: float = 0.03
_ACES_C: float = 2.43
_ACES_D: float = 0.59
_ACES_E: float = 0.14


def tonemap_aces(rgb_hdr: np.ndarray) -> np.ndarray:
    """ACES Filmic tonemap (Narkowicz 2015 单变量近似)。

    Args:
        rgb_hdr: 非负 HDR RGB 数组。

    Returns:
        与输入同形状的数组，落在 `[0, ~1.033]`，再 clip 到 `[0, 1]`。

    Formula:
        ```
        x → clip(x · (a·x + b) / (x · (c·x + d) + e), 0, 1)
        a=2.51, b=0.03, c=2.43, d=0.59, e=0.14
        ```

    Notes:
        相对 Reinhard 的关键性质：

        - x=0 → 0（一致）
        - x=1 → 0.77（Reinhard 是 0.5）—— **中调更亮**
        - x=10 → 0.95（Reinhard 是 0.91）—— **高光区不饱和、仍有细节**
        - x=∞ → ~1.033（数学上限，clip 到 1）

        视觉效果：HDR 跨度大时（如本项目 V2 主验收 HDR max/p99 ≈ 100）
        ACES 让中段保留更多动态范围，而 Reinhard 会把所有中调压到 0.5 以下。
        多普勒方向性、衰减曲线、外缘软化都需要中调动态范围才能可见。
    """
    safe = np.maximum(rgb_hdr, 0.0)
    numerator = safe * (_ACES_A * safe + _ACES_B)
    denominator = safe * (_ACES_C * safe + _ACES_D) + _ACES_E
    out = numerator / np.maximum(denominator, 1e-12)
    # Narkowicz 形式的渐近上限为 a/c ≈ 1.033，clip 到 [0, 1]
    return np.clip(out, 0.0, 1.0)


def tonemap(
    rgb_hdr: np.ndarray,
    params: DiskV2PaletteParams,
) -> np.ndarray:
    """HDR → LDR 色调映射。当前仅支持 Reinhard（X1 ACES 已撤回）。

    Args:
        rgb_hdr: 任意形状的非负实数数组（最后一维一般是 3，但函数对形状不挑剔）。
            语义上是经过 V2 体积积分得到的高动态范围线性强度。
        params: `DiskV2PaletteParams`，决定 `tonemap_mode`。

    Returns:
        与输入同形状的数组，落在 `[0, 1]`。

    Formula:
        Reinhard：`x / (1 + x)`，简单稳健。

    Physical Meaning:
        把无界 HDR 强度压到 `[0, 1]` 区间，避免后处理时被硬截断。

    Notes:
        - 对负数输入：先 clip 到 0，再做映射，避免除零。
        - ACES Filmic 已在 X1 (2026-06-14) 撤回：其 x→0 时斜率≈0.21 的低值
          响应让"黑底 + 高亮"场景背景被抬亮成灰雾。函数本体 `tonemap_aces`
          保留供未来 + black pedestal 方案启用。
    """

    safe_hdr = np.maximum(_to_array(rgb_hdr), 0.0)
    if params.tonemap_mode == "reinhard":
        out = tonemap_reinhard(safe_hdr)
    else:
        # ACES 在 params.__post_init__ 已被拦截；防御性兜底。
        raise ValueError(f"unsupported tonemap_mode: {params.tonemap_mode!r}")
    return _restore_shape(out, rgb_hdr)


def apply_exposure(
    rgb_hdr: np.ndarray,
    exposure_scale: float,
) -> np.ndarray:
    """对 HDR 线性强度应用曝光缩放。

    Args:
        rgb_hdr: HDR RGB 数组或标量数组。
        exposure_scale: 曝光缩放，通常等于 `1 / white_point`。

    Returns:
        与输入同形状的曝光后 HDR 数组。
    """
    out = _to_array(rgb_hdr) * float(exposure_scale)
    return _restore_shape(out.astype(np.float64), rgb_hdr)


def gamma_correct(
    rgb_linear: np.ndarray,
    params: DiskV2PaletteParams,
) -> np.ndarray:
    """sRGB 伽马校正：把线性 RGB 转为感知空间的 RGB。

    Args:
        rgb_linear: 落在 `[0, 1]` 的线性 RGB 数组。
        params: `DiskV2PaletteParams`，提供 `gamma`。

    Returns:
        与输入同形状的数组，仍在 `[0, 1]`。

    Formula:
        ```
        out = clip(rgb_linear, 0, 1) ** (1 / gamma)
        ```

    Notes:
        - 对负输入：先 clip 到 0，再幂运算（避免负底数幂）。
        - 默认 `gamma = 2.2`。严格 sRGB 标准用 2.4 + 分段；这里用 2.2 近似。
    """

    safe_linear = np.clip(_to_array(rgb_linear), 0.0, 1.0)
    out = np.power(safe_linear, 1.0 / params.gamma)
    return _restore_shape(out, rgb_linear)


def render_hdr_to_ldr(
    rgb_hdr: np.ndarray,
    params: DiskV2PaletteParams,
) -> np.ndarray:
    """显示链路出口：HDR 线性 RGB → 经色调映射 + 伽马校正后的 LDR RGB。

    Args:
        rgb_hdr: 任意形状的非负实数数组。
        params: `DiskV2PaletteParams`。

    Returns:
        与输入同形状的数组，落在 `[0, 1]`。

    Formula:
        ```
        rgb_ldr = gamma_correct(tonemap(rgb_hdr))
        ```

    Notes:
        这是 V2 渲染管线的最后一步。Bloom 必须在调用本函数**之前**完成
        （即在 HDR 域），否则会丢失高动态范围的真实辉光感。
    """

    return gamma_correct(tonemap(rgb_hdr, params), params)


def apply_palette(
    intensity_hdr: float | np.ndarray,
    T_K: float | np.ndarray,
    params: DiskV2PaletteParams,
    *,
    T_peak_K: float | None = None,
    T_outer_K: float | None = None,
) -> np.ndarray:
    """把 HDR 强度乘上由温度决定的 palette 颜色，得到 HDR RGB。

    Args:
        intensity_hdr: 非负 HDR 强度，形状任意。语义上是 V2 体积积分对单
            一光线累积出的标量强度（或每通道强度的预先平均）。
        T_K: 与 `intensity_hdr` 广播兼容的温度数组，单位 K。
        params: `DiskV2PaletteParams`。

    Returns:
        形状为 `(..., 3)` 的 HDR RGB 数组。

    Formula:
        ```
        rgb_hdr = palette_color(T_K, params) · intensity_hdr
        ```

    Notes:
        - `palette_color` 返回值落在 `[0, 1]`；强度本身决定 HDR 量级。
        - 调用 `tonemap` / `render_hdr_to_ldr` 才会把结果压到 `[0, 1]`。
    """

    color = palette_color(T_K, params, T_peak_K=T_peak_K, T_outer_K=T_outer_K)
    intensity_arr = _to_array(intensity_hdr)
    # color 形状 (..., 3)；intensity 形状 (...)；广播相乘。
    return color * intensity_arr[..., None]
