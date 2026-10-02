"""Disk V2 后处理链（v2.3 S7）。

替代 `taichi_render.py` 中的 `_disk_tonemap_kernel` + `_bloom_kernel` + `_compose_kernel`。
全部 NumPy 实现（1080p 约 0.3 s），视频性能不足时再移到 Taichi。

链路顺序（与参考实现一致）：

1. 白平衡（von Kries 增益，`palette.white_balance_gain`）
2. 高光 bloom（HDR 域，只散射超过阈值的亮度；轴向色散 = 逐通道模糊半径缩放）
3. 镶边（饱和高光外缘的蓝紫失焦环）
4. 横向色散（R 通道放大、B 通道缩小的径向错位）
5. 保色度 ACES 色调映射（只对亮度做 ACES，RGB 等比缩放）
6. sRGB 编码

所有函数接受 `(H, W, 3)` 线性 HDR RGB，返回同形状。
"""

from __future__ import annotations

import numpy as np

from .palette import white_balance_gain


def hdr_luminance(rgb_hdr: np.ndarray) -> np.ndarray:
    """计算 HDR RGB 的 BT.709 亮度。

    Args:
        rgb_hdr: 形状 `(..., 3)` 的非负线性 HDR RGB。

    Returns:
        与输入去掉最后一维同形状的亮度标量场（float64，≥ 0）。

    Formula:
        `L = 0.2126·R + 0.7152·G + 0.0722·B`
    """
    rgb = np.asarray(rgb_hdr, dtype=np.float64)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

# ---------------------------------------------------------------------------
# 盒式模糊（近似高斯）
# ---------------------------------------------------------------------------
def _box_blur(x: np.ndarray, rad: int) -> np.ndarray:
    """三次盒式模糊（近似高斯）。`rad` 为每侧像素数。

    Args:
        x: `(H, W, C)` 数组。
        rad: 模糊半径（像素），< 1 时原样返回。

    Returns:
        与 `x` 同形状的模糊结果。
    """
    if rad < 1:
        return x
    y = x
    for _ in range(3):
        for ax in (0, 1):
            pad_spec = [(rad + 1, rad) if a == ax else (0, 0) for a in range(x.ndim)]
            c = np.cumsum(np.pad(y, pad_spec, mode="edge"), axis=ax)
            n = y.shape[ax]
            hi = [slice(None)] * x.ndim
            lo = [slice(None)] * x.ndim
            hi[ax] = slice(2 * rad + 1, 2 * rad + 1 + n)
            lo[ax] = slice(0, n)
            y = (c[tuple(hi)] - c[tuple(lo)]) / (2 * rad + 1)
    return y


# ---------------------------------------------------------------------------
# 1. 白平衡
# ---------------------------------------------------------------------------
def apply_white_balance(
    hdr: np.ndarray,
    white_balance_K: float,
) -> np.ndarray:
    """对线性 HDR RGB 应用 von Kries 白平衡增益。

    Args:
        hdr: `(H, W, 3)` 线性 HDR RGB。
        white_balance_K: 白平衡色温（K）。6600 K ≈ 中性白点，增益 ≈ 1。

    Returns:
        白平衡后的 HDR（亮度量级不变）。
    """
    if white_balance_K >= 6599.0:
        return hdr
    return hdr * white_balance_gain(white_balance_K)[None, None, :]


def compute_auto_wb_gain(
    hdr: np.ndarray,
    alpha: np.ndarray | None = None,
    percentile: float = 99.0,
    warm_bias: float = 0.0,
) -> np.ndarray:
    """从 HDR 自动估计 von Kries 白平衡增益。

    取盘区（或全图）亮度前 `percentile`% 的像素，计算平均色度，
    返回使该色度呈精确中性的增益（BT.709 亮度不变）。

    Args:
        hdr: `(H, W, 3)` 线性 HDR RGB。
        alpha: 可选 `(H, W)` 盘区不透明度 mask（> 0.1 视为盘区）。
        percentile: 亮度分位数（只取最亮的前 x% 像素做色度参考）。
        warm_bias: 暖色偏移 `[0, 1]`。0 = 完全中性；> 0 让白点偏暖
            （R 增益 × (1+bias)、B 增益 × (1−bias)，模拟"Interstellar 金色"）。

    Returns:
        `(3,)` von Kries 增益（BT.709 亮度不变）。
    """
    w = np.array([0.2126, 0.7152, 0.0722])
    lum = hdr @ w
    mask = lum > 1e-10
    if alpha is not None:
        mask &= alpha > 0.1
    if not mask.any():
        return np.ones(3)
    v = lum[mask]
    if v.size < 10:
        return np.ones(3)
    bright = lum >= np.percentile(v, percentile)
    m = mask & bright
    if not m.any():
        m = mask
    mean_rgb = hdr[m].mean(0)
    lum_mean = float(mean_rgb @ w)
    if lum_mean < 1e-10:
        return np.ones(3)
    # 归一化到亮度=1（与 blackbody_color 同约定）
    c = mean_rgb / lum_mean
    gain = 1.0 / np.maximum(c, 1e-3)
    # 暖色偏移：R 增益提高、B 增益降低（保持亮度不变的重归一）
    if warm_bias > 0:
        gain = gain * np.array([1.0 + warm_bias, 1.0, 1.0 - warm_bias])
    # BT.709 亮度不变归一
    return gain * lum_mean / float(gain @ mean_rgb)


# ---------------------------------------------------------------------------
# 2. 高光 bloom（轴向色散）
# ---------------------------------------------------------------------------
def apply_bloom(
    hdr: np.ndarray,
    threshold: float = 0.3,
    gain: float = 4.0,
    axial_scale: tuple[float, float, float] = (1.0, 1.0, 1.15),
) -> np.ndarray:
    """高光 bloom：只散射超过阈值的亮度（HDR 域）。

    逐通道不同模糊半径模拟轴向色散（绿光合焦，红蓝轻微失焦）。

    Args:
        hdr: `(H, W, 3)` 线性 HDR RGB。
        threshold: 高光阈值（越低光晕越多）。
        gain: bloom 增益。
        axial_scale: R/G/B 通道模糊半径倍率。R、B 同时外扩会形成品红光晕，
            因此默认只让蓝光稍外扩 (1, 1, 1.15)。

    Returns:
        加了 bloom 的 HDR。
    """
    h = hdr.shape[0]
    src = np.maximum(hdr - threshold, 0.0)
    bloom = np.zeros_like(hdr)
    for c in range(3):
        sc = axial_scale[c]
        ch = src[..., c : c + 1]
        bloom[..., c : c + 1] = (
            0.25 * _box_blur(ch, max(1, int(h / 120 * sc)))
            + 0.35 * _box_blur(ch, max(2, int(h / 25 * sc)))
            + 0.40 * _box_blur(ch, max(4, int(h / 7 * sc)))
        )
    return hdr + gain * bloom


# ---------------------------------------------------------------------------
# 3. 镶边（高光外缘偏蓝失焦环）
# ---------------------------------------------------------------------------
def apply_fringe(
    hdr: np.ndarray,
    fringe_strength: float = 0.4,
    fringe_threshold: float = 0.7,
    fringe_color: tuple[float, float, float] = (0.3, 0.45, 1.0),
) -> np.ndarray:
    """饱和高光外缘的蓝紫失焦环。

    只作用于超过 `fringe_threshold` 的亮度区域的外缘。

    Args:
        hdr: `(H, W, 3)` 线性 HDR RGB。
        fringe_strength: 强度，0 关闭。
        fringe_threshold: 高光阈值。
        fringe_color: 镶边颜色（蓝紫为主）。

    Returns:
        加了镶边的 HDR。
    """
    if fringe_strength <= 0:
        return hdr
    h = hdr.shape[0]
    lum = hdr @ np.array([0.2126, 0.7152, 0.0722])
    hs = np.maximum(lum - fringe_threshold, 0.0)[..., None]
    ring = np.maximum(
        _box_blur(hs, max(2, h // 90)) - _box_blur(hs, max(1, h // 400)), 0.0
    )
    return hdr + fringe_strength * ring * np.asarray(fringe_color, dtype=hdr.dtype)


# ---------------------------------------------------------------------------
# 4. 横向色散
# ---------------------------------------------------------------------------
def apply_lateral_ca(
    hdr: np.ndarray,
    strength: float = 0.0025,
) -> np.ndarray:
    """横向色散：R 通道放大、B 通道缩小的径向错位。

    Args:
        hdr: `(H, W, 3)` 线性 HDR RGB。
        strength: 径向缩放比例（0 关闭）。

    Returns:
        色散后的 HDR。
    """
    if strength <= 0:
        return hdr
    h, w, _ = hdr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    out = hdr.copy()
    for c, scale in ((0, 1.0 + strength), (2, 1.0 - strength)):
        sy = cy + (yy - cy) / scale
        sx = cx + (xx - cx) / scale
        y0 = np.clip(np.floor(sy).astype(int), 0, h - 2)
        x0 = np.clip(np.floor(sx).astype(int), 0, w - 2)
        fy = np.clip(sy - y0, 0, 1)
        fx = np.clip(sx - x0, 0, 1)
        ch = hdr[..., c]
        out[..., c] = (
            (1 - fy) * ((1 - fx) * ch[y0, x0] + fx * ch[y0, x0 + 1])
            + fy * ((1 - fx) * ch[y0 + 1, x0] + fx * ch[y0 + 1, x0 + 1])
        )
    return out


# ---------------------------------------------------------------------------
# 5. 保色度 ACES 色调映射
# ---------------------------------------------------------------------------
_ACES_A = 2.51
_ACES_B = 0.03
_ACES_C = 2.43
_ACES_D = 0.59
_ACES_E = 0.14


def tonemap_chroma_aces(
    hdr: np.ndarray,
    white_blend: float = 0.12,
) -> np.ndarray:
    """保色度 ACES 色调映射。

    只对亮度做 ACES，RGB 等比缩放；超出色域的通道按比例收回并向白少量混合
    （模拟传感器高光饱和）。

    Args:
        hdr: `(H, W, 3)` 非负线性 HDR RGB。
        white_blend: 超色域向白混合斜率（0 = 纯色度保持，越大高光越白）。

    Returns:
        `(H, W, 3)`，`[0, 1]` 线性 LDR。
    """
    lum = hdr @ np.array([0.2126, 0.7152, 0.0722])
    lum_t = (lum * (_ACES_A * lum + _ACES_B)) / (lum * (_ACES_C * lum + _ACES_D) + _ACES_E)
    y = hdr * (lum_t / np.maximum(lum, 1e-6))[..., None]
    m = y.max(-1, keepdims=True)
    w_white = np.clip((m - 1.0) * white_blend, 0.0, 0.25)
    y = y / np.maximum(m, 1.0)
    return np.clip(y * (1 - w_white) + w_white, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 5.5 饱和度调整（ACES 后、sRGB 前）
# ---------------------------------------------------------------------------
def adjust_saturation(rgb: np.ndarray, saturation: float = 1.0) -> np.ndarray:
    """调整色彩饱和度（围绕 BT.709 亮度）。

    Args:
        rgb: `(H, W, 3)` RGB，`[0, 1]` 线性。
        saturation: 1.0 = 不变；< 1 降饱和（→金色/中性）；> 1 增饱和。

    Returns:
        调整后的 RGB，`[0, 1]`。
    """
    if saturation == 1.0:
        return rgb
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    return np.clip(lum[..., None] + saturation * (rgb - lum[..., None]), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 6. sRGB 编码
# ---------------------------------------------------------------------------
def srgb_decode(x: np.ndarray) -> np.ndarray:
    """sRGB 编码值 → 线性光（IEC 61966-2-1，`srgb_encode` 的逆）。

    Args:
        x: sRGB 编码 RGB，`[0, 1]`，任意形状。

    Returns:
        同形状线性 RGB，`[0, 1]`。

    Formula:
        `x ≤ 0.04045：x/12.92`；否则 `((x + 0.055)/1.055)^2.4`

    Physical Meaning:
        天空盒 PNG（与程序星空）存的是 sRGB 编码值，必须先解码成线性光才能与盘的
        线性 HDR 相加并参与 bloom / 色调映射，否则相当于叠加两次伽马。
    """
    return np.where(x <= 0.04045, x / 12.92, np.power((np.clip(x, 0, 1) + 0.055) / 1.055, 2.4))


def srgb_encode(x: np.ndarray) -> np.ndarray:
    """线性 RGB → sRGB 编码。

    Args:
        x: `(H, W, 3)` 线性 RGB，`[0, 1]`。

    Returns:
        sRGB 编码后的 RGB，`[0, 1]`。
    """
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(np.clip(x, 0, 1), 1 / 2.4) - 0.055,
    )


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def postfx_params_defaults() -> dict:
    """S7 后处理默认参数（与参考实现预设 M 一致）。"""
    return {
        "white_balance_K": 5000.0,
        "bloom_threshold": 0.3,
        "bloom_gain": 4.0,
        "axial_scale": (1.0, 1.0, 1.15),
        "fringe_strength": 0.4,
        "fringe_threshold": 0.7,
        "fringe_color": (0.3, 0.45, 1.0),
        "lateral_ca": 0.0025,
        "white_blend": 0.12,
    "saturation": 1.0,
    }


def postfx(
    hdr: np.ndarray,
    exposure: float = 1.0,
    auto_wb: bool = False,
    auto_wb_warm_bias: float = 0.0,
    **params,
) -> np.ndarray:
    """完整后处理链：曝光 → WB → bloom → 镶边 → 色散 → ACES → sRGB。

    Args:
        hdr: `(H, W, 3)` 线性 HDR RGB。
        exposure: 曝光缩放。
        auto_wb: True 时从 HDR 自动估计白平衡（忽略 `white_balance_K`）。
        auto_wb_warm_bias: 自动 WB 的暖色偏移 `[0, 1]`（0 = 中性；0.1 ≈ Interstellar 金）。
        **params: 覆盖 `postfx_params_defaults` 中的参数。

    Returns:
        `(H, W, 3)` uint8 RGB。
    """
    p = {**postfx_params_defaults(), **params}
    x = hdr * exposure
    if auto_wb:
        gain = compute_auto_wb_gain(x, percentile=99.0, warm_bias=auto_wb_warm_bias)
        x = x * gain[None, None, :]
    else:
        x = apply_white_balance(x, p["white_balance_K"])
    x = apply_bloom(x, p["bloom_threshold"], p["bloom_gain"], p["axial_scale"])
    x = apply_fringe(x, p["fringe_strength"], p["fringe_threshold"], p["fringe_color"])
    x = apply_lateral_ca(x, p["lateral_ca"])
    x = tonemap_chroma_aces(x, p["white_blend"])
    x = adjust_saturation(x, p["saturation"])
    x = srgb_encode(x)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
