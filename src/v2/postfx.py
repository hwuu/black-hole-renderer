"""Disk V2 后处理链（v2.3 S7）。

替代 `taichi_render.py` 中的 `_disk_tonemap_kernel` + `_bloom_kernel` + `_compose_kernel`。
全部 NumPy 实现（1080p 约 0.3 s），视频性能不足时再移到 Taichi。

链路顺序（成像原理见 `docs/imaging_model.md`）：

1. 白平衡（von Kries 增益，`palette.white_balance_gain`）
2. 镜头 PSF（默认 `lens_model="psf"`）：能量守恒的长尾散射（眩光）+ 逐通道核宽
   （轴向色差，自然产生高光蓝边），作用于全部场景光、无阈值。
   `lens_model="legacy"` 时为参考实现的"高光 bloom + 镶边"两步（会额外加光，仅作对照）。
3. 横向色散（R 通道放大、B 通道缩小的径向错位）
4. 保色度 ACES 色调映射（只对亮度做 ACES，RGB 等比缩放）
5. sRGB 编码

白平衡是逐通道增益、镜头 PSF 是逐通道线性卷积，两者可交换，因此 PSF 放在白平衡之后
与"先镜头、后传感器/ISP"的物理顺序等价。

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


def _box_blur_zero(x: np.ndarray, rad: int) -> np.ndarray:
    """三次盒式模糊（近似高斯），边界补 0。

    Args:
        x: `(H, W, C)` 数组。
        rad: 模糊半径（像素），< 1 时原样返回。

    Returns:
        与 `x` 同形状的模糊结果；总和 ≤ 输入总和（散射出画面的部分丢弃）。

    Notes:
        与 `_box_blur`（复制边缘像素补边）不同：复制边缘会把画面边缘的亮内容
        重复计入，测试图上能量放大到 1.41×；镜头散射到画面外的光应当丢失，
        因此镜头 PSF 用补 0 版本，保证能量只减不增。
    """
    if rad < 1:
        return x
    y = x
    for _ in range(3):
        for ax in (0, 1):
            pad_spec = [(rad + 1, rad) if a == ax else (0, 0) for a in range(x.ndim)]
            c = np.cumsum(np.pad(y, pad_spec, mode="constant"), axis=ax)
            n = y.shape[ax]
            hi = [slice(None)] * x.ndim
            lo = [slice(None)] * x.ndim
            hi[ax] = slice(2 * rad + 1, 2 * rad + 1 + n)
            lo[ax] = slice(0, n)
            y = (c[tuple(hi)] - c[tuple(lo)]) / (2 * rad + 1)
    return y


# 镜头眩光长尾核：(半径 = 画面高度 / d, 能量权重)。多尺度高斯叠加，权重随尺度递减，
# 近似真实镜头 PSF 尾部 ~1/r^n 的衰减；权重和为 1（核归一化）。
LENS_TAIL = ((60.0, 0.40), (20.0, 0.30), (7.0, 0.20), (3.0, 0.10))


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
        white_balance_K: 白平衡色温（K，1000–40000，黑体色度表范围）。色温为该值的黑体显示为白色；
            约 6600 K 时增益 ≈ 1；更低 → 画面更冷（中和暖光），更高 → 画面更暖。

    Returns:
        白平衡后的 HDR，`(H, W, 3)`（亮度量级基本不变）。

    Formula:
        `out_c = hdr_c · gain_c`，`gain_c = 1 / rgb_c(T_wb)`（`palette.white_balance_gain`）
    """
    return hdr * white_balance_gain(white_balance_K)[None, None, :]


# ---------------------------------------------------------------------------
# 2. 高光 bloom（轴向色散）
# ---------------------------------------------------------------------------
def apply_bloom(
    hdr: np.ndarray,
    threshold: float = 0.3,
    gain: float = 4.0,
    axial_scale: tuple[float, float, float] = (1.0, 1.0, 1.15),
    luma_threshold: bool = True,
) -> np.ndarray:
    """高光 bloom：只散射超过阈值的亮度（HDR 域）。

    逐通道不同模糊半径模拟轴向色散（绿光合焦，红蓝轻微失焦）。

    Args:
        hdr: `(H, W, 3)` 线性 HDR RGB。
        threshold: 高光阈值（线性 HDR，越低光晕越多）。
        gain: bloom 增益（散射光叠回原图的倍率）。
        axial_scale: R/G/B 通道模糊半径倍率。R、B 同时外扩会形成品红光晕，
            因此默认只让蓝光稍外扩 (1, 1, 1.15)。
        luma_threshold: 高光提取方式。True（默认）= 按亮度扣阈值并保持像素色度；
            False = 逐通道扣阈值（参考实现旧行为）。

    Returns:
        加了 bloom 的 HDR，`(H, W, 3)`，非负，≥ 输入。

    Formula:
        luma_threshold=True：  src = hdr · max(L − th, 0) / L，L = 0.2126R + 0.7152G + 0.0722B
        luma_threshold=False： src_c = max(hdr_c − th, 0)
        bloom_c = Σ_k w_k · Blur(src_c, r_k · axial_scale_c)，w = (0.25, 0.35, 0.40)
        out = hdr + gain · bloom

    Physical Meaning:
        镜头内散射只把超出阈值的那部分光扩散开；散射光与源像素同色。

    Simplifications:
        逐通道扣阈值会让 G/B 偏低的金色像素在散射光里只剩 R，把金色区染成橙红，
        因此默认改为按亮度扣阈值；三级盒式模糊近似镜头点扩散函数。
    """
    h = hdr.shape[0]
    if luma_threshold:
        # 按亮度扣阈值：超出部分按原色度散射，不改变色相
        lum = hdr @ np.array([0.2126, 0.7152, 0.0722], dtype=hdr.dtype)
        src = hdr * (np.maximum(lum - threshold, 0.0) / np.maximum(lum, 1e-12))[..., None]
    else:
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
# 2'. 镜头 PSF（默认镜头模型：能量守恒眩光 + 轴向色差）
# ---------------------------------------------------------------------------
def apply_lens_psf(
    hdr: np.ndarray,
    glare: float = 0.4,
    axial_scale: tuple[float, float, float] = (1.0, 1.0, 1.15),
    tail: tuple[tuple[float, float], ...] = LENS_TAIL,
) -> np.ndarray:
    """镜头点扩散函数（PSF）：把每个点的光按"尖锐核心 + 长尾"摊开。

    Args:
        hdr: `(H, W, 3)` 线性 HDR RGB（场景光，已曝光）。允许任意非负值，不设阈值。
        glare: 眩光强度 ε ∈ [0, 1]：每个点散射到长尾里的能量比例。0 = 理想镜头
            （原样返回）；好镜头约 0.02，柔光镜（Pro-Mist 类）约 0.2~0.4；默认 0.4。
        axial_scale: R/G/B 通道长尾核的半径倍率（轴向色差）：蓝光核略宽，高反差边缘
            外侧自然出现淡蓝边。默认 (1, 1, 1.15)。
        tail: 长尾核 `((d_k, w_k), ...)`：第 k 个高斯分量半径 = 画面高度 / d_k，
            能量权重 w_k（Σ w_k = 1）。

    Returns:
        `(H, W, 3)` 线性 HDR，非负；总能量 ≤ 输入（差额为散射到画面外的光，
        默认参数下 720p 约 1.6%）。成片大亮区内部基本不变，亮区外侧的暗处被摊入光晕。

    Formula:
        out_c = (1 − ε) · x_c + ε · (K_c ∗ x_c)
        K_c = Σ_k w_k · G(r_k · s_c)，r_k = H / d_k，s_c = axial_scale[c]，
        G(r) 为半径 r 的三次盒式模糊（≈ 高斯），边界补 0。

    Physical Meaning:
        真实镜头的成像是场景光与 PSF 的卷积：绝大部分能量落在中心一点，约 ε 的能量
        因镜片散射、灰尘、滤镜形成平滑长尾（眩光 / 辉光）。线性、能量守恒、作用于所有
        光——辉光只在背景足够暗处（天空、黑洞阴影）显著，这是对比度的结果，不需要阈值。
        蓝光核略宽对应轴向色差，替代参考实现中单独叠加的"镶边"。

    Simplifications:
        PSF 视为空间不变（横向色差另由 `apply_lateral_ca` 处理）；长尾用 4 个高斯近似；
        不模拟衍射星芒与鬼影。
    """
    if glare <= 0:
        return hdr
    h = hdr.shape[0]
    out = np.empty_like(hdr)
    for c, sc in enumerate(axial_scale):
        ch = hdr[..., c : c + 1]
        k = sum(w * _box_blur_zero(ch, max(1, int(h / d * sc))) for d, w in tail)
        out[..., c : c + 1] = (1.0 - glare) * ch + glare * k
    return out


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


def tonemap_film(
    hdr: np.ndarray,
    film_response: float = 0.0,
    white_blend: float = 0.12,
) -> np.ndarray:
    """胶片响应色调映射：保色度 ACES 与逐通道 ACES 按 `film_response` 线性混合。

    Args:
        hdr: `(H, W, 3)` 非负线性 HDR RGB（已曝光、白平衡、经过镜头）。
        film_response: 胶片响应 m ∈ [0, 1]。0 = 保色度 ACES（与 `tonemap_chroma_aces` 逐位一致）；
            1 = 三个通道各自过 ACES 曲线；中间值线性混合。
        white_blend: 传给保色度 ACES 的超色域向白混合斜率。

    Returns:
        `(H, W, 3)`，`[0, 1]` 线性 LDR，形状同输入。灰色输入在任何 m 下仍为灰色。

    Formula:
        ```
        A(x)       = x(2.51x + 0.03) / (x(2.43x + 0.59) + 0.14)       （ACES 拟合曲线，截断到 [0, 1]）
        T_chroma   = tonemap_chroma_aces(x, white_blend)               （只压亮度 L，RGB 等比缩放）
        T_channel  = (A(x_R), A(x_G), A(x_B))                          （逐通道）
        y          = (1 − m)·T_chroma + m·T_channel
        ```

    Physical Meaning:
        胶片的三层乳剂（数码传感器的 R/G/B 光电位）各有一条特性曲线、各自饱和：强通道先到肩部，
        弱通道继续增长，于是同一颜色越亮饱和度越低，最亮处趋近白色（"通往白色的路径"）。
        保色度 ACES 只压亮度，最亮处止于浅金、不会烧白；m 控制向真实胶片响应靠近的程度。

    Simplifications:
        三层共用同一条 ACES 曲线（不模拟各层感光度、趾部差异与层间串扰）；
        逐通道曲线的趾部会略微提高暗部饱和度（与胶片一致）。
    """
    m = float(film_response)
    if not 0.0 <= m <= 1.0:
        raise ValueError(f"film_response must be in [0, 1], got {film_response!r}")
    y = tonemap_chroma_aces(hdr, white_blend)
    if m == 0.0:
        return y
    x = np.maximum(hdr, 0.0)
    y_ch = np.clip((x * (_ACES_A * x + _ACES_B)) / (x * (_ACES_C * x + _ACES_D) + _ACES_E), 0.0, 1.0)
    return (1.0 - m) * y + m * y_ch


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
    """后处理默认参数。

    `lens_model="psf"`（默认）只用 `lens_glare` / `axial_scale`；`bloom_*` / `fringe_*` 仅在
    `lens_model="legacy"`（参考实现预设 M 的 bloom + 镶边）时生效。
    """
    return {
        # 白平衡色温（ISP 层）：4500 K 的黑体显示为白色（盘面浅金、最热处为白）；4000 K 偏冷发白，
        # 参考实现为 5000 K（整体偏暖黄）
        "white_balance_K": 4500.0,
        # 镜头模型："psf" = 能量守恒镜头 PSF（默认）；"legacy" = 阈值 bloom + 镶边（额外加光）
        "lens_model": "psf",
        # 眩光强度 ε：每个点散射到长尾的能量比例（柔光镜量级）
        "lens_glare": 0.5,
        "bloom_threshold": 0.3,
        "bloom_gain": 4.0,
        # True = 按亮度扣阈值（保色度，默认）；False = 逐通道扣阈值（参考实现旧行为，光晕偏红）
        "bloom_luma_threshold": True,
        "axial_scale": (1.0, 1.0, 1.15),
        "fringe_strength": 0.4,
        "fringe_threshold": 0.7,
        "fringe_color": (0.3, 0.45, 1.0),
        "lateral_ca": 0.0025,
        "white_blend": 0.12,
        # 胶片响应 m（ISP 层）：0 = 保色度 ACES（高光止于浅金）；1 = 逐通道 ACES（各层独立饱和，高光趋白）
        "film_response": 0.0,
    "saturation": 1.0,
    }


def postfx(
    hdr: np.ndarray,
    exposure: float = 1.0,
    **params,
) -> np.ndarray:
    """完整后处理链：曝光 → WB → 镜头 PSF（或 legacy：bloom + 镶边）→ 横向色散 → ACES（胶片响应）→ sRGB。

    Args:
        hdr: `(H, W, 3)` 线性 HDR RGB。
        exposure: 曝光缩放。
        **params: 覆盖 `postfx_params_defaults` 中的参数。

    Returns:
        `(H, W, 3)` uint8 RGB。
    """
    p = {**postfx_params_defaults(), **params}
    x = hdr * exposure
    x = apply_white_balance(x, p["white_balance_K"])
    if p["lens_model"] == "psf":
        x = apply_lens_psf(x, p["lens_glare"], p["axial_scale"])
    elif p["lens_model"] == "legacy":
        x = apply_bloom(x, p["bloom_threshold"], p["bloom_gain"], p["axial_scale"],
                        p["bloom_luma_threshold"])
        x = apply_fringe(x, p["fringe_strength"], p["fringe_threshold"], p["fringe_color"])
    else:
        raise ValueError(f"lens_model must be 'psf' or 'legacy', got {p['lens_model']!r}")
    x = apply_lateral_ca(x, p["lateral_ca"])
    x = tonemap_film(x, p["film_response"], p["white_blend"])
    x = adjust_saturation(x, p["saturation"])
    x = srgb_encode(x)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
