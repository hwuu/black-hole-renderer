"""Disk V2 黑体颜色与亮度（CIE 查找表）。

本模块只处理"物理量 → 像素颜色 / 亮度"的映射，不涉及物理场定义本身：

- `blackbody_color(T_K)`：温度 → 线性 sRGB(D65) 黑体色度（普朗克谱 × CIE 1931
  色匹配函数 → XYZ → sRGB，负值裁 0、BT.709 亮度归一为 1）。预计算 log T 等
  间距查找表（v2.3 之前用 Tanner Helland 分段拟合——其输出是 sRGB 编码值，
  被当线性光使用会叠加两次伽马、白平衡下易偏青/品红）。
- `blackbody_luminance(T_K)`：温度 → 可见光亮度 `Y(T) = ∫ B_λ(T)·ȳ(λ) dλ`，
  log T 间距查找表。观测谱为温度 g·T 的黑体（I_ν/ν³ 洛伦兹不变），因此频移
  后的观测亮度就是 `Y(g·T)`，替代旧的 `(T/T_peak)^p` 与 550 nm 单色近似。
- `white_balance_gain(T_wb)`：von Kries 白平衡增益，使色温 T_wb 的黑体呈精确
  中性 (1,1,1)；不保证其他温度黑体的 BT.709 亮度（von Kries 固有属性）。

Notes:
    所有函数都是纯 NumPy 实现，作为参考实现。Taichi 端
    （`taichi_impl.blackbody_color_ti` 等）以同一张查找表做 parity。
"""

from __future__ import annotations

import math

import numpy as np

from ._array_utils import _to_array

# ---------------------------------------------------------------------------
# CIE 黑体查找表（模块级常量，NumPy 与 Taichi 共用）
# ---------------------------------------------------------------------------
_BB_LUT_N: int = 512
"""黑体查找表项数（log T 等间距）。"""

_BB_T_MIN_K: float = 1000.0
_BB_T_MAX_K: float = 40000.0
"""色度表温度范围（K）。盘内有效色温（T_peak ≈ 4500 K × g_cap）都在范围内。"""

_LNY_T_MIN_K: float = 300.0
_LNY_T_MAX_K: float = 60000.0
"""亮度表温度范围（K）。需要覆盖红移后的低温（g < 1）。"""

_LAM_NM = np.linspace(380.0, 780.0, 801)
"""CIE 积分波长网格（nm）。"""


def _cie_cmf(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CIE 1931 2° 色匹配函数（Wyman, Sloan & Shirley 2013 多瓣高斯解析近似）。

    Args:
        lam: 波长（nm）。

    Returns:
        `(x̄, ȳ, z̄)`，与 `lam` 同形状，均 ≥ 0。

    Formula:
        每个通道是若干 `exp(−0.5((λ−μ)/σ)²)` 的加权和，σ 在 μ 两侧取不同值。
    """
    def g(x, mu, s1, s2):
        sig = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / sig) ** 2)

    xb = (1.056 * g(lam, 599.8, 37.9, 31.0) + 0.362 * g(lam, 442.0, 16.0, 26.7)
          - 0.065 * g(lam, 501.1, 20.4, 26.2))
    yb = 0.821 * g(lam, 568.8, 46.9, 40.5) + 0.286 * g(lam, 530.9, 16.3, 31.1)
    zb = 1.217 * g(lam, 437.0, 11.8, 36.0) + 0.681 * g(lam, 459.0, 26.0, 13.8)
    return xb, yb, zb


_XYZ_TO_SRGB_D65 = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])
"""XYZ → 线性 sRGB(D65) 矩阵。"""

_BT709 = np.array([0.2126, 0.7152, 0.0722])
"""BT.709 亮度权重。"""


def _planck(lam: np.ndarray, t_k: float) -> np.ndarray:
    """普朗克谱 `B_λ ∝ λ⁻⁵ / (exp(hc/(λkT)) − 1)`（任意单位）。

    Args:
        lam: 波长（nm）。
        t_k: 温度（K），> 0。

    Returns:
        与 `lam` 同形状的谱强度。`hc/(λk)` 用 nm-K 标度 `1.4388e7`。
    """
    x = np.minimum(1.4388e7 / (lam * t_k), 700.0)
    return lam ** -5 / np.expm1(x)


def _blackbody_rgb_exact(t_k: float) -> np.ndarray:
    """单温度黑体的线性 sRGB(D65) 色度（直接数值积分，用于构建查找表）。

    Args:
        t_k: 温度（K）。

    Returns:
        形状 `(3,)` 的 RGB，每通道 ≥ 0、BT.709 亮度 = 1。黑体轨迹极低温端
        小段在 sRGB 色域外（G 为负），裁 0。

    Formula:
        ```
        XYZ = ∫ B_λ(T)·(x̄, ȳ, z̄) dλ；RGB = M_sRGB · XYZ；
        RGB ← RGB / (w·RGB)，负通道裁 0 后重归一。
        ```
    """
    xb, yb, zb = _cie_cmf(_LAM_NM)
    spec = _planck(_LAM_NM, t_k)
    xyz = np.array([(spec * xb).sum(), (spec * yb).sum(), (spec * zb).sum()])
    rgb = _XYZ_TO_SRGB_D65 @ xyz
    rgb = np.maximum(rgb, 0.0)
    return rgb / max(float(rgb @ _BT709), 1e-30)


def _blackbody_luminance_exact(t_k: float) -> float:
    """单温度黑体的可见光亮度 `Y(T) = ∫ B_λ(T)·ȳ dλ`（任意单位，> 0）。

    Args:
        t_k: 温度（K），> 0。

    Returns:
        标量亮度。与色度无关的独立量，决定"多亮"；观测时被频移缩放。
    """
    _, yb, _ = _cie_cmf(_LAM_NM)
    return float((_planck(_LAM_NM, t_k) * yb).sum())


def _build_bb_lut() -> tuple[np.ndarray, np.ndarray]:
    """构建黑体查找表（模块导入时执行一次）。

    Returns:
        `(bb_lut, lny_lut)`：
        - `bb_lut`：`(_BB_LUT_N, 3)` 色度表，log T ∈ [log 1000, log 40000] 等间距；
        - `lny_lut`：`(_BB_LUT_N,)` 的 `ln Y(T)` 表，log T ∈ [log 300, log 60000] 等间距。
    """
    ts_color = np.exp(np.linspace(math.log(_BB_T_MIN_K), math.log(_BB_T_MAX_K), _BB_LUT_N))
    bb_lut = np.stack([_blackbody_rgb_exact(t) for t in ts_color]).astype(np.float64)
    ts_lum = np.exp(np.linspace(math.log(_LNY_T_MIN_K), math.log(_LNY_T_MAX_K), _BB_LUT_N))
    lny_lut = np.array(
        [math.log(max(_blackbody_luminance_exact(t), 1e-300)) for t in ts_lum]
    )
    return bb_lut, lny_lut


BB_LUT, LNY_LUT = _build_bb_lut()
"""模块级查找表；`taichi_impl.DiskV2Taichi.__init__` 把它们上传为 Taichi field。"""


def _lut_lookup(lut: np.ndarray, t_k: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    """log T 等间距查找表的线性插值。

    Args:
        lut: `(N,)` 或 `(N, 3)` 查找表。
        t_k: 温度（K），标量或数组；越界按端点裁剪。
        t_min, t_max: 表的温度范围（K）。

    Returns:
        与 `t_k` 广播后的插值结果；`lut` 为 `(N, 3)` 时形状为 `(..., 3)`。
    """
    t_arr = _to_array(t_k)
    u = (np.log(np.clip(t_arr, t_min, t_max)) - math.log(t_min)) / (math.log(t_max) - math.log(t_min))
    f = u * (_BB_LUT_N - 1)
    i0 = np.clip(np.floor(f).astype(np.int64), 0, _BB_LUT_N - 2)
    w = (f - i0)[..., None] if lut.ndim == 2 else (f - i0)
    if lut.ndim == 2:
        return lut[i0] * (1.0 - w) + lut[i0 + 1] * w
    return lut[i0] * (1.0 - w) + lut[i0 + 1] * w


def blackbody_color(
    T_K: float | np.ndarray,
) -> np.ndarray:
    """温度 → 黑体色度（线性 sRGB D65，BT.709 亮度归一为 1）。

    Args:
        T_K: 温度，单位开氏度 K。可以是标量或任意形状数组。

    Returns:
        最后一维大小为 3 的 RGB 数组，每通道 ≥ 0、亮度 = 1。
        `T ≤ 0`（盘外 / 边界）返回全 0 RGB，不抛错。标量输入返回 `(3,)`。

    Physical Meaning:
        把黑体谱经人眼（CIE 1931 色匹配）映射到显示器色度空间；
        只描述"什么颜色"，不描述"多亮"（亮度见 `blackbody_luminance`）。

    Simplifications:
        - CIE 匹配函数用 Wyman 2013 解析近似，非查表精确值。
        - XYZ → sRGB 后负通道裁 0（黑体轨迹极低温端略超 sRGB 色域）。
        - 512 项 log T 查找表线性插值（相对直接积分误差 < 3%）。
    """
    T_arr = _to_array(T_K)
    scalar = np.ndim(T_K) == 0
    rgb = np.zeros(T_arr.shape + (3,), dtype=np.float64)
    mask = T_arr > 0.0
    if np.any(mask):
        rgb[mask] = _lut_lookup(BB_LUT, T_arr[mask], _BB_T_MIN_K, _BB_T_MAX_K)
    if scalar:
        return rgb.reshape(3)
    return rgb


def blackbody_luminance(
    T_K: float | np.ndarray,
) -> float | np.ndarray:
    """温度 → 黑体可见光亮度 `Y(T) = ∫ B_λ(T)·ȳ(λ) dλ`（任意单位）。

    Args:
        T_K: 温度（K），标量或数组。

    Returns:
        `Y(T)`，≥ 0；`T ≤ 0` 返回 0。标量输入返回 float。
        频移后的观测亮度即 `Y(g·T)`（观测谱仍是黑体，温度 g·T）。

    Physical Meaning:
        "多亮"的物理量。替代旧 `(T/T_peak)^p` 旋钮（4500 K 盘在可见光处于
        Wien 指数段，Y 随温度的等效指数 ≈ hν̄/kT ≈ 6，不是固定 p）。

    Simplifications:
        - 512 项 ln Y 查找表线性插值（相对直接积分误差 < 1%）。
        - 单一 ȳ 积分代替全光谱 RGB 加权（亮度与色度分离）。
    """
    T_arr = _to_array(T_K)
    out = np.zeros(T_arr.shape, dtype=np.float64)
    mask = T_arr > 0.0
    if np.any(mask):
        ln_y = _lut_lookup(LNY_LUT, T_arr[mask], _LNY_T_MIN_K, _LNY_T_MAX_K)
        out[mask] = np.exp(ln_y)
    if np.ndim(T_K) == 0:
        return float(out)
    return out


def white_balance_gain(
    T_wb: float,
) -> np.ndarray:
    """von Kries 白平衡增益：让色温 `T_wb` 的黑体呈中性，BT.709 亮度不变。

    Args:
        T_wb: 相机白平衡色温（K）。6600 K ≈ 表的中性白点，增益接近 1。

    Returns:
        形状 `(3,)` 的 RGB 增益，全为正。渲染时对 HDR 线性 RGB 逐通道相乘。

    Formula:
        ```
        gain_c = 1 / rgb_c(T_wb)
        ```

    Physical Meaning:
        相机按场景色温设定白点：色温 T_wb 的黑体经增益后成为精确中性 (1,1,1)，
        低于该色温的暖色被中和、相对冷暖差更易辨认。这是标准 von Kries 对角
        变换；它保持锥响应，**不**逐谱保持 BT.709 亮度（其他温度的黑体平衡后
        luma 有百分之几的物理性漂移），曝光由 `postfx` 单独控制。
    """
    c = np.asarray(blackbody_color(float(T_wb)))
    return 1.0 / np.maximum(c, 1e-3)
