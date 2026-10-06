"""小尺度温度湍流：剪切级联向小尺度延伸的温度起伏（几何构造、权重与 NumPy 参考实现）。

相机贴近盘面时，近处云层沿视线被平均成均匀的雾。本模块在主云剪切级联最细八度之后再延伸
E 个八度，只调制温度、不调制密度：

```
tf_c ← tf_c·f_T，f_T = exp(σ_T·n_T − 2σ_T²·V)          （⟨f_T⁴⟩ = 1，平均热辐射通量守恒）
n_T  = Σ_e a_e·w_e·ν_e / N，V = Σ_e (a_e·w_e)² / Σ_e a_e²
```

各八度按像素足迹做频率钳制（比像素小的八度淡出为其均值 0），经过强透镜的光线按偏折角淡出。
模型、推导与原型实测见 `docs/plans/v2_temperature_turbulence_plan.md`；
Taichi 实现见 `taichi_impl.DiskV2Taichi._temp_turb_*`，与本模块逐值一致。
"""

import math
from typing import Callable, Tuple

import numpy as np

from .shear_cascade import ShearCascadeGeometry, shear_cascade_geometry

# 径向噪声坐标的原点 ln 20：使最细八度（K ≈ 1620）在 r ∈ [3, 30] 内的坐标保持在 10³ 量级，
# 避免单精度插值台阶形成条纹（方案 §3.5）。原点是世界常数，与相机无关。
LN_R_ANCHOR: float = math.log(20.0)
# 固定坐标偏移：使温度起伏与主云（共用刚体环带种子 ox、oz）以及各八度之间去相关
U_OFFSET: float = 1000.0
W_OFFSET: float = 23.0
W_OFFSET_STEP: float = 5.0
# 延伸八度数上限：E = 3 时最细八度方位周期约 1.2×10⁴，接近单精度台阶出现条纹的量级（方案 §3.5）
MAX_OCTAVES: int = 2
# 像素足迹的路径长度下限（r_s）：只用于频率钳制与步长控制，使相机附近 λ → 0 时足迹不退化为 0；
# 不影响光线积分本身的路径长度
MIN_PATH_LENGTH: float = 1.0e-3


def smoothstep01(s):
    """截断到 [0, 1] 的三次平滑阶跃 `3s² − 2s³`。

    Args:
        s: 标量或数组，任意实数。

    Returns:
        与输入同形状，值域 [0, 1]；s ≤ 0 为 0，s ≥ 1 为 1，中间单调递增且一阶导数连续。

    Formula:
        `sstep(s) = 3t² − 2t³`，`t = clamp(s, 0, 1)`
    """
    t = np.clip(s, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def temp_turb_geometry(shear_k0: float, shear_octaves: int, shear_ar_small: float, shear_tilt_k: float,
                       n_octaves: int, disk_spin: float) -> ShearCascadeGeometry:
    """温度湍流延伸八度的噪声坐标系数（接在主云剪切级联最细八度之后）。

    Args:
        shear_k0: 主云剪切级联最大八度的 `ln r` 频率（格 / 单位 ln r，> 0）。
        shear_octaves: 主云剪切级联八度数（≥ 1）。
        shear_ar_small: 主云最细八度的长宽比（≥ 1）；延伸八度统一取该值。
        shear_tilt_k: 拖尾倾角系数（≥ 0），与主云相同。
        n_octaves: 延伸八度数 E（1 ≤ E ≤ `MAX_OCTAVES`）。
        disk_spin: 盘旋转方向（+1 / −1）。

    Returns:
        `ShearCascadeGeometry`，各字段长度为 E；第 e 个八度的频率为 `shear_k0·3^(shear_octaves + e)`。

    Raises:
        ValueError: `n_octaves` 越界；其余参数由 `shear_cascade_geometry` 校验。

    Formula:
        `shear_cascade_geometry(shear_k0·3^shear_octaves, E, shear_ar_small, shear_ar_small, shear_tilt_k, spin)`

    Physical Meaning:
        湍流级联继续向小尺度延伸，团块形状沿用开普勒剪切取形规律。

    Simplifications:
        延伸八度的长宽比不再随尺度变化（统一取主云最细八度的长宽比）。
    """
    if not 1 <= n_octaves <= MAX_OCTAVES:
        raise ValueError(f"n_octaves must be in [1, {MAX_OCTAVES}]")
    k_first = float(shear_k0) * 3.0 ** int(shear_octaves)
    return shear_cascade_geometry(k_first, n_octaves, shear_ar_small, shear_ar_small, shear_tilt_k, disk_spin)


def temp_turb_gains(n_octaves: int, gain: float) -> Tuple[float, ...]:
    """延伸八度的幅度 `a_e = gain^e`（a_0 = 1）。

    Args:
        n_octaves: 延伸八度数 E（1 ≤ E ≤ `MAX_OCTAVES`）。
        gain: 逐八度幅度比（(0, 1]）。

    Returns:
        长度为 E 的元组，首项为 1，逐项乘 `gain`，值域 (0, 1]。

    Raises:
        ValueError: `n_octaves` 或 `gain` 越界。

    Formula:
        `a_e = gain^e`，e = 0 … E−1

    Physical Meaning:
        湍流起伏幅度随尺度减小而减弱；默认 0.69 ≈ 3^(−1/3) 为 Kolmogorov 标度（尺度缩小到 1/3）。

    Simplifications:
        幅度比与尺度无关（纯幂律）。
    """
    if not 1 <= n_octaves <= MAX_OCTAVES:
        raise ValueError(f"n_octaves must be in [1, {MAX_OCTAVES}]")
    if not 0.0 < gain <= 1.0:
        raise ValueError("gain must be in (0, 1]")
    return tuple(float(gain) ** e for e in range(n_octaves))


def temp_turb_lens_weight(delta_rad, lens_rad: float):
    """透镜权重 L：光线累计偏折角越大，像素足迹的直线近似越不可信，延伸八度越弱。

    Args:
        delta_rad: 累计偏折角 δ（rad，≥ 0），即当前光线方向与离开相机时方向的夹角；标量或数组。
        lens_rad: 淡出角 δ₀（rad，> 0）。

    Returns:
        与 `delta_rad` 同形状，值域 [0, 1]：δ ≤ δ₀/2 时为 1，δ ≥ δ₀ 时为 0，中间单调递减。

    Formula:
        `L = 1 − sstep((δ − δ₀/2) / (δ₀/2))`

    Physical Meaning:
        强透镜把一个像素映射到气体中大得多的面积，真实足迹远大于 λθ；淡出细八度避免锯齿与亮斑。

    Simplifications:
        用累计偏折角代替完整的光线映射（工程近似，阈值为经验值）。
    """
    half = 0.5 * float(lens_rad)
    return 1.0 - smoothstep01((np.asarray(delta_rad, dtype=np.float64) - half) / half)


def temp_turb_clamp_weights(r, foot, lens_w, geom: ShearCascadeGeometry, clamp_px: float) -> np.ndarray:
    """各延伸八度的钳制权重 `w_e = L·sstep((c_e/F − K)/K)`。

    Args:
        r: 盘局部柱坐标半径（r_s，> 0）；标量或数组。
        foot: 像素足迹 F（r_s，> 0；调用方保证，渲染器取 `max(λ, MIN_PATH_LENGTH)·θ`），与 `r` 可广播。
        lens_w: 透镜权重 L（[0, 1]），与 `r` 可广播。
        geom: `temp_turb_geometry` 的返回值。
        clamp_px: 钳制阈值 K（> 0，格宽 / 足迹；取自已校验的 `DiskV2VolumeParams.temp_turb_clamp_px`）。

    Returns:
        形状为 `broadcast(r, foot, lens_w).shape + (E,)` 的数组，值域 [0, 1]：
        格宽 ≤ K 个足迹时为 0，≥ 2K 个足迹时为 L。

    Formula:
        `c_e = r / (|N₀₀|_e·K_e)`（径向噪声坐标 `|N₀₀|_e·K_e·ln r` 的一格换算成 r_s）

    Physical Meaning:
        频率钳制：比像素足迹还小的八度用其统计均值 0 代替，气体场本身与相机无关。

    Simplifications:
        只按径向格宽判断（方位格宽为径向的 `shear_ar_small` 倍，更晚淡出）；有限足迹内的局部平均视为 0。
    """
    r_, f_, l_ = np.broadcast_arrays(np.asarray(r, dtype=np.float64), np.asarray(foot, dtype=np.float64),
                                     np.asarray(lens_w, dtype=np.float64))
    k = float(clamp_px)
    out = np.empty(r_.shape + (len(geom.kr),))
    for e in range(len(geom.kr)):
        cell = r_ / (geom.u_scale[e] * geom.kr[e])
        out[..., e] = l_ * smoothstep01((cell / f_ - k) / k)
    return out


def temp_turb_variance(weights, gains: Tuple[float, ...]):
    """钳制后 n_T 的方差 `V = Σ_e (a_e·w_e)² / Σ_e a_e²`。

    Args:
        weights: 钳制权重，最后一维长度为 E（`temp_turb_clamp_weights` 的返回值）。
        gains: 八度幅度 `a_e`（`temp_turb_gains` 的返回值）。

    Returns:
        形状为 `weights.shape[:-1]` 的数组，值域 [0, 1]；全部 w_e = 1 时为 1，全部为 0 时为 0。

    Formula:
        各八度噪声近似独立、方差同为 σ_ν² 时，`N² = σ_ν²·Σa_e²`，
        `Var(Σ a_e·w_e·ν_e) / N² = Σ(a_e·w_e)² / Σa_e² = V`。

    Physical Meaning:
        频率钳制后像素内仍可分辨的温度起伏占全尺度起伏的方差比例；用于温度倍率的通量守恒补偿。

    Simplifications:
        各八度视为独立、同方差（单测检查混合后的样本方差与 V 一致）。
    """
    a = np.asarray(gains, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return ((a * w) ** 2).sum(axis=-1) / (a * a).sum()


def temp_turb_temperature(sigma: float, n, var):
    """温度湍流的温度倍率 `f_T = exp(σ_T·n_T − 2σ_T²·V)`（围绕 1，⟨f_T⁴⟩ = 1）。

    Args:
        sigma: 温度起伏强度 σ_T（≥ 0）：全部 w_e = 1 时 ln T 起伏的标准差。
        n: 归一化起伏 n_T（零均值，方差约为 `var`）；标量或数组。
        var: 钳制后的方差 V（[0, 1]），与 `n` 可广播。

    Returns:
        与 `broadcast(n, var)` 同形状，> 0，围绕 1 波动；σ_T = 0 或 n = V = 0 时精确为 1。

    Formula:
        n_T ~ N(0, V) 时 `⟨exp(4σ·n)⟩ = exp(8σ²V)`，所以 `⟨f_T⁴⟩ = ⟨exp(4σn − 8σ²V)⟩ = 1`。

    Physical Meaning:
        温度的对数正态起伏，按平均热辐射通量 ⟨T⁴⟩ 守恒归一（黑体通量 ∝ T⁴）。

    Simplifications:
        值噪声之和视为高斯（工程近似）。
    """
    s = float(sigma)
    return np.exp(s * np.asarray(n, dtype=np.float64) - 2.0 * s * s * np.asarray(var, dtype=np.float64))


def temp_turb_pattern_np(lnr, phi, zr, ox: float, oz: float, geom: ShearCascadeGeometry,
                         gains: Tuple[float, ...], weights, disk_spin: float,
                         noise: Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]) -> np.ndarray:
    """单个图案（一条刚体环带、一个种子相位）的未归一化温度起伏 `Σ_e a_e·w_e·ν_e`。

    Args:
        lnr: `ln r`（r 为盘局部柱坐标半径，r_s）；标量或数组。
        phi: 带内流坐标 φ'（rad，已扣除刚体环带转动），与 `lnr` 可广播。
        zr: 无量纲高度 `z / r`，与 `lnr` 可广播。
        ox / oz: 刚体环带种子偏移（与主云共用）。
        geom: `temp_turb_geometry` 的返回值。
        gains: 八度幅度 `a_e`。
        weights: 钳制权重，形状 `broadcast(lnr, phi, zr).shape + (E,)`，或可广播到该形状。
        disk_spin: 盘旋转方向（+1 / −1），须与构造 `geom` 时一致。
        noise: 三维值噪声 `noise(x, y, z, period)`，y 方向周期为 `period`，输出 `[-1, 1]`、零均值。

    Returns:
        与输入广播形状相同的数组，零均值；除以标定常数 N 后为单位方差（全部 w_e = 1 时）。

    Formula:
        ```
        u0  = K_e·(ln r − ln 20)
        ν_e = noise(|N₀₀|_e·u0 + ox + 1000,  φ'/2π·P_e + spin·s_e·u0,  K_e·z/r + oz + 23 + 5e;  P_e)
        ```
        权重为 0 的八度贡献为 0（参考实现仍计算噪声再乘 0；Taichi 实现跳过，结果相同）。

    Physical Meaning:
        湍流级联最小尺度上的温度团块，形状按开普勒剪切取形，竖直方向与径向尺度相当。

    Simplifications:
        加性值噪声（非乘性级联）；竖直方向各向同性（`K_e·z/r`）。
    """
    lnr, phi, zr = np.broadcast_arrays(np.asarray(lnr, dtype=np.float64), np.asarray(phi, dtype=np.float64),
                                       np.asarray(zr, dtype=np.float64))
    w = np.broadcast_to(np.asarray(weights, dtype=np.float64), lnr.shape + (len(geom.kr),))
    s = np.zeros(lnr.shape)
    for e in range(len(geom.kr)):
        u0 = geom.kr[e] * (lnr - LN_R_ANCHOR)
        x = geom.u_scale[e] * u0 + ox + U_OFFSET
        y = phi / (2.0 * math.pi) * geom.period[e] + disk_spin * geom.shear[e] * u0
        z = geom.kr[e] * zr + oz + W_OFFSET + W_OFFSET_STEP * e
        n = noise(x, y, z, geom.period[e])
        s = s + gains[e] * w[..., e] * n
    return s
