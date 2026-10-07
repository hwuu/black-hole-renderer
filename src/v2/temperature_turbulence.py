"""小尺度温度湍流：剪切级联向小尺度延伸的温度起伏（几何构造、权重与 NumPy 参考实现）。

相机贴近盘面时，近处云层沿视线被平均成均匀的雾。本模块在主云剪切级联最细八度之后再延伸
E 个八度，只调制温度、不调制密度：

```
tf_c ← tf_c·f_T，f_T = exp(σ_l·n_T − 2σ_l²·V)          （⟨f_T⁴⟩ = 1，平均热辐射通量守恒）
σ_l  = σ_T·m(ĉ)                                          （间歇性：强度随主云密度 ĉ）
n_T  = Σ_e a_e·w_e·ν_e / N，V = Σ_e (a_e·w_e)² / Σ_e a_e²
```

各八度按像素足迹做频率钳制（比像素小的八度淡出为其均值 0），经过强透镜的光线按偏折角淡出；
噪声坐标按半频梯度噪声扭曲，打散格子的行列排布。
模型、推导与原型实测见 `docs/plans/v2_temperature_turbulence_plan.md`；
Taichi 实现见 `taichi_impl.DiskV2Taichi._temp_turb_*`，与本模块逐值一致。
"""

import math
from typing import Callable, Optional, Tuple

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
# 间歇性：主云密度 ĉ 的截断上限（防止极少数高密度点的起伏过强）
INTERMITTENCY_CN_MAX: float = 3.0
# 坐标扭曲：gnoise 的实测标准差（noise_ti.gnoise 文档），把扭曲幅度换算成"格"
GNOISE_STD: float = 0.19
# 坐标扭曲噪声的固定偏移（与图案本身去相关；两个分量之间去相关）
WARP_X_OFFSETS: Tuple[float, float] = (7.1, 13.7)
WARP_Z_OFFSETS: Tuple[float, float] = (1.7, 9.3)
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
                       n_octaves: int, disk_spin: float, n_coarse: int = 0,
                       shear_ar_big: Optional[float] = None) -> ShearCascadeGeometry:
    """温度湍流各八度的噪声坐标系数：可选的 S 个粗尺度八度在前，E 个延伸八度在后。

    Args:
        shear_k0: 主云剪切级联最大八度的 `ln r` 频率（格 / 单位 ln r，> 0）。
        shear_octaves: 主云剪切级联八度数 N（≥ 1）。
        shear_ar_small: 主云最细八度的长宽比（≥ 1）；延伸八度统一取该值。
        shear_tilt_k: 拖尾倾角系数（≥ 0），与主云相同。
        n_octaves: 向小尺度延伸的八度数 E（1 ≤ E ≤ `MAX_OCTAVES`）。
        disk_spin: 盘旋转方向（+1 / −1）。
        n_coarse: 向粗尺度延伸的八度数 S（0 ≤ S ≤ N）：温度起伏另取主云最细的 S 个八度的尺度。0 = 只有延伸八度。
        shear_ar_big: 主云最大八度的长宽比（≥ 1）；S > 0 时必填，粗尺度八度的几何与主云对应八度相同。

    Returns:
        `ShearCascadeGeometry`，各字段长度为 S + E，按频率从低到高排列：
        前 S 项与主云剪切级联第 N−S … N−1 个八度相同（频率 `shear_k0·3^k`），
        后 E 项为延伸八度（频率 `shear_k0·3^(N + e)`）。S = 0 时与只取延伸八度完全相同。

    Raises:
        ValueError: `n_octaves` 或 `n_coarse` 越界、S > 0 时缺 `shear_ar_big`；其余参数由 `shear_cascade_geometry` 校验。

    Formula:
        延伸八度：`shear_cascade_geometry(shear_k0·3^N, E, AR_small, AR_small, tilt_k, spin)`
        粗尺度八度：`shear_cascade_geometry(shear_k0, N, AR_big, AR_small, tilt_k, spin)[N−S : N]`

    Physical Meaning:
        湍流级联继续向小尺度延伸，团块形状沿用开普勒剪切取形规律；粗尺度八度使温度起伏也出现
        主云尺度的大片冷暖斑块，大斑块内部再由更细的八度细分（分形结构），远景中也能看到起伏。

    Simplifications:
        延伸八度的长宽比不再随尺度变化（统一取主云最细八度的长宽比）；粗尺度八度与主云同尺度、
        同几何，但噪声坐标偏移不同，起伏与主云密度不相关。
    """
    if not 1 <= n_octaves <= MAX_OCTAVES:
        raise ValueError(f"n_octaves must be in [1, {MAX_OCTAVES}]")
    if not 0 <= n_coarse <= shear_octaves:
        raise ValueError(f"n_coarse must be in [0, shear_octaves = {shear_octaves}]")
    k_first = float(shear_k0) * 3.0 ** int(shear_octaves)
    fine = shear_cascade_geometry(k_first, n_octaves, shear_ar_small, shear_ar_small, shear_tilt_k, disk_spin)
    if n_coarse == 0:
        return fine
    if shear_ar_big is None:
        raise ValueError("shear_ar_big is required when n_coarse > 0")
    main = shear_cascade_geometry(shear_k0, shear_octaves, shear_ar_big, shear_ar_small, shear_tilt_k, disk_spin)
    sl = slice(shear_octaves - n_coarse, shear_octaves)
    return ShearCascadeGeometry(*(tuple(getattr(main, f)[sl]) + tuple(getattr(fine, f))
                                  for f in ("kr", "period", "u_scale", "shear", "aspect", "tilt_rad")))


def temp_turb_gains(n_octaves: int, gain: float, n_coarse: int = 0,
                    max_coarse: Optional[int] = None) -> Tuple[float, ...]:
    """温度湍流各八度的幅度 `a_e = gain^e`（a_0 = 1，e 从最粗的八度起算）。

    Args:
        n_octaves: 向小尺度延伸的八度数 E（1 ≤ E ≤ `MAX_OCTAVES`）。
        gain: 逐八度幅度比（(0, 1]）；1 = 各八度等幅。
        n_coarse: 向粗尺度延伸的八度数 S（≥ 0），见 `temp_turb_geometry`。
        max_coarse: S 的上限（通常为主云八度数 N）；None = 只校验 S ≥ 0。

    Returns:
        长度为 S + E 的元组，首项为 1，逐项乘 `gain`，值域 (0, 1]；顺序与 `temp_turb_geometry` 相同。

    Raises:
        ValueError: `n_octaves`、`n_coarse` 或 `gain` 越界。

    Formula:
        `a_e = gain^e`，e = 0 … S+E−1

    Physical Meaning:
        湍流起伏幅度随尺度减小而减弱；0.69 ≈ 3^(−1/3) 为 Kolmogorov 标度（尺度缩小到 1/3）。
        含粗尺度八度时取 1（等幅）：按 Kolmogorov 标度逐级减弱时，6 个八度的最细一级只有最粗一级的 0.69⁵ ≈ 16%，近处细节消失。

    Simplifications:
        幅度比与尺度无关（纯幂律）。
    """
    if not 1 <= n_octaves <= MAX_OCTAVES:
        raise ValueError(f"n_octaves must be in [1, {MAX_OCTAVES}]")
    if n_coarse < 0 or (max_coarse is not None and n_coarse > max_coarse):
        raise ValueError(f"n_coarse must be in [0, {max_coarse}]")
    if not 0.0 < gain <= 1.0:
        raise ValueError("gain must be in (0, 1]")
    return tuple(float(gain) ** e for e in range(n_coarse + n_octaves))


def temp_turb_warp_period(period: int) -> int:
    """坐标扭曲噪声的方位周期 `P_w = max(1, round(P/2))`（约为该八度频率的一半）。

    Args:
        period: 该八度的方位周期 P（正整数）。

    Returns:
        正整数 P_w；扭曲噪声的方位坐标取 `y·P_w/P`，绕盘一圈前进 P_w 格，φ 方向无缝。
    """
    return max(1, int(round(period / 2)))


def temp_turb_intermittency_norm(cn, gamma: float) -> float:
    """间歇性的归一化常数 `M = √⟨clip(ĉ, 0, 3)^{2γ}⟩`（γ = 0 时为 1）。

    Args:
        cn: 归一化主云密度 ĉ = c/⟨c⟩ 的样本（数组，≥ 0；构造期取 ⟨c⟩ 的标定样本）。
        gamma: 间歇性指数 γ（[0, 3]）。

    Returns:
        正标量；使 `⟨m(ĉ)²⟩ = 1`，即局部强度的全盘均方根等于 σ_T。

    Formula:
        `M = √mean(clip(ĉ, 0, 3)^{2γ})`

    Physical Meaning:
        保持 σ_T 的含义为"全盘均方根强度"，间歇性只重新分配强度、不改变总体水平。

    Simplifications:
        用 z = 0 的有限样本估计全盘平均。
    """
    if gamma == 0.0:
        return 1.0
    q = np.clip(np.asarray(cn, dtype=np.float64), 0.0, INTERMITTENCY_CN_MAX)
    return float(max(np.sqrt(np.mean(q ** (2.0 * gamma))), 1e-6))


def temp_turb_intermittency_factor(cn, gamma: float, m_norm: float):
    """间歇性的局部强度系数 m(ĉ)（局部强度 σ_l = σ_T·m）。

    Args:
        cn: 采样点的归一化主云密度 ĉ（标量或数组）。
        gamma: 间歇性指数 γ（[0, 3]）；0 = 处处同强。
        m_norm: 归一化常数 M（`temp_turb_intermittency_norm` 的返回值）。

    Returns:
        与 `cn` 同形状，≥ 0；γ = 0 时恒为 1，否则随 ĉ 单调不减、ĉ ≥ 3 时取最大值 3^γ/M。

    Formula:
        `m(ĉ) = 1`（γ = 0）；`m(ĉ) = clip(ĉ, 0, 3)^γ / M`（γ > 0）

    Physical Meaning:
        湍流耗散发热集中在密度高的团块里，稀薄处起伏弱：温度起伏的强度成片分布（间歇性）。

    Simplifications:
        用主云密度代替湍流耗散率；γ 与截断上限 3 由画面选定。
    """
    if gamma == 0.0:
        return np.ones_like(np.asarray(cn, dtype=np.float64))
    q = np.clip(np.asarray(cn, dtype=np.float64), 0.0, INTERMITTENCY_CN_MAX)
    return q ** gamma / m_norm


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


def temp_turb_temperature(sigma, n, var):
    """温度湍流的温度倍率 `f_T = exp(σ_l·n_T − 2σ_l²·V)`（围绕 1，⟨f_T⁴⟩ = 1）。

    Args:
        sigma: 局部强度 σ_l（≥ 0，标量或与 `n` 可广播的数组）：全部 w_e = 1 时该处 ln T 起伏的标准差，
            即 σ_T·m(ĉ)。
        n: 归一化起伏 n_T（零均值，方差约为 `var`）；标量或数组。
        var: 钳制后的方差 V（[0, 1]），与 `n` 可广播。

    Returns:
        与 `broadcast(sigma, n, var)` 同形状，> 0，围绕 1 波动；σ_l = 0 或 n = V = 0 时精确为 1。

    Formula:
        n_T ~ N(0, V) 时 `⟨exp(4σ·n)⟩ = exp(8σ²V)`，所以 `⟨f_T⁴⟩ = ⟨exp(4σn − 8σ²V)⟩ = 1`。

    Physical Meaning:
        温度的对数正态起伏，按平均热辐射通量 ⟨T⁴⟩ 守恒归一（黑体通量 ∝ T⁴）。

    Simplifications:
        值噪声之和视为高斯（工程近似）。
    """
    s = np.asarray(sigma, dtype=np.float64)
    return np.exp(s * np.asarray(n, dtype=np.float64) - 2.0 * s * s * np.asarray(var, dtype=np.float64))


def temp_turb_pattern_np(lnr, phi, zr, ox: float, oz: float, geom: ShearCascadeGeometry,
                         gains: Tuple[float, ...], weights, disk_spin: float,
                         noise: Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray],
                         warp: float = 0.0,
                         warp_noise: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]] = None
                         ) -> np.ndarray:
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
        warp: 坐标扭曲幅度 A（格，[0, 1)）；0 = 不扭曲（`warp_noise` 可为 None）。
        warp_noise: 三维梯度噪声 `warp_noise(x, y, z, period)`（与 `noise_ti.gnoise` 一致），`warp > 0` 时必须提供。

    Returns:
        与输入广播形状相同的数组，零均值；除以标定常数 N 后在标定分布上为单位方差（全部 w_e = 1 时）。

    Formula:
        ```
        u0  = K_e·(ln r − ln 20)
        (x, y, z) = (|N₀₀|_e·u0 + ox + 1000,  φ'/2π·P_e + spin·s_e·u0,  K_e·z/r + oz + 23 + 5e)
        d_x = gnoise(½x + 7.1, y·P_w/P_e, ½z + 1.7; P_w)，d_y = gnoise(½x + 13.7, y·P_w/P_e, ½z + 9.3; P_w)
        ν_e = noise(x + A·d_x/0.19,  y + A·d_y/0.19,  z;  P_e)          （A = 0 时不扭曲）
        ```
        权重为 0 的八度贡献为 0（参考实现仍计算噪声再乘 0；Taichi 实现跳过，结果相同）。

    Physical Meaning:
        湍流级联最小尺度上的温度团块，形状按开普勒剪切取形，竖直方向与径向尺度相当。

    Simplifications:
        加性值噪声（非乘性级联）；竖直方向各向同性（`K_e·z/r`）；扭曲对特征的局部压缩未计入钳制权重。
    """
    if warp > 0.0 and warp_noise is None:
        raise ValueError("warp > 0 requires warp_noise")
    lnr, phi, zr = np.broadcast_arrays(np.asarray(lnr, dtype=np.float64), np.asarray(phi, dtype=np.float64),
                                       np.asarray(zr, dtype=np.float64))
    w = np.broadcast_to(np.asarray(weights, dtype=np.float64), lnr.shape + (len(geom.kr),))
    s = np.zeros(lnr.shape)
    for e in range(len(geom.kr)):
        u0 = geom.kr[e] * (lnr - LN_R_ANCHOR)
        x = geom.u_scale[e] * u0 + ox + U_OFFSET
        y = phi / (2.0 * math.pi) * geom.period[e] + disk_spin * geom.shear[e] * u0
        z = geom.kr[e] * zr + oz + W_OFFSET + W_OFFSET_STEP * e
        if warp > 0.0:
            pw = temp_turb_warp_period(geom.period[e])
            yw = y * (pw / geom.period[e])
            dx = warp_noise(0.5 * x + WARP_X_OFFSETS[0], yw, 0.5 * z + WARP_Z_OFFSETS[0], pw)
            dy = warp_noise(0.5 * x + WARP_X_OFFSETS[1], yw, 0.5 * z + WARP_Z_OFFSETS[1], pw)
            x = x + warp / GNOISE_STD * dx
            y = y + warp / GNOISE_STD * dy
        n = noise(x, y, z, geom.period[e])
        s = s + gains[e] * w[..., e] * n
    return s
