"""吸积盘真实感参考实现（独立脚本，不依赖 render.py / disk_v2）。

用途：
1. V2 体积云雾 + 动态旋转方案的**视觉验收基准**（docs/plans/v2_volumetric_video_plan.md §2）。
   V2 移植后需与本脚本默认输出等价。
2. 物理自检工具：`physcheck` 逐像素对比严格 GR 频移、旋转方向、亮侧 = 蓝移侧。

模型要点：
- Schwarzschild 笛卡尔等效势光追（d²x/dλ² = −1.5 L² x / r⁵）+ 体积发射-吸收积分。
- 3D 程序化密度：薄核心层（lognormal × ridged 丝 × 覆盖率）+ 7 层重叠烟雾 + 大尺度低频调制。
- 动态：mode 3 刚体环（ln r 分带，带内以带中心 Ω 刚体旋转，结构按 K_RIGID 个本地周期换种子），不卷绕。
- 相对论：本地静止观者方向的 Doppler × 引力红移；亮度用 550nm Planck 比值，颜色用 CIE 黑体色。
- 后处理：白平衡 → 高光 bloom（轴向色散）→ 镶边 → 横向色散 → 保色度 ACES → sRGB。

单位：r_s = c = 1，M = 0.5。时间单位 r_s / c。

子命令（默认即定稿预设 H、r_out = 30、相机 dist = 60 / fov = 38°）：
    python scripts/proto_disk_reference.py frame --ss 2     # 1080p 单帧
    python scripts/proto_disk_reference.py video            # 慢速实时视频（内缘 16 s 一圈）
    python scripts/proto_disk_reference.py longrun          # naive vs mode 3 长时 time-lapse（上下对比）
    python scripts/proto_disk_reference.py physcheck        # 物理自检
    python scripts/proto_disk_reference.py stats            # 中面结构统计
    python scripts/proto_disk_reference.py winding          # face-on 卷绕对比图

输出目录：output/proto/
"""

import argparse
import math
import os
import time

import numpy as np
import taichi as ti

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "proto")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 盘参数
# ---------------------------------------------------------------------------
R_IN = 3.0            # ISCO (r_s)
R_OUT = 14.0
HR = 0.012            # 标高 H / r（薄盘）
T_PEAK_VIS = 4500.0   # 可视色温峰值（K），Interstellar 风格暖金色；不是物理 1e7 K
COLOR_SAT = 1.0       # 色度饱和度增强（1 = 不增强；线性化后黑体色本身已足够饱和）
TAU0 = 2.0            # 参考半径处 face-on 竖直光学深度
SIGMA_LN = 1.6        # 对数正态密度涨落强度（湍流密度 PDF 近似 lognormal）
RIDGE_POW = 6.0       # 丝状结构锐度（每个八度）
FIL_GAIN = 6.0        # 丝状项增益（ridged 均值约 0.1）
COV_RIDGE = 3.0       # 覆盖率中丝状项权重：丝所在处不挖空
FIL_MIX = 0.6         # 丝状项在密度中的权重
COV_LO = -0.1       # 覆盖率阈值：混合涨落低于此处为空洞
COV_HI = 0.7         # 覆盖率阈值：高于此处完全覆盖

# 云层：5 层薄片（k = −2..2），中心 z_k = k·CL_SPACING·r，厚度 σ = CL_WIDTH·r，
# 密度从中面向外按 exp(−CL_DECAY·|k|) 递减；沿用"厚盘"版的噪声尺度
N_CL_HALF = 2         # 层编号 k ∈ [−2, 2]
CL_SPACING = 0.018    # 层间距 / r
CL_WIDTH = 0.006      # 单层高斯厚度 σ / r（间距/厚度 = 3，层可分辨）
CL_DECAY = 0.6        # 层密度衰减：A_k ∝ exp(−CL_DECAY·|k|) → 1 : 0.55 : 0.30
CL_EXTENT = N_CL_HALF * CL_SPACING + 3.0 * CL_WIDTH   # 云层竖直包络 / r
HR_CLOUD = CL_EXTENT / 3.5                              # 兼容：步长控制用
CLOUD_COL = 0.25      # 云层总柱密度 / 核心柱密度
FR_C = 14.0           # 云层径向频率（厚盘版参数）
NPHI_C = 8            # 云层方位周期
FZ_C = 0.9            # 云层竖直频率（每个云层标高）
SIGMA_C = 1.0         # 云层 lognormal 强度
CLOUD_C0 = 0.25       # 云团覆盖率 sigmoid 中心：n_c 高于此处为云
CLOUD_SOFT = 0.4      # sigmoid 过渡宽度（越大边缘越柔）
T_CLUMP = 0.06        # 温度对密度涨落的弱响应：T ← T·(1 + T_CLUMP·tanh(n_a))

# 外观旋钮（预设覆盖，见 PRESETS）
EMIT_POW = 4.0        # 发射率温度指数 j ∝ ρ·(T/T_peak)^EMIT_POW；4 为 bolometric，越小外圈越不暗
LOWF_SIGMA = 0.0      # 低频大尺度 lognormal 调制强度（0 关闭）
FR_L = 3.0            # 低频噪声径向频率（ln r 每单位），尺度 Δln r ≈ 0.33
NPHI_L = 3            # 低频噪声方位周期
DLN_L = math.log(1.7) # 低频层刚体带宽（宽带，避免被细带切成环）
K_RIGID_L = 6.0       # 低频结构种子寿命（本地轨道周期）
CLOUD_EMIT = 1.0      # 云层单位密度发射率 / 核心（<1 = 偏冷、以吸收为主的烟雾，背光呈暗色丝缕）

# 各层振幅归一：Σ_k A_k = 1，使云层总柱密度 = CLOUD_COL × 核心柱密度
CL_AMP_NORM = 1.0 / sum(math.exp(-CL_DECAY * abs(k)) for k in range(-N_CL_HALF, N_CL_HALF + 1))

# 噪声各向异性：径向尺度 Δln r ≈ 1/FR，方位尺度 Δφ ≈ 2π/NPHI
FR = 24.0            # 方位/径向各向异性 ≈ 2π·FR/NPHI ≈ 15（Bruneton 约 10-20）
NPHI = 10             # 必须为整数（φ 方向周期噪声）
WARP = 0.9           # 域扭曲强度（噪声格点单位），打破格点规律
FZ = 0.9              # 每个标高内的噪声频率

# 分带平流（mode 2：带内按本地 Ω 平流 + 周期重置）
DLN = math.log(1.45)            # 带宽（ln r）
LNR0 = math.log(R_IN) - 2 * DLN  # 第 0 带中心
K_LIFE_DEFAULT = 0.35           # 结构寿命 = K_LIFE 个本地轨道周期

# 刚体环（mode 3：Bruneton 式，每带以 Ω(r_b) 刚体旋转 + 慢速换种子）
DLN_R = math.log(1.22)
LNR0_R = math.log(R_IN) - 2 * DLN_R
K_RIGID_DEFAULT = 4.0           # 刚体环的种子寿命（本地轨道周期），不影响卷绕；越大形状保持越久


def omega_k(r):
    """Schwarzschild 圆轨道坐标角速度 Ω = sqrt(M / r^3)，M = 0.5。"""
    return math.sqrt(0.5 / r ** 3)


def period_k(r):
    """本地轨道周期 2π / Ω(r)。"""
    return 2 * math.pi / omega_k(r)


ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=0)

params = ti.field(ti.f32, shape=12)  # 0 k_life, 1 normA, 2 normB, 3 kappa, 4 emit, 5 k_rigid, 6 doppler_lum, 7 doppler_color, 8 t_peak, 9 normC
cam_f = ti.Vector.field(3, ti.f32, shape=4)  # pos, fwd, right, up
BB_N = 512
bb_lut = ti.Vector.field(3, ti.f32, shape=BB_N)  # CIE 黑体线性 sRGB 色度表，log T ∈ [log 1000, log 40000]


# ---------------------------------------------------------------------------
# 噪声
# ---------------------------------------------------------------------------
@ti.func
def _hash_u32(x):
    """整数 hash（lowbias32 变体）。"""
    x ^= x >> ti.u32(16)
    x *= ti.u32(0x7feb352d)
    x ^= x >> ti.u32(15)
    x *= ti.u32(0x2c1b3c6d)
    x ^= x >> ti.u32(16)
    return x


@ti.func
def _hash3(ix, iy, iz):
    """三维整数格点 hash。"""
    h = _hash_u32(ti.cast(ix + 4096, ti.u32) * ti.u32(1597334677))
    h = _hash_u32(h ^ (ti.cast(iy + 4096, ti.u32) * ti.u32(1103515245)))
    h = _hash_u32(h ^ (ti.cast(iz + 4096, ti.u32) * ti.u32(1234567891)))
    return h


@ti.func
def _hashf(a, b, c):
    """hash → [0, 1) 浮点。"""
    return ti.cast(_hash3(a, b, c) & ti.u32(0xFFFFFF), ti.f32) / 16777216.0


@ti.func
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@ti.func
def _corner(xi, yi, zi, dx, dy, dz, period):
    """格点梯度 · 偏移向量；y 方向按 period 取模实现 φ 周期。"""
    yy = ((yi % period) + period) % period
    h = _hash3(xi, yy, zi)
    gx = ti.cast(h & ti.u32(1023), ti.f32) / 511.5 - 1.0
    gy = ti.cast((h >> ti.u32(10)) & ti.u32(1023), ti.f32) / 511.5 - 1.0
    gz = ti.cast((h >> ti.u32(20)) & ti.u32(1023), ti.f32) / 511.5 - 1.0
    return gx * dx + gy * dy + gz * dz


@ti.func
def gnoise(x, y, z, period):
    """3D 梯度噪声，y 方向周期为 period（整数）。输出约 [-1, 1]，零均值。"""
    fx0 = ti.floor(x)
    fy0 = ti.floor(y)
    fz0 = ti.floor(z)
    xi = ti.cast(fx0, ti.i32)
    yi = ti.cast(fy0, ti.i32)
    zi = ti.cast(fz0, ti.i32)
    fx = x - fx0
    fy = y - fy0
    fz = z - fz0
    ux = _fade(fx)
    uy = _fade(fy)
    uz = _fade(fz)
    n000 = _corner(xi, yi, zi, fx, fy, fz, period)
    n100 = _corner(xi + 1, yi, zi, fx - 1.0, fy, fz, period)
    n010 = _corner(xi, yi + 1, zi, fx, fy - 1.0, fz, period)
    n110 = _corner(xi + 1, yi + 1, zi, fx - 1.0, fy - 1.0, fz, period)
    n001 = _corner(xi, yi, zi + 1, fx, fy, fz - 1.0, period)
    n101 = _corner(xi + 1, yi, zi + 1, fx - 1.0, fy, fz - 1.0, period)
    n011 = _corner(xi, yi + 1, zi + 1, fx, fy - 1.0, fz - 1.0, period)
    n111 = _corner(xi + 1, yi + 1, zi + 1, fx - 1.0, fy - 1.0, fz - 1.0, period)
    nx00 = n000 + ux * (n100 - n000)
    nx10 = n010 + ux * (n110 - n010)
    nx01 = n001 + ux * (n101 - n001)
    nx11 = n011 + ux * (n111 - n011)
    nxy0 = nx00 + uy * (nx10 - nx00)
    nxy1 = nx01 + uy * (nx11 - nx01)
    return nxy0 + uz * (nxy1 - nxy0)


@ti.func
def fbm_a(x, y, z):
    """5 八度 fBm（主密度涨落）。"""
    s = 0.0
    for o in ti.static(range(5)):
        f = 2 ** o
        s += (0.5 ** o) * gnoise(x * f, y * f, z * f, NPHI * f)
    return s


@ti.func
def fbm_b(x, y, z):
    """4 八度 ridged multifractal（丝状源），值域约 [0, 1]。

    每个八度单独取脊 (1 − |n|)^RIDGE_POW 再加权求和，频谱比"单个 fBm 取脊"宽，
    避免丝的间距集中在一个特征尺度上（看起来"太规律"）。
    """
    s = 0.0
    wsum = 0.0
    for o in ti.static(range(4)):
        f = 2 ** (o + 1)
        n = gnoise(x * f, y * f, z * f, NPHI * f)
        rdg = ti.pow(ti.max(1.0 - 1.6 * ti.abs(n), 0.0), RIDGE_POW)
        s += (0.7 ** o) * rdg
        wsum += 0.7 ** o
    return s / wsum


@ti.func
def _eval_pair(lnr, phi0, zeta, ox, oz):
    """在拉格朗日坐标 (ln r, φ0, z/H) 上求两路 fBm。"""
    x = lnr * FR + ox
    y = phi0 / (2.0 * math.pi) * NPHI
    z = zeta * FZ + oz
    # 域扭曲：用低频噪声扰动 (x, y)，消除梯度噪声格点零值造成的等间距环纹
    wx = gnoise(x * 0.37 + 5.1, y * 0.5, z * 0.5 + 3.3, NPHI // 2)
    wy = gnoise(x * 0.37 + 9.7, y * 0.5, z * 0.5 + 7.9, NPHI // 2)
    x = x + WARP * wx
    y = y + 0.6 * WARP * wy
    return fbm_a(x, y, z), fbm_b(x + 31.7, y, z + 17.3)


@ti.func
def fbm_c(x, y, z):
    """5 八度 fBm（云层），φ 周期 NPHI_C。"""
    s = 0.0
    for o in ti.static(range(5)):
        f = 2 ** o
        s += (0.5 ** o) * gnoise(x * f, y * f, z * f, NPHI_C * f)
    return s


@ti.func
def _eval_cloud(lnr, phi0, zeta, ox, oz, layer_off):
    """云层噪声，在 (ln r, φ0, 层内 z/σ) 上求值；layer_off 使各层独立。"""
    x = lnr * FR_C + ox + 57.3 + layer_off
    y = phi0 / (2.0 * math.pi) * NPHI_C
    z = zeta * FZ_C + oz + 41.9 + 0.37 * layer_off
    wx = gnoise(x * 0.37 + 2.1, y * 0.5, z * 0.5 + 1.3, NPHI_C // 2)
    return fbm_c(x + WARP * wx, y, z)


@ti.func
def _eval_layer(layer: ti.template(), lnr, phi0, zeta, ox, oz, loff):
    """layer 0 = 核心层（主涨落 + 丝）；layer 1 = 云层（只有主涨落，loff 为运行时层偏移）。"""
    a = 0.0
    b = 0.0
    if ti.static(layer == 0):
        a, b = _eval_pair(lnr, phi0, zeta, ox, oz)
    else:
        a = _eval_cloud(lnr, phi0, zeta, ox, oz, loff)
    return a, b


@ti.func
def _fbm_low(x, y, z):
    """3 八度低频 fBm，φ 周期 NPHI_L。"""
    s = 0.0
    for o in ti.static(range(3)):
        f = 2 ** o
        s += (0.5 ** o) * gnoise(x * f, y * f, z * f, NPHI_L * f)
    return s


@ti.func
def turb_low(r, phi, t):
    """大尺度低频调制场（单位方差）。

    宽带刚体环：带宽 DLN_L，带内以带中心 Ω 刚体旋转，种子寿命 K_RIGID_L 个本地周期，
    两带 cos²/sin² 混合 × 两相位 sin² 交叉淡化，方差保持归一。
    宽带使大尺度结构不被细带切成同心环。
    """
    lnr = ti.log(r)
    fb = (lnr - math.log(R_IN)) / DLN_L
    b0 = ti.floor(fb)
    fbf = fb - b0
    acc = 0.0
    wsq = 0.0
    for db in ti.static(range(2)):
        bi = ti.cast(b0, ti.i32) + db
        wb = ti.cos(0.5 * math.pi * fbf) ** 2
        if db == 1:
            wb = ti.sin(0.5 * math.pi * fbf) ** 2
        r_b = ti.exp(math.log(R_IN) + (ti.cast(bi, ti.f32) + 0.5) * DLN_L)
        om_b = ti.sqrt(0.5 / (r_b * r_b * r_b))
        t_life = K_RIGID_L * 2.0 * math.pi / om_b
        phi0 = phi - om_b * t - _hashf(bi, 23, 1) * 2.0 * math.pi
        for p in ti.static(range(2)):
            s = t / t_life + _hashf(bi, 29, 5) + 0.5 * p
            cyc = ti.floor(s)
            wp = ti.sin(math.pi * (s - cyc)) ** 2
            ci = ti.cast(cyc, ti.i32)
            n = _fbm_low(lnr * FR_L + _hashf(bi, ci, 7 + p) * 97.0,
                         phi0 / (2.0 * math.pi) * NPHI_L,
                         _hashf(bi, ci, 11 + p) * 97.0)
            w = wb * wp
            acc += w * n
            wsq += w * w
    return acc / ti.sqrt(ti.max(wsq, 1e-6)) / params[10]


@ti.func
def turb_pair(r, phi, zeta, t, mode: ti.template(), layer: ti.template(), loff):
    """湍流场（两路，归一化到约单位方差）。

    mode 0: naive —— φ0 = φ − Ω(r) t，t 无界 → 卷绕。
    mode 1: 逐半径相位 —— 周期 T(r)=k·P(r)，相位随 r 变化（缺陷版）。
    mode 2: 分带 —— ln r 分带，每带全局周期 T_b=k·P(r_b)，带间/相位间方差保持混合。
    mode 3: 刚体环 —— ln r 分带，每带以 Ω(r_b) 刚体旋转（无带内剪切），慢速换种子。
    """
    k_life = params[0]
    lnr = ti.log(r)
    om = ti.sqrt(0.5 / (r * r * r))
    acc_a = 0.0
    acc_b = 0.0
    wsq = 0.0
    if ti.static(mode == 0):
        a, b = _eval_layer(layer, lnr, phi - om * t, zeta, 0.0, 0.0, loff)
        acc_a = a
        acc_b = b
        wsq = 1.0
    elif ti.static(mode == 1):
        s_base = om * t / (2.0 * math.pi * k_life)
        for p in ti.static(range(2)):
            s = s_base + 0.5 * p
            cyc = ti.floor(s)
            fr = s - cyc
            w = ti.sin(math.pi * fr) ** 2
            tau = fr * k_life * 2.0 * math.pi / om
            ci = ti.cast(cyc, ti.i32)
            ox = _hashf(7, ci, 2 * p) * 97.0
            oz = _hashf(7, ci, 2 * p + 1) * 97.0
            a, b = _eval_layer(layer, lnr, phi - om * tau, zeta, ox, oz, loff)
            acc_a += w * a
            acc_b += w * b
            wsq += w * w
    elif ti.static(mode == 3):
        k_rigid = params[5]
        # 带边界用单调 1D 噪声扰动，使带间距不规则（斜率扰动 < 基础斜率，保持单调）
        fb = (lnr - LNR0_R) / DLN_R + 0.35 * gnoise(lnr * 4.0, 0.37, 11.3, 8)
        b0 = ti.floor(fb)
        fbf = fb - b0
        for db in ti.static(range(2)):
            bi = ti.cast(b0, ti.i32) + db
            wb = ti.cos(0.5 * math.pi * fbf) ** 2
            if db == 1:
                wb = ti.sin(0.5 * math.pi * fbf) ** 2
            r_b = ti.exp(LNR0_R + ti.cast(bi, ti.f32) * DLN_R)
            om_b = ti.sqrt(0.5 / (r_b * r_b * r_b))
            t_life = k_rigid * 2.0 * math.pi / om_b
            ph0 = _hashf(bi, 13, 5)
            phi_rigid = phi - om_b * t - _hashf(bi, 17, 1) * 2.0 * math.pi
            for p in ti.static(range(2)):
                s = t / t_life + ph0 + 0.5 * p
                cyc = ti.floor(s)
                fr = s - cyc
                wp = ti.sin(math.pi * fr) ** 2
                ci = ti.cast(cyc, ti.i32)
                ox = _hashf(bi, ci, 2 * p) * 97.0
                oz = _hashf(bi, ci, 2 * p + 1) * 97.0
                a, b = _eval_layer(layer, lnr, phi_rigid, zeta, ox, oz, loff)
                w = wb * wp
                acc_a += w * a
                acc_b += w * b
                wsq += w * w
    else:
        fb = (lnr - LNR0) / DLN
        b0 = ti.floor(fb)
        fbf = fb - b0
        for db in ti.static(range(2)):
            bi = ti.cast(b0, ti.i32) + db
            wb = ti.cos(0.5 * math.pi * fbf) ** 2
            if db == 1:
                wb = ti.sin(0.5 * math.pi * fbf) ** 2
            lnr_b = LNR0 + ti.cast(bi, ti.f32) * DLN
            r_b = ti.exp(lnr_b)
            om_b = ti.sqrt(0.5 / (r_b * r_b * r_b))
            t_life = k_life * 2.0 * math.pi / om_b
            ph0 = _hashf(bi, 11, 3)  # 每带随机相位，避免所有带同时重置
            for p in ti.static(range(2)):
                s = t / t_life + ph0 + 0.5 * p
                cyc = ti.floor(s)
                fr = s - cyc
                wp = ti.sin(math.pi * fr) ** 2
                tau = fr * t_life
                ci = ti.cast(cyc, ti.i32)
                ox = _hashf(bi, ci, 2 * p) * 97.0
                oz = _hashf(bi, ci, 2 * p + 1) * 97.0
                a, b = _eval_layer(layer, lnr, phi - om * tau, zeta, ox, oz, loff)
                w = wb * wp
                acc_a += w * a
                acc_b += w * b
                wsq += w * w
    inv = 1.0 / ti.sqrt(ti.max(wsq, 1e-6))
    norm = params[1]
    if ti.static(layer == 1):
        norm = params[9]
    return acc_a * inv / norm, acc_b


# ---------------------------------------------------------------------------
# 盘物理场
# ---------------------------------------------------------------------------
@ti.func
def _smoothstep(e0, e1, x):
    t = ti.min(ti.max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.func
def radial_env(r):
    """中面基础密度：ρ ∝ r^-1.5 · (1 − sqrt(r_in/r))^0.5 · 外缘收口。"""
    out = 0.0
    if r > R_IN:
        inner = ti.sqrt(ti.max(1.0 - ti.sqrt(R_IN / r), 0.0))
        outer = 1.0 - _smoothstep(0.72 * R_OUT, R_OUT, r)
        out = ti.pow(r / R_IN, -1.5) * inner * outer
    return out


@ti.func
def temperature(r):
    """Novikov-Thorne 形状温度：T ∝ r^-3/4 (1 − sqrt(r_in/r))^1/4，峰值归一到 T_PEAK_VIS。"""
    out = 0.0
    if r > R_IN:
        f = ti.pow(r, -0.75) * ti.pow(ti.max(1.0 - ti.sqrt(R_IN / r), 0.0), 0.25)
        r_pk = 49.0 / 36.0 * R_IN
        f_pk = ti.pow(r_pk, -0.75) * ti.pow(1.0 - ti.sqrt(R_IN / r_pk), 0.25)
        out = params[8] * f / f_pk
    return out


@ti.func
def density(r, z, phi, t, mode: ti.template()):
    """3D 密度 = 薄核心层 + 云层。返回 (ρ_total, 温度调制因子, ρ_cloud)；盘外 (0, 1, 0)。

    两者都乘大尺度低频调制 LN_L = exp(LOWF_SIGMA·n_L − LOWF_SIGMA²/2)。

    核心层：ρ_env(r)·exp(−ζ²/2)·lognormal(n_a)·丝(n_b)·覆盖率，ζ = z/(HR·r)
    云层：  CLOUD_COL·(HR/HR_CLOUD)·ρ_env(r)·exp(−ζc²/2)·lognormal(n_c)·云团覆盖率，ζc = z/(HR_CLOUD·r)
            系数 HR/HR_CLOUD 使云层柱密度 = CLOUD_COL × 核心柱密度。
    """
    rho = 0.0
    rho_c = 0.0
    tmod = 1.0
    if r > R_IN and r < R_OUT:
        env_r = radial_env(r)
        if ti.static(LOWF_SIGMA > 0.0):
            nl = turb_low(r, phi, t)
            env_r *= ti.exp(LOWF_SIGMA * nl - 0.5 * LOWF_SIGMA * LOWF_SIGMA)
        zeta = z / (HR * r)
        if ti.abs(zeta) < 4.0:
            env = env_r * ti.exp(-0.5 * zeta * zeta)
            if env > 1e-5:
                na, nb = turb_pair(r, phi, zeta, t, mode, 0, 0.0)
                logn = ti.exp(SIGMA_LN * na - 0.5 * SIGMA_LN * SIGMA_LN)
                fil = (1.0 - FIL_MIX) + FIL_MIX * nb * FIL_GAIN
                cov = _smoothstep(COV_LO, COV_HI, na + COV_RIDGE * (nb - 0.1))
                rho += env * logn * fil * cov
                tmod = 1.0 + T_CLUMP * ti.tanh(na)
        if ti.abs(z) < CL_EXTENT * r:
            # 5 层薄片：运行时循环（只内联一份噪声代码），只算 |z − z_k| < 3σ 的层
            for kk in range(2 * N_CL_HALF + 1):
                k = ti.cast(kk - N_CL_HALF, ti.f32)
                dz = (z - k * CL_SPACING * r) / (CL_WIDTH * r)
                if ti.abs(dz) < 3.0:
                    amp = CL_AMP_NORM * ti.exp(-CL_DECAY * ti.abs(k))
                    env_c = CLOUD_COL * (HR / CL_WIDTH) * amp * env_r * ti.exp(-0.5 * dz * dz)
                    nc, _unused = turb_pair(r, phi, dz, t, mode, 1, 131.7 * ti.cast(kk + 1, ti.f32))
                    cov = 1.0 / (1.0 + ti.exp(-(nc - CLOUD_C0) / CLOUD_SOFT))
                    rho_c += env_c * ti.exp(SIGMA_C * nc - 0.5 * SIGMA_C * SIGMA_C) * cov
    return rho + rho_c, tmod, rho_c


@ti.func
def _srgb_to_linear(v):
    """sRGB 编码值 → 线性光（IEC 61966-2-1）。"""
    out = v / 12.92
    if v > 0.04045:
        out = ti.pow((v + 0.055) / 1.055, 2.4)
    return out


@ti.func
def blackbody_rgb(t_k):
    """黑体色度（线性 sRGB，亮度归一）：查 CIE 普朗克表 bb_lut，log T 线性插值。"""
    u = (ti.log(ti.min(ti.max(t_k, 1000.0), 40000.0)) - math.log(1000.0)) / (math.log(40000.0) - math.log(1000.0))
    f = u * (BB_N - 1)
    i0 = ti.min(ti.cast(ti.floor(f), ti.i32), BB_N - 2)
    w = f - ti.cast(i0, ti.f32)
    lin = bb_lut[i0] * (1.0 - w) + bb_lut[i0 + 1] * w
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    c = ti.max(lum + COLOR_SAT * (lin - lum), 0.0)
    lum2 = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    return c / ti.max(lum2, 1e-3)


@ti.func
def g_factor(pos, dir_, r_obs):
    """g = ν_obs/ν_em。静止观者圆轨道速度 β = sqrt(M/(r−2M))，盘逆时针（+z 视角）。"""
    r = ti.sqrt(pos[0] * pos[0] + pos[1] * pos[1])
    r_e = ti.max(r, R_IN)
    beta = ti.min(ti.sqrt(0.5 / (r_e - 1.0)), 0.99)
    gam = 1.0 / ti.sqrt(1.0 - beta * beta)
    v_hat = ti.Vector([-pos[1], pos[0], 0.0]) / ti.max(r, 1e-6)
    # 光子传播方向（坐标方向）→ 本地静止观者方向：
    # 径向局部长度 dl_r = dr / sqrt(1 − r_s/r)，切向 dl_t = r dφ，
    # 故 tanψ_local = sqrt(1 − r_s/R) · tanψ_coord（ψ 为与径向夹角）。
    k = -dir_.normalized()
    r3 = pos.norm()
    r_hat = pos / ti.max(r3, 1e-6)
    k_rad = k.dot(r_hat) * r_hat
    k_tan = (k - k_rad) * ti.sqrt(ti.max(1.0 - 1.0 / r3, 1e-6))
    k_loc = (k_rad + k_tan).normalized()
    cos_th = v_hat.dot(k_loc)
    g_dop = 1.0 / (gam * (1.0 - beta * cos_th))
    g_grav = ti.sqrt(1.0 - 1.0 / ti.sqrt(r_e * r_e + pos[2] * pos[2])) / ti.sqrt(1.0 - 1.0 / r_obs)
    return g_dop * g_grav


@ti.func
def g_factor_coord(pos, dir_, r_obs):
    """旧做法：cosθ 直接用坐标方向（未做本地静止观者变换），仅用于 physcheck 对照。"""
    r = ti.sqrt(pos[0] * pos[0] + pos[1] * pos[1])
    r_e = ti.max(r, R_IN)
    beta = ti.min(ti.sqrt(0.5 / (r_e - 1.0)), 0.99)
    gam = 1.0 / ti.sqrt(1.0 - beta * beta)
    v_hat = ti.Vector([-pos[1], pos[0], 0.0]) / ti.max(r, 1e-6)
    cos_th = v_hat.dot(-dir_.normalized())
    g_grav = ti.sqrt(1.0 - 1.0 / ti.sqrt(r_e * r_e + pos[2] * pos[2])) / ti.sqrt(1.0 - 1.0 / r_obs)
    return g_grav / (gam * (1.0 - beta * cos_th))


PLANCK_X = 26160.0   # h c / (λ k)，λ = 550 nm，单位 K


@ti.func
def band_boost(t_em, g):
    """可见光波段强度的 Doppler/引力频移增强因子 B_ν(ν, gT) / B_ν(ν, T)。

    Planck 不变性：I_ν,obs = g³ I_ν,em(ν/g) = B_ν(ν, g·T)，即观测谱仍是温度 g·T 的黑体。
    在 λ=550nm 取比值，有效指数约 hν/kT ≈ 5-6，比 bolometric 的 g⁴ 更陡。
    """
    x_em = ti.min(PLANCK_X / ti.max(t_em, 500.0), 80.0)
    x_obs = ti.min(PLANCK_X / ti.max(t_em * g, 500.0), 80.0)
    return (ti.exp(x_em) - 1.0) / (ti.exp(x_obs) - 1.0)


@ti.func
def star_sky(d):
    """稀疏程序星空。"""
    dn = d.normalized()
    u = (ti.atan2(dn[1], dn[0]) / (2.0 * math.pi) + 0.5) * 2048.0
    v = ti.acos(ti.min(ti.max(dn[2], -1.0), 1.0)) / math.pi * 1024.0
    h = _hashf(ti.cast(u, ti.i32), ti.cast(v, ti.i32), 5)
    c = ti.Vector([0.0, 0.0, 0.0])
    if h < 0.004:
        br = ti.pow(_hashf(ti.cast(u, ti.i32), ti.cast(v, ti.i32), 9), 3.0) * 0.6
        c = ti.Vector([br, br * 0.95, br * 1.05])
    return c


# ---------------------------------------------------------------------------
# 光追
# ---------------------------------------------------------------------------
img = None


@ti.func
def accel(p, l2):
    """d²x/dλ² = −1.5 L² x / r⁵。"""
    r2 = p.dot(p)
    return -1.5 * l2 * p / (r2 * r2 * ti.sqrt(r2))


@ti.kernel
def render_kernel(out: ti.template(), t: ti.f32, mode: ti.template(), fov_v: ti.f32,
                  dbg: ti.template(), dbg_out: ti.template()):
    w = out.shape[0]
    hgt = out.shape[1]
    cp = cam_f[0]
    fw = cam_f[1]
    rt = cam_f[2]
    up = cam_f[3]
    tan_h = ti.tan(0.5 * fov_v)
    aspect = ti.cast(w, ti.f32) / ti.cast(hgt, ti.f32)
    r_obs = cp.norm()
    kappa = params[3]
    emit = params[4]
    for i, j in out:
        sx = ((ti.cast(i, ti.f32) + 0.5) / w * 2.0 - 1.0) * tan_h * aspect
        sy = (1.0 - (ti.cast(j, ti.f32) + 0.5) / hgt * 2.0) * tan_h
        d = (fw + sx * rt + sy * up).normalized()
        p = cp
        l2 = p.cross(d).norm_sqr()
        acc = ti.Vector([0.0, 0.0, 0.0])
        trans = 1.0
        escaped = False
        first = True
        # 光子守恒量：L = p × d 严格守恒；|d|² − L²/r³ 守恒 → 无穷远处 |d_inf|
        lz = p.cross(d)[2]
        d_inf = ti.sqrt(ti.max(1.0 - l2 / (cp.norm() ** 3), 1e-6))
        for _ in range(6000):
            r = p.norm()
            h = ti.min(0.06 * r, 2.0)
            if r < 3.0:
                h = ti.min(h, 0.02 + 0.06 * (r - 1.0))
            rc = ti.sqrt(p[0] * p[0] + p[1] * p[1])
            if rc > R_IN * 0.95 and rc < R_OUT * 1.02:
                if ti.abs(p[2]) < 4.5 * HR * rc + 0.05:
                    h = ti.min(h, ti.max(0.3 * HR * rc, 0.02))
                elif ti.abs(p[2]) < 1.1 * CL_EXTENT * rc + 0.05:
                    h = ti.min(h, ti.max(0.4 * CL_WIDTH * rc, 0.02))
            k1p = d
            k1d = accel(p, l2)
            k2p = d + 0.5 * h * k1d
            k2d = accel(p + 0.5 * h * k1p, l2)
            k3p = d + 0.5 * h * k2d
            k3d = accel(p + 0.5 * h * k2p, l2)
            k4p = d + h * k3d
            k4d = accel(p + h * k3p, l2)
            pn = p + h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
            dn = d + h / 6.0 * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)
            # 段中点做体积采样
            pm = 0.5 * (p + pn)
            rm = ti.sqrt(pm[0] * pm[0] + pm[1] * pm[1])
            if rm > R_IN and rm < R_OUT and ti.abs(pm[2]) < ti.max(CL_EXTENT, 4.0 * HR) * rm:
                rho, tmod, rho_cl = density(rm, pm[2], ti.atan2(pm[1], pm[0]), t, mode)
                if rho > 1e-6:
                    ds = (pn - p).norm()
                    a = 1.0 - ti.exp(-kappa * rho * ds)
                    tk = temperature(rm) * tmod
                    # 频移强度旋钮：亮度用 g^s_lum，颜色用 g^s_color；s=1 为物理值，s=0 关闭
                    g_phys = g_factor(pm, 0.5 * (d + dn), r_obs)
                    g_lum = ti.pow(g_phys, params[6])
                    g_col = ti.pow(g_phys, params[7])
                    # 发射率：核心按 1，云层按 CLOUD_EMIT（偏冷烟雾以吸收为主）
                    em_frac = 1.0 - (1.0 - CLOUD_EMIT) * rho_cl / rho
                    s_em = emit * em_frac * ti.pow(tk / params[8], EMIT_POW)
                    acc += trans * a * s_em * band_boost(tk, g_lum) * blackbody_rgb(tk * g_col)
                    if ti.static(dbg):
                        if first and ti.abs(pm[2]) < 0.05:
                            # 赤道面圆轨道 → 远处静止观者的严格 GR 频移（光子真实动量为 −d）
                            om_e = ti.sqrt(0.5 / (rm * rm * rm))
                            g_ex = ti.sqrt(1.0 - 1.5 / rm) / (1.0 + om_e * lz / d_inf) / ti.sqrt(1.0 - 1.0 / r_obs)
                            dbg_out[i, j] = ti.Vector([g_phys, g_ex, g_factor_coord(pm, 0.5 * (d + dn), r_obs), rm])
                            first = False
                    trans *= 1.0 - a
            p = pn
            d = dn
            rn = p.norm()
            if rn < 1.0 or trans < 1e-3:
                trans = 0.0
                break
            if rn > 90.0 and p.dot(d) > 0.0:
                escaped = True
                break
        sky = ti.Vector([0.0, 0.0, 0.0])
        if escaped:
            sky = star_sky(d)
        out[i, j] = acc + trans * sky


@ti.kernel
def polar_kernel(out: ti.template(), t: ti.f32, mode: ti.template()):
    """在 (ln r, φ) 极坐标网格上取中面 log 密度，用于 face-on 图和倾角指标。"""
    nr = out.shape[0]
    nphi = out.shape[1]
    lr0 = ti.log(R_IN * 1.05)
    lr1 = ti.log(R_OUT * 0.95)
    for i, j in out:
        lnr = lr0 + (lr1 - lr0) * (ti.cast(i, ti.f32) + 0.5) / nr
        phi = (ti.cast(j, ti.f32) + 0.5) / nphi * 2.0 * math.pi - math.pi
        na, nb = turb_pair(ti.exp(lnr), phi, 0.0, t, mode, 0, 0.0)
        fil = (1.0 - FIL_MIX) + FIL_MIX * nb * FIL_GAIN
        out[i, j] = SIGMA_LN * na + ti.log(fil)


@ti.kernel
def density_polar_kernel(out: ti.template(), t: ti.f32, mode: ti.template()):
    """中面结构因子 ρ/ρ_env 在 (ln r, φ) 网格上的分布（诊断用）。"""
    nr = out.shape[0]
    nphi = out.shape[1]
    lr0 = ti.log(R_IN * 1.15)
    lr1 = ti.log(R_OUT * 0.7)
    for i, j in out:
        r = ti.exp(lr0 + (lr1 - lr0) * (ti.cast(i, ti.f32) + 0.5) / nr)
        phi = (ti.cast(j, ti.f32) + 0.5) / nphi * 2.0 * math.pi - math.pi
        rho, _tm, _rc = density(r, 0.0, phi, t, mode)
        out[i, j] = rho / ti.max(radial_env(r), 1e-9)


@ti.kernel
def raw_stats_low_kernel(out: ti.template()):
    """低频 fBm 未归一化值，用于标定方差。"""
    for i, j in out:
        out[i, j] = _fbm_low(i / 16.0, j / out.shape[1] * NPHI_L, (i * 7 + j * 13) % 17 / 5.0)


@ti.kernel
def raw_stats_cloud_kernel(out: ti.template()):
    """云层 fbm 未归一化值，用于标定方差。"""
    nr = out.shape[0]
    nphi = out.shape[1]
    for i, j in ti.ndrange(nr, nphi):
        lnr = ti.log(R_IN) + (ti.log(R_OUT) - ti.log(R_IN)) * (i + 0.5) / nr
        phi = (j + 0.5) / nphi * 2.0 * math.pi
        zeta = ti.cast((i * 7 + j * 13) % 17, ti.f32) / 17.0 * 4.0 - 2.0
        out[i, j] = _eval_cloud(lnr, phi, zeta, 0.0, 0.0, 0.0)


@ti.kernel
def raw_stats_kernel(out: ti.template()):
    """naive, t=0 下未归一化 fbm 两路，用于标定方差。"""
    nr = out.shape[0]
    nphi = out.shape[1]
    for i, j in ti.ndrange(nr, nphi):
        lnr = ti.log(R_IN) + (ti.log(R_OUT) - ti.log(R_IN)) * (i + 0.5) / nr
        phi = (j + 0.5) / nphi * 2.0 * math.pi
        zeta = ti.cast((i * 7 + j * 13) % 17, ti.f32) / 17.0 * 4.0 - 2.0
        a, b = _eval_pair(lnr, phi, zeta, 0.0, 0.0)
        out[i, j] = ti.Vector([a, b])


# ---------------------------------------------------------------------------
# Python 侧
# ---------------------------------------------------------------------------
def setup(k_life, k_rigid=K_RIGID_DEFAULT, doppler_lum=0.5, doppler_color=2.2, t_peak=T_PEAK_VIS):
    """标定噪声方差、吸收系数、发射尺度；doppler_* 为频移强度指数（1 = 物理）。"""
    params[0] = k_life
    params[5] = k_rigid
    params[6] = doppler_lum
    params[7] = doppler_color
    params[8] = t_peak
    params[1] = 1.0
    params[2] = 1.0
    buf = ti.Vector.field(2, ti.f32, shape=(256, 512))
    raw_stats_kernel(buf)
    arr = buf.to_numpy()
    params[1] = float(arr[..., 0].std())
    params[2] = float(arr[..., 1].std())
    cbuf = ti.field(ti.f32, shape=(256, 512))
    raw_stats_cloud_kernel(cbuf)
    params[9] = float(cbuf.to_numpy().std())
    lbuf = ti.field(ti.f32, shape=(256, 512))
    raw_stats_low_kernel(lbuf)
    params[10] = float(lbuf.to_numpy().std())
    build_bb_lut()
    # κ：使 r_ref 处 face-on 竖直光学深度 ≈ TAU0（湍流平均因子约 1）
    r_ref = 6.0
    inner = math.sqrt(max(1 - math.sqrt(R_IN / r_ref), 0))
    env_ref = (r_ref / R_IN) ** -1.5 * inner
    params[3] = TAU0 / (env_ref * math.sqrt(2 * math.pi) * HR * r_ref)
    params[4] = 1.0


def set_camera(dist, elev_deg, azim_deg=-90.0, roll_deg=0.0):
    """相机在球面上看向原点。azim=-90 表示在 -y 轴。"""
    e = math.radians(elev_deg)
    a = math.radians(azim_deg)
    pos = np.array([dist * math.cos(e) * math.cos(a), dist * math.cos(e) * math.sin(a), dist * math.sin(e)])
    fwd = -pos / np.linalg.norm(pos)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rr = math.radians(roll_deg)
    right, up = right * math.cos(rr) + up * math.sin(rr), up * math.cos(rr) - right * math.sin(rr)
    for k, v in enumerate([pos, fwd, right, up]):
        cam_f[k] = v.astype(np.float32).tolist()


def box_blur(x, rad):
    """沿两轴的三次 box blur（近似高斯）。x: (H, W, 3)。"""
    if rad < 1:
        return x
    y = x
    for _ in range(3):
        for ax in (0, 1):
            c = np.cumsum(np.pad(y, [(rad + 1, rad) if a == ax else (0, 0) for a in range(3)], mode="edge"), axis=ax)
            n = y.shape[ax]
            sl_hi = [slice(None)] * 3
            sl_lo = [slice(None)] * 3
            sl_hi[ax] = slice(2 * rad + 1, 2 * rad + 1 + n)
            sl_lo[ax] = slice(0, n)
            y = (c[tuple(sl_hi)] - c[tuple(sl_lo)]) / (2 * rad + 1)
    return y


BLOOM_SRC = 0.3       # bloom 只取曝光后亮度超过此值的部分（高光：逼近侧边缘、光子环、内缘）
BLOOM_GAIN = 4.0      # 高光散射增益
CA_AXIAL = (1.0, 1.0, 1.15)  # 轴向色散：R/G/B 光晕半径倍率（只让蓝光稍外扩；R 与 B 同时外扩会形成品红/紫色光晕）
CA_FRINGE = 0.4              # 紫边强度：饱和高光外缘的蓝色失焦环
FRINGE_SRC = 0.7             # 紫边只来自曝光后亮度超过此值的饱和高光
FRINGE_COLOR = (0.3, 0.45, 1.0)  # 镶边颜色（偏蓝，避免逼近侧被染紫）
CA_LATERAL = 0.0025          # 横向色散：R 通道放大、B 通道缩小的径向比例（画面边缘红/蓝镶边）


WHITE_BALANCE_K = 5000.0   # 相机白平衡色温（K）；6600 为不校正
WHITE_BLEND = 0.12         # 超色域高光向白混合斜率（模拟传感器饱和；越大高光越白、越吃色）


def _cie_cmf(lam):
    """CIE 1931 2° 色匹配函数的多瓣高斯解析近似（Wyman, Sloan & Shirley 2013）。lam 单位 nm。"""
    def g(x, mu, s1, s2):
        sig = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / sig) ** 2)
    xb = 1.056 * g(lam, 599.8, 37.9, 31.0) + 0.362 * g(lam, 442.0, 16.0, 26.7) - 0.065 * g(lam, 501.1, 20.4, 26.2)
    yb = 0.821 * g(lam, 568.8, 46.9, 40.5) + 0.286 * g(lam, 530.9, 16.3, 31.1)
    zb = 1.217 * g(lam, 437.0, 11.8, 36.0) + 0.681 * g(lam, 459.0, 26.0, 13.8)
    return xb, yb, zb


def blackbody_rgb_np(t_k):
    """普朗克谱 → CIE XYZ → 线性 sRGB(D65)，负值（色域外）裁 0，亮度归一。

    Formula: B_λ(T) ∝ λ⁻⁵ / (exp(hc/(λkT)) − 1)；XYZ = ∫ B_λ·cmf dλ；RGB = M_sRGB · XYZ
    Physical Meaning: 温度 T 的黑体在标准观察者与 sRGB 显示下的真实色度（物理白点 D65 ≈ 6500K）。
    """
    lam = np.linspace(380.0, 780.0, 401)
    x = 1.4388e7 / (lam * t_k)
    spec = lam ** -5 / np.expm1(np.minimum(x, 700.0))
    xb, yb, zb = _cie_cmf(lam)
    xyz = np.array([(spec * xb).sum(), (spec * yb).sum(), (spec * zb).sum()])
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.maximum(m @ xyz, 0.0)
    return rgb / max(float(rgb @ [0.2126, 0.7152, 0.0722]), 1e-30)


def build_bb_lut():
    """把 blackbody_rgb_np 采样到 bb_lut（log T 等间距）。"""
    ts = np.exp(np.linspace(math.log(1000.0), math.log(40000.0), BB_N))
    bb_lut.from_numpy(np.stack([blackbody_rgb_np(t) for t in ts]).astype(np.float32))


def white_balance_gain(t_wb):
    """von Kries 白平衡增益：让色温 t_wb 的黑体呈中性灰，亮度不变。

    Formula: gain_c = 1 / rgb_c(T_wb)，再整体缩放使 Σ w_c·gain_c·rgb_c = Σ w_c·rgb_c 的亮度权重不变。
    Physical Meaning: 相机按场景色温设定白点；低于 6600K 时暖色被中和，频移造成的相对冷暖差更易辨认。
    """
    c = blackbody_rgb_np(t_wb)
    gain = 1.0 / np.maximum(c, 1e-3)
    w = np.array([0.2126, 0.7152, 0.0722])
    return gain * (w @ c) / (w @ (gain * c)) / (w @ c) * (w @ c)


def lateral_ca(x, k):
    """横向色散：R 通道以画面中心为原点放大 (1+k)，B 通道缩小 (1−k)，双线性重采样。x: (H, W, 3)。"""
    if k <= 0:
        return x
    h, w, _ = x.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    out = x.copy()
    for c, scale in ((0, 1.0 + k), (2, 1.0 - k)):
        sy = cy + (yy - cy) / scale
        sx = cx + (xx - cx) / scale
        y0 = np.clip(np.floor(sy).astype(int), 0, h - 2)
        x0 = np.clip(np.floor(sx).astype(int), 0, w - 2)
        fy = np.clip(sy - y0, 0, 1)
        fx = np.clip(sx - x0, 0, 1)
        ch = x[..., c]
        out[..., c] = ((1 - fy) * ((1 - fx) * ch[y0, x0] + fx * ch[y0, x0 + 1])
                       + fy * ((1 - fx) * ch[y0 + 1, x0] + fx * ch[y0 + 1, x0 + 1]))
    return out


def tonemap(hdr, exposure):
    """曝光 → 能量守恒 bloom（HDR 域 PSF 卷积）→ 保色度 ACES → sRGB。hdr: (W, H, 3) Taichi 布局。

    bloom：只让高光（> BLOOM_SRC）经 PSF 翼部散射，x' = x + G·PSF∗max(x − S, 0)。
    避免"整盘都在散光"给盘面罩一层纱，把云雾对比和蓝移冲淡。
    """
    x = np.transpose(hdr, (1, 0, 2)) * exposure
    if WHITE_BALANCE_K < 6599.0:
        x = x * white_balance_gain(WHITE_BALANCE_K)
    h = x.shape[0]
    src = np.maximum(x - BLOOM_SRC, 0.0)
    bloom = np.zeros_like(x)
    for c in range(3):
        sc = CA_AXIAL[c]
        ch = src[..., c:c + 1]
        # 近核（紫边主要来源，半径按通道放大）+ 远翼（大范围光晕）
        bloom[..., c:c + 1] = (0.25 * box_blur(ch, max(1, int(h / 120 * sc)))
                               + 0.35 * box_blur(ch, max(2, int(h / 25 * sc)))
                               + 0.40 * box_blur(ch, max(4, int(h / 7 * sc))))
    x = x + BLOOM_GAIN * bloom
    if CA_FRINGE > 0:
        # 紫边：饱和高光（消色差）经"蓝紫失焦核 − 合焦核"得到外缘环，染成蓝紫色
        lum_x = x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722
        hs = np.maximum(lum_x - FRINGE_SRC, 0.0)[..., None]
        ring = np.maximum(box_blur(hs, max(2, h // 90)) - box_blur(hs, max(1, h // 400)), 0.0)
        x = x + CA_FRINGE * ring * np.array(FRINGE_COLOR, dtype=x.dtype)
    x = lateral_ca(x, CA_LATERAL)
    # 保色度色调映射：只对亮度做 ACES，RGB 按比例缩放，避免高光被逐通道压成白色
    lum = x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722
    lum_t = (lum * (2.51 * lum + 0.03)) / (lum * (2.43 * lum + 0.59) + 0.14)
    y = x * (lum_t / np.maximum(lum, 1e-6))[..., None]
    # 超出色域的通道按比例收回，并向白色做少量混合（真实传感器高光饱和）
    m = y.max(-1, keepdims=True)
    w_white = np.clip((m - 1.0) * WHITE_BLEND, 0.0, 0.25)
    y = y / np.maximum(m, 1.0)
    y = np.clip(y * (1 - w_white) + w_white, 0.0, 1.0)
    y = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8)


EXP_TARGET = 0.6   # 自动曝光：盘区亮度 p99.9 映射到此值（越大整体越亮，高光越易压白）


def auto_exposure(hdr, pct=99.9, target=None):
    """把盘区亮度的 pct 分位映射到 target（默认 EXP_TARGET）。"""
    if target is None:
        target = EXP_TARGET
    lum = hdr[..., 0] * 0.2126 + hdr[..., 1] * 0.7152 + hdr[..., 2] * 0.0722
    v = lum[lum > 1e-4]
    return target / float(np.percentile(v, pct)) if v.size else 1.0


def save_png(path, rgb):
    from PIL import Image
    Image.fromarray(rgb).save(path)
    print("saved", path)


class VideoWriter:
    """逐帧写 H.264（避免整段视频驻留内存）。"""

    def __init__(self, path, fps):
        import imageio.v3 as iio
        self.path = path
        self._f = iio.imopen(path, "w", plugin="pyav")
        self._f.init_video_stream("libx264", fps=fps)
        self.n = 0

    def write(self, frame):
        self._f.write_frame(frame)
        self.n += 1

    def close(self):
        self._f.close()
        print("saved", self.path, self.n, "frames")


def label(frame, text):
    """左上角叠加文字标签。"""
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame)
    ImageDraw.Draw(im).text((10, 8), text, fill=(230, 230, 230))
    return np.asarray(im)


CAM = dict(dist=30.0, elev_deg=7.0)
FOV_V = math.radians(34.0)


_FIELDS = {}


def render_hdr(w, h, t, mode, ss=1):
    """渲染 HDR；ss>1 时按 ss×ss 超采样后盒式下采样。返回 (w, h, 3)。"""
    key = (w * ss, h * ss)
    if key not in _FIELDS:
        _FIELDS[key] = ti.Vector.field(3, ti.f32, shape=key)
    field = _FIELDS[key]
    render_kernel(field, t, mode, FOV_V, False, field)
    arr = field.to_numpy()
    if ss > 1:
        arr = arr.reshape(w, ss, h, ss, 3).mean(axis=(1, 3))
    return arr


def cmd_frame(a):
    set_camera(**CAM)
    t0 = time.time()
    hdr = render_hdr(a.w, a.h, a.t, a.mode, a.ss)
    print(f"render {a.w}x{a.h} ss={a.ss}: {time.time() - t0:.2f}s (含编译)")
    t0 = time.time()
    hdr = render_hdr(a.w, a.h, a.t, a.mode, a.ss)
    print(f"render {a.w}x{a.h} ss={a.ss}: {time.time() - t0:.2f}s (纯运行)")
    lum = hdr[..., 0] * 0.2126 + hdr[..., 1] * 0.7152 + hdr[..., 2] * 0.0722
    half = hdr.shape[0] // 2
    print(f"左/右半幅总通量比 = {lum[:half].sum() / max(lum[half:].sum(), 1e-9):.2f}"
          f"  盘覆盖率 = {(lum > 1e-4).mean():.2f}")
    exp = auto_exposure(hdr)
    save_png(os.path.join(OUT_DIR, f"frame_{a.preset}_m{a.mode}_R{R_OUT:g}_T{a.t_peak:g}_L{a.doppler_lum:g}_C{a.doppler_color:g}_{a.w}x{a.h}.png"), tonemap(hdr, exp))
    np.save(os.path.join(OUT_DIR, "frame_hdr.npy"), hdr)


def cmd_video(a):
    """真实时间：内缘轨道周期 = a.orbit_s 秒视频。"""
    set_camera(**CAM)
    dt = period_k(R_IN) / (a.orbit_s * a.fps)
    vw = VideoWriter(os.path.join(OUT_DIR, f"demo_slow_m{a.mode}.mp4"), a.fps)
    exp = None
    t0 = time.time()
    for f in range(a.n):
        hdr = render_hdr(a.w, a.h, a.t0 + f * dt, a.mode, a.ss)
        if exp is None:
            exp = auto_exposure(hdr)
        frame = tonemap(hdr, exp)
        vw.write(frame)
        if f == 0:
            save_png(os.path.join(OUT_DIR, f"demo_slow_m{a.mode}_first.png"), frame)
        if f % 48 == 0:
            print(f"frame {f}/{a.n}  {time.time() - t0:.1f}s")
    vw.close()


def cmd_longrun(a):
    """time-lapse：a.orbits 个内缘轨道周期；左 naive，右分带。"""
    set_camera(**CAM)
    t_total = a.orbits * period_k(R_IN)
    vw = VideoWriter(os.path.join(OUT_DIR, f"longrun_naive_vs_m{a.mode}.mp4"), a.fps)
    exp = None
    t0 = time.time()
    frame = None
    for f in range(a.n):
        t = a.t0 + t_total * f / (a.n - 1)
        hn = render_hdr(a.w, a.h, t, 0)
        hb = render_hdr(a.w, a.h, t, a.mode)
        if exp is None:
            exp = auto_exposure(hb)
        orbit_txt = f"t = {(t - a.t0) / period_k(R_IN):.2f} inner orbits"
        top = label(tonemap(hn, exp), f"CURRENT (naive advection)   {orbit_txt}")
        bottom = label(tonemap(hb, exp), f"NEW (mode {a.mode}: rigid rings)   {orbit_txt}")
        frame = np.concatenate([top, np.full((4, a.w, 3), 60, np.uint8), bottom], axis=0)
        vw.write(frame)
        if f % 48 == 0:
            print(f"frame {f}/{a.n}  {time.time() - t0:.1f}s")
    vw.close()
    save_png(os.path.join(OUT_DIR, f"longrun_m{a.mode}_last.png"), frame)


def pitch_deg(logf, lr_span):
    """由 (ln r, φ) 网格上的梯度估计平均螺旋倾角（度）。

    对条纹 f(φ − k ln r)：∂f/∂lnr = −k f'，∂f/∂φ = f'，tan i = |∂f/∂φ| / |∂f/∂lnr| = 1/k。
    i = 90° 表示各向同性/径向条纹，i → 0 表示卷成同心圆。
    """
    nr, nphi = logf.shape
    d_lnr = lr_span / nr
    d_phi = 2 * math.pi / nphi
    g_r = np.diff(logf, axis=0)[:, :-1] / d_lnr
    g_p = np.diff(logf, axis=1)[:-1, :] / d_phi
    return math.degrees(math.atan(math.sqrt((g_p ** 2).mean() / max((g_r ** 2).mean(), 1e-12))))


def polar_to_faceon(logf, n=360):
    """(ln r, φ) → face-on 笛卡尔灰度图。"""
    nr, nphi = logf.shape
    ys, xs = np.mgrid[0:n, 0:n]
    x = (xs + 0.5) / n * 2 - 1
    y = 1 - (ys + 0.5) / n * 2
    r = np.hypot(x, y) * R_OUT
    phi = np.arctan2(y, x)
    lr0, lr1 = math.log(R_IN * 1.05), math.log(R_OUT * 0.95)
    with np.errstate(divide="ignore"):
        ii = ((np.log(np.maximum(r, 1e-6)) - lr0) / (lr1 - lr0) * nr).astype(int)
    jj = ((phi + math.pi) / (2 * math.pi) * nphi).astype(int) % nphi
    valid = (ii >= 0) & (ii < nr)
    v = np.zeros((n, n))
    v[valid] = logf[ii[valid], jj[valid]]
    img_ = np.clip((v + 2.0) / 4.5, 0, 1)
    img_[~valid] = 0
    return (img_ * 255).astype(np.uint8)


def structure_stats(t=2000.0, mode=3):
    """中面结构统计：空洞率、空洞平均弧长、径向规律性（周期峰值 / 中位谱）。"""
    nr, nphi = 1024, 4096
    buf = ti.field(ti.f32, shape=(nr, nphi))
    density_polar_kernel(buf, t, mode)
    f = buf.to_numpy()
    lr0, lr1 = math.log(R_IN * 1.15), math.log(R_OUT * 0.7)
    r = np.exp(np.linspace(lr0, lr1, nr))
    void = f < 0.1 * np.median(f)
    runs = []
    for i in range(0, nr, 16):
        row = np.concatenate([void[i], [False]])
        edges = np.flatnonzero(np.diff(np.concatenate([[0], row.astype(int)])))
        lengths = edges[1::2] - edges[0::2]
        runs.extend((lengths * 2 * math.pi / nphi * r[i]).tolist())
    # 径向规律性：沿每条径向线（固定 φ）对 log 密度做自相关，取第一个零点之后的最大次峰。
    # 准周期（等间距环）→ 次峰高（>0.3）；宽带随机 → 次峰低（<0.15）
    g = np.log(np.maximum(f, 1e-3))
    g = g - g.mean(0, keepdims=True)
    spec = np.abs(np.fft.rfft(g, n=2 * nr, axis=0)) ** 2
    ac = np.fft.irfft(spec.mean(1))[:nr]
    ac = ac / ac[0]
    z0 = int(np.argmax(ac < 0)) if np.any(ac < 0) else nr // 4
    second = float(ac[z0:nr // 4].max()) if z0 < nr // 4 else 0.0
    fwhm_lag = int(np.argmax(ac < 0.5))
    return dict(void_frac=float(void.mean()), void_arc=float(np.mean(runs)) if runs else 0.0,
                periodicity=second, corr_len_rs=float(fwhm_lag * (lr1 - lr0) / nr * 6.0))


def cmd_stats(a):
    st = structure_stats(a.t, a.mode)
    print(f"空洞率 {st['void_frac']*100:.1f}%  空洞平均弧长 {st['void_arc']:.2f} r_s  "
          f"径向自相关次峰 {st['periodicity']:.3f}  径向相关长度(@r=6) {st['corr_len_rs']:.3f} r_s")


def cmd_physcheck(a):
    """物理自检：GR 解析频移对比、近/远侧纹理运动方向、亮侧 = 蓝移侧。"""
    set_camera(**CAM)
    w, h = 640, 360
    col = ti.Vector.field(3, ti.f32, shape=(w, h))
    dbg = ti.Vector.field(4, ti.f32, shape=(w, h))
    dbg.fill(0.0)
    render_kernel(col, a.t, a.mode, FOV_V, True, dbg)
    dd = dbg.to_numpy()
    hit = dd[..., 3] > 0
    g_new, g_ex, g_old = dd[..., 0][hit], dd[..., 1][hit], dd[..., 2][hit]
    err_new = np.abs(g_new / g_ex - 1)
    err_old = np.abs(g_old / g_ex - 1)
    print(f"[1] 频移 vs 严格 GR（{hit.sum()} 个赤道面命中像素）")
    print(f"    新(本地方向): 平均误差 {err_new.mean()*100:.2f}%  最大 {err_new.max()*100:.2f}%")
    print(f"    旧(坐标方向): 平均误差 {err_old.mean()*100:.2f}%  最大 {err_old.max()*100:.2f}%")
    xs = np.arange(w)[:, None].repeat(h, 1)[hit]
    left, right = xs < w * 0.35, xs > w * 0.65
    print(f"    左侧 g 均值 {g_ex[left].mean():.3f}（期望 >1）  右侧 {g_ex[right].mean():.3f}（期望 <1）"
          f"  符号一致率 {(np.sign(g_new - 1) == np.sign(g_ex - 1)).mean()*100:.1f}%")
    # [2] 纹理运动：两帧互相关求水平位移
    dt = period_k(8.0) / 50.0
    f0 = render_hdr(w, h, a.t, a.mode)
    f1 = render_hdr(w, h, a.t + dt, a.mode)
    lum0 = f0 @ np.array([0.2126, 0.7152, 0.0722])
    lum1 = f1 @ np.array([0.2126, 0.7152, 0.0722])
    rows_lum = lum0.sum(0)
    def shift_x(r0, r1):
        """行平均后的一维互相关水平位移（像素，正 = 向右）。"""
        c0, c1 = int(w * .2), int(w * .8)
        s0 = np.log(lum0[c0:c1, r0:r1] + 1e-3).mean(1)
        s1 = np.log(lum1[c0:c1, r0:r1] + 1e-3).mean(1)
        ker = np.ones(21) / 21  # 高通：去掉沿 x 的慢变包络
        s0 = s0 - np.convolve(s0, ker, "same")
        s1 = s1 - np.convolve(s1, ker, "same")
        cc = [np.dot(s0[70:-70], np.roll(s1, -k)[70:-70]) for k in range(-60, 61)]
        return int(np.argmax(cc)) - 60
    # 近侧外盘（r≈10）在图像中位于中心下方 10·sin(elev) / 像素尺度 处
    pix = 2 * math.tan(FOV_V / 2) * CAM["dist"] / h
    near_row = h // 2 + int(10 * math.sin(math.radians(CAM["elev_deg"])) / pix)
    sn = shift_x(near_row - 3, near_row + 4)
    # 盘面极坐标：纹理沿 φ 的位移（逆时针 = 正）
    nr_, nph_ = 256, 2048
    pb = ti.field(ti.f32, shape=(nr_, nph_))
    density_polar_kernel(pb, a.t, a.mode); q0 = np.log(pb.to_numpy() + 1e-4)
    density_polar_kernel(pb, a.t + dt, a.mode); q1 = np.log(pb.to_numpy() + 1e-4)
    spec = (np.fft.fft(q0, axis=1).conj() * np.fft.fft(q1, axis=1)).sum(0)
    kshift = int(np.argmax(np.fft.ifft(spec).real))
    kshift = kshift - nph_ if kshift > nph_ // 2 else kshift
    print(f"[2] 纹理运动（Δt = {dt:.2f}）：盘面 Δφ = {kshift * 360 / nph_:+.2f}°（期望 >0，逆时针，与 [1] 的旋转方向一致）"
          f"；图像近侧外盘(行 {near_row}) 位移 {sn:+d} px（期望 >0，向右）")
    # [3] 亮侧 = 蓝移侧
    ldr = tonemap(f0, auto_exposure(f0)).astype(float)
    L = ldr @ [0.2126, 0.7152, 0.0722]
    lm = L[:, :w // 3] > 8; rm_ = L[:, 2 * w // 3:] > 8
    lb = ldr[:, :w // 3][lm]; rb = ldr[:, 2 * w // 3:][rm_]
    print(f"[3] 左1/3 亮度 {L[:, :w//3][lm].mean():.0f}  B/R {lb[:,2].mean()/lb[:,0].mean():.2f} | "
          f"右1/3 亮度 {L[:, 2*w//3:][rm_].mean():.0f}  B/R {rb[:,2].mean()/rb[:,0].mean():.2f}（期望：左侧更亮且 B/R 更高）")
    # [4] Planck 增强单调
    tt = np.array([2500.0, 4500.0, 7000.0]); gg = np.array([0.6, 1.0, 1.5])
    xb = lambda T: np.minimum(PLANCK_X / T, 80)
    ok = all(((np.exp(xb(T)) - 1) / (np.exp(xb(T * gg)) - 1)).tolist() == sorted(((np.exp(xb(T)) - 1) / (np.exp(xb(T * gg)) - 1)).tolist()) for T in tt)
    print(f"[4] Planck 波段增强随 g 单调递增: {ok}")


def cmd_winding(a):
    from PIL import Image, ImageDraw
    nr, nphi = 1024, 2048
    buf = ti.field(ti.f32, shape=(nr, nphi))
    lr_span = math.log(R_OUT * 0.95) - math.log(R_IN * 1.05)
    orbits = [0.0, 1.0, 3.0, 10.0, 30.0]
    names = ["naive", "per-radius phase", "log-r bands", "rigid rings"]
    p_in = period_k(R_IN)
    tiles = []
    print(f"内缘 P={p_in:.1f}, 外缘 P={period_k(R_OUT):.1f} (r_s/c)")
    print("平均螺旋倾角（度，越小越卷；inner/mid/outer 三段）：")
    for mode in range(4):
        row = []
        for o in orbits:
            polar_kernel(buf, a.t0 + o * p_in, mode)
            lf = buf.to_numpy()
            thirds = [pitch_deg(lf[k * nr // 3:(k + 1) * nr // 3], lr_span / 3) for k in range(3)]
            print(f"  {names[mode]:>17s} @ {o:5.1f} orbits: " + " / ".join(f"{v:5.1f}" for v in thirds))
            row.append(polar_to_faceon(lf))
        tiles.append(row)
    n = tiles[0][0].shape[0]
    pad = 24
    canvas = Image.new("L", (len(orbits) * n, len(names) * (n + pad)), 0)
    draw = ImageDraw.Draw(canvas)
    for mi, row in enumerate(tiles):
        for oi, tile in enumerate(row):
            canvas.paste(Image.fromarray(tile), (oi * n, mi * (n + pad) + pad))
            draw.text((oi * n + 6, mi * (n + pad) + 4), f"{names[mi]}  t={orbits[oi]:g} inner orbits", fill=255)
    path = os.path.join(OUT_DIR, "winding_compare.png")
    canvas.save(path)
    print("saved", path)


PRESETS = {
    # 当前版（对照）：仅换 CIE 黑体 + 去品红光晕
    "base": {},
    # A：外圈变暗放缓 + 大尺度低频明暗
    "A": dict(EMIT_POW=2.5, LOWF_SIGMA=0.6),
    # B：A + 核心纹理更稀疏、低频更强
    "B": dict(EMIT_POW=2.5, LOWF_SIGMA=0.9, FR=16.0, NPHI=7, SIGMA_LN=1.3, FIL_GAIN=5.0),
    # C：B + 5 层烟雾拉开、加厚、偏冷吸收、蓬松（低各向异性）
    "C": dict(EMIT_POW=2.5, LOWF_SIGMA=0.9, FR=16.0, NPHI=7, SIGMA_LN=1.3, FIL_GAIN=5.0,
              CL_SPACING=0.03, CL_WIDTH=0.012, CLOUD_COL=0.7, CLOUD_EMIT=0.25,
              FR_C=8.0, NPHI_C=10, SIGMA_C=1.2, CLOUD_C0=0.4, CLOUD_SOFT=0.35),
    # D：B + 连续烟雾冕（7 层重叠 ≈ 连续），偏冷吸收
    "D": dict(EMIT_POW=2.5, LOWF_SIGMA=0.9, FR=16.0, NPHI=7, SIGMA_LN=1.3, FIL_GAIN=5.0,
              N_CL_HALF=3, CL_SPACING=0.014, CL_WIDTH=0.011, CL_DECAY=0.35, CLOUD_COL=0.8, CLOUD_EMIT=0.25,
              FR_C=8.0, NPHI_C=10, SIGMA_C=1.2, CLOUD_C0=0.4, CLOUD_SOFT=0.35),
    # E：D，但烟雾也发光（暖色辉光烟雾）
    "E": dict(EMIT_POW=2.5, LOWF_SIGMA=0.9, FR=16.0, NPHI=7, SIGMA_LN=1.3, FIL_GAIN=5.0,
              N_CL_HALF=3, CL_SPACING=0.014, CL_WIDTH=0.011, CL_DECAY=0.35, CLOUD_COL=0.8, CLOUD_EMIT=0.7,
              FR_C=8.0, NPHI_C=10, SIGMA_C=1.2, CLOUD_C0=0.4, CLOUD_SOFT=0.35),
    # F：D/E 之间（烟雾半发光），外圈衰减更缓
    "F": dict(EMIT_POW=2.0, LOWF_SIGMA=0.9, FR=16.0, NPHI=7, SIGMA_LN=1.3, FIL_GAIN=5.0,
              N_CL_HALF=3, CL_SPACING=0.014, CL_WIDTH=0.011, CL_DECAY=0.35, CLOUD_COL=0.8, CLOUD_EMIT=0.45,
              FR_C=8.0, NPHI_C=10, SIGMA_C=1.2, CLOUD_C0=0.4, CLOUD_SOFT=0.35, EXP_TARGET=0.7),
    # G：F 再进一步
    "G": dict(EMIT_POW=1.7, LOWF_SIGMA=0.9, FR=16.0, NPHI=7, SIGMA_LN=1.3, FIL_GAIN=5.0,
              N_CL_HALF=3, CL_SPACING=0.014, CL_WIDTH=0.011, CL_DECAY=0.35, CLOUD_COL=0.8, CLOUD_EMIT=0.45,
              FR_C=8.0, NPHI_C=10, SIGMA_C=1.2, CLOUD_C0=0.4, CLOUD_SOFT=0.35, EXP_TARGET=0.75),
    # H：外圈衰减同 D/E（2.5），烟雾偏 D（半吸收），整体亮度略提
    "H": dict(EMIT_POW=2.5, LOWF_SIGMA=0.9, FR=16.0, NPHI=7, SIGMA_LN=1.3, FIL_GAIN=5.0,
              N_CL_HALF=3, CL_SPACING=0.014, CL_WIDTH=0.011, CL_DECAY=0.35, CLOUD_COL=0.8, CLOUD_EMIT=0.33,
              FR_C=8.0, NPHI_C=10, SIGMA_C=1.2, CLOUD_C0=0.4, CLOUD_SOFT=0.35, EXP_TARGET=0.7),
}


def apply_preset(name):
    """把预设写入模块全局（须在任何 kernel 编译前调用），并重算派生常量。"""
    g = globals()
    g.update(PRESETS[name])
    g["CL_EXTENT"] = g["N_CL_HALF"] * g["CL_SPACING"] + 3.0 * g["CL_WIDTH"]
    g["HR_CLOUD"] = g["CL_EXTENT"] / 3.5
    g["CL_AMP_NORM"] = 1.0 / sum(math.exp(-g["CL_DECAY"] * abs(k)) for k in range(-g["N_CL_HALF"], g["N_CL_HALF"] + 1))


def main():
    global BLOOM_SRC, BLOOM_GAIN, CA_LATERAL, CA_AXIAL, CA_FRINGE, WHITE_BALANCE_K, R_OUT, FOV_V
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frame", "video", "longrun", "winding", "stats", "physcheck"])
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--t", type=float, default=2000.0)
    ap.add_argument("--t0", type=float, default=2000.0)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--orbit_s", type=float, default=16.0, help="内缘一圈对应的视频秒数")
    ap.add_argument("--orbits", type=float, default=20.0, help="longrun 覆盖的内缘圈数")
    ap.add_argument("--k_life", type=float, default=K_LIFE_DEFAULT)
    ap.add_argument("--k_rigid", type=float, default=K_RIGID_DEFAULT)
    ap.add_argument("--doppler_lum", type=float, default=0.5,
                    help="亮度频移强度 s，亮度用 g^s：1 物理，0 关闭")
    ap.add_argument("--doppler_color", type=float, default=2.2,
                    help="颜色频移强度 s，色温用 T·g^s：1 物理，0 关闭")
    ap.add_argument("--t_peak", type=float, default=T_PEAK_VIS, help="盘峰值可视色温（K）")
    ap.add_argument("--bloom_src", type=float, default=BLOOM_SRC, help="bloom 高光阈值（越低光晕越多、盘面越糊）")
    ap.add_argument("--bloom_gain", type=float, default=BLOOM_GAIN, help="bloom 增益")
    ap.add_argument("--ca_lateral", type=float, default=CA_LATERAL, help="横向色散强度（0 关闭）")
    ap.add_argument("--ca_axial", type=float, default=1.0, help="轴向色散（光晕按通道半径差）倍率，0 关闭")
    ap.add_argument("--ca_fringe", type=float, default=CA_FRINGE, help="紫边强度，0 关闭")
    ap.add_argument("--wb", type=float, default=WHITE_BALANCE_K, help="相机白平衡色温（K），6600 为不校正")
    ap.add_argument("--r_out", type=float, default=30.0, help="盘外半径（r_s）；须在首次渲染前设置")
    ap.add_argument("--dist", type=float, default=60.0, help="相机距离（r_s）")
    ap.add_argument("--fov", type=float, default=38.0, help="竖直视场角（度）")
    ap.add_argument("--preset", default="H", choices=list(PRESETS), help="外观预设（H 为定稿）")
    ap.add_argument("--ss", type=int, default=1, help="超采样倍率（每轴）")
    ap.add_argument("--elev", type=float, default=CAM["elev_deg"], help="相机仰角（度）")
    ap.add_argument("--mode", type=int, default=3, help="0 naive / 1 逐半径相位 / 2 分带平流 / 3 刚体环")
    a = ap.parse_args()
    R_OUT = a.r_out
    apply_preset(a.preset)
    setup(a.k_life, a.k_rigid, a.doppler_lum, a.doppler_color, a.t_peak)
    CAM["elev_deg"] = a.elev
    CAM["dist"] = a.dist
    FOV_V = math.radians(a.fov)
    BLOOM_SRC, BLOOM_GAIN = a.bloom_src, a.bloom_gain
    CA_LATERAL = a.ca_lateral
    CA_AXIAL = tuple(1.0 + (v - 1.0) * a.ca_axial for v in CA_AXIAL)
    CA_FRINGE = a.ca_fringe
    WHITE_BALANCE_K = a.wb
    {"frame": cmd_frame, "video": cmd_video, "longrun": cmd_longrun, "winding": cmd_winding, "stats": cmd_stats, "physcheck": cmd_physcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
