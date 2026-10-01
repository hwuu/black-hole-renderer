"""吸积盘真实感参考实现（独立脚本，不依赖 render.py / disk_v2）。

用途：
1. V2 体积云雾 + 动态旋转方案的**视觉验收基准**（docs/plans/v2_volumetric_video_plan.md §2）。
   V2 移植后需与本脚本默认输出（预设 M、mode 3）等价。
2. 物理自检工具：`physcheck` 逐像素对比严格 GR 频移、旋转方向、亮侧 = 蓝移侧。
3. `params` 子命令：按三层（基本参数 / 物理模型 / 视觉调节）列出全部参数的当前值与物理值。

参数三层（详见 params 与方案 §2）：
- 第 1 层 基本参数：黑洞质量 M、吸积率 Ṁ（→ Page–Thorne 绝对通量推出 T_peak）、盘内外半径、相机、分辨率、视频速度。
- 第 2 层 物理模型：Page–Thorne 温度、SS 外区 H(r)/Σ(r) + 竖直高斯、观测亮度 Y(g·T)、灰大气、湍流起伏、
  盘风烟雾（温度比）、开普勒尘埃、光行时间、静止观者相机。
- 第 3 层 视觉调节：多普勒强度、灰大气强度、核心光学深度、盘厚、大尺度明暗、程序化结构、曝光、白平衡、bloom、色散。

预设：
- M（默认，定稿）：上述全部物理修正；多普勒 0.55 / 1.5、曝光 0.9。
- Mphys：M 的第 3 层艺术偏离全取物理值（另需 --doppler_lum 1 --doppler_color 1）。
- L / J2 / H：旧定稿，可退回（命令见下）。其余预设为调参过程中的对照。

单位：r_s = c = 1，M = 0.5。时间单位 r_s / c。

常用命令（默认即定稿：预设 M、mode 3、r_out = 30、相机 dist = 40 / fov = 38°）：
    python scripts/proto_disk_reference.py frame --ss 2     # 1080p 单帧
    python scripts/proto_disk_reference.py video            # 慢速实时视频（内缘 16 s 一圈）
    python scripts/proto_disk_reference.py physcheck        # 物理自检
    python scripts/proto_disk_reference.py params           # 三层参数表

退回旧定稿（与历史图逐像素一致）：
    ... frame --ss 2 --preset L  --doppler_lum 0.75
    ... frame --ss 2 --preset J2 --dist 60 --doppler_lum 0.5 --doppler_color 2.2
    ... frame --ss 2 --preset H  --dist 60 --doppler_lum 0.5 --doppler_color 2.2

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

# ---------------- 模型 I（借鉴 NPGS / Baopinsui 最新 shader，单位 r_s）----------------
DISK_MODEL = 0        # 0 = 定稿 H（薄核心 + 7 层烟雾）；1 = 模型 I
THIN_I = 0.22         # 基础半厚度（r_s，绝对尺度，不按 H ∝ r）
HOPPER_I = 0.008      # 厚度随半径的张开斜率：H_geo = THIN_I + HOPPER_I·(R − R_IN)
SHAPE_A = 0.9         # 截面轮廓 Shape(x, a, b) = k·x^a·(1 − x)^b：两端收口的透镜状截面
SHAPE_B = 1.5
EFF_SPAN = 5.0        # 有效半径压缩尺度（r_s）：大盘时轮廓峰值靠近内区
KR_I = 0.2            # 主云噪声的径向 / 竖直基频（每 r_s），两者同尺度 → 盘内部有真实竖直结构
NPHI_I = 2            # 主云噪声方位基频（整数周期，无接缝）
L0_I = 3.0            # 主云噪声起始八度（倍率 3），使用 [L0, L0+2) 两个八度
CON_I = 80.0          # 乘性级联的软阈值对比度：out = log(1 + (Π(1+0.1 n))^CON)
KT_I = 2.0            # 厚度扰动噪声径向基频（每 r_s）
NPHI_T = 9            # 厚度扰动噪声方位基频
LT0_I = 0.7           # 厚度扰动噪声起始八度（[LT0, LT0+2)）
DUST_S = 1.0          # 尘埃源函数（相对主云表面增亮因子的量级）
SURF_LO = 0.2         # 表面增亮：发射 × (SURF_LO + SURF_K·|z|/H)，中心暗、表面亮（光学厚盘只见表面）
SURF_K = 2.0
TAU_I = 1.5           # r = 6 处 face-on 竖直光学深度（用于标定吸收系数）
DUST_EM = 0.02        # 内区尘埃发射（相对主云）
DUST_ON = 1
V_IN = 0.04           # mode 4 螺线内流径向速度（c）；螺线倾角 tan i = V_IN / (rΩ)
ARM_W = 0.0           # mode 4 外圈密度波旋臂强度（0 关闭）
R_ARM = 10.0          # 旋臂图样角速度取 Ω(R_ARM)，整体自转
K_LOG = 2.0           # 旋臂对数螺线系数：θ_arm = φ − Ω_arm·t + K_LOG·ln r（拖尾）
H_IN_I = 0.03         # 盘体内采样步长（r_s），配合起点抖动 + 超采样做蒙特卡洛积分
LIGHT_DELAY = 0       # 1 = 采样时间 t_emit = t − 光程（光行时间）
SMOKE_I = 0.0         # I+H：叠加 H 的多层半吸收烟雾，烟雾柱密度 = SMOKE_I × Σ(r)（0 关闭）
SMOKE_S = 0.33        # 烟雾源函数（相对核心；H 的 CLOUD_EMIT）
GREY_ATM = 0          # 1 = 灰大气竖直温度 T⁴ = ¾·T_eff⁴·(τ_z + ⅔)（τ_z 为到表面的竖直光学深度）→ 临边昏暗
# ---- 第 1 层：基本参数（M、Ṁ 推出 T_peak）----
BH_MASS_MSUN = 1.0e8  # 黑洞质量（太阳质量）；图像与尺度无关，只经 Ṁ 决定温度
MDOT_EDD = 1.7e-6     # 吸积率（爱丁顿吸积率倍数，η = 1 − sqrt(8/9)）
T_FROM_MDOT = 0       # 1 = T_peak 由 M、Ṁ 经 Page–Thorne 绝对通量推出；0 = 用 T_PEAK_VIS
# ---- 第 2 层：物理结构 ----
PHYS_STRUCT = 0       # 1 = Shakura–Sunyaev 外区（Kramers）结构：H/r ∝ r^{1/8} f^{3/20}，Σ ∝ r^{-3/4} f^{7/10}，竖直高斯
HR_REF = 0.027        # r = R_REF 处 H/r（SS 理论对这里的 M、Ṁ 给出 ≪ 0.01；0.027 为视觉取值，见三层参数表）
R_REF = 10.0
SURF_NOISE = 0.6      # 表面起伏幅度：H_s = H·(1 − SURF_NOISE + SURF_NOISE·softsat(tn))
DUST_KEPLER = 0       # 1 = 尘埃按当地开普勒刚体环旋转（与主云同流场），并去掉重复计算的掠射增亮
SMOKE_TR = 0.0        # > 0：烟雾（盘风团块）温度 = SMOKE_TR·T(r)，取代亮度比例 SMOKE_S
STATIC_CAM = 0        # 1 = 相机为静止观者本地标架：tanψ_coord = tanψ_local / sqrt(1 − r_s/r)
GREY_MIX = 1.0        # 灰大气强度：温度倍率 = 1 + GREY_MIX·(T_grey/T_eff − 1)；0 = 竖直均匀，1 = 完整灰大气
GREY_CAP = 1.19       # 灰大气温度倍率上限 = τ≈2 处的值：光线从起伏侧壁斜入时，平行平面近似会误判为深层高温
CORE_OPAC = 1.0       # 核心吸收倍率（相对 TAU_I 标定）；> 1 → 不透明核心（烟雾 / 尘埃不受影响）
CORE_FLOOR = 0.0      # > 0：核心密度 = FLOOR + (1 − FLOOR)·c/⟨c⟩（温和起伏、无空洞）；0 = 原乘性级联（有空洞）
DT_I = 0.0            # 核心温度起伏：T ← T·(1 + DT_I·(c/⟨c⟩ − 1))（发热率起伏）
TEMP_PT = 0           # 1 = Page–Thorne 相对论温度剖面（数值积分查表），0 = 牛顿近似
PHYS_LUM = 0          # 1 = 源函数亮度用物理可见光亮度 Y(g·T)/Y(T_peak)（普朗克谱 × CIE ȳ 积分），取代 (T/T_peak)^EMIT_POW·B_550

F_REF_SS = 1.0 - math.sqrt(R_IN / 10.0)   # SS 剖面归一：f(R_REF)，R_REF = 10

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

params = ti.field(ti.f32, shape=14)   # 12: kappa_I  # 0 k_life, 1 normA, 2 normB, 3 kappa, 4 emit, 5 k_rigid, 6 doppler_lum, 7 doppler_color, 8 t_peak, 9 normC
cam_f = ti.Vector.field(3, ti.f32, shape=4)  # pos, fwd, right, up
jit_seed = ti.field(ti.i32, shape=())        # 每次渲染递增，用于步长抖动
BB_N = 512
bb_lut = ti.Vector.field(3, ti.f32, shape=BB_N)  # CIE 黑体线性 sRGB 色度表，log T ∈ [log 1000, log 40000]
LNY_T0, LNY_T1 = 300.0, 60000.0
tpt_lut = ti.field(ti.f32, shape=BB_N)            # Page–Thorne T(r)/T_peak，r ∈ [R_IN, R_OUT] 等间距
lny_lut = ti.field(ti.f32, shape=BB_N)            # ln Y(T)：黑体可见光亮度（CIE ȳ 积分），log T ∈ [log 300, log 60000]


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
        if ti.static(TEMP_PT == 1):
            u = ti.min((r - R_IN) / (R_OUT - R_IN), 1.0) * (BB_N - 1)
            i0 = ti.min(ti.cast(ti.floor(u), ti.i32), BB_N - 2)
            w = u - ti.cast(i0, ti.f32)
            out = params[8] * (tpt_lut[i0] * (1.0 - w) + tpt_lut[i0 + 1] * w)
        else:
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
def ln_luminance(t_k):
    """黑体可见光亮度的对数 ln Y(T)，查表 lny_lut（log T 线性插值）。"""
    u = (ti.log(ti.min(ti.max(t_k, LNY_T0), LNY_T1)) - math.log(LNY_T0)) / (math.log(LNY_T1) - math.log(LNY_T0))
    f = u * (BB_N - 1)
    i0 = ti.min(ti.cast(ti.floor(f), ti.i32), BB_N - 2)
    w = f - ti.cast(i0, ti.f32)
    return lny_lut[i0] * (1.0 - w) + lny_lut[i0 + 1] * w


@ti.func
def _smooth3(t):
    return t * t * (3.0 - 2.0 * t)


@ti.func
def vnoise(x, y, z, period):
    """3D value noise（格点随机值 + 三次平滑插值），y 方向周期 period（整数）。值域约 [-1, 1]。"""
    fx0 = ti.floor(x)
    fy0 = ti.floor(y)
    fz0 = ti.floor(z)
    xi = ti.cast(fx0, ti.i32)
    yi = ti.cast(fy0, ti.i32)
    zi = ti.cast(fz0, ti.i32)
    ux = _smooth3(x - fx0)
    uy = _smooth3(y - fy0)
    uz = _smooth3(z - fz0)
    y0 = ((yi % period) + period) % period
    y1 = (((yi + 1) % period) + period) % period
    v000 = _hashf(xi, y0, zi) * 2.0 - 1.0
    v100 = _hashf(xi + 1, y0, zi) * 2.0 - 1.0
    v010 = _hashf(xi, y1, zi) * 2.0 - 1.0
    v110 = _hashf(xi + 1, y1, zi) * 2.0 - 1.0
    v001 = _hashf(xi, y0, zi + 1) * 2.0 - 1.0
    v101 = _hashf(xi + 1, y0, zi + 1) * 2.0 - 1.0
    v011 = _hashf(xi, y1, zi + 1) * 2.0 - 1.0
    v111 = _hashf(xi + 1, y1, zi + 1) * 2.0 - 1.0
    a0 = v000 + ux * (v100 - v000)
    a1 = v010 + ux * (v110 - v010)
    a2 = v001 + ux * (v101 - v001)
    a3 = v011 + ux * (v111 - v011)
    b0 = a0 + uy * (a1 - a0)
    b1 = a2 + uy * (a3 - a2)
    return b0 + uz * (b1 - b0)


@ti.func
def _softplus(x):
    out = x
    if x < 20.0:
        out = ti.log(1.0 + ti.exp(x))
    return out


@ti.func
def cascade(x, y, z, per_y, l0, l1, con):
    """乘性级联噪声（倍率 3，八度等权，支持小数层级）+ 高对比软阈值。

    Formula: S = Σ_l w_l · ln(1 + 0.1·n(3^l·p))；out = ln(1 + exp(con·S)) = ln(1 + (Π(1+0.1 w n))^con)
    Returns: ≥ 0，大部分区域接近 0（暗缝），少数区域线性变亮（丝缕）。
    """
    s = 0.0
    i0 = ti.cast(ti.floor(l0), ti.i32)
    for k in range(5):
        lv = i0 + k
        lvf = ti.cast(lv, ti.f32)
        w = ti.min(ti.max(ti.min(l1, lvf + 1.0) - ti.max(l0, lvf), 0.0), 1.0)
        if w > 0.0 and lv >= 0:
            f = 1.0
            per = per_y
            for _q in range(lv):
                f *= 3.0
                per *= 3
            n = vnoise(x * f, y * f, z * f, per)
            s += ti.log(1.0 + 0.1 * n * w)
    return _softplus(con * s)


@ti.func
def shape_profile(x):
    """Shape(x, a, b) = k·x^a·(1 − x)^b，峰值归一为 1。"""
    a = SHAPE_A
    b = SHAPE_B
    k = ti.pow(a + b, a + b) / (ti.pow(a, a) * ti.pow(b, b))
    xx = ti.min(ti.max(x, 0.0), 1.0)
    return k * ti.pow(xx, a) * ti.pow(1.0 - xx, b)


@ti.func
def eff_radius(r):
    """有效半径：x = (R − R_IN)/span 经非线性压缩，大盘时截面峰值靠近内区（NPGS EffectiveRadius）。"""
    x = ti.min(ti.max((r - R_IN) / (R_OUT - R_IN), 0.0), 1.0)
    a = ti.max(1.0, (R_OUT - R_IN) / EFF_SPAN)
    e = x
    if a > 1.0001:
        e = (-1.0 + ti.sqrt(ti.max(0.0, 1.0 + 4.0 * a * a * x - 4.0 * x * a))) / (2.0 * a - 2.0)
    return e


@ti.func
def ss_half_thickness(r):
    """SS 外区（气压主导、Kramers 不透明度）标高：H = HR_REF·r·(r/R_REF)^{1/8}·(f/f_ref)^{3/20}，f = 1 − sqrt(r_in/r)。"""
    fr = ti.max(1.0 - ti.sqrt(R_IN / ti.max(r, R_IN)), 1e-6)
    return HR_REF * r * ti.pow(r / R_REF, 0.125) * ti.pow(fr / F_REF_SS, 0.15)


@ti.func
def ss_surface_density(r):
    """SS 外区柱密度形状：Σ ∝ (r/R_REF)^{-3/4}·(f/f_ref)^{7/10}，外缘截断 smoothstep(0.72·R_OUT, R_OUT)。"""
    fr = ti.max(1.0 - ti.sqrt(R_IN / ti.max(r, R_IN)), 1e-6)
    return ti.pow(r / R_REF, -0.75) * ti.pow(fr / F_REF_SS, 0.7) * (1.0 - _smoothstep(0.72 * R_OUT, R_OUT, r))


@ti.func
def erfc_pos(x):
    """erfc(x)，x ≥ 0（Abramowitz & Stegun 7.1.26，误差 < 1.5e-7）。"""
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return y * ti.exp(-x * x)


@ti.func
def core_zmax(r):
    """核心盘竖直包络（采样剔除用）。"""
    out = THIN_I + HOPPER_I * (r - R_IN) + 0.01
    if ti.static(PHYS_STRUCT == 1):
        out = 3.0 * ss_half_thickness(r) + 0.01
    return out


@ti.func
def _flow_noise(ru, th, z, ox, oz, lev_cut, con):
    """在流坐标 (ru, th) 上求 (主云, 厚度扰动) 两路级联噪声。"""
    c = cascade(KR_I * ru + ox, th / (2.0 * math.pi) * NPHI_I, KR_I * z + oz, NPHI_I,
                L0_I - lev_cut, L0_I + 2.0 - lev_cut, con)
    tn = cascade(KT_I * ru + ox + 17.0, th / (2.0 * math.pi) * NPHI_T, oz + 5.0, NPHI_T,
                 LT0_I, LT0_I + 2.0, CON_I)
    return c, tn


@ti.func
def flow_I(r, phi, z, t, mode: ti.template()):
    """模型 I 的流动噪声：mode 3 刚体环（两带 × 两相位混合）；mode 4 螺线内流（单次求值）。

    mode 4 不变量（盘逆时针、径向内流 V_IN）：u = r + V_IN·t，θ' = φ + Ψ(r)，
    Ψ(r) = ∫Ω/V_IN dr = −sqrt(2) / (V_IN·sqrt(r))。沿流线 du/dt = dθ'/dt = 0，
    切向角速度恰为当地 Ω(r)，图样永不卷紧（倾角 tan i = V_IN/(rΩ)）。
    Returns: (主云 c, 厚度扰动 tn)。
    """
    r_rg = 2.0 * r
    lev_cut = 0.91 * ti.log(1.0 + 0.066 * ti.max(0.0, r_rg - 10.0))   # 外圈减少八度
    con = CON_I - 80.0 * ti.log(1.0 + 0.006 * ti.max(0.0, r_rg - 10.0))  # 外圈降低对比度
    c = 0.0
    tn = 0.0
    if ti.static(mode == 4):
        psi = -ti.sqrt(2.0) / (V_IN * ti.sqrt(r))
        c, tn = _flow_noise(r + V_IN * t, phi + psi, z, 0.0, 0.0, lev_cut, con)
    else:
        lnr = ti.log(r)
        fb = (lnr - LNR0_R) / DLN_R + 0.35 * gnoise(lnr * 4.0, 0.37, 11.3, 8)
        b0 = ti.floor(fb)
        fbf = fb - b0
        k_rigid = params[5]
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
                wp = ti.sin(math.pi * (s - cyc)) ** 2
                ci = ti.cast(cyc, ti.i32)
                cc, tt = _flow_noise(r, phi_rigid, z, _hashf(bi, ci, 2 * p) * 97.0,
                                     _hashf(bi, ci, 2 * p + 1) * 97.0, lev_cut, con)
                c += wb * wp * cc
                tn += wb * wp * tt
    return c, tn


@ti.func
def dust_flow(r, phi, z, t):
    """尘埃噪声：与主云相同的 mode 3 刚体环流场（两带 × 两相位），各带以当地 Ω(r_b) 旋转。"""
    lnr = ti.log(r)
    fb = (lnr - LNR0_R) / DLN_R + 0.35 * gnoise(lnr * 4.0, 0.37, 11.3, 8)
    b0 = ti.floor(fb)
    fbf = fb - b0
    k_rigid = params[5]
    out = 0.0
    for db in ti.static(range(2)):
        bi = ti.cast(b0, ti.i32) + db
        wb = ti.cos(0.5 * math.pi * fbf) ** 2
        if db == 1:
            wb = ti.sin(0.5 * math.pi * fbf) ** 2
        r_b = ti.exp(LNR0_R + ti.cast(bi, ti.f32) * DLN_R)
        om_b = ti.sqrt(0.5 / (r_b * r_b * r_b))
        t_life = k_rigid * 2.0 * math.pi / om_b
        ph0 = _hashf(bi, 19, 3)
        phi_rigid = phi - om_b * t - _hashf(bi, 23, 7) * 2.0 * math.pi
        for p in ti.static(range(2)):
            s = t / t_life + ph0 + 0.5 * p
            cyc = ti.floor(s)
            wp = ti.sin(math.pi * (s - cyc)) ** 2
            ci = ti.cast(cyc, ti.i32)
            ox = _hashf(bi, ci, 40 + 2 * p) * 97.0
            oz = _hashf(bi, ci, 41 + 2 * p) * 97.0
            out += wb * wp * cascade(2.0 * r + ox, phi_rigid / (2.0 * math.pi) * 9.0, 2.0 * z + oz, 9, 0.0, 6.0, 80.0)
    return out


@ti.func
def density_I(r, z, phi, t, mode: ti.template(), dir_z):
    """模型 I：返回 (核心发射, 核心温度倍率, 核心吸收, 其他发射, 其他吸收, 烟雾发射)；盘外全 0（温度倍率 1）。

    核心吸收在渲染核中再乘 CORE_OPAC；"其他" = 尘埃（+ SMOKE_TR = 0 时的烟雾），温度取当地 T(r)；
    SMOKE_TR > 0 时烟雾发射单列，温度取 SMOKE_TR·T(r)。
    PHYS_STRUCT = 1：SS 外区 H(r)、Σ(r) + 竖直高斯（见 ss_half_thickness / ss_surface_density）。

    柱密度：Σ(r) = radial_env(r)·r，ρ_s = vm·Σ/H_s（几何与物理分离）。
    截面：H_geo = THIN_I + HOPPER_I·(R − R_IN)，D = Shape(eff)，H_cap = H_geo·D，
          H_s = H_cap·(0.4 + 0.6·softsat(tn))（噪声调制的表面高度 → 云顶轮廓）。
    体内：vm = 1 − |z|/H_s，ρ_s = 0.7·vm·D²。
    吸收：c·ρ_s·arm；发射 = 吸收 × (SURF_LO + SURF_K·|z|/H_s)（局部热平衡；外缘收口由 Σ(r) 的 radial_env 承担）。
    尘埃：内区整体自转的稀薄尘埃，吸收 DUST_EM·…，源函数 DUST_S × 掠射增亮 sqrt(1 − dir_z²)。
    """
    em = 0.0
    ab = 0.0
    em_c = 0.0
    em_s = 0.0
    ab_c_out = 0.0
    tf_c = 1.0
    if r > R_IN and r < R_OUT:
        h_geo = 0.0
        h_cap = 0.0
        sig = 0.0
        zc = 0.0
        if ti.static(PHYS_STRUCT == 1):
            h_geo = ss_half_thickness(r)
            h_cap = h_geo
            sig = ss_surface_density(r)
            zc = 3.0 * h_cap
        else:
            h_geo = THIN_I + HOPPER_I * (r - R_IN)
            h_cap = h_geo * shape_profile(eff_radius(r))
            sig = radial_env(r) * r
            zc = h_cap
        xi = (r - R_IN) / ti.min(R_OUT - R_IN, 6.0)
        dust_bound = h_geo * ti.max(0.0, 1.0 - 5.0 * xi * xi)
        az = ti.abs(z)
        if ti.static(LOWF_SIGMA > 0.0):
            nl = turb_low(r, phi, t)
            sig *= ti.exp(LOWF_SIGMA * nl - 0.5 * LOWF_SIGMA * LOWF_SIGMA)
        if ti.static(SMOKE_I > 0.0):
            # H 的多层烟雾（k = −N..N，中心 z_k = k·CL_SPACING·r，层厚 CL_WIDTH·r），偏冷、以吸收为主
            if az < CL_EXTENT * r:
                for kk in range(2 * N_CL_HALF + 1):
                    kf = ti.cast(kk - N_CL_HALF, ti.f32)
                    dz = (z - kf * CL_SPACING * r) / (CL_WIDTH * r)
                    if ti.abs(dz) < 3.0:
                        amp = CL_AMP_NORM * ti.exp(-CL_DECAY * ti.abs(kf))
                        rho_c = SMOKE_I * sig * amp * ti.exp(-0.5 * dz * dz) / (2.5066283 * CL_WIDTH * r)
                        nc, _unused = turb_pair(r, phi, dz, t, mode, 1, 131.7 * ti.cast(kk + 1, ti.f32))
                        cov = 1.0 / (1.0 + ti.exp(-(nc - CLOUD_C0) / CLOUD_SOFT))
                        ab_c = rho_c * ti.exp(SIGMA_C * nc - 0.5 * SIGMA_C * SIGMA_C) * cov
                        ab += ab_c
                        if ti.static(SMOKE_TR > 0.0):
                            em_s += ab_c
                        else:
                            em += ab_c * SMOKE_S
        if az < ti.max(zc, dust_bound):
            if az < zc:
                c, tn = flow_I(r, phi, z, t, mode)
                softsat = 1.0 - 1.0 / (ti.max(tn, 0.0) + 1.0)
                h_s = ti.max(h_cap * (1.0 - SURF_NOISE + SURF_NOISE * softsat), 1e-6)
                zs = h_s
                if ti.static(PHYS_STRUCT == 1):
                    zs = 3.0 * h_s
                if az < zs:
                    vm = 1.0 - az / h_s
                    rho_s = 0.0
                    if ti.static(PHYS_STRUCT == 1):
                        # 等温静力平衡：ρ = Σ/(sqrt(2π)·H_s)·exp(−z²/2H_s²)，∫ρ dz = Σ（与表面起伏无关）
                        rho_s = sig * ti.exp(-0.5 * (az / h_s) ** 2) / (2.5066283 * h_s)
                    else:
                        # 三角竖直剖面 ∫vm dz = H_s，故 ρ = vm·Σ/H_s
                        rho_s = vm * sig / h_s
                    arm = 1.0
                    if ti.static(mode == 4 and ARM_W > 0.0):
                        om_a = ti.sqrt(0.5 / (R_ARM * R_ARM * R_ARM))
                        th_a = phi - om_a * t + K_LOG * ti.log(r)
                        spir = cascade(0.05 * r, th_a / (2.0 * math.pi) * 2.0, KR_I * z, 2, 1.0, 2.0, 80.0)
                        arm = 1.0 + (ti.min(ti.max(1.05 * spir - 0.5, 0.0), 3.0) - 1.0) * ARM_W * _smoothstep(5.0, 12.0, r)
                    # 局部热平衡：吸收 ∝ 密度；发射 = 吸收 × 源函数（表面增亮 × 外缘收口），
                    # 光学厚处亮度只由温度决定（避免 NPGS 原式 S ∝ 1/ρ 导致“越密越暗”）
                    cfac = c
                    if ti.static(CORE_FLOOR > 0.0):
                        cfac = CORE_FLOOR + (1.0 - CORE_FLOOR) * c / params[11]
                    ab_b = cfac * rho_s * arm
                    ab_c_out = ab_b
                    em_c = ab_b * (SURF_LO + SURF_K * az / h_s)
                    if ti.static(GREY_ATM == 1):
                        # 到表面的竖直光学深度（三角剖面解析）：τ_z = κ·CORE_OPAC·cfac·arm·Σ'·(1 − |z|/H_s)²/2
                        tau_z = params[12] * CORE_OPAC * cfac * arm * sig * vm * vm * 0.5
                        if ti.static(PHYS_STRUCT == 1):
                            # 高斯剖面：τ_z = κ·CORE_OPAC·cfac·arm·Σ'/2·erfc(|z| / (sqrt(2)·H_s))
                            tau_z = params[12] * CORE_OPAC * cfac * arm * sig * 0.5 * erfc_pos(az / (1.4142136 * h_s))
                        tf_c = 1.0 + GREY_MIX * (ti.min(ti.pow(0.75 * (tau_z + 2.0 / 3.0), 0.25), GREY_CAP) - 1.0)
                    if ti.static(DT_I > 0.0):
                        tf_c *= ti.min(ti.max(1.0 + DT_I * (c / params[11] - 1.0), 0.7), 1.3)
            if ti.static(DUST_ON == 1):
                if az < dust_bound:
                    di = ti.max(1.0 - (z / ti.max(dust_bound, 1e-6)) ** 2, 0.0)
                    if ti.static(DUST_KEPLER == 1):
                        dn = dust_flow(r, phi, z, t)
                        ab_d = DUST_EM * di * dn
                        ab += ab_d
                        em += ab_d * DUST_S
                    else:
                        th_d = phi - (2.0 / 3.0) * ti.sqrt(0.5 / (R_IN * R_IN * R_IN)) * t
                        dn = cascade(2.0 * r, th_d / (2.0 * math.pi) * 9.0, 2.0 * z, 9, 0.0, 6.0, 80.0)
                        ab_d = DUST_EM * di * dn
                        ab += ab_d
                        em += ab_d * DUST_S * ti.sqrt(ti.max(0.0, 1.0001 - dir_z * dir_z))
    return em_c, tf_c, ab_c_out, em, ab, em_s


@ti.kernel
def cmean_kernel_I(out: ti.template(), t: ti.f32, mode: ti.template()):
    """核心主云 c 在盘面若干随机点的样本，用于求 ⟨c⟩。"""
    for i in out:
        r = R_IN + 0.5 + _hashf(i, 3, 7) * (ti.min(R_OUT, 20.0) - R_IN - 0.5)
        phi = _hashf(i, 5, 11) * 2.0 * math.pi
        c, tn = flow_I(r, phi, 0.0, t, mode)
        out[i] = c


@ti.kernel
def column_kernel_I(out: ti.template(), t: ti.f32, mode: ti.template()):
    """r ∈ [5.5, 6.5] 处竖直吸收柱 ∫ab dz 的样本，用于标定 κ_I。"""
    for i in out:
        r = 5.5 + ti.cast(i % 16, ti.f32) / 16.0
        phi = ti.cast(i, ti.f32) * 0.61803 * 2.0 * math.pi
        col = 0.0
        zmax = core_zmax(r)
        for k in range(200):
            z = -zmax + (ti.cast(k, ti.f32) + 0.5) / 200.0 * 2.0 * zmax
            em_c, tf_c, ab_c, em_o, ab_o, em_s = density_I(r, z, phi, t, mode, 0.0)
            col += (ab_c + ab_o) * 2.0 * zmax / 200.0
        out[i] = col


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
        if ti.static(STATIC_CAM == 1):
            # 静止观者本地方向 → 坐标方向：tanψ_coord = tanψ_local / sqrt(1 − r_s/r)（ψ 为与径向夹角）
            r0 = cp.norm()
            rh = cp / r0
            d_rad = d.dot(rh) * rh
            d = (d_rad + (d - d_rad) / ti.sqrt(1.0 - 1.0 / r0)).normalized()
        p = cp
        l2 = p.cross(d).norm_sqr()
        acc = ti.Vector([0.0, 0.0, 0.0])
        trans = 1.0
        escaped = False
        first = True
        # 光子守恒量：L = p × d 严格守恒；|d|² − L²/r³ 守恒 → 无穷远处 |d_inf|
        lz = p.cross(d)[2]
        d_inf = ti.sqrt(ti.max(1.0 - l2 / (cp.norm() ** 3), 1e-6))
        lam = 0.0
        jit = _hashf(i, j, jit_seed[None])
        for it in range(8000):
            r = p.norm()
            rc = ti.sqrt(p[0] * p[0] + p[1] * p[1])
            h = ti.min(0.06 * r, 2.0)
            if ti.static(DISK_MODEL == 1):
                # 步长为位置的连续函数（避免条纹）：远场 ∝ r，近黑洞线性收缩，盘体附近按到盘包围体距离线性放大
                h = ti.min(h, 0.02 + 0.06 * ti.max(r - 1.0, 0.0))
                # 核心盘：细步长 H_IN_I，离开核心包络后线性放大
                zb = THIN_I + HOPPER_I * ti.max(rc - R_IN, 0.0) + 0.02
                if ti.static(PHYS_STRUCT == 1):
                    zb = 3.0 * ss_half_thickness(ti.max(rc, R_IN)) + 0.02
                rad_out = ti.max(R_IN * 0.95 - rc, 0.0) + ti.max(rc - R_OUT * 1.02, 0.0)
                dslab = ti.max(ti.abs(p[2]) - zb, 0.0) + rad_out
                h = ti.min(h, H_IN_I + 0.3 * dslab)
                if ti.static(SMOKE_I > 0.0):
                    # 烟雾层：步长 ∝ 层厚（与 H 一致），离开烟雾包络后线性放大；三者取 min，仍是位置的连续函数
                    dsm = ti.max(ti.abs(p[2]) - (CL_EXTENT * rc + 0.02), 0.0) + rad_out
                    h_sm = ti.max(0.4 * CL_WIDTH * rc, H_IN_I) + 0.3 * dsm
                    h = ti.min(h, h_sm)
                if it == 0:
                    h *= jit
            else:
                if r < 3.0:
                    h = ti.min(h, 0.02 + 0.06 * (r - 1.0))
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
            ds_seg = (pn - p).norm()
            lam += ds_seg
            if ti.static(DISK_MODEL == 1):
                zlim = core_zmax(rm)
                if ti.static(SMOKE_I > 0.0):
                    zlim = ti.max(zlim, CL_EXTENT * rm)
                if rm > R_IN and rm < R_OUT and ti.abs(pm[2]) < zlim:
                    t_s = t
                    if ti.static(LIGHT_DELAY == 1):
                        t_s = t - (lam - 0.5 * ds_seg)
                    dm = (0.5 * (d + dn)).normalized()
                    em_c, tf_c, ab_c, em, ab_o, em_s = density_I(rm, pm[2], ti.atan2(pm[1], pm[0]), t_s, mode, dm[2])
                    if em + em_c + em_s + ab_o + ab_c > 1e-7:
                        tk = temperature(rm)
                        g_phys = g_factor(pm, 0.5 * (d + dn), r_obs)
                        g_lum = ti.pow(g_phys, params[6])
                        g_col = ti.pow(g_phys, params[7])
                        src = ti.Vector([0.0, 0.0, 0.0])
                        src_c = ti.Vector([0.0, 0.0, 0.0])
                        src_s = ti.Vector([0.0, 0.0, 0.0])
                        if ti.static(PHYS_LUM == 1):
                            # I_ν,obs = B_ν(g·T)（I_ν/ν³ 不变）→ 观测亮度 = Y(g·T)；色度取 χ(T·g_col)
                            src = emit * ti.exp(ln_luminance(tk * g_lum) - params[13]) * blackbody_rgb(tk * g_col)
                            tc = tk * tf_c
                            src_c = emit * ti.exp(ln_luminance(tc * g_lum) - params[13]) * blackbody_rgb(tc * g_col)
                            if ti.static(SMOKE_TR > 0.0):
                                ts_ = tk * SMOKE_TR
                                src_s = emit * ti.exp(ln_luminance(ts_ * g_lum) - params[13]) * blackbody_rgb(ts_ * g_col)
                        else:
                            src = emit * ti.pow(tk / params[8], EMIT_POW) * band_boost(tk, g_lum) * blackbody_rgb(tk * g_col)
                            src_c = src
                        acc += trans * ds_seg * (em * src + CORE_OPAC * em_c * src_c + em_s * src_s)
                        if ti.static(dbg):
                            if first and ti.abs(pm[2]) < 0.05:
                                om_e = ti.sqrt(0.5 / (rm * rm * rm))
                                g_ex = ti.sqrt(1.0 - 1.5 / rm) / (1.0 + om_e * lz / d_inf) / ti.sqrt(1.0 - 1.0 / r_obs)
                                dbg_out[i, j] = ti.Vector([g_phys, g_ex, g_factor_coord(pm, 0.5 * (d + dn), r_obs), rm])
                                first = False
                        trans *= ti.exp(-params[12] * (ab_o + CORE_OPAC * ab_c) * ds_seg)
            elif rm > R_IN and rm < R_OUT and ti.abs(pm[2]) < ti.max(CL_EXTENT, 4.0 * HR) * rm:
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
        if ti.static(DISK_MODEL == 1):
            em_c, tf_c, ab_c, em_o, ab_o, em_s = density_I(r, 0.0, phi, t, mode, 0.0)
            out[i, j] = (ab_c + ab_o) / ti.max(radial_env(r) * r, 1e-9)
        else:
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
    params[13] = math.log(luminance_np(t_peak))
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


def luminance_np(t_k):
    """黑体可见光亮度 Y(T) = ∫ B_λ(T)·ȳ(λ) dλ（任意单位，λ ∈ [380, 780] nm）。"""
    lam = np.linspace(380.0, 780.0, 801)
    x = 1.4388e7 / (lam * t_k)
    spec = lam ** -5 / np.expm1(np.minimum(x, 700.0))
    return float((spec * _cie_cmf(lam)[1]).sum())


def build_bb_lut():
    """把 blackbody_rgb_np 采样到 bb_lut，把 ln Y(T) 采样到 lny_lut（log T 等间距）。"""
    ts = np.exp(np.linspace(math.log(1000.0), math.log(40000.0), BB_N))
    bb_lut.from_numpy(np.stack([blackbody_rgb_np(t) for t in ts]).astype(np.float32))
    ty = np.exp(np.linspace(math.log(LNY_T0), math.log(LNY_T1), BB_N))
    lny_lut.from_numpy(np.array([math.log(max(luminance_np(t), 1e-300)) for t in ty], dtype=np.float32))


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
    jit_seed[None] = jit_seed[None] + 1
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
    save_png(os.path.join(OUT_DIR, f"frame_{a.preset}{a.tag}_m{a.mode}_R{R_OUT:g}_T{a.t_peak:.0f}_L{a.doppler_lum:g}_C{a.doppler_color:g}_{a.w}x{a.h}.png"), tonemap(hdr, exp))
    np.save(os.path.join(OUT_DIR, "frame_hdr.npy"), hdr)


def cmd_video(a):
    """真实时间：内缘轨道周期 = a.orbit_s 秒视频。"""
    set_camera(**CAM)
    dt = period_k(R_IN) / (a.orbit_s * a.fps)
    vw = VideoWriter(os.path.join(OUT_DIR, f"slow_{a.preset}_m{a.mode}.mp4"), a.fps)
    exp = None
    t0 = time.time()
    for f in range(a.n):
        hdr = render_hdr(a.w, a.h, a.t0 + f * dt, a.mode, a.ss)
        if exp is None:
            exp = auto_exposure(hdr)
        frame = tonemap(hdr, exp)
        vw.write(frame)
        if f % 48 == 0:
            print(f"frame {f}/{a.n}  {time.time() - t0:.1f}s")
    vw.close()


def cmd_longrun(a):
    """time-lapse：a.orbits 个内缘轨道周期；左 naive，右分带。"""
    set_camera(**CAM)
    t_total = a.orbits * period_k(R_IN)
    vw = VideoWriter(os.path.join(OUT_DIR, f"longrun_{a.preset}_m{a.cmp_mode}_vs_m{a.mode}.mp4"), a.fps)
    exp = None
    t0 = time.time()
    frame = None
    for f in range(a.n):
        t = a.t0 + t_total * f / (a.n - 1)
        hn = render_hdr(a.w, a.h, t, a.cmp_mode)
        hb = render_hdr(a.w, a.h, t, a.mode)
        if exp is None:
            exp = auto_exposure(hb)
        orbit_txt = f"t = {(t - a.t0) / period_k(R_IN):.2f} inner orbits"
        names = {0: "naive advection", 3: "mode 3: rigid rings", 4: "mode 4: spiral inflow"}
        top = label(tonemap(hn, exp), f"TOP: {names.get(a.cmp_mode, a.cmp_mode)}   {orbit_txt}")
        bottom = label(tonemap(hb, exp), f"BOTTOM: {names.get(a.mode, a.mode)}   {orbit_txt}")
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
    # I：模型 I（透镜状绝对厚度 + 噪声调制表面 + 乘性级联噪声 + 表面增亮 + 内区尘埃），mode 3 刚体环
    "I": dict(DISK_MODEL=1, EMIT_POW=2.5, EXP_TARGET=0.7, LIGHT_DELAY=1),
    # Is：模型 I + mode 4 螺线内流 + 外圈旋臂（运行时需 --mode 4）
    "Is": dict(DISK_MODEL=1, EMIT_POW=2.5, EXP_TARGET=0.7, LIGHT_DELAY=1, ARM_W=0.6),
    # J1：I + H 的 7 层半吸收烟雾
    "J1": dict(DISK_MODEL=1, EMIT_POW=2.5, EXP_TARGET=0.7, LIGHT_DELAY=1, SMOKE_I=0.8, SMOKE_S=0.33, N_CL_HALF=3, CL_SPACING=0.014, CL_WIDTH=0.011, CL_DECAY=0.35, FR_C=8.0, NPHI_C=10, SIGMA_C=1.2, CLOUD_C0=0.4, CLOUD_SOFT=0.35),
    # J2：J1 + H 的大尺度低频明暗 + 对比度 80 → 50（柔化）
    "J2": dict(DISK_MODEL=1, EMIT_POW=2.5, EXP_TARGET=0.7, LIGHT_DELAY=1, SMOKE_I=0.8, SMOKE_S=0.33, N_CL_HALF=3, CL_SPACING=0.014, CL_WIDTH=0.011, CL_DECAY=0.35, FR_C=8.0, NPHI_C=10, SIGMA_C=1.2, CLOUD_C0=0.4, CLOUD_SOFT=0.35,
               LOWF_SIGMA=0.9, CON_I=50.0),
    # J3：J2 再柔一档（对比度 35，烟雾稍淡）
    "J3": dict(DISK_MODEL=1, EMIT_POW=2.5, EXP_TARGET=0.7, LIGHT_DELAY=1, SMOKE_I=0.6, SMOKE_S=0.33, N_CL_HALF=3, CL_SPACING=0.014, CL_WIDTH=0.011, CL_DECAY=0.35, FR_C=8.0, NPHI_C=10, SIGMA_C=1.2, CLOUD_C0=0.4, CLOUD_SOFT=0.35,
               LOWF_SIGMA=0.9, CON_I=35.0),
    # K1：J2 + 物理可见光亮度 Y(g·T)（去掉 EMIT_POW 旋钮），保留表面增亮
    "K1": dict(DISK_MODEL=1, EXP_TARGET=0.7, LIGHT_DELAY=1, SMOKE_I=0.8, SMOKE_S=0.33, N_CL_HALF=3, CL_SPACING=0.014, CL_WIDTH=0.011, CL_DECAY=0.35, FR_C=8.0, NPHI_C=10, SIGMA_C=1.2, CLOUD_C0=0.4, CLOUD_SOFT=0.35, LOWF_SIGMA=0.9, CON_I=50.0, PHYS_LUM=1),
    # K2：K1 + 去掉表面增亮（竖直方向源函数均匀，近似等温表面）
    # L：J2 + 物理亮度 + Page–Thorne + 灰大气（强度 0.5）+ 核心 τ≈3 温和起伏 + 温度起伏 5%（多普勒 0.75 / 1.5 由 CLI 默认给出）
    "L": dict(DISK_MODEL=1, EXP_TARGET=0.7, LIGHT_DELAY=1, SMOKE_I=0.8, SMOKE_S=0.33, N_CL_HALF=3, CL_SPACING=0.014, CL_WIDTH=0.011, CL_DECAY=0.35, FR_C=8.0, NPHI_C=10, SIGMA_C=1.2, CLOUD_C0=0.4, CLOUD_SOFT=0.35, LOWF_SIGMA=0.9, CON_I=50.0, PHYS_LUM=1, TEMP_PT=1, GREY_ATM=1, GREY_MIX=0.5, GREY_CAP=1.19, SURF_LO=1.0, SURF_K=0.0,
              CORE_OPAC=2.0, CORE_FLOOR=0.35, DT_I=0.05),
    "K2": dict(DISK_MODEL=1, EXP_TARGET=0.7, LIGHT_DELAY=1, SMOKE_I=0.8, SMOKE_S=0.33, N_CL_HALF=3, CL_SPACING=0.014, CL_WIDTH=0.011, CL_DECAY=0.35, FR_C=8.0, NPHI_C=10, SIGMA_C=1.2, CLOUD_C0=0.4, CLOUD_SOFT=0.35, LOWF_SIGMA=0.9, CON_I=50.0, PHYS_LUM=1, SURF_LO=1.0, SURF_K=0.0),
}


def page_thorne_T(r_rs):
    """Schwarzschild Page–Thorne 相对论薄盘有效温度形状（未归一），r 单位 r_s（M = 0.5）。

    Formula（赤道圆轨道，几何单位）：
        E = (1 − 2M/r)/sqrt(1 − 3M/r),  L = sqrt(M r)/sqrt(1 − 3M/r),  Ω = sqrt(M/r³)
        F(r) ∝ −Ω'(r) / (E − ΩL)² / r · ∫_{r_in}^{r} (E − ΩL)·L'(r') dr'
        T ∝ F^{1/4}
    """
    m = 0.5
    r = np.asarray(r_rs, dtype=np.float64)
    rr = np.linspace(R_IN, max(float(r.max()), R_IN + 1e-3), 20001)
    e = (1 - 2 * m / rr) / np.sqrt(1 - 3 * m / rr)
    l = np.sqrt(m * rr) / np.sqrt(1 - 3 * m / rr)
    om = np.sqrt(m / rr ** 3)
    dl = np.gradient(l, rr)
    dom = np.gradient(om, rr)
    integrand = (e - om * l) * dl
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(rr))])
    flux = -dom / (e - om * l) ** 2 / rr * integ
    return np.interp(r, rr, np.maximum(flux, 0.0) ** 0.25)


def page_thorne_flux_geom(r_rs):
    """Page–Thorne 通量（几何单位，r 单位 r_s，M = 0.5）：F_geom = −Ω'/(E − ΩL)²/r·∫(E − ΩL)L' dr。

    物理通量 F = Ṁ c² / (4π r_s²) · F_geom（大 r 极限回到 3GMṀ/(8πR³)(1 − sqrt(r_in/r))）。
    """
    m = 0.5
    r = np.asarray(r_rs, dtype=np.float64)
    rr = np.linspace(R_IN, max(float(r.max()), R_IN + 1e-3), 20001)
    e = (1 - 2 * m / rr) / np.sqrt(1 - 3 * m / rr)
    l = np.sqrt(m * rr) / np.sqrt(1 - 3 * m / rr)
    om = np.sqrt(m / rr ** 3)
    integrand = (e - om * l) * np.gradient(l, rr)
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(rr))])
    flux = -np.gradient(om, rr) / (e - om * l) ** 2 / rr * integ
    return np.interp(r, rr, np.maximum(flux, 0.0))


def derive_t_peak(m_msun, mdot_edd):
    """由黑洞质量与吸积率推出 Page–Thorne 盘的峰值有效温度（K）。

    Formula：r_s = 2GM/c²，L_Edd = 4πGM m_p c/σ_T，Ṁ = mdot_edd·L_Edd/(η c²)，η = 1 − sqrt(8/9)，
             T_peak = (max F / σ_SB)^{1/4}，F = Ṁ c²/(4π r_s²)·F_geom。
    Simplifications：标准薄盘（低吸积率下真实流体应为 ADAF，此处仍按薄盘处理）。
    """
    g_, c_, msun, sig_sb, m_p, s_t = 6.674e-8, 2.998e10, 1.989e33, 5.6704e-5, 1.6726e-24, 6.6524e-25
    m = m_msun * msun
    rs = 2 * g_ * m / c_ ** 2
    eta = 1 - math.sqrt(8 / 9)
    mdot = mdot_edd * 4 * math.pi * g_ * m * m_p * c_ / s_t / (eta * c_ ** 2)
    f_geom = float(page_thorne_flux_geom(np.linspace(R_IN, 20.0, 4001)).max())
    return (mdot * c_ ** 2 / (4 * math.pi * rs ** 2) * f_geom / sig_sb) ** 0.25


def build_pt_lut():
    rs = np.linspace(R_IN, R_OUT, BB_N)
    tt = page_thorne_T(rs)
    tpt_lut.from_numpy((tt / tt.max()).astype(np.float32))
    print(f"[PT] Page–Thorne 温度峰值位于 r = {rs[int(np.argmax(tt))]:.2f} r_s（牛顿近似 {49 / 36 * R_IN:.2f}）")

# M（定稿）：L + 物理修正（SS 外区 H/Σ + 竖直高斯、尘埃开普勒、烟雾温度比、静止观者相机）+ T_peak 由 M、Ṁ 推出 + 曝光 0.9
PRESETS["M"] = dict(PRESETS["L"], PHYS_STRUCT=1, DUST_KEPLER=1, SMOKE_TR=0.85, STATIC_CAM=1, T_FROM_MDOT=1, EXP_TARGET=0.9)
# Mphys：M 的第 3 层艺术偏离全部取物理值（灰大气 1、核心 τ≈100、大尺度明暗 0.3；多普勒 1/1 由 CLI 给出）
PRESETS["Mphys"] = dict(PRESETS["M"], GREY_MIX=1.0, CORE_OPAC=67.0, LOWF_SIGMA=0.3)


def calibrate_I(mode):
    """标定 ⟨c⟩ 与模型 I 吸收系数（r ≈ 6 处 face-on 竖直光学深度 = TAU_I，不含 CORE_OPAC）。"""
    if TEMP_PT == 1:
        build_pt_lut()
    params[11] = 1.0
    cb = ti.field(ti.f32, shape=4096)
    cmean_kernel_I(cb, 2000.0, mode)
    params[11] = max(float(cb.to_numpy().mean()), 1e-6)
    buf = ti.field(ti.f32, shape=256)
    column_kernel_I(buf, 2000.0, mode)
    col = float(buf.to_numpy().mean())
    params[12] = TAU_I / max(col, 1e-12)
    print(f"[I] ⟨c⟩ = {params[11]:.3g}，吸收柱均值 {col:.4g} → κ_I = {params[12]:.4g}（核心 ×{CORE_OPAC:g}）")


def apply_preset(name):
    """把预设写入模块全局（须在任何 kernel 编译前调用），并重算派生常量。"""
    g = globals()
    g.update(PRESETS[name])
    g["CL_EXTENT"] = g["N_CL_HALF"] * g["CL_SPACING"] + 3.0 * g["CL_WIDTH"]
    g["HR_CLOUD"] = g["CL_EXTENT"] / 3.5
    g["CL_AMP_NORM"] = 1.0 / sum(math.exp(-g["CL_DECAY"] * abs(k)) for k in range(-g["N_CL_HALF"], g["N_CL_HALF"] + 1))


# 三层参数表：(层, 名称, CLI / 全局, 物理值或说明, 含义)
LAYER_TABLE = [
    (1, "黑洞质量 M", "--bh_mass / BH_MASS_MSUN", "场景设定", "太阳质量；图像与尺度无关，只经 Ṁ 决定温度"),
    (1, "吸积率 Ṁ", "--mdot_edd / MDOT_EDD", "场景设定", "爱丁顿倍数；与 M 一起推出 T_peak（Page–Thorne 绝对通量）"),
    (1, "自旋 a", "（固定）", "0", "史瓦西黑洞"),
    (1, "盘内半径", "R_IN", "ISCO = 3 r_s", "自动"),
    (1, "盘外半径", "--r_out", "场景设定", "r_s"),
    (1, "相机距离 / 仰角 / 视场", "--dist / --elev / --fov", "场景设定", "r_s / 度 / 竖直视场（度）"),
    (1, "分辨率 / 超采样", "--w --h / --ss", "—", ""),
    (1, "视频速度 / 帧率", "--orbit_s / --fps", "—", "内缘一圈对应的视频秒数"),
    (2, "温度剖面", "TEMP_PT", "1（Page–Thorne）", "0 = 牛顿近似"),
    (2, "峰值温度覆盖", "--t_peak", "0（由 M、Ṁ 推出）", "非 0 时直接指定（K）"),
    (2, "盘结构", "PHYS_STRUCT", "1", "SS 外区：H/r ∝ r^{1/8} f^{3/20}，Σ ∝ r^{-3/4} f^{7/10}，竖直高斯"),
    (2, "亮度规律", "PHYS_LUM", "1", "观测亮度 Y(g·T)（普朗克谱 × CIE ȳ）"),
    (2, "灰大气", "GREY_ATM", "1", "T⁴ = ¾T_eff⁴(τ_z + ⅔)，临边昏暗"),
    (2, "密度起伏下限", "CORE_FLOOR", "0.35", "核心密度 = FLOOR + (1 − FLOOR)·c/⟨c⟩，无空洞"),
    (2, "温度起伏 δT", "DT_I", "0.05", "湍流发热率起伏"),
    (2, "结构寿命", "--k_rigid", "4", "本地轨道周期"),
    (2, "烟雾柱密度 / 温度比", "SMOKE_I / SMOKE_TR", "0.8 / 0.85", "盘风团块：柱密度 = SMOKE_I·Σ，温度 = SMOKE_TR·T"),
    (2, "尘埃", "DUST_EM / DUST_KEPLER", "0.02 / 1", "内区稀薄尘埃，按开普勒刚体环旋转"),
    (2, "光行时间", "LIGHT_DELAY", "1", ""),
    (2, "相机标架", "STATIC_CAM", "1", "静止观者本地标架"),
    (3, "多普勒亮度强度", "--doppler_lum", "物理 1", "亮度用 Y(g^s·T)"),
    (3, "多普勒颜色强度", "--doppler_color", "物理 1", "色度用 χ(T·g^s)"),
    (3, "灰大气强度", "GREY_MIX", "物理 1", ""),
    (3, "核心光学深度倍率", "CORE_OPAC", "物理 ≫ 100（τ = 1.5 × 倍率）", "τ≈3 为通透感的视觉取值"),
    (3, "盘厚 H/r（r = 10）", "HR_REF", "SS 对本场景 ≪ 0.01", "0.027 为视觉取值"),
    (3, "大尺度明暗幅度", "LOWF_SIGMA", "约 0.3", "lognormal σ"),
    (3, "程序化结构", "CON_I / KR_I / L0_I / SURF_NOISE / CL_* / DLN_R", "—", "噪声对比度、频率、八度、表面起伏、烟雾分层、刚体环带宽"),
    (3, "曝光", "--exp / EXP_TARGET", "—", "盘区亮度 p99.9 映射值"),
    (3, "白平衡", "--wb", "—", "K"),
    (3, "bloom / 色散", "--bloom_src --bloom_gain --ca_*", "—", "相机效果"),
]


def cmd_params(a):
    """打印三层参数表（含当前值）。"""
    cur = {"--doppler_lum": a.doppler_lum, "--doppler_color": a.doppler_color, "--t_peak": f"{a.t_peak:.0f} K",
           "--r_out": a.r_out, "--dist / --elev / --fov": f"{a.dist} / {a.elev} / {a.fov}", "--wb": a.wb,
           "--k_rigid": a.k_rigid, "--bh_mass / BH_MASS_MSUN": BH_MASS_MSUN, "--mdot_edd / MDOT_EDD": MDOT_EDD,
           "--exp / EXP_TARGET": EXP_TARGET, "--orbit_s / --fps": f"{a.orbit_s} / {a.fps}",
           "--w --h / --ss": f"{a.w}x{a.h} / {a.ss}"}
    names = {1: "第 1 层  基本参数", 2: "第 2 层  物理模型（右列为物理默认值）", 3: "第 3 层  视觉调节（右列为物理值，用于对照）"}
    for layer in (1, 2, 3):
        print("\n" + names[layer])
        for lv, name, key, phys, desc in LAYER_TABLE:
            if lv != layer:
                continue
            v = cur.get(key)
            if v is None and key.isidentifier() and key in globals():
                v = globals()[key]
            if v is None and " / " in key and all(k.strip() in globals() for k in key.split(" / ")):
                v = " / ".join(str(globals()[k.strip()]) for k in key.split(" / "))
            print(f"  {name:<20s} {key:<34s} 当前 {str(v if v is not None else '—'):<14s} {phys:<26s} {desc}")


def main():
    global BLOOM_SRC, BLOOM_GAIN, CA_LATERAL, CA_AXIAL, CA_FRINGE, WHITE_BALANCE_K, R_OUT, FOV_V
    ap = argparse.ArgumentParser(description="吸积盘参考实现；参数分三层，见 params 子命令")
    ap.add_argument("cmd", choices=["frame", "video", "longrun", "winding", "stats", "physcheck", "params"])
    g1 = ap.add_argument_group("第 1 层：基本参数")
    g1.add_argument("--bh_mass", type=float, default=None, help="黑洞质量（太阳质量）")
    g1.add_argument("--mdot_edd", type=float, default=None, help="吸积率（爱丁顿倍数）")
    g1.add_argument("--r_out", type=float, default=30.0, help="盘外半径（r_s）")
    g1.add_argument("--dist", type=float, default=40.0, help="相机距离（r_s）")
    g1.add_argument("--elev", type=float, default=CAM["elev_deg"], help="相机仰角（度）")
    g1.add_argument("--fov", type=float, default=38.0, help="竖直视场角（度）")
    g1.add_argument("--w", type=int, default=1920)
    g1.add_argument("--h", type=int, default=1080)
    g1.add_argument("--ss", type=int, default=1, help="超采样倍率（每轴）")
    g1.add_argument("--t", type=float, default=2000.0, help="单帧物理时间（r_s/c）")
    g1.add_argument("--t0", type=float, default=2000.0, help="视频起始物理时间")
    g1.add_argument("--n", type=int, default=240, help="视频帧数")
    g1.add_argument("--fps", type=int, default=24)
    g1.add_argument("--orbit_s", type=float, default=16.0, help="内缘一圈对应的视频秒数")
    g1.add_argument("--orbits", type=float, default=20.0, help="longrun 覆盖的内缘圈数")
    g2 = ap.add_argument_group("第 2 层：物理模型（其余用 --set NAME=VALUE，见 params）")
    g2.add_argument("--t_peak", type=float, default=0.0, help="覆盖盘峰值温度（K）；0 = 由 M、Ṁ 推出（或预设的固定值）")
    g2.add_argument("--k_rigid", type=float, default=K_RIGID_DEFAULT, help="结构寿命（本地轨道周期）")
    g2.add_argument("--k_life", type=float, default=K_LIFE_DEFAULT, help="mode 2 结构寿命（对照用）")
    g2.add_argument("--mode", type=int, default=3, help="平流：3 刚体环（定稿）/ 4 螺线内流 / 0 naive / 1、2 对照")
    g3 = ap.add_argument_group("第 3 层：视觉调节")
    g3.add_argument("--preset", default="M", choices=list(PRESETS), help="外观预设（M 为定稿；L、J2、H 为旧定稿）")
    g3.add_argument("--doppler_lum", type=float, default=0.55, help="多普勒亮度强度 s：亮度用 Y(g^s·T)；1 = 物理")
    g3.add_argument("--doppler_color", type=float, default=1.5, help="多普勒颜色强度 s：色度用 χ(T·g^s)；1 = 物理")
    g3.add_argument("--exp", type=float, default=None, help="曝光：盘区亮度 p99.9 映射值；默认由预设决定")
    g3.add_argument("--wb", type=float, default=WHITE_BALANCE_K, help="相机白平衡色温（K），6600 为不校正")
    g3.add_argument("--bloom_src", type=float, default=BLOOM_SRC, help="bloom 高光阈值（越低光晕越多）")
    g3.add_argument("--bloom_gain", type=float, default=BLOOM_GAIN, help="bloom 增益")
    g3.add_argument("--ca_lateral", type=float, default=CA_LATERAL, help="横向色散强度（0 关闭）")
    g3.add_argument("--ca_axial", type=float, default=1.0, help="轴向色散倍率，0 关闭")
    g3.add_argument("--ca_fringe", type=float, default=CA_FRINGE, help="高光镶边强度，0 关闭")
    gx = ap.add_argument_group("其他")
    gx.add_argument("--set", nargs="*", default=[], help="覆盖模块全局，如 GREY_MIX=1 CORE_OPAC=20")
    gx.add_argument("--tag", default="", help="输出文件名后缀")
    gx.add_argument("--cmp_mode", type=int, default=0, help="longrun 上半画面的对照 mode")
    a = ap.parse_args()
    R_OUT = a.r_out
    apply_preset(a.preset)
    for kv in a.set:
        k_, v_ = kv.split("=")
        globals()[k_] = type(globals()[k_])(float(v_)) if isinstance(globals()[k_], (int, float)) else v_
    if a.bh_mass is not None:
        globals()["BH_MASS_MSUN"] = a.bh_mass
    if a.mdot_edd is not None:
        globals()["MDOT_EDD"] = a.mdot_edd
    if a.exp is not None:
        globals()["EXP_TARGET"] = a.exp
    if a.t_peak <= 0.0:
        a.t_peak = derive_t_peak(BH_MASS_MSUN, MDOT_EDD) if T_FROM_MDOT == 1 else T_PEAK_VIS
        if T_FROM_MDOT == 1:
            print(f"[第1层] M = {BH_MASS_MSUN:.3g} M☉，Ṁ = {MDOT_EDD:.3g} Ṁ_Edd → T_peak = {a.t_peak:.0f} K")
    if a.cmd == "params":
        cmd_params(a)
        return
    setup(a.k_life, a.k_rigid, a.doppler_lum, a.doppler_color, a.t_peak)
    if DISK_MODEL == 1:
        calibrate_I(a.mode)
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
