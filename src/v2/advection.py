"""Disk V2 刚体环平流（v2.3 S4）。

mode 3 动态核心：盘按 ln r 分带，带内结构以带中心角速度 Ω_b **刚体**旋转
（带内无剪切 → 任意时长不卷绕）；每带两套噪声种子相位差半周期，三角权重
交叉淡化（种子切换瞬间权重为 0，无跳变）。来自参考实现预设 M 的 `turb_pair` mode 3。

坐标与符号约定：

- 盘局部坐标：盘面 z = 0，从 +z 看逆时针旋转；`phi` 为方位角。
- 带坐标：`fb = (ln r − lnr0)/dln + 0.35·gnoise(4·ln r, 0.37, 11.3, 8)`，
  1D 噪声扰动使带间距不规则（扰动斜率 < 基础斜率 1/dln，保持单调）。
- 带 b 的常量（f64 预计算）：`r_b = exp(lnr0 + b·dln)`、`Ω_b = sqrt(M/r_b³)`
  （M = 0.5）、`T_life_b = k_rigid·2π/Ω_b`、随机方位偏移 `φ_b`、种子相位 `ph0_b`。
- 相位 p ∈ {0, 1}：`s = t/T_life + ph0 + 0.5p`，`frac = s − floor(s)`，
  权重 `wp = sin²(π·frac)`；种子偏移 `ox/oz = hashf(b, cyc, 2p/2p+1)·97`。

float32 精度处理（长视频必需）：

- `Ω_b·t` 与 `t/T_life` 随 t 无界增长，f32 相位精度会损失。本模块在 CPU 用
  float64 每帧预计算 `rot_b = (Ω_b·t) mod 2π`、`frac_p`、`cyc_p mod 4096`
  （`RigidRingBands.phase_table`），上传为小 field（带数 ~20），kernel 只查表。
- 光行时间（采样时刻 `t_s = t − delay`，delay 为沿光线累计光程）由 kernel 以
  `Ω_b·delay`（小量，f32 精确）在查表值上做差分修正；种子周期跨越只发生在
  `frac ≈ 0` 附近，此时该种子权重 ≈ 0，索引错配无视觉影响。
"""

import math
from dataclasses import dataclass

import numpy as np
import taichi as ti

from .noise_ti import gnoise, hashf

BAND_PERTURB_AMP: float = 0.35
"""带坐标 1D 扰动幅度。约束：`AMP·max|d gnoise/d ln r| < 1/dln` 保持单调
（gnoise 单位格距最大斜率实测 < 4，`1/ln(1.22) ≈ 5`，裕量充足）。"""

BAND_PERTURB_FREQ: float = 4.0
"""带坐标扰动频率（每 ln r 单位）。"""

_CYCLE_MOD: int = 4096
"""种子周期索引的模（防 f32 大整数精度丢失；只影响哈希取值分布，不影响时序）。"""


@ti.func
def band_coordinate(lnr, lnr0, dln):
    """ln r → 带坐标 fb（带中心在整数处；相邻带以 cos²/sin² 混合）。

    Args:
        lnr: `ln r`（盘局部坐标，r 单位 r_s）。
        lnr0: 第 0 带中心的 `ln r`。
        dln: 带宽（ln r）。

    Returns:
        带坐标 fb（float）；`floor(fb)` 为带索引，`fb − floor(fb)` 为带内位置。
        fb 关于 lnr 严格单调（见 `BAND_PERTURB_AMP`）。

    Formula:
        `fb = (lnr − lnr0)/dln + AMP·gnoise(FREQ·lnr, 0.37, 11.3, 8)`
    """
    perturb = BAND_PERTURB_AMP * gnoise(BAND_PERTURB_FREQ * lnr, 0.37, 11.3, 8)
    return (lnr - lnr0) / dln + perturb


@ti.func
def band_mix_weights(fbf):
    """带内位置 → 相邻两带混合权重（cos² / sin²，和恒为 1）。

    Args:
        fbf: 带内位置 `[0, 1)`。

    Returns:
        `(w_lo, w_hi)`，均 `[0, 1]`，`w_lo + w_hi = 1`。
    """
    w_lo = ti.cos(0.5 * math.pi * fbf) ** 2
    w_hi = ti.sin(0.5 * math.pi * fbf) ** 2
    return w_lo, w_hi


@ti.func
def phase_weight(frac):
    """种子相位进度 → 三角交叉淡化权重 `sin²(π·frac)`。

    Args:
        frac: 种子周期进度 `[0, 1)`。

    Returns:
        权重 `[0, 1]`；`frac = 0`（种子切换瞬间）与 `frac → 1` 时为 0，
        `frac = 0.5` 时为 1 —— 切换点权重为 0，种子更替无跳变。
    """
    return ti.sin(math.pi * frac) ** 2


@dataclass(frozen=True)
class RigidRingBands:
    """刚体环常量表（CPU，float64）与每帧相位表。

    Args:
        r_in, r_out: 盘内外半径（r_s）；带索引范围覆盖
            `[ln r_in − 2·dln, ln r_out + dln]`。
        dln: 带宽（ln r）。
        k_rigid: 结构种子寿命（本地轨道周期数）。
        spin: 旋转方向符号，`+1` 逆时针、`-1` 顺时针（从盘法向 +z 看）；只作用于带角速度
            `om_b`，种子寿命恒为正。

    Physical Meaning:
        "结构有有限寿命、被湍流不断重建"的平流骨架：带内刚体（形状保持），
        带间 / 种子间平滑混合（不卷绕、不跳变）。
    """

    r_in: float
    r_out: float
    dln: float
    k_rigid: float
    spin: float = 1.0
    phi_b_hash: tuple = (17, 1)
    """φ_b 随机偏移的哈希流 `(b, c)`（区分使用同一带网格的不同结构层）。"""
    ph0_hash: tuple = (13, 5)
    """种子相位 ph0 的哈希流 `(b, c)`。"""
    lnr0_bands: float = -2.0
    """带网格原点：`lnr0 = ln r_in + lnr0_bands·dln`（核心 = −2；低频层 = 0）。"""
    center_frac: float = 0.0
    """带中心在带坐标中的偏移：`r_b = exp(lnr0 + (b + center_frac)·dln)`（低频层 = 0.5）。

    与参考实现保持一致：核心层 `r_b = exp(LNR0_R + b·DLN_R)`，
    低频层 `r_b = exp(ln R_IN + (b + 0.5)·DLN_L)`。
    """

    def __post_init__(self) -> None:
        lnr0 = math.log(self.r_in) + self.lnr0_bands * self.dln
        b_lo = int(math.floor((math.log(self.r_in) - lnr0) / self.dln)) - 1
        b_hi = int(math.ceil((math.log(self.r_out) + self.dln - lnr0) / self.dln)) + 1
        bs = np.arange(b_lo, b_hi + 1, dtype=np.float64)
        object.__setattr__(self, "lnr0", lnr0)
        object.__setattr__(self, "b_lo", b_lo)
        object.__setattr__(self, "b_hi", b_hi)
        r_b = np.exp(lnr0 + (bs + self.center_frac) * self.dln)
        object.__setattr__(self, "r_b", r_b)
        om_abs = np.sqrt(0.5 / r_b ** 3)
        object.__setattr__(self, "om_b", float(self.spin) * om_abs)
        object.__setattr__(self, "t_life", self.k_rigid * 2.0 * math.pi / om_abs)
        object.__setattr__(self, "phi_b", np.array(
            [_hashf_py(int(b), *self.phi_b_hash) * 2.0 * math.pi for b in bs]))
        object.__setattr__(self, "ph0", np.array(
            [_hashf_py(int(b), *self.ph0_hash) for b in bs]))

    @property
    def n_bands(self) -> int:
        """带数（含边界余量）。"""
        return self.b_hi - self.b_lo + 1

    def band_of_radius(self, r: float) -> int:
        """半径 → 主带索引（忽略扰动的一阶估计，供测试/诊断用）。"""
        return int(math.floor((math.log(r) - self.lnr0) / self.dln))

    def phase_table(self, t: float) -> dict:
        """某物理时刻 t 的相位表（float64）。

        Args:
            t: 物理时间（r_s/c），任意大。

        Returns:
            dict：
            - `rot`：`(n_bands,)`，`(Ω_b·t) mod 2π`（kernel 侧减去即可得流坐标）。
            - `frac`：`(n_bands, 2)`，两个种子相位的周期进度 `[0, 1)`。
            - `cyc`：`(n_bands, 2)` int，种子周期索引 `mod 4096`（哈希用）。
            - `t_life`, `om_b`, `phi_b`, `ph0`：常量表副本。

        Notes:
            全部 float64 计算；kernel 消费端按 f32 存储（frac ∈ [0,1)、
            rot ∈ [0, 2π)，f32 表示精度 ~1e-7，远优于长视频下直接
            `Ω_b·t` 的 f32 误差（t = 1e6 时 ~1e-2 rad）。
        """
        rot = (self.om_b * float(t)) % (2.0 * math.pi)
        s = float(t) / self.t_life[:, None] + self.ph0[:, None] + 0.5 * np.array([0.0, 1.0])
        cyc = np.floor(s)
        frac = s - cyc
        return {
            "rot": rot,
            "frac": frac,
            "cyc": (cyc.astype(np.int64) % _CYCLE_MOD).astype(np.int32),
            "t_life": self.t_life,
            "om_b": self.om_b,
            "phi_b": self.phi_b,
            "ph0": self.ph0,
        }


@dataclass
class RigidRingFields:
    """相位表的 Taichi field 容器（`upload` 后供 kernel 查表）。

    Attributes:
        rot: `(n,)` f32，`(Ω_b·t) mod 2π`。
        frac: `(n, 2)` f32，两个种子相位的周期进度 `[0, 1)`。
        cyc: `(n, 2)` i32，种子周期索引 `mod 4096`。
        t_life / om_b / phi_b: `(n,)` f32 常量表。
        b_lo: field 下标 0 对应的带索引（kernel 侧 `bi − b_lo` 寻址）。
    """

    rot: object
    frac: object
    cyc: object
    t_life: object
    om_b: object
    phi_b: object
    b_lo: int


def make_fields(bands: RigidRingBands) -> RigidRingFields:
    """按带数创建 Taichi field。

    Args:
        bands: `RigidRingBands` 常量表。

    Returns:
        未上传数据的 `RigidRingFields`（调用 `upload` 填充）。
    """
    n = bands.n_bands
    return RigidRingFields(
        rot=ti.field(dtype=ti.f32, shape=n),
        frac=ti.Vector.field(2, dtype=ti.f32, shape=n),
        cyc=ti.Vector.field(2, dtype=ti.i32, shape=n),
        t_life=ti.field(dtype=ti.f32, shape=n),
        om_b=ti.field(dtype=ti.f32, shape=n),
        phi_b=ti.field(dtype=ti.f32, shape=n),
        b_lo=bands.b_lo,
    )


def upload(fields: RigidRingFields, table: dict) -> None:
    """把 `phase_table(t)` 的结果上传进 field（每帧一次）。

    Args:
        fields: `make_fields` 的容器。
        table: `RigidRingBands.phase_table(t)` 的返回值。
    """
    fields.rot.from_numpy(table["rot"].astype(np.float32))
    fields.frac.from_numpy(table["frac"].astype(np.float32))
    fields.cyc.from_numpy(table["cyc"].astype(np.int32))
    fields.t_life.from_numpy(table["t_life"].astype(np.float32))
    fields.om_b.from_numpy(table["om_b"].astype(np.float32))
    fields.phi_b.from_numpy(table["phi_b"].astype(np.float32))


def _hashf_py(a: int, b: int, c: int) -> float:
    """`noise_ti.hashf` 的 Python 镜像（构造期用，避免依赖 Taichi 运行时）。"""
    x = (a + 4096) & 0xFFFFFFFF
    h = _h32(x * 1597334677)
    h = _h32(h ^ (((b + 4096) & 0xFFFFFFFF) * 1103515245))
    h = _h32(h ^ (((c + 4096) & 0xFFFFFFFF) * 1234567891))
    return (h & 0xFFFFFF) / 16777216.0


def _h32(x: int) -> int:
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x ^= x >> 16
    return x
