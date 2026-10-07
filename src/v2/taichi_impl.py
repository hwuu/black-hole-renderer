"""Disk V2 的 Taichi 实现：体积密度场、相对论频移与黑体查找表。

本模块提供 `DiskV2Renderer` 主光追 kernel 所需的 `@ti.func` 构件：

- 频移：`disk_g_factor_ti`（静止观者本地方向 + 圆轨道 Doppler + 引力红移），
  与 `src.v2.relativity.disk_g_factor`（NumPy 参考）parity。
- 刚体环平流相位的光行时间换算：`_delayed_rot` / `_delayed_seed`。
- `DiskV2Taichi`：把 `DiskV2Params` / `DiskV2VolumeParams` 平铺为 Python 标量
  （Taichi `@ti.func` 不接受 dataclass 常量），上传 Page–Thorne 温度表、
  CIE 黑体色度 / 亮度表与刚体环相位表，标定噪声归一化因子、⟨c⟩ 与 κ，
  并提供统一气体模型的体积密度场 `density_I`（盘面高斯核心 + 指数大气，同一湍流场）。

模型与公式见 `docs/plans/v2_unified_gas_plan.md`；温度、刚体环平流与带混合沿用参考实现预设 M。
"""


import math

import numpy as np
import taichi as ti

from .palette import (
    _blackbody_luminance_exact,
    BB_LUT,
    LNY_LUT,
    _BB_LUT_N,
    _BB_T_MAX_K,
    _BB_T_MIN_K,
    _LNY_T_MAX_K,
    _LNY_T_MIN_K,
)
from .advection import RigidRingBands, make_fields, upload as adv_upload
from .noise_ti import (
    cascade,
    cascade_fast,
    fbm_gradient,
    fbm_gradient_fast,
    gnoise,
    gnoise_fast,
    hashf as _hashf,
    softplus,
    vnoise,
    vnoise_fast,
)
from .params import DiskV2Params, DiskV2VolumeParams
from .shear_cascade import shear_cascade_geometry
from .temperature_turbulence import (
    GNOISE_STD as _TT_GNOISE_STD,
    INTERMITTENCY_CN_MAX as _TT_CN_MAX,
    LN_R_ANCHOR as _TT_LN_R_ANCHOR,
    WARP_X_OFFSETS as _TT_WARP_X,
    WARP_Z_OFFSETS as _TT_WARP_Z,
    U_OFFSET as _TT_U_OFFSET,
    W_OFFSET as _TT_W_OFFSET,
    W_OFFSET_STEP as _TT_W_OFFSET_STEP,
    temp_turb_gains,
    temp_turb_geometry,
    temp_turb_intermittency_norm,
    temp_turb_warp_period,
)

# 厚度扰动方位拉长 az_stretch_t = 1 时方位周期等于基础周期 n 的参考半径（r_s）：r_b = 6·s 处 n_φ = n。
# 取 6（≈ ISCO 外侧亮区）使内区特征与参考实现相同，外圈按 r_b/(6·s) 增加周期。
_AZ_REF_R = 6.0


@ti.func
def schwarzschild_orbital_beta_ti(r, rs, eps, beta_cap):
    """本地静止观者测得的 Schwarzschild 赤道圆轨道速度（与 `relativity.orbital_beta_local` 一致）。

    Args:
        r: 发射半径（r_s）。
        rs: Schwarzschild 半径。
        eps: 分母 `1 − 2M/r` 下限。
        beta_cap: 速度上限。

    Returns:
        `β = sqrt(M/r) / sqrt(1 − 2M/r)`，范围 `[0, beta_cap]`；ISCO 处 0.5。
    """
    safe_r = ti.max(r, rs + 1e-6)
    mass = 0.5 * rs
    denom = ti.sqrt(ti.max(1.0 - 2.0 * mass / safe_r, eps))
    beta = ti.sqrt(mass / safe_r) / denom
    return ti.min(ti.max(beta, 0.0), beta_cap)


@ti.func
def local_photon_direction_ti(pos, k_coord, rs):
    """光子坐标方向 → 本地静止观者单位方向（与 `relativity.local_photon_direction` 一致）。

    Args:
        pos: 位置向量（r_s）。
        k_coord: 坐标方向（不必归一）。
        rs: Schwarzschild 半径。

    Returns:
        单位向量；`tanψ_local = sqrt(1 − r_s/r) · tanψ_coord`。
    """
    r = ti.max(pos.norm(), 1e-6)
    r_hat = pos / r
    k_rad = k_coord.dot(r_hat) * r_hat
    k_tan = (k_coord - k_rad) * ti.sqrt(ti.max(1.0 - rs / r, 1e-12))
    return (k_rad + k_tan).normalized()


@ti.func
def disk_g_factor_ti(pos, trace_dir, r_obs, rs, spin):
    """盘面圆轨道发射体的频移 g（`relativity.disk_g_factor` 在 spin = +1 时逐值一致）。

    Args:
        pos: 盘局部坐标发射点（盘面 z = 0）。
        trace_dir: 反向追踪光线方向（光子真实方向为其反向）。
        r_obs: 静止观察者半径。
        rs: Schwarzschild 半径。
        spin: 旋转方向符号（+1 逆时针 / −1 顺时针，从盘法向 +z 看）。

    Returns:
        `g = g_grav / (γ (1 − β cosθ_local))`。
    """
    big_r = ti.max(ti.sqrt(pos[0] * pos[0] + pos[1] * pos[1]), 1e-6)
    r3 = ti.max(pos.norm(), rs + 1e-6)
    beta = schwarzschild_orbital_beta_ti(ti.max(big_r, 3.0 * rs), rs, 1e-6, 0.99)
    v_hat = spin * ti.Vector([-pos[1], pos[0], 0.0]) / big_r
    cos_th = v_hat.dot(local_photon_direction_ti(pos, -trace_dir, rs))
    gamma = 1.0 / ti.sqrt(1.0 - beta * beta)
    g_grav = ti.sqrt(ti.max(1.0 - rs / r3, 1e-12)) / ti.sqrt(ti.max(1.0 - rs / r_obs, 1e-12))
    return g_grav / (gamma * (1.0 - beta * cos_th))



@ti.func
def _delayed_rot(rot, om_b, t_delay):
    """刚体环带在光行时间延迟后的转角。

    Args:
        rot: 帧时刻转角 `(Ω_b·t) mod 2π`（rad）。
        om_b: 带中心角速度 Ω_b（rad / (r_s/c)）。
        t_delay: 延迟 `Δt ≥ 0`（r_s/c）。

    Returns:
        标量转角 `Ω_b·(t − Δt)`（rad，未取模；噪声 φ 方向周期为整数，取模无影响）。

    Formula:
        `rot_d = rot − Ω_b·Δt`
    """
    return rot - om_b * t_delay


@ti.func
def _delayed_seed(frac, cyc, t_life, t_delay):
    """刚体环种子相位在光行时间延迟后的 (周期进度, 周期索引)。

    Args:
        frac: 帧时刻周期进度 `s − floor(s) ∈ [0, 1)`，`s = t/t_life + ph0 + p/2`。
        cyc: 帧时刻周期索引 `floor(s) mod 4096`（i32）。
        t_life: 种子寿命（r_s/c）。
        t_delay: 延迟 `Δt ≥ 0`（r_s/c）。

    Returns:
        `(frac_d ∈ [0, 1), cyc_d ∈ [0, 4096))`：采样时刻 `t − Δt` 的进度与索引。

    Formula:
        ```
        s_d    = frac − Δt / t_life
        frac_d = s_d − floor(s_d)
        cyc_d  = (cyc + floor(s_d)) mod 4096
        ```

    Notes:
        只在 f32 下处理小量 `Δt/t_life`（光程 ≲ 100，远小于长视频的 t），
        保持相位表"长视频精度"的设计。
    """
    s_d = frac - t_delay / t_life
    k = ti.floor(s_d)
    cyc_d = ((cyc + ti.cast(k, ti.i32)) % 4096 + 4096) % 4096
    return s_d - k, cyc_d


@ti.func
def _ti_smoothstep(edge0, edge1, x):
    """三次平滑插值（参考实现 `_smoothstep` 同式）。

    Args:
        edge0: 平滑区起点（必须 < edge1）。
        edge1: 平滑区终点。
        x: 输入标量。

    Returns:
        `[0, 1]` 区间的标量。`x <= edge0` 返回 0；`x >= edge1` 返回 1；
        中间用三次多项式平滑过渡。

    Formula:
        ```
        t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
        out = t² (3 - 2t)
        ```
    """
    denom = edge1 - edge0
    t = (x - edge0) / denom
    t = ti.min(ti.max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.data_oriented
class DiskV2Taichi:
    """Disk V2 体积密度场的 Taichi 端句柄。

    Args:
        params: `DiskV2Params`（盘内外半径，r_s）。
        volume_params: `DiskV2VolumeParams`（预设 M 的物理模型与视觉参数）。

    Notes:
        构造时：平铺标量参数 → 上传黑体查找表 → 构造刚体环相位表 → 上传 Page–Thorne
        温度表 → 推出 T_peak → 标定噪声归一化因子、⟨c⟩、κ。每帧由渲染器调用
        `update_advection(t)` 上传当帧相位表。
    """

    def __init__(self, params: DiskV2Params, volume_params: DiskV2VolumeParams, opt_level: int = 0) -> None:
        self.params = params
        # 把 dataclass 的标量字段平铺为 self._<name>，便于 @ti.func 内访问。
        # Taichi 不接受 dataclass 作为 runtime 常量，必须用 Python float。
        self._r_in = float(params.r_in)
        self._r_out = float(params.r_out)
        self._spin = float(params.disk_spin)
        # 优化级别（编译期常量）：≥ 1 时噪声改用逐位一致的快速实现，并共享每采样点的带信息
        self._opt = int(opt_level)
        self._init_luts()
        self.volume_params = volume_params
        self._init_volume_params(volume_params)

    def _init_luts(self) -> None:
        """上传 CIE 黑体色度 / 亮度查找表（与 `palette.py` 共用同一份数据 → parity 恒成立）。"""
        self._bb_lut = ti.Vector.field(3, dtype=ti.f32, shape=_BB_LUT_N)
        self._lny_lut = ti.field(dtype=ti.f32, shape=_BB_LUT_N)
        self._bb_lut.from_numpy(BB_LUT.astype(np.float32))
        self._lny_lut.from_numpy(LNY_LUT.astype(np.float32))



    def _init_volume_params(self, vp: DiskV2VolumeParams) -> None:
        """把 DiskV2VolumeParams 平铺为 self._xxx（Taichi @ti.func 限制）+ 标定。"""
        import taichi as ti

        # SS 结构
        # 盘厚缩放：核心标高乘 thickness_scale（柱密度不变，见 DiskV2VolumeParams）
        self._thick = float(vp.thickness_scale)
        self._hr_ref = float(vp.hr_ref) * self._thick
        self._r_ref_vol = float(vp.r_ref)
        self._f_ref_ss = 1.0 - math.sqrt(self._r_in / self._r_ref_vol)
        self._surf_noise = float(vp.surf_noise)
        # 灰大气
        self._grey_mix = float(vp.grey_mix)
        self._grey_cap = float(vp.grey_cap)
        self._core_floor = float(vp.core_floor)
        self._dt_i = float(vp.dt_i)
        self._surf_lo = float(vp.surf_lo)
        self._surf_k = float(vp.surf_k)
        # 大气（统一气体模型的指数尾巴；含义与公式见 DiskV2VolumeParams 同名字段）
        self._atm_frac = float(vp.atm_frac)
        self._atm_h = float(vp.atm_height)
        self._atm_ext = float(vp.atm_extent)
        self._atm_c0 = float(vp.atm_cov_c0)
        self._atm_soft = float(vp.atm_cov_soft)
        self._atm_fsig = float(vp.atm_fine_sigma)
        self._atm_ffz = float(vp.atm_fine_fz)
        # 小尺度起伏噪声的原始 std（_calibrate_volume 中标定，标定前取 1）
        self._atm_fine_norm = 1.0
        self._abs_sca_q = float(vp.abs_scatter_ratio)
        self._scatter_j = float(vp.scatter_j)
        # 噪声归一化因子（原始 fBm 的实测 std；_calibrate_volume 中标定，标定前取 1）
        self._low_norm = 1.0
        # 低频
        self._lowf_sigma = float(vp.lowf_sigma)
        self._fr_l = float(vp.fr_l)
        self._nphi_l = int(vp.nphi_l)
        self._dln_l = float(vp.dln_l)
        self._k_rigid_l = float(vp.k_rigid_l)
        # 厚度扰动级联
        self._kt_i = float(vp.kt_i)
        self._nphi_t = int(vp.nphi_t)
        self._lt0_i = float(vp.lt0_i)
        self._con_t = float(vp.con_t)
        self._az_ref_r = _AZ_REF_R * float(vp.az_stretch_t)  # 0 = 方位周期固定（参考实现）
        self._oct_gain = float(vp.core_oct_gain)
        self._seam_fix = bool(vp.band_seam_fix)
        # 主云剪切级联：各八度噪声格的周期 / 剪切 / 拉伸在构造期解析算好，kernel 内为常数
        _geom = shear_cascade_geometry(vp.shear_k0, vp.shear_octaves, vp.shear_ar_big, vp.shear_ar_small,
                                       vp.shear_tilt_k, float(self.params.disk_spin))
        self._sc_n = len(_geom.kr)
        self._sc_kr = list(_geom.kr)
        self._sc_per = list(_geom.period)
        self._sc_ua = list(_geom.u_scale)
        self._sc_c = list(_geom.shear)
        self._sc_g = [float(vp.core_oct_gain) ** k for k in range(self._sc_n)]
        self._sc_con = float(vp.shear_con)
        # 小尺度温度湍流：剪切级联最细八度之后再延伸 E 个八度，只调制温度
        # （σ_T = 0 时 _tt_n = 0、相关代码在编译期移除；见 docs/plans/v2_temperature_turbulence_plan.md）
        self._tt_sig = float(vp.temp_turb_sigma)
        self._tt_on = self._tt_sig > 0.0
        # 八度顺序：temp_turb_coarse 个粗尺度八度（主云尺度）在前，temp_turb_octaves 个延伸八度在后
        _tg = temp_turb_geometry(vp.shear_k0, vp.shear_octaves, vp.shear_ar_small, vp.shear_tilt_k,
                                 vp.temp_turb_octaves, float(self.params.disk_spin),
                                 n_coarse=vp.temp_turb_coarse, shear_ar_big=vp.shear_ar_big)
        self._tt_n = len(_tg.kr) if self._tt_on else 0
        self._tt_kr = list(_tg.kr)
        self._tt_per = list(_tg.period)
        self._tt_ua = list(_tg.u_scale)
        self._tt_c = list(_tg.shear)
        self._tt_a = list(temp_turb_gains(vp.temp_turb_octaves, vp.temp_turb_gain, n_coarse=vp.temp_turb_coarse))
        self._tt_a2_sum = sum(a * a for a in self._tt_a)
        # 径向格宽 c_e = r·_tt_cell[e]（径向噪声坐标 |N₀₀|·K·ln r 的一格换算成 r_s）
        self._tt_cell = [1.0 / (ua * kr) for ua, kr in zip(self._tt_ua, self._tt_kr)]
        self._tt_k = float(vp.temp_turb_clamp_px)
        self._tt_lens_half = 0.5 * math.radians(vp.temp_turb_lens_deg)
        # 单图案标准差 N（_calibrate_volume 中标定，标定前取 1）
        self._tt_norm = 1.0
        # 间歇性：局部强度 σ_l = σ_T·clip(ĉ, 0, 3)^γ / M（γ = 0 时编译期移除；M 在 ⟨c⟩ 标定后计算，标定前取 1）
        self._tt_gam = float(vp.temp_turb_intermittency)
        self._tt_intm = 1.0
        # 坐标扭曲：幅度 A/0.19（gnoise 标准差换算成格）；扭曲噪声周期 P_w 与方位坐标缩放 P_w/P
        self._tt_warp = float(vp.temp_turb_warp) / _TT_GNOISE_STD
        self._tt_wper = [temp_turb_warp_period(p) for p in self._tt_per]
        self._tt_wsc = [pw / p for pw, p in zip(self._tt_wper, self._tt_per)]
        self._core_contrast = float(vp.core_contrast)
        # 保方差混合的单图案均值 m（主云 c / 厚度扰动 tn；_calibrate_volume 中标定，标定前取 1）
        self._mc_single = 1.0
        self._mt_single = 1.0
        # 刚体环
        self._dln_r = float(vp.dln_r)
        self._k_rigid_vol = float(vp.k_rigid)
        self._lnr0_r = math.log(self._r_in) - 2 * self._dln_r

        # 刚体环平流表（核心 / 低频各一套，不同哈希流）
        self._adv_core = RigidRingBands(self._r_in, self._r_out, self._dln_r, self._k_rigid_vol,
                                        spin=self._spin)
        self._adv_core_f = make_fields(self._adv_core)
        # 低频层用宽带
        self._adv_low = RigidRingBands(
            self._r_in, self._r_out, self._dln_l, self._k_rigid_l,
            spin=self._spin, phi_b_hash=(23, 1), ph0_hash=(29, 5),
            lnr0_bands=0.0, center_frac=0.5,
        )
        self._adv_low_f = make_fields(self._adv_low)

        # Page–Thorne 温度 LUT
        self._pt_lut = ti.field(dtype=ti.f32, shape=_BB_LUT_N)
        from .physical_fields import build_page_thorne_lut
        build_page_thorne_lut(self._pt_lut, self._r_in, self._r_out, _BB_LUT_N)

        # T_peak（第 1 层：M、Mdot 推出）
        if vp.t_peak_override_K > 0:
            self._t_peak_vol = float(vp.t_peak_override_K)
        else:
            from .physical_fields import derive_t_peak
            self._t_peak_vol = derive_t_peak(vp.bh_mass_msun, vp.mdot_edd)
        # 亮度温度倍率 s：亮度按 Y(s·g·T)/Y(s·T_peak) 计算（1 = 物理，见 DiskV2VolumeParams）
        self._lum_ts = float(vp.lum_temp_scale)
        # ln Y(s·T_peak)（Y(s·g·T)/Y(s·T_peak) 用；s = 1 时即 ln Y(T_peak)）
        self._ln_y_peak = math.log(
            max(_blackbody_luminance_exact(self._lum_ts * self._t_peak_vol), 1e-300)
        )

        # κ 预设 1.0（density_I 的灰大气 tau_z 在标定期间引用 κ）
        self._kappa_vol = 1.0
        # 上传标定时刻的平流相位表（density_I / _flow_I 查表需要）
        self.update_advection(2000.0)
        # 噪声标定（⟨c⟩ 与 κ）
        self._calibrate_volume()

        # 光行时间 / 静止观者相机
        self._light_delay = bool(vp.light_delay)
        self._static_cam = bool(vp.static_cam)

    def _calibrate_volume(self) -> None:
        """标定噪声归一化因子、⟨c⟩（主云级联均值）与 κ（吸收系数）。

        Formula:
            ```
            low_norm   = std(fbm_3oct(i/16, j/512·NPHI_L, ((7i+13j) mod 17)/5))
            fine_norm  = std(_atm_fine_pattern(ln r_i, φ_j, ζ_ij; 0, 0))，ζ_ij = ((7i+13j) mod 17)/17·4 − 2
            m_c, m_t   = mean(单图案 c_k, tn_k)    （仅 band_seam_fix；保方差混合的均值项）
            ⟨c⟩        = mean(c(r, φ, z=0))
            κ          = TAU_I / mean(∫(ab_c + ab_a) dz)，r ∈ [5.5, 6.5]，
                         z ∈ ±max(3H + 0.01, atm_extent·H_a)，800 个中点
            ```
            低频采样网格与参考实现 `raw_stats_low_kernel` 一致（256 × 512）。

        Physical Meaning:
            大尺度明暗参数 `lowf_sigma` 按"噪声单位方差"定义；不归一化时原始 fBm 的 std 只有约 0.2，
            大尺度明暗只剩 1/5。大气小尺度起伏强度 `atm_fine_sigma` 同样按单位方差定义。
            κ 按核心 + 大气的总竖直柱标定，`tau_i` 即 r ≈ 6 处的总 τ⊥。

        Simplifications:
            std 在 t = 2000、无刚体环相位下标定，视作全时段常数（fBm 统计平稳）。
            单图案均值 m 用 4096 个随机 (r, φ, 种子偏移) 样本估计（r ∈ [r_in + 0.5, min(r_out, 20)]，
            与 ⟨c⟩ 同区间），视作与半径无关的常数。800 个中点保证大气指数尾巴（H_a ≫ H）与
            薄核心都被充分采样。
        """
        import taichi as ti

        # 噪声归一化：必须先于 κ 标定（κ 的吸收柱依赖归一化后的低频调制）
        n_r, n_phi = 256, 512
        lbuf = ti.field(dtype=ti.f32, shape=(n_r, n_phi))
        fbuf = ti.field(dtype=ti.f32, shape=(n_r, n_phi))
        ln_r_in = math.log(self._r_in)
        ln_r_out = math.log(self._r_out)

        @ti.kernel
        def _raw_noise(out_l: ti.template(), out_f: ti.template()):
            for i, j in out_l:
                out_l[i, j] = self._fbm(
                    ti.cast(i, ti.f32) / 16.0,
                    ti.cast(j, ti.f32) / n_phi * self._nphi_l,
                    ti.cast((i * 7 + j * 13) % 17, ti.f32) / 5.0,
                    self._nphi_l, 3, 0.5)
                lnr = ln_r_in + (ln_r_out - ln_r_in) * (ti.cast(i, ti.f32) + 0.5) / n_r
                phi = (ti.cast(j, ti.f32) + 0.5) / n_phi * 2.0 * math.pi
                zeta = ti.cast((i * 7 + j * 13) % 17, ti.f32) / 17.0 * 4.0 - 2.0
                out_f[i, j] = self._atm_fine_pattern(lnr, phi, zeta, 0.0, 0.0)

        _raw_noise(lbuf, fbuf)
        self._low_norm = max(float(lbuf.to_numpy().std()), 1e-6)
        self._atm_fine_norm = max(float(fbuf.to_numpy().std()), 1e-6)

        # 温度湍流单图案标准差 N：全部 w_e = 1（足迹取 1e-9 r_s、L = 1），种子偏移 0，与上面同一 (ln r, φ) 网格；
        # z/r = ζ/100（ζ 同大气小尺度起伏），使竖直噪声坐标跨越多个格——值噪声在整数格点平面上方差偏大
        if self._tt_on:
            tbuf = ti.field(dtype=ti.f32, shape=(n_r, n_phi))

            @ti.kernel
            def _raw_temp_turb(out: ti.template()):
                for i, j in out:
                    lnr = ln_r_in + (ln_r_out - ln_r_in) * (ti.cast(i, ti.f32) + 0.5) / n_r
                    phi = (ti.cast(j, ti.f32) + 0.5) / n_phi * 2.0 * math.pi
                    zr = (ti.cast((i * 7 + j * 13) % 17, ti.f32) / 17.0 * 4.0 - 2.0) / 100.0
                    out[i, j] = self._temp_turb_pattern(lnr, ti.exp(lnr), phi, zr, 0.0, 0.0, 1e-9, 1.0)

            _raw_temp_turb(tbuf)
            self._tt_norm = max(float(tbuf.to_numpy().std()), 1e-6)

        # 保方差混合的单图案均值 m：必须先于 ⟨c⟩（⟨c⟩ 经 _flow_I 使用 m）
        if self._seam_fix:
            mbuf = ti.Vector.field(2, dtype=ti.f32, shape=4096)

            @ti.kernel
            def _single_mean(out: ti.template()):
                for i in out:
                    r = self._r_in + 0.5 + _hashf(i, 13, 7) * (ti.min(self._r_out, 20.0) - self._r_in - 0.5)
                    phi = _hashf(i, 15, 11) * 2.0 * math.pi
                    bi = ti.cast(ti.floor(self._band_coord(ti.log(r), phi)), ti.i32)
                    n_t = self._band_nphi(bi)
                    cc, tt = self._flow_noise_core(r, phi, 0.0, _hashf(i, 17, 3) * 97.0,
                                                   _hashf(i, 19, 5) * 97.0, n_t)
                    out[i] = ti.Vector([cc, tt])

            _single_mean(mbuf)
            m = mbuf.to_numpy().mean(0)
            self._mc_single, self._mt_single = float(m[0]), float(m[1])

        # ⟨c⟩
        cbuf = ti.field(dtype=ti.f32, shape=4096)  # 与参考实现 calibrate_I 同样本数

        @ti.kernel
        def _cmean(out: ti.template()):
            for i in out:
                r = self._r_in + 0.5 + _hashf(i, 3, 7) * (
                    ti.min(self._r_out, 20.0) - self._r_in - 0.5
                )
                phi = _hashf(i, 5, 11) * 2.0 * math.pi
                c, tn = self._flow_I(r, phi, 0.0, 0.0)
                out[i] = c

        _cmean(cbuf)
        self._c_mean = max(float(cbuf.to_numpy().mean()), 1e-6)
        # 温度湍流间歇性的归一化常数 M：同一组样本上使 ⟨m(ĉ)²⟩ = 1
        if self._tt_on:
            self._tt_intm = temp_turb_intermittency_norm(cbuf.to_numpy() / self._c_mean, self._tt_gam)

        # κ
        kbuf = ti.field(dtype=ti.f32, shape=256)

        @ti.kernel
        def _column(out: ti.template()):
            for i in out:
                r = 5.5 + ti.cast(i % 16, ti.f32) / 16.0
                phi = ti.cast(i, ti.f32) * 0.61803 * 2.0 * math.pi
                zmax = ti.max(3.0 * self._ss_half_thickness(r) + 0.01, self._atm_ext * self._atm_h * r)
                col = 0.0
                for k in range(800):
                    z = -zmax + (ti.cast(k, ti.f32) + 0.5) / 800.0 * 2.0 * zmax
                    # 透镜权重 0：温度湍流不参与 κ 标定（且只改温度，不影响吸收柱）
                    em_c, tf_c, ab_c, ab_a, em_a, sc_a = self.density_I(r, z, phi, 0.0, 1.0, 0.0)
                    col += (ab_c + ab_a) * 2.0 * zmax / 800.0
                out[i] = col

        _column(kbuf)
        col_mean = max(float(kbuf.to_numpy().mean()), 1e-12)
        self._kappa_vol = self.volume_params.tau_i / col_mean
        print(f"[S5] ⟨c⟩ = {self._c_mean:.3g}，吸收柱均值 = {col_mean:.4g} → κ = {self._kappa_vol:.4g}")

    def update_advection(self, t: float) -> None:
        """每帧上传刚体环相位表（核心 / 低频）。

        Args:
            t: 帧时刻（r_s/c）。
        """
        adv_upload(self._adv_core_f, self._adv_core.phase_table(t))
        adv_upload(self._adv_low_f, self._adv_low.phase_table(t))

    # ---- SS 结构 ----

    # ---- 噪声分派（优化级别 ≥ 1 用快速实现，输出逐位一致） ----

    @ti.func
    def _gn(self, x, y, z, period):
        """梯度噪声：级别 0 用 `gnoise`，级别 ≥ 1 用逐位一致的 `gnoise_fast`。"""
        v = 0.0
        if ti.static(self._opt >= 1):
            v = gnoise_fast(x, y, z, period)
        else:
            v = gnoise(x, y, z, period)
        return v

    @ti.func
    def _casc(self, x, y, z, per_y, l0, l1, con, gain):
        """乘性级联：级别 0 用 `cascade`，级别 ≥ 1 用逐位一致的 `cascade_fast`（参数同 `cascade`）。"""
        v = 0.0
        if ti.static(self._opt >= 1):
            v = cascade_fast(x, y, z, per_y, l0, l1, con, gain)
        else:
            v = cascade(x, y, z, per_y, l0, l1, con, gain)
        return v

    @ti.func
    def _fbm(self, x, y, z, period, octaves: ti.template(), gain):
        """梯度噪声 fBm：级别 0 用 `fbm_gradient`，级别 ≥ 1 用逐位一致的 `fbm_gradient_fast`。"""
        v = 0.0
        if ti.static(self._opt >= 1):
            v = fbm_gradient_fast(x, y, z, period, octaves, gain)
        else:
            v = fbm_gradient(x, y, z, period, octaves, gain)
        return v

    @ti.func
    def _band_info(self, lnr, phi, t_delay):
        """主云与厚度扰动共用的刚体环带信息（优化级别 ≥ 1：每采样点只算一次）。

        Args:
            lnr: `ln r`。
            phi: 盘局部方位角（rad）。
            t_delay: 光行时间延迟（r_s/c）。

        Returns:
            `(w, ph, ox, oz, nb)`：`w` 为 4 路（带 × 种子相位，下标 `2·db + p`）混合权重 `wb·wp`，
            越界的带权重为 0；`ph` 为两条带的流坐标 φ；`ox`、`oz` 为 4 路种子偏移；
            `nb` 为两条带的厚度扰动方位周期 `n_t`（下标 `db`，见 `_band_nphi`）。
            与 `_flow_I` 内部的同名量逐位一致。
        """
        fb = self._band_coord(lnr, phi)
        b0 = ti.floor(fb)
        fbf = fb - b0
        w = ti.Vector([0.0, 0.0, 0.0, 0.0])
        ox = ti.Vector([0.0, 0.0, 0.0, 0.0])
        oz = ti.Vector([0.0, 0.0, 0.0, 0.0])
        ph = ti.Vector([0.0, 0.0])
        nb = ti.Vector([0.0, 0.0])
        for db in ti.static(range(2)):
            bi = ti.cast(b0, ti.i32) + db
            wb = ti.cos(0.5 * math.pi * fbf) ** 2
            if db == 1:
                wb = ti.sin(0.5 * math.pi * fbf) ** 2
            nb[db] = self._band_nphi(bi)
            idx = bi - self._adv_core_f.b_lo
            if 0 <= idx < self._adv_core_f.rot.shape[0]:
                ph[db] = phi - _delayed_rot(self._adv_core_f.rot[idx], self._adv_core_f.om_b[idx], t_delay) - self._adv_core_f.phi_b[idx]
                for p in ti.static(range(2)):
                    fr, cyc = _delayed_seed(self._adv_core_f.frac[idx][p], self._adv_core_f.cyc[idx][p], self._adv_core_f.t_life[idx], t_delay)
                    wp = ti.sin(math.pi * fr) ** 2
                    w[2 * db + p] = wb * wp
                    ox[2 * db + p] = _hashf(bi, cyc, 2 * p) * 97.0
                    oz[2 * db + p] = _hashf(bi, cyc, 2 * p + 1) * 97.0
        return w, ph, ox, oz, nb

    @ti.func
    def _flow_shared(self, r, z, w, ph, ox, oz, nb):
        """`_flow_I` 的共享带信息版本（逐位一致；跳过权重为 0 的组合，加 0 不改变结果）。

        Args:
            r, z: 盘局部半径与高度（r_s）。
            w, ph, ox, oz, nb: `_band_info` 的返回值。

        Returns:
            `(c, tn)`：主云级联值与厚度扰动级联值（≥ 0），混合方式同 `_flow_I`。
        """
        c = 0.0
        tn = 0.0
        wsq = 0.0
        for k in ti.static(range(4)):
            if w[k] != 0.0:
                cc, tt = self._flow_noise_core(r, ph[k // 2], z, ox[k], oz[k], nb[k // 2])
                if ti.static(self._seam_fix):
                    c += w[k] * (cc - self._mc_single)
                    tn += w[k] * (tt - self._mt_single)
                    wsq += w[k] * w[k]
                else:
                    c += w[k] * cc
                    tn += w[k] * tt
        if ti.static(self._seam_fix):
            c, tn = self._blend_finish(c, tn, wsq)
        return c, tn

    @ti.func
    def _ss_half_thickness(self, r):
        """SS 外区标高 H = HR_REF·r·(r/r_ref)^{1/8}·(f/f_ref)^{3/20}。"""
        fr = ti.max(1.0 - ti.sqrt(self._r_in / ti.max(r, self._r_in)), 1e-6)
        return self._hr_ref * r * ti.pow(r / self._r_ref_vol, 0.125) * ti.pow(fr / self._f_ref_ss, 0.15)

    @ti.func
    def _ss_surface_density(self, r):
        """SS 外区柱密度 Σ ∝ (r/r_ref)^{-3/4}·(f/f_ref)^{7/10}·外缘截断。"""
        fr = ti.max(1.0 - ti.sqrt(self._r_in / ti.max(r, self._r_in)), 1e-6)
        outer = 1.0 - _ti_smoothstep(0.72 * self._r_out, self._r_out, r)
        return ti.pow(r / self._r_ref_vol, -0.75) * ti.pow(fr / self._f_ref_ss, 0.7) * outer

    @ti.func
    def _erfc_pos(self, x):
        """erfc(x)，x ≥ 0（A&S 7.1.26，误差 < 1.5e-7）。"""
        t = 1.0 / (1.0 + 0.3275911 * x)
        y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        return y * ti.exp(-x * x)

    @ti.func
    def _page_thorne_temperature(self, r):
        """Page–Thorne 相对论温度 T(r)（查表，线性 r 插值——与 LUT 构建一致）。"""
        u = (ti.min(ti.max(r, self._r_in), self._r_out) - self._r_in) / (
            self._r_out - self._r_in
        )
        f = u * (_BB_LUT_N - 1)
        # 两端钳制（与 palette._lut_lookup 的 np.clip 一致）：运行时 f32 的 log 与编译期常量的 log
        # 在表下限处可相差 1 ulp，使 u 变成小负数、floor 后 i0 = −1；不钳下端会在 CUDA 上越界崩溃
        i0 = ti.max(ti.min(ti.cast(ti.floor(f), ti.i32), _BB_LUT_N - 2), 0)
        w = f - ti.cast(i0, ti.f32)
        return self._t_peak_vol * (self._pt_lut[i0] * (1.0 - w) + self._pt_lut[i0 + 1] * w)

    # ---- 主云流动噪声 ----

    @ti.func
    def _band_coord(self, lnr, phi):
        """刚体环带坐标 fb（整数 = 带中心，相邻整数之间按 cos² / sin² 混合两带）。

        Args:
            lnr: `ln r`（r 单位 r_s）。
            phi: 盘局部方位角（rad）；仅 `band_seam_fix` 时参与。

        Returns:
            标量带坐标，`floor(fb)` 为下侧带号。

        Formula:
            ```
            fb = (ln r − ln r0) / dln_r + 0.35·gn(4·ln r, y, 11.3; 周期 8)
            y  = 0.37（参考实现，边界为正圆）  或  8·φ/(2π)（接缝修复，边界随方位起伏，每圈 8 个周期）
            ```

        Physical Meaning:
            刚体环带是流场的分段近似；带边界只是数值构造，不应在图像上留下同心圆。
            边界随 φ 起伏后，任一圆周上的点分属不同的带相位，接缝被打散。
        """
        fb = 0.0
        if ti.static(self._seam_fix):
            fb = (lnr - self._lnr0_r) / self._dln_r + 0.35 * self._gn(lnr * 4.0, phi / (2.0 * math.pi) * 8.0, 11.3, 8)
        else:
            fb = (lnr - self._lnr0_r) / self._dln_r + 0.35 * self._gn(lnr * 4.0, 0.37, 11.3, 8)
        return fb

    @ti.func
    def _blend_finish(self, c, tn, wsq):
        """保方差混合的收尾：由去均值加权和恢复主云 c 与厚度扰动 tn。

        Args:
            c, tn: 去均值加权和 `Σw_k(c_k − m_c)`、`Σw_k(tn_k − m_t)`。
            wsq: 权重平方和 `Σw_k²`（≥ 0）。

        Returns:
            `(c, tn)`：标量，≥ 0；`wsq ≈ 0`（所有带越界）时为 `(0, 0)`。

        Formula:
            ```
            c  = max(m_c + α·Σw_k(c_k − m_c) / √Σw_k², 0)
            tn = max(m_t +   Σw_k(tn_k − m_t) / √Σw_k², 0)
            ```
            α = `core_contrast`；m_c、m_t 为 `_calibrate_volume` 标定的单图案均值。

        Physical Meaning:
            k 个独立图案的线性混合 Σw_k·c_k 方差为 Σw_k²·Var（Σw_k = 1 时 ≤ Var），
            除以 √Σw_k² 使混合后的起伏方差在带内、带间处处相等（无接缝），均值保持 m。

        Simplifications:
            各图案视作独立同分布；截断到 0 只影响 α·起伏 < −m 的少数深暗缝。
        """
        oc = 0.0
        ot = 0.0
        if wsq > 1e-12:
            inv = 1.0 / ti.sqrt(wsq)
            oc = ti.max(self._mc_single + self._core_contrast * c * inv, 0.0)
            ot = ti.max(self._mt_single + tn * inv, 0.0)
        return oc, ot

    @ti.func
    def _band_nphi(self, bi):
        """核心刚体环带 bi 的厚度扰动方位基频周期 `n_t`（浮点存整数）。

        Args:
            bi: 带号（i32）；带中心半径 `r_b = exp(ln r0 + bi·dln_r)`。

        Returns:
            标量 `n_t`，正整数值的 f32；`az_stretch_t = 0` 时恒为 `nphi_t`。

        Formula:
            ```
            n_φ(b) = max(n, round(n · r_b / (6·s)))，s = az_stretch_t，n = nphi_t
            ```
            方位特征尺寸 ≈ 2π·r_b / (n_φ·3^l) ≈ 2π·6·s / (n·3^l)，与半径无关。

        Physical Meaning:
            开普勒剪切 q = −dlnΩ/dlnr = 3/2 与半径无关，湍流团块被剪切拉长的比例也与半径无关。

        Simplifications:
            周期按带取整（φ 无缝要求整数周期），相邻带周期可能差 1，由带间混合平滑。
        """
        n_t = float(self._nphi_t)
        if ti.static(self._az_ref_r > 0.0):
            q = ti.exp(self._lnr0_r + ti.cast(bi, ti.f32) * self._dln_r) / self._az_ref_r
            n_t = ti.max(n_t, ti.round(n_t * q))
        return n_t

    @ti.func
    def _flow_noise_core(self, ru, th, z, ox, oz, n_t):
        """主云（剪切级联）+ 厚度扰动（乘性级联）两路图案（单个带 × 种子相位）。

        Args:
            ru, z: 半径与高度（r_s）。
            th: 带内流坐标 φ（rad）。
            ox, oz: 种子偏移。
            n_t: 该带的厚度扰动方位周期（`_band_nphi`）。

        Returns:
            `(c_k, tn_k)`：标量，≥ 0。`c_k` 为主云密度结构，`tn_k` 为表面标高扰动。

        Formula:
            ```
            c_k  = _casc_shear(ln r, φ, z/r; ox, oz)
            tn_k = cascade(kt_i·r + ox + 17, φ/2π·n_t, oz + 5; 周期 n_t, 八度 [lt0_i, lt0_i + 2], con_t, γ)
            ```
        """
        c = self._casc_shear(ti.log(ru), th, z / ru, ox, oz)
        if ti.static(self._az_ref_r == 0.0):
            # 周期固定时用编译期常数（运行期变量会改变编译器的常数折叠）
            n_t = float(self._nphi_t)
        tn = self._casc(self._kt_i * ru + ox + 17.0, th / (2.0 * math.pi) * n_t,
                        oz + 5.0, ti.cast(n_t, ti.i32), self._lt0_i, self._lt0_i + 2.0, self._con_t,
                        self._oct_gain)
        return c, tn

    @ti.func
    def _casc_shear(self, lnr, th, zr, ox, oz):
        """对数极坐标剪切级联（与 `shear_cascade.shear_cascade_np` 逐值一致）。

        Args:
            lnr: `ln r`（r 单位 r_s）。
            th: 带内流坐标 φ（rad，已扣除刚体环旋转）。
            zr: 无量纲高度 `z / r`。
            ox, oz: 种子偏移。

        Returns:
            标量，`≥ 0`（由调用方按 ⟨c⟩ 归一）。

        Formula:
            `softplus(con·Σ_k ln(1 + 0.1·γ^k·n_k))`，`n_k = vnoise(u_k + ox, v_k, K_k·zr + oz, P_k)`，
            `u_k = |N₀₀|_k·K_k·ln r`，`v_k = φ/2π·P_k + spin·s_k·K_k·ln r`（系数见 `shear_cascade_geometry`）。

        Physical Meaning:
            主云的乘性级联，每个尺度的团块按开普勒剪切取形（方位拉长、略向拖尾倾斜）。

        Simplifications:
            竖直方向各向同性（`K_k·z/r`）；优化级别 ≥ 1 用 `vnoise_fast`（与 `vnoise` 逐值一致的快速版）。
        """
        s = 0.0
        for k in ti.static(range(self._sc_n)):
            u0 = self._sc_kr[k] * lnr
            u = self._sc_ua[k] * u0
            v = th / (2.0 * math.pi) * self._sc_per[k] + self._spin * self._sc_c[k] * u0
            n = 0.0
            if ti.static(self._opt >= 1):
                n = vnoise_fast(u + ox, v, self._sc_kr[k] * zr + oz, self._sc_per[k])
            else:
                n = vnoise(u + ox, v, self._sc_kr[k] * zr + oz, self._sc_per[k])
            s += ti.log(1.0 + 0.1 * n * self._sc_g[k])
        return softplus(self._sc_con * s)

    @ti.func
    def _flow_I(self, r, phi, z, t_delay):
        """mode 3 刚体环：两带 × 两相位混合，各带以 Ω(r_b) 刚体旋转。

        Args:
            r, phi, z: 盘局部柱坐标（r_s, rad, r_s）。
            t_delay: 光行时间延迟 `Δt ≥ 0`（r_s/c）；采样时刻 = 帧时刻 − Δt。
                0 表示直接使用帧时刻相位表。

        Returns:
            `(c, tn)`：主云级联值（≥ 0）与厚度扰动级联值（≥ 0）。

        Formula:
            ```
            参考实现：      c = Σ_k w_k·c_k                       （w_k = wb·wp，Σw_k = 1）
            band_seam_fix： c = m + α·Σ_k w_k(c_k − m) / √Σw_k²  （见 `_blend_finish`）
            ```
        """
        lnr = ti.log(r)
        fb = self._band_coord(lnr, phi)
        b0 = ti.floor(fb)
        fbf = fb - b0
        c = 0.0
        tn = 0.0
        wsq = 0.0
        for db in ti.static(range(2)):
            bi = ti.cast(b0, ti.i32) + db
            wb = ti.cos(0.5 * math.pi * fbf) ** 2
            if db == 1:
                wb = ti.sin(0.5 * math.pi * fbf) ** 2
            # 用相位表查表（长视频精度）
            idx = bi - self._adv_core_f.b_lo
            if 0 <= idx < self._adv_core_f.rot.shape[0]:
                phi_rigid = phi - _delayed_rot(self._adv_core_f.rot[idx], self._adv_core_f.om_b[idx], t_delay) - self._adv_core_f.phi_b[idx]
                for p in ti.static(range(2)):
                    fr, cyc = _delayed_seed(self._adv_core_f.frac[idx][p], self._adv_core_f.cyc[idx][p], self._adv_core_f.t_life[idx], t_delay)
                    wp = ti.sin(math.pi * fr) ** 2
                    ox = _hashf(bi, cyc, 2 * p) * 97.0
                    oz = _hashf(bi, cyc, 2 * p + 1) * 97.0
                    n_t = self._band_nphi(bi)
                    cc, tt = self._flow_noise_core(r, phi_rigid, z, ox, oz, n_t)
                    if ti.static(self._seam_fix):
                        c += wb * wp * (cc - self._mc_single)
                        tn += wb * wp * (tt - self._mt_single)
                        wsq += (wb * wp) * (wb * wp)
                    else:
                        c += wb * wp * cc
                        tn += wb * wp * tt
        if ti.static(self._seam_fix):
            c, tn = self._blend_finish(c, tn, wsq)
        return c, tn

    # ---- 低频调制 ----

    @ti.func
    def _turb_low(self, r, phi, t_delay):
        """大尺度低频调制场（宽带刚体环，单位方差）。

        Args:
            r, phi: 盘局部柱坐标。
            t_delay: 光行时间延迟 `Δt ≥ 0`（同 `_flow_I`）。

        Returns:
            标量，零均值、约单位方差；调用方以 `exp(σ_L·n − σ_L²/2)` 作 lognormal 调制。

        Formula:
            `n_L = Σ w·n / sqrt(Σ w²) / low_norm`，`low_norm` 为原始 3 八度 fBm 的实测 std。
        """
        lnr = ti.log(r)
        fb = (lnr - ti.log(self._r_in)) / self._dln_l
        b0 = ti.floor(fb)
        fbf = fb - b0
        acc = 0.0
        wsq = 0.0
        for db in ti.static(range(2)):
            bi = ti.cast(b0, ti.i32) + db
            wb = ti.cos(0.5 * math.pi * fbf) ** 2
            if db == 1:
                wb = ti.sin(0.5 * math.pi * fbf) ** 2
            idx = bi - self._adv_low_f.b_lo
            if 0 <= idx < self._adv_low_f.rot.shape[0]:
                phi0 = phi - _delayed_rot(self._adv_low_f.rot[idx], self._adv_low_f.om_b[idx], t_delay) - self._adv_low_f.phi_b[idx]
                for p in ti.static(range(2)):
                    fr, cyc = _delayed_seed(self._adv_low_f.frac[idx][p], self._adv_low_f.cyc[idx][p], self._adv_low_f.t_life[idx], t_delay)
                    wp = ti.sin(math.pi * fr) ** 2
                    ox = _hashf(bi, cyc, 7 + p) * 97.0
                    oz = _hashf(bi, cyc, 11 + p) * 97.0
                    n = self._fbm(lnr * self._fr_l + ox,
                                     phi0 / (2.0 * math.pi) * self._nphi_l,
                                     oz, self._nphi_l, 3, 0.5)
                    w = wb * wp
                    acc += w * n
                    wsq += w * w
        return acc / ti.sqrt(ti.max(wsq, 1e-6)) / self._low_norm

    # ---- 大气小尺度起伏 ----

    @ti.func
    def _atm_fine_pattern(self, lnr, phi0, zeta, ox, oz):
        """大气小尺度起伏的单个图案（剪切级联几何的加性值噪声，未归一化）。

        Args:
            lnr: `ln r`（r 单位 r_s）。
            phi0: 带内流坐标 φ（rad，已扣除刚体环旋转）。
            zeta: 竖直噪声坐标 `atm_fine_fz·z/H_a`（无量纲）。
            ox, oz: 种子偏移（与主云同一组，靠固定偏移 57.3 / 41.9 / 3.1·k 去相关）。

        Returns:
            标量，零均值；原始 std 约 0.3–0.5（由 `_calibrate_volume` 标定为 `_atm_fine_norm`）。

        Formula:
            ```
            n = Σ_k γ^k · vnoise(|N₀₀|_k·K_k·ln r + ox + 57.3,  φ/2π·P_k + spin·s_k·K_k·ln r,
                                 0.9·ζ + oz + 41.9 + 3.1·k;  周期 P_k)
            ```
            K_k、P_k、|N₀₀|_k、s_k 与主云剪切级联（`_casc_shear`）相同，γ = `core_oct_gain`。

        Physical Meaning:
            大气团块与盘面湍流同一套剪切取形（方位拉长、长宽比随尺度变化），但竖直方向独立变化，
            使大气随高度逐渐与下方盘面去相关。

        Simplifications:
            加性 fBm（非乘性级联）；0.9 为原型沿用的竖直格比例。
        """
        s = 0.0
        for k in ti.static(range(self._sc_n)):
            u0 = self._sc_kr[k] * lnr
            u = self._sc_ua[k] * u0 + ox + 57.3
            v = phi0 / (2.0 * math.pi) * self._sc_per[k] + self._spin * self._sc_c[k] * u0
            w = zeta * 0.9 + oz + 41.9 + 3.1 * k
            n = 0.0
            if ti.static(self._opt >= 1):
                n = vnoise_fast(u, v, w, self._sc_per[k])
            else:
                n = vnoise(u, v, w, self._sc_per[k])
            s += self._sc_g[k] * n
        return s

    @ti.func
    def _atm_fine_I(self, r, phi, zeta, t_delay):
        """大气小尺度起伏场（刚体环两带 × 两相位混合，单位方差）。

        Args:
            r, phi: 盘局部柱坐标（r_s, rad）。
            zeta: 竖直噪声坐标 `atm_fine_fz·z/H_a`。
            t_delay: 光行时间延迟 `Δt ≥ 0`（同 `_flow_I`）。

        Returns:
            标量，零均值、约单位方差；调用方以 `exp(σ_a·n − σ_a²/2)` 作保均值 lognormal 调制。

        Formula:
            `n = Σ w_k·n_k / √Σw_k² / fine_norm`（w_k 为带 × 相位权重，与 `_flow_I` 共用平流表与种子）。
        """
        lnr = ti.log(r)
        fb = self._band_coord(lnr, phi)
        b0 = ti.floor(fb)
        fbf = fb - b0
        acc = 0.0
        wsq = 0.0
        for db in ti.static(range(2)):
            bi = ti.cast(b0, ti.i32) + db
            wb = ti.cos(0.5 * math.pi * fbf) ** 2
            if db == 1:
                wb = ti.sin(0.5 * math.pi * fbf) ** 2
            idx = bi - self._adv_core_f.b_lo
            if 0 <= idx < self._adv_core_f.rot.shape[0]:
                phi_rigid = phi - _delayed_rot(self._adv_core_f.rot[idx], self._adv_core_f.om_b[idx], t_delay) - self._adv_core_f.phi_b[idx]
                for p in ti.static(range(2)):
                    fr, cyc = _delayed_seed(self._adv_core_f.frac[idx][p], self._adv_core_f.cyc[idx][p], self._adv_core_f.t_life[idx], t_delay)
                    wp = ti.sin(math.pi * fr) ** 2
                    ox = _hashf(bi, cyc, 2 * p) * 97.0
                    oz = _hashf(bi, cyc, 2 * p + 1) * 97.0
                    n = self._atm_fine_pattern(lnr, phi_rigid, zeta, ox, oz)
                    w = wb * wp
                    acc += w * n
                    wsq += w * w
        return acc / ti.sqrt(ti.max(wsq, 1e-6)) / self._atm_fine_norm

    @ti.func
    def _atm_fine_shared(self, lnr, zeta, w, ph, ox, oz):
        """`_atm_fine_I` 的共享带信息版本（优化级别 ≥ 1；逐位一致，跳过权重为 0 的组合）。

        Args:
            lnr: `ln r`。
            zeta: 竖直噪声坐标 `atm_fine_fz·z/H_a`。
            w, ph, ox, oz: `_band_info` 的返回值。

        Returns:
            标量，零均值、约单位方差。
        """
        acc = 0.0
        wsq = 0.0
        for k in ti.static(range(4)):
            if w[k] != 0.0:
                n = self._atm_fine_pattern(lnr, ph[k // 2], zeta, ox[k], oz[k])
                acc += w[k] * n
                wsq += w[k] * w[k]
        return acc / ti.sqrt(ti.max(wsq, 1e-6)) / self._atm_fine_norm

    # ---- 小尺度温度湍流（docs/plans/v2_temperature_turbulence_plan.md） ----

    @ti.func
    def _temp_turb_weight(self, r, foot, lens_w, e: ti.template()):
        """第 e 个延伸八度的钳制权重（与 `temperature_turbulence.temp_turb_clamp_weights` 逐值一致）。

        Args:
            r: 盘局部柱坐标半径（r_s，> 0）。
            foot: 像素足迹 F（r_s，> 0）。
            lens_w: 透镜权重 L（[0, 1]）。
            e: 八度下标（编译期常量）。

        Returns:
            标量，[0, 1]：格宽 ≤ K 个足迹时为 0，≥ 2K 个足迹时为 L。

        Formula:
            `w_e = L·sstep((c_e/F − K)/K)`，`c_e = r/(|N₀₀|_e·K_e)`，`sstep(s) = 3s² − 2s³`（s 截断到 [0, 1]）

        Physical Meaning:
            频率钳制：比像素足迹还小的八度淡出为其统计均值 0；强透镜光线（L → 0）全部淡出。

        Simplifications:
            只按径向格宽判断（方位格宽更大、更晚淡出）。
        """
        t = ti.min(ti.max((r * self._tt_cell[e] / foot - self._tt_k) / self._tt_k, 0.0), 1.0)
        return lens_w * t * t * (3.0 - 2.0 * t)

    @ti.func
    def _temp_turb_variance(self, r, foot, lens_w):
        """钳制后 n_T 的方差 `V = Σ_e (a_e·w_e)² / Σ_e a_e²`（与 `temp_turb_variance` 逐值一致）。

        Args:
            r, foot, lens_w: 同 `_temp_turb_weight`。

        Returns:
            标量，[0, 1]：全部 w_e = 1 时为 1，全部为 0 时为 0。

        Physical Meaning:
            像素内仍可分辨的温度起伏占全尺度起伏的方差比例，供 f_T 的通量守恒补偿使用。

        Simplifications:
            各八度视为独立、同方差（见 `temperature_turbulence.temp_turb_variance`）。
        """
        v = 0.0
        for e in ti.static(range(self._tt_n)):
            v += (self._tt_a[e] * self._temp_turb_weight(r, foot, lens_w, e)) ** 2
        return v / self._tt_a2_sum

    @ti.func
    def _temp_turb_pattern(self, lnr, r, phi0, zr, ox, oz, foot, lens_w):
        """单个图案的未归一化温度起伏 `Σ_e a_e·w_e·ν_e`（与 `temp_turb_pattern_np` 逐值一致）。

        Args:
            lnr, r: `ln r` 与 r（盘局部柱坐标半径，r_s；调用方已算好，避免重复取对数）。
            phi0: 带内流坐标 φ'（rad，已扣除刚体环带转动）。
            zr: 无量纲高度 `z / r`。
            ox, oz: 刚体环带种子偏移（与主云共用）。
            foot, lens_w: 像素足迹（r_s）与透镜权重，见 `_temp_turb_weight`。

        Returns:
            标量，零均值；除以 `_tt_norm` 后在全部 w_e = 1 时为单位方差。

        Formula:
            ```
            u0  = K_e·(ln r − ln 20)
            (x, y, z) = (|N₀₀|_e·u0 + ox + 1000,  φ'/2π·P_e + spin·s_e·u0,  K_e·z/r + oz + 23 + 5e)
            d_x = gnoise(½x + 7.1, y·P_w/P_e, ½z + 1.7; P_w)，d_y = gnoise(½x + 13.7, y·P_w/P_e, ½z + 9.3; P_w)
            ν_e = vnoise(x + A·d_x/0.19,  y + A·d_y/0.19,  z;  P_e)          （A = 0 时编译期移除扭曲）
            ```
            权重为 0 的八度跳过（结果相同）。

        Physical Meaning:
            湍流级联最小尺度上的温度团块；坐标扭曲打散值噪声格子的行列排布，使团块成为不规则的丝缕。

        Simplifications:
            加性值噪声；竖直方向各向同性（`K_e·z/r`）；扭曲对特征的局部压缩未计入钳制权重；
            优化级别 ≥ 1 用 `vnoise_fast` / `gnoise_fast`（与 `vnoise` / `gnoise` 逐位一致）。
        """
        s = 0.0
        for e in ti.static(range(self._tt_n)):
            w = self._temp_turb_weight(r, foot, lens_w, e)
            if w > 0.0:
                u0 = self._tt_kr[e] * (lnr - _TT_LN_R_ANCHOR)
                x = self._tt_ua[e] * u0 + ox + _TT_U_OFFSET
                y = phi0 / (2.0 * math.pi) * self._tt_per[e] + self._spin * self._tt_c[e] * u0
                z = self._tt_kr[e] * zr + oz + _TT_W_OFFSET + _TT_W_OFFSET_STEP * e
                if ti.static(self._tt_warp > 0.0):
                    yw = y * self._tt_wsc[e]
                    dx = self._gn(0.5 * x + _TT_WARP_X[0], yw, 0.5 * z + _TT_WARP_Z[0], self._tt_wper[e])
                    dy = self._gn(0.5 * x + _TT_WARP_X[1], yw, 0.5 * z + _TT_WARP_Z[1], self._tt_wper[e])
                    x += self._tt_warp * dx
                    y += self._tt_warp * dy
                n = 0.0
                if ti.static(self._opt >= 1):
                    n = vnoise_fast(x, y, z, self._tt_per[e])
                else:
                    n = vnoise(x, y, z, self._tt_per[e])
                s += self._tt_a[e] * w * n
        return s

    @ti.func
    def _temp_turb_I(self, r, phi, zr, t_delay, foot, lens_w):
        """温度湍流的归一化起伏 n_T（刚体环两带 × 两相位混合；优化级别 0 的参考路径）。

        Args:
            r, phi: 盘局部柱坐标（r_s, rad）。
            zr: 无量纲高度 `z / r`。
            t_delay: 光行时间延迟 `Δt ≥ 0`（同 `_flow_I`）。
            foot, lens_w: 像素足迹（r_s）与透镜权重。

        Returns:
            标量，零均值，方差约为 `_temp_turb_variance(r, foot, lens_w)`。

        Formula:
            `n_T = Σ_k ω_k·q_k / √Σω_k² / N`（ω_k 为带 × 相位权重，与 `_atm_fine_I` 共用平流表与种子；
            q_k 为 `_temp_turb_pattern`，N = `_tt_norm`）。

        Physical Meaning:
            温度团块随刚体环带转动、按带重新播种，视频中不卷绕；保方差混合使带间接缝处起伏强度不变。

        Simplifications:
            四个图案共用同一组钳制权重，混合后方差仍为 V。
        """
        lnr = ti.log(r)
        fb = self._band_coord(lnr, phi)
        b0 = ti.floor(fb)
        fbf = fb - b0
        acc = 0.0
        wsq = 0.0
        for db in ti.static(range(2)):
            bi = ti.cast(b0, ti.i32) + db
            wb = ti.cos(0.5 * math.pi * fbf) ** 2
            if db == 1:
                wb = ti.sin(0.5 * math.pi * fbf) ** 2
            idx = bi - self._adv_core_f.b_lo
            if 0 <= idx < self._adv_core_f.rot.shape[0]:
                phi_rigid = phi - _delayed_rot(self._adv_core_f.rot[idx], self._adv_core_f.om_b[idx], t_delay) - self._adv_core_f.phi_b[idx]
                for p in ti.static(range(2)):
                    fr, cyc = _delayed_seed(self._adv_core_f.frac[idx][p], self._adv_core_f.cyc[idx][p], self._adv_core_f.t_life[idx], t_delay)
                    wp = ti.sin(math.pi * fr) ** 2
                    ox = _hashf(bi, cyc, 2 * p) * 97.0
                    oz = _hashf(bi, cyc, 2 * p + 1) * 97.0
                    w = wb * wp
                    acc += w * self._temp_turb_pattern(lnr, r, phi_rigid, zr, ox, oz, foot, lens_w)
                    wsq += w * w
        return acc / ti.sqrt(ti.max(wsq, 1e-6)) / self._tt_norm

    @ti.func
    def _temp_turb_shared(self, lnr, r, zr, w, ph, ox, oz, foot, lens_w):
        """`_temp_turb_I` 的共享带信息版本（优化级别 ≥ 1；跳过权重为 0 的组合，结果相同）。

        Args:
            lnr, r: `ln r` 与 r（r_s）。
            zr: 无量纲高度 `z / r`。
            w, ph, ox, oz: `_band_info` 的返回值。
            foot, lens_w: 像素足迹（r_s）与透镜权重。

        Returns:
            标量，零均值，方差约为 `_temp_turb_variance(r, foot, lens_w)`；与 `_temp_turb_I` 在浮点舍入内一致。

        Formula:
            同 `_temp_turb_I`，带信息由 `_band_info` 预先算好。
        """
        acc = 0.0
        wsq = 0.0
        for k in ti.static(range(4)):
            if w[k] != 0.0:
                acc += w[k] * self._temp_turb_pattern(lnr, r, ph[k // 2], zr, ox[k], oz[k], foot, lens_w)
                wsq += w[k] * w[k]
        return acc / ti.sqrt(ti.max(wsq, 1e-6)) / self._tt_norm

    @ti.func
    def _temp_turb_lens_weight(self, cos_delta):
        """透镜权重 L（与 `temperature_turbulence.temp_turb_lens_weight` 逐值一致）。

        Args:
            cos_delta: 当前光线方向与离开相机时方向的夹角余弦（两者均为单位向量）。

        Returns:
            标量，[0, 1]：δ ≤ δ₀/2 时为 1，δ ≥ δ₀ 时为 0。

        Formula:
            `δ = acos(clamp(cos_delta, −1, 1))`，`L = 1 − sstep((δ − δ₀/2)/(δ₀/2))`

        Physical Meaning:
            累计偏折角越大，像素足迹 λθ 的直线近似越不可信，温度湍流越弱。

        Simplifications:
            用累计偏折角代替完整的光线映射；δ₀ 为经验阈值。
        """
        d = ti.acos(ti.min(ti.max(cos_delta, -1.0), 1.0))
        t = ti.min(ti.max((d - self._tt_lens_half) / self._tt_lens_half, 0.0), 1.0)
        return 1.0 - t * t * (3.0 - 2.0 * t)

    # ---- 体积密度场（统一气体模型） ----

    @ti.func
    def density_I(self, r, z, phi, t_delay, foot, lens_w):
        """统一气体模型的体积密度场：盘面（高斯核心）与大气（指数尾巴）是同一团气体、同一湍流场。

        Args:
            r, z, phi: 盘局部柱坐标（r_s, r_s, rad）；z 为离中面高度，可正可负。
            t_delay: 光行时间延迟 `Δt ≥ 0`（r_s/c）；采样时刻 = 帧时刻 − Δt，
                标定与无光行时间渲染时传 0。
            foot: 像素足迹 F（r_s，> 0）：一个输出像素在采样点处覆盖的尺寸，供温度湍流的频率钳制使用。
            lens_w: 透镜权重 L（[0, 1]）：0 = 温度湍流全部淡出（标定时传 0）。
                温度湍流关闭（`temp_turb_sigma = 0`）时两者不参与计算。

        Returns:
            6 元组标量 `(em_c, tf_c, ab_c, ab_a, em_a, sc_a)`：
            `em_c ≥ 0` 核心发射权重；`tf_c` 温度倍率（核心与大气共用，围绕 1）：温度湍流关闭时
            `∈ [0.7, 1.3·grey_cap]`，开启时再乘对数正态因子 f_T，只保证 > 0；`ab_c ≥ 0` 核心吸收（无量纲，渲染核乘 κ）；`ab_a ≥ 0` 大气消光（吸收 + 散射）；
            `em_a = (1 − ω)·ab_a ∈ [0, ab_a]` 大气热发射权重；
            `sc_a = ω·ab_a·(1 − e^{−τ_c}) ∈ [0, ab_a − em_a]` 大气散射权重（渲染核乘 `scatter_j·S_disk`）。
            盘外或大气顶以上为 `(0, 1, 0, 0, 0, 0)`。

        Formula:
            ```
            ρ      = Σ(r)·[ cfac·exp(−z²/2H_s²)/(√(2π)·H_s) + A·ĉ·cov(ĉ)·m(n)·exp(−|z|/H_a)/(2H_a) ]
            m(n)   = exp(σ_a·n − σ_a²/2)，n = 小尺度起伏（单位方差，`_atm_fine_I`；σ_a = 0 时 m = 1）
            ĉ      = c/⟨c⟩，cfac = f + (1 − f)·ĉ                     （f = core_floor）
            cov(ĉ) = 1/(1 + exp(−(ĉ − c0)/soft))                     （大气覆盖软阈值）
            ω      = 1/(1 + q·ρ_a/ρ_mid)，ρ_mid = Σ/(√(2π)·H)         （q = abs_scatter_ratio）
            τ_z    = κ·[ cfac·Σ·½·erfc(|z|/(√2·H_s)) + ½·A·Σ·ĉ·cov·exp(−|z|/H_a) ]
            tf     = 1 + grey_mix·(min((¾(τ_z + ⅔))^{1/4}, grey_cap) − 1)，再乘 clamp(1 + dt_i·(ĉ − 1), 0.7, 1.3)
            f_T    = exp(σ_l·n_T − 2σ_l²·V)，tf ← tf·f_T                （温度湍流，`_temp_turb_*`；σ_T = 0 时无此项）
            σ_l    = σ_T·clip(ĉ, 0, 3)^γ / M                           （间歇性；γ = 0 时 σ_l = σ_T）
            τ_c    = κ·cfac·Σ                                         （当地核心整柱光学深度）
            J      = scatter_j·S_disk·(1 − e^{−τ_c})                  （渲染核：sc_a·scatter_j·S_disk）
            ```
            Σ(r) 为 SS 柱密度乘大尺度 lognormal 调制；H 为 SS 标高，H_s 为带表面起伏的标高
            （H·(1 − surf_noise + surf_noise·softsat(tn))），H_a = atm_height·r；A = atm_frac。
            核心截断于 |z| < 3H_s，大气截断于 |z| < max(3H, atm_extent·H_a)。

        Physical Meaning:
            同一团气体在竖直方向的两段剖面：等温静力平衡的致密核心 + 被加热、上浮的稀薄大气。
            温度湍流是级联最小尺度上的温度起伏，按平均热辐射通量 ⟨T⁴⟩ 守恒归一，只改热发射、不改密度与吸收。
            大气密度跟随下方湍流（ĉ），稀处出现空隙；温度由上方光学深度按灰大气规律连续给出；
            散射比例 ω 由密度决定（Kramers 吸收 ∝ ρ，电子散射与 ρ 无关），稀薄大气以散射为主。
            散射的入射光来自下方盘面：光学深度为 τ_c 的核心层发出的强度是 S·(1 − e^{−τ_c})，
            核心稀薄处（τ_c ≪ 1）照不亮上方大气。

        Simplifications:
            κ_abs 的 T^{−3.5} 依赖忽略；大气无独立速度场（与核心同刚体环平流）；τ_z 用平行平面近似，
            其大气项不含 m(n)（取局部平均柱）；
            J 只取正下方核心（半个天空、各向同性，不做方向积分与多次散射），cfac 取采样点处的值。
        """
        em_c = 0.0
        tf_c = 1.0
        ab_c = 0.0
        ab_a = 0.0
        em_a = 0.0
        sc_a = 0.0
        if r > self._r_in and r < self._r_out:
            h_geo = self._ss_half_thickness(r)
            sig = self._ss_surface_density(r)
            # 大尺度低频 lognormal 调制（保均值）
            nl = self._turb_low(r, phi, t_delay)
            sig *= ti.exp(self._lowf_sigma * nl - 0.5 * self._lowf_sigma * self._lowf_sigma)
            az = ti.abs(z)
            h_a = self._atm_h * r
            z_top = ti.max(3.0 * h_geo, self._atm_ext * h_a)
            if az < z_top:
                lnr = ti.log(r)
                c = 0.0
                tn = 0.0
                na = 0.0  # 大气小尺度起伏（零均值、单位方差）
                nt = 0.0  # 温度湍流起伏 n_T（零均值，方差 tv）
                tv = 0.0  # 钳制后的方差 V；全部八度淡出时为 0，n_T 不计算
                if ti.static(self._tt_on):
                    tv = self._temp_turb_variance(r, foot, lens_w)
                if ti.static(self._opt >= 1):
                    # 优化级别 ≥ 1：带信息每采样点只算一次（与 _flow_I / _atm_fine_I 逐位一致）
                    bw, bph, box, boz, bnb = self._band_info(lnr, phi, t_delay)
                    c, tn = self._flow_shared(r, z, bw, bph, box, boz, bnb)
                    if ti.static(self._atm_fsig > 0.0):
                        na = self._atm_fine_shared(lnr, self._atm_ffz * z / h_a, bw, bph, box, boz)
                    if ti.static(self._tt_on):
                        if tv > 0.0:
                            nt = self._temp_turb_shared(lnr, r, z / r, bw, bph, box, boz, foot, lens_w)
                else:
                    c, tn = self._flow_I(r, phi, z, t_delay)
                    if ti.static(self._atm_fsig > 0.0):
                        na = self._atm_fine_I(r, phi, self._atm_ffz * z / h_a, t_delay)
                    if ti.static(self._tt_on):
                        if tv > 0.0:
                            nt = self._temp_turb_I(r, phi, z / r, t_delay, foot, lens_w)
                cn = c / self._c_mean
                softsat = 1.0 - 1.0 / (ti.max(tn, 0.0) + 1.0)
                h_s = ti.max(h_geo * (1.0 - self._surf_noise + self._surf_noise * softsat), 1e-6)
                cfac = self._core_floor + (1.0 - self._core_floor) * cn
                # 核心：等温静力平衡 ρ = cfac·Σ/(√(2π)·H_s)·exp(−z²/2H_s²)
                if az < 3.0 * h_s:
                    rho_s = sig * ti.exp(-0.5 * (az / h_s) ** 2) / (2.5066283 * h_s)
                    ab_c = cfac * rho_s
                    em_c = ab_c * (self._surf_lo + self._surf_k * az / h_s)
                # 大气：指数尾巴，柱密度 A·Σ·ĉ·cov(ĉ)，跟随同一湍流场
                cov = 1.0 / (1.0 + ti.exp(-(cn - self._atm_c0) / self._atm_soft))
                col_a = self._atm_frac * sig * cn * cov
                ab_a = col_a * ti.exp(-az / h_a) / (2.0 * h_a)
                if ti.static(self._atm_fsig > 0.0):
                    # 保均值对数正态：⟨exp(σn − σ²/2)⟩ = 1，柱密度期望不变
                    ab_a *= ti.exp(self._atm_fsig * na - 0.5 * self._atm_fsig * self._atm_fsig)
                # 散射反照率 ω(ρ)：稀处以电子散射为主
                rho_mid = sig / (2.5066283 * h_geo)
                omega = 1.0 / (1.0 + self._abs_sca_q * ab_a / ti.max(rho_mid, 1e-12))
                em_a = (1.0 - omega) * ab_a
                # 散射权重：入射光为下方核心层的发射率 (1 − e^{−τ_c})·S_disk
                sc_a = (ab_a - em_a) * (1.0 - ti.exp(-self._kappa_vol * cfac * sig))
                # 灰大气温度：上方柱 τ_z（核心 erfc + 大气指数）连续
                tau_z = self._kappa_vol * (cfac * sig * 0.5 * self._erfc_pos(az / (1.4142136 * h_s))
                                           + 0.5 * col_a * ti.exp(-az / h_a))
                tf_c = 1.0 + self._grey_mix * (
                    ti.min(ti.pow(0.75 * (tau_z + 2.0 / 3.0), 0.25), self._grey_cap) - 1.0)
                tf_c *= ti.min(ti.max(1.0 + self._dt_i * (cn - 1.0), 0.7), 1.3)
                if ti.static(self._tt_on):
                    # 间歇性：局部强度随主云密度（浓处起伏强、稀处平静；⟨m²⟩ = 1）
                    sl = self._tt_sig
                    if ti.static(self._tt_gam > 0.0):
                        sl = self._tt_sig * ti.pow(ti.min(ti.max(cn, 0.0), _TT_CN_MAX), self._tt_gam) / self._tt_intm
                    # 平均通量守恒的对数正态温度起伏：⟨f_T⁴⟩ = 1；tv = 0 时 nt = 0，f_T 精确为 1
                    tf_c *= ti.exp(sl * nt - 2.0 * sl * sl * tv)
        return em_c, tf_c, ab_c, ab_a, em_a, sc_a

    @ti.func
    def blackbody_color_ti(self, T_K):
        """温度 → 黑体色度（线性 sRGB D65，查 `palette.BB_LUT`，与 NumPy parity）。

        Args:
            T_K: 温度（K）。

        Returns:
            `(3,)` RGB 向量，每通道 ≥ 0、BT.709 亮度 = 1；`T_K ≤ 0` 返回 0。
        """
        rgb = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        if T_K > 0.0:
            u = (ti.log(ti.min(ti.max(T_K, _BB_T_MIN_K), _BB_T_MAX_K)) - ti.log(_BB_T_MIN_K)) / (
                ti.log(_BB_T_MAX_K) - ti.log(_BB_T_MIN_K)
            )
            f = u * (_BB_LUT_N - 1)
            i0 = ti.max(ti.min(ti.cast(ti.floor(f), ti.i32), _BB_LUT_N - 2), 0)  # 两端钳制，见 _page_thorne_temperature
            w = f - ti.cast(i0, ti.f32)
            rgb = self._bb_lut[i0] * (1.0 - w) + self._bb_lut[i0 + 1] * w
        return rgb

    @ti.func
    def blackbody_luminance_ti(self, T_K):
        """温度 → 黑体可见光亮度的对数 `ln Y(T)`（查 `palette.LNY_LUT`）。

        Args:
            T_K: 温度（K）。

        Returns:
            `ln Y(T)`；`T_K ≤ 0` 返回 0（调用方以 `exp(lnY − lnY_peak)` 计算比值）。
            频移后的观测亮度比值为 `exp(lnY(g·T) − lnY(T_peak)) = Y(g·T)/Y(T_peak)`。

        Notes:
            返回 `ln Y` 而非 `Y`（与参考实现 `ln_luminance` 一致）——
            kernel 用 `exp(lnY(T) − lnY(T_peak))` 得到无量纲比值。
            若直接返回 `Y`，`exp(Y − lnY_peak)` 会产生 ~exp(32) ≈ 1e14 的错误量级。
        """
        out = 0.0
        if T_K > 0.0:
            u = (ti.log(ti.min(ti.max(T_K, _LNY_T_MIN_K), _LNY_T_MAX_K)) - ti.log(_LNY_T_MIN_K)) / (
                ti.log(_LNY_T_MAX_K) - ti.log(_LNY_T_MIN_K)
            )
            f = u * (_BB_LUT_N - 1)
            i0 = ti.max(ti.min(ti.cast(ti.floor(f), ti.i32), _BB_LUT_N - 2), 0)  # 两端钳制，见 _page_thorne_temperature
            w = f - ti.cast(i0, ti.f32)
            out = self._lny_lut[i0] * (1.0 - w) + self._lny_lut[i0 + 1] * w
        return out
