"""Disk V2 的 Taichi 实现：体积密度场、相对论频移与黑体查找表。

本模块提供 `DiskV2Renderer` 主光追 kernel 所需的 `@ti.func` 构件：

- 频移：`disk_g_factor_ti`（静止观者本地方向 + 圆轨道 Doppler + 引力红移），
  与 `src.v2.relativity.disk_g_factor`（NumPy 参考）parity。
- 刚体环平流相位的光行时间换算：`_delayed_rot` / `_delayed_seed`。
- `DiskV2Taichi`：把 `DiskV2Params` / `DiskV2VolumeParams` 平铺为 Python 标量
  （Taichi `@ti.func` 不接受 dataclass 常量），上传 Page–Thorne 温度表、
  CIE 黑体色度 / 亮度表与刚体环相位表，标定噪声归一化因子、⟨c⟩ 与 κ，
  并提供体积密度场 `density_I`。

所有公式与参考实现 `scripts/proto_disk_reference.py`（预设 M）逐段一致，
一致性由 `scripts/compare_v2_proto.py` 与 `tests/unit/test_disk_v2_proto_parity.py` 保护。
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
)
from .params import DiskV2Params, DiskV2VolumeParams

# core_az_stretch = 1 时方位周期等于基础周期 n 的参考半径（r_s）：r_b = 6·s 处 n_φ = n。
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
        # 盘厚缩放：标高与烟雾层厚度同乘 thickness_scale（柱密度不变，见 DiskV2VolumeParams）
        self._thick = float(vp.thickness_scale)
        self._hr_ref = float(vp.hr_ref) * self._thick
        self._r_ref_vol = float(vp.r_ref)
        self._f_ref_ss = 1.0 - math.sqrt(self._r_in / self._r_ref_vol)
        self._surf_noise = float(vp.surf_noise)
        # 灰大气
        self._grey_mix = float(vp.grey_mix)
        self._grey_cap = float(vp.grey_cap)
        self._core_opac = float(vp.core_opac)
        self._core_floor = float(vp.core_floor)
        self._dt_i = float(vp.dt_i)
        self._surf_lo = float(vp.surf_lo)
        self._surf_k = float(vp.surf_k)
        # 烟雾
        self._smoke_i = float(vp.smoke_i)
        self._smoke_tr = float(vp.smoke_tr)
        self._smoke_s = float(vp.smoke_s)
        self._n_cl_half = int(vp.n_cl_half)
        self._cl_spacing = float(vp.cl_spacing) * self._thick
        self._cl_width = float(vp.cl_width) * self._thick
        self._cl_decay = float(vp.cl_decay)
        self._cl_amp_norm = 1.0 / sum(
            math.exp(-self._cl_decay * abs(k)) for k in range(-self._n_cl_half, self._n_cl_half + 1)
        )
        self._fr_c = float(vp.fr_c)
        self._nphi_c = int(vp.nphi_c)
        self._fz_c = float(vp.fz_c)
        self._sigma_c = float(vp.sigma_c)
        self._cloud_c0 = float(vp.cloud_c0)
        self._cloud_soft = float(vp.cloud_soft)
        self._smoke_on = self._smoke_i > 0.0
        # 烟雾竖直包络 / r：CL_EXTENT = N·间距 + 3·层厚（采样剔除与步长细化用，与参考实现一致）
        self._cl_extent = self._n_cl_half * self._cl_spacing + 3.0 * self._cl_width
        # 噪声归一化因子（原始 fBm 的实测 std；_calibrate_volume 中标定，标定前取 1）
        self._smoke_norm = 1.0
        self._low_norm = 1.0
        # 低频
        self._lowf_sigma = float(vp.lowf_sigma)
        self._fr_l = float(vp.fr_l)
        self._nphi_l = int(vp.nphi_l)
        self._dln_l = float(vp.dln_l)
        self._k_rigid_l = float(vp.k_rigid_l)
        # 主云
        self._kr_i = float(vp.kr_i)
        self._nphi_i = int(vp.nphi_i)
        self._l0_i = float(vp.l0_i)
        self._con_i = float(vp.con_i)
        self._kt_i = float(vp.kt_i)
        self._nphi_t = int(vp.nphi_t)
        self._lt0_i = float(vp.lt0_i)
        # 外圈细节（含义与公式见 DiskV2VolumeParams 同名字段）
        self._az_ref_r = _AZ_REF_R * float(vp.core_az_stretch)  # 0 = 方位周期固定（参考实现）
        self._oct_gain = float(vp.core_oct_gain)
        self._seam_fix = bool(vp.band_seam_fix)
        self._core_contrast = float(vp.core_contrast)
        self._outer_fade = float(vp.outer_detail_fade)
        # 保方差混合的单图案均值 m（主云 c / 厚度扰动 tn；_calibrate_volume 中标定，标定前取 1）
        self._mc_single = 1.0
        self._mt_single = 1.0
        # 尘埃
        self._dust_em = float(vp.dust_em)
        self._dust_s = float(vp.dust_s)
        self._dust_on = bool(vp.dust_on)
        self._dust_kepler = bool(vp.dust_kepler)
        # 刚体环
        self._dln_r = float(vp.dln_r)
        self._k_rigid_vol = float(vp.k_rigid)
        self._lnr0_r = math.log(self._r_in) - 2 * self._dln_r

        # 刚体环平流表（核心 / 尘埃 / 低频各一套；尘埃与低频用不同哈希流）
        self._adv_core = RigidRingBands(self._r_in, self._r_out, self._dln_r, self._k_rigid_vol,
                                        spin=self._spin)
        self._adv_core_f = make_fields(self._adv_core)
        self._adv_dust = RigidRingBands(
            self._r_in, self._r_out, self._dln_r, self._k_rigid_vol,
            spin=self._spin, phi_b_hash=(23, 7), ph0_hash=(19, 3),
        )
        self._adv_dust_f = make_fields(self._adv_dust)
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
            smoke_norm = std(_eval_cloud(ln r, φ, ζ; 0, 0, 0))     （原始烟雾 fBm）
            low_norm   = std(fbm_3oct(i/16, j/512·NPHI_L, ((7i+13j) mod 17)/5))
            m_c, m_t   = mean(单图案 c_k, tn_k)    （仅 band_seam_fix；保方差混合的均值项）
            ⟨c⟩        = mean(c(r, φ, z=0))
            κ          = TAU_I / mean(∫(ab_c + ab_o) dz)，r ∈ [5.5, 6.5]
            ```
            采样网格与参考实现 `raw_stats_cloud_kernel` / `raw_stats_low_kernel`
            完全一致（256 × 512），保证两边归一化因子相同。

        Physical Meaning:
            烟雾层与大尺度明暗的参数（`sigma_c`、`cloud_c0`、`lowf_sigma`）都是按
            "噪声单位方差"定义的；不归一化时原始 fBm 的 std 只有约 0.2，烟雾覆盖率
            会从约 35% 掉到约 3%，大尺度明暗只剩 1/5。

        Simplifications:
            std 在 t = 2000、无刚体环相位下标定，视作全时段常数（fBm 统计平稳）。
            单图案均值 m 用 4096 个随机 (r, φ, 种子偏移) 样本估计（r ∈ [r_in + 0.5, min(r_out, 20)]，
            与 ⟨c⟩ 同区间），视作与半径无关的常数。
        """
        import taichi as ti

        # 噪声归一化：必须先于 κ 标定（κ 的吸收柱依赖归一化后的烟雾与低频调制）
        n_r, n_phi = 256, 512
        sbuf = ti.field(dtype=ti.f32, shape=(n_r, n_phi))
        lbuf = ti.field(dtype=ti.f32, shape=(n_r, n_phi))
        ln_r_in = math.log(self._r_in)
        ln_r_out = math.log(self._r_out)

        @ti.kernel
        def _raw_noise(out_s: ti.template(), out_l: ti.template()):
            for i, j in out_s:
                lnr = ln_r_in + (ln_r_out - ln_r_in) * (ti.cast(i, ti.f32) + 0.5) / n_r
                phi = (ti.cast(j, ti.f32) + 0.5) / n_phi * 2.0 * math.pi
                zeta = ti.cast((i * 7 + j * 13) % 17, ti.f32) / 17.0 * 4.0 - 2.0
                out_s[i, j] = self._eval_cloud(lnr, phi, zeta, 0.0, 0.0, 0.0)
                out_l[i, j] = self._fbm(
                    ti.cast(i, ti.f32) / 16.0,
                    ti.cast(j, ti.f32) / n_phi * self._nphi_l,
                    ti.cast((i * 7 + j * 13) % 17, ti.f32) / 5.0,
                    self._nphi_l, 3, 0.5)

        _raw_noise(sbuf, lbuf)
        self._smoke_norm = max(float(sbuf.to_numpy().std()), 1e-6)
        self._low_norm = max(float(lbuf.to_numpy().std()), 1e-6)

        # 保方差混合的单图案均值 m：必须先于 ⟨c⟩（⟨c⟩ 经 _flow_I 使用 m）
        if self._seam_fix:
            mbuf = ti.Vector.field(2, dtype=ti.f32, shape=4096)

            @ti.kernel
            def _single_mean(out: ti.template()):
                for i in out:
                    r = self._r_in + 0.5 + _hashf(i, 13, 7) * (ti.min(self._r_out, 20.0) - self._r_in - 0.5)
                    phi = _hashf(i, 15, 11) * 2.0 * math.pi
                    lev_cut, con = self._outer_cut(r)
                    bi = ti.cast(ti.floor(self._band_coord(ti.log(r), phi)), ti.i32)
                    n_i, n_t = self._band_nphi(bi)
                    cc, tt = self._flow_noise_core(r, phi, 0.0, _hashf(i, 17, 3) * 97.0,
                                                   _hashf(i, 19, 5) * 97.0, lev_cut, con, n_i, n_t)
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

        # κ
        kbuf = ti.field(dtype=ti.f32, shape=256)

        @ti.kernel
        def _column(out: ti.template()):
            for i in out:
                r = 5.5 + ti.cast(i % 16, ti.f32) / 16.0
                phi = ti.cast(i, ti.f32) * 0.61803 * 2.0 * math.pi
                zmax = 3.0 * self._ss_half_thickness(r) + 0.01
                col = 0.0
                for k in range(200):
                    z = -zmax + (ti.cast(k, ti.f32) + 0.5) / 200.0 * 2.0 * zmax
                    em_c, tf_c, ab_c, em_o, ab_o, em_s = self.density_I(
                        r, z, phi, 0.0, 0.0)
                    col += (ab_c + ab_o) * 2.0 * zmax / 200.0
                out[i] = col

        _column(kbuf)
        col_mean = max(float(kbuf.to_numpy().mean()), 1e-12)
        self._kappa_vol = self.volume_params.tau_i / col_mean
        print(f"[S5] ⟨c⟩ = {self._c_mean:.3g}，吸收柱均值 = {col_mean:.4g} → κ = {self._kappa_vol:.4g}")

    def update_advection(self, t: float) -> None:
        """每帧上传刚体环相位表（核心 / 尘埃 / 低频）。"""
        adv_upload(self._adv_core_f, self._adv_core.phase_table(t))
        adv_upload(self._adv_dust_f, self._adv_dust.phase_table(t))
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
        """核心层与烟雾层共用的刚体环带信息（优化级别 ≥ 1：每采样点只算一次）。

        Args:
            lnr: `ln r`。
            phi: 盘局部方位角（rad）。
            t_delay: 光行时间延迟（r_s/c）。

        Returns:
            `(w, ph, ox, oz, nb)`：`w` 为 4 路（带 × 种子相位，下标 `2·db + p`）混合权重 `wb·wp`，
            越界的带权重为 0；`ph` 为两条带的流坐标 φ；`ox`、`oz` 为 4 路种子偏移；
            `nb` 为两条带的方位周期 `(n_i, n_t)`（下标 `2·db`、`2·db + 1`，见 `_band_nphi`）。
            与 `_flow_I` / `_turb_pair_smoke` 内部的同名量逐位一致。
        """
        fb = self._band_coord(lnr, phi)
        b0 = ti.floor(fb)
        fbf = fb - b0
        w = ti.Vector([0.0, 0.0, 0.0, 0.0])
        ox = ti.Vector([0.0, 0.0, 0.0, 0.0])
        oz = ti.Vector([0.0, 0.0, 0.0, 0.0])
        ph = ti.Vector([0.0, 0.0])
        nb = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for db in ti.static(range(2)):
            bi = ti.cast(b0, ti.i32) + db
            wb = ti.cos(0.5 * math.pi * fbf) ** 2
            if db == 1:
                wb = ti.sin(0.5 * math.pi * fbf) ** 2
            n_i, n_t = self._band_nphi(bi)
            nb[2 * db] = n_i
            nb[2 * db + 1] = n_t
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
        lev_cut, con = self._outer_cut(r)
        c = 0.0
        tn = 0.0
        wsq = 0.0
        for k in ti.static(range(4)):
            if w[k] != 0.0:
                cc, tt = self._flow_noise_core(r, ph[k // 2], z, ox[k], oz[k], lev_cut, con,
                                               nb[2 * (k // 2)], nb[2 * (k // 2) + 1])
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
    def _smoke_shared(self, lnr, zeta, loff, w, ph, ox, oz):
        """`_turb_pair_smoke` 的共享带信息版本（逐位一致，单位方差）。

        Args:
            lnr: `ln r`。
            zeta: 层内竖直坐标 `(z − z_k)/σ_k`。
            loff: 层偏移。
            w, ph, ox, oz: `_band_info` 的返回值。

        Returns:
            标量，零均值、约单位方差。
        """
        acc = 0.0
        wsq = 0.0
        for k in ti.static(range(4)):
            if w[k] != 0.0:
                n = self._eval_cloud(lnr, ph[k // 2], zeta, ox[k], oz[k], loff)
                acc += w[k] * n
                wsq += w[k] * w[k]
        return acc / ti.sqrt(ti.max(wsq, 1e-6)) / self._smoke_norm

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
        i0 = ti.min(ti.cast(ti.floor(f), ti.i32), _BB_LUT_N - 2)
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
        """核心刚体环带 bi 的方位基频周期 `(n_i, n_t)`（主云 / 厚度扰动，浮点存整数）。

        Args:
            bi: 带号（i32）；带中心半径 `r_b = exp(ln r0 + bi·dln_r)`。

        Returns:
            `(n_i, n_t)`：标量，正整数值的 f32；`core_az_stretch = 0` 时恒为 `(nphi_i, nphi_t)`。

        Formula:
            ```
            n_φ(b) = max(n, round(n · r_b / (6·s)))，s = core_az_stretch，n = nphi_i 或 nphi_t
            ```
            方位特征尺寸 ≈ 2π·r_b / (n_φ·3^l) ≈ 2π·6·s / (n·3^l)，与半径无关；径向尺寸 ∝ 1/kr_i 也与
            半径无关，故长宽比恒定，且约与 s 成正比。

        Physical Meaning:
            开普勒剪切 q = −dlnΩ/dlnr = 3/2 与半径无关，湍流团块被剪切拉长的比例也与半径无关。

        Simplifications:
            周期按带取整（φ 无缝要求整数周期），相邻带周期可能差 1，由带间混合平滑。
        """
        n_i = float(self._nphi_i)
        n_t = float(self._nphi_t)
        if ti.static(self._az_ref_r > 0.0):
            q = ti.exp(self._lnr0_r + ti.cast(bi, ti.f32) * self._dln_r) / self._az_ref_r
            n_i = ti.max(n_i, ti.round(n_i * q))
            n_t = ti.max(n_t, ti.round(n_t * q))
        return n_i, n_t

    @ti.func
    def _outer_cut(self, r):
        """主云级联的外圈细节衰减：截掉的起始八度数与级联对比度。

        Args:
            r: 盘局部半径（r_s）。

        Returns:
            `(lev_cut, con)`：标量；`lev_cut ≥ 0`（八度数），`con ≤ con_i`；r ≤ 5 时为 `(0, con_i)`。

        Formula:
            ```
            lev_cut = f · 0.91 · ln(1 + 0.066·max(0, 2r − 10))
            con     = con_i − f · 80 · ln(1 + 0.006·max(0, 2r − 10))
            ```
            f = `outer_detail_fade`；f = 1 与参考实现逐位一致（乘 1.0 精确）。

        Physical Meaning:
            参考实现为外圈降细节的视觉经验式；f = 0 关闭（外圈保留全部八度与对比度）。
        """
        x = ti.max(0.0, 2.0 * r - 10.0)
        lev_cut = self._outer_fade * 0.91 * ti.log(1.0 + 0.066 * x)
        con = self._con_i - self._outer_fade * 80.0 * ti.log(1.0 + 0.006 * x)
        return lev_cut, con

    @ti.func
    def _flow_noise_core(self, ru, th, z, ox, oz, lev_cut, con, n_i, n_t):
        """主云 + 厚度扰动两路级联（单个带 × 种子相位的图案）。

        Args:
            ru, z: 半径与高度（r_s）。
            th: 带内流坐标 φ（rad）。
            ox, oz: 种子偏移。
            lev_cut, con: `_outer_cut` 的返回值。
            n_i, n_t: 该带的方位周期（`_band_nphi`）。

        Returns:
            `(c_k, tn_k)`：标量，≥ 0。
        """
        if ti.static(self._az_ref_r == 0.0):
            # 周期固定时用编译期常数（与参考实现逐位一致；运行期变量会改变编译器的常数折叠）
            n_i = float(self._nphi_i)
            n_t = float(self._nphi_t)
        c = self._casc(self._kr_i * ru + ox, th / (2.0 * math.pi) * n_i,
                    self._kr_i * z + oz, ti.cast(n_i, ti.i32),
                    self._l0_i - lev_cut, self._l0_i + 2.0 - lev_cut, con, self._oct_gain)
        tn = self._casc(self._kt_i * ru + ox + 17.0, th / (2.0 * math.pi) * n_t,
                     oz + 5.0, ti.cast(n_t, ti.i32), self._lt0_i, self._lt0_i + 2.0, self._con_i,
                     self._oct_gain)
        return c, tn

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
        lev_cut, con = self._outer_cut(r)
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
                    n_i, n_t = self._band_nphi(bi)
                    cc, tt = self._flow_noise_core(r, phi_rigid, z, ox, oz, lev_cut, con, n_i, n_t)
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

    # ---- 尘埃 ----

    @ti.func
    def _dust_flow(self, r, phi, z, t_delay):
        """尘埃噪声：与主云相同的刚体环流场，不同哈希流。

        Args:
            r, phi, z: 盘局部柱坐标。
            t_delay: 光行时间延迟 `Δt ≥ 0`（同 `_flow_I`）。

        Returns:
            标量级联值（≥ 0），尘埃密度的结构因子。
        """
        lnr = ti.log(r)
        fb = self._band_coord(lnr, phi)
        b0 = ti.floor(fb)
        fbf = fb - b0
        out = 0.0
        for db in ti.static(range(2)):
            bi = ti.cast(b0, ti.i32) + db
            wb = ti.cos(0.5 * math.pi * fbf) ** 2
            if db == 1:
                wb = ti.sin(0.5 * math.pi * fbf) ** 2
            idx = bi - self._adv_dust_f.b_lo
            if 0 <= idx < self._adv_dust_f.rot.shape[0]:
                phi_rigid = phi - _delayed_rot(self._adv_dust_f.rot[idx], self._adv_dust_f.om_b[idx], t_delay) - self._adv_dust_f.phi_b[idx]
                for p in ti.static(range(2)):
                    fr, cyc = _delayed_seed(self._adv_dust_f.frac[idx][p], self._adv_dust_f.cyc[idx][p], self._adv_dust_f.t_life[idx], t_delay)
                    wp = ti.sin(math.pi * fr) ** 2
                    ox = _hashf(bi, cyc, 40 + 2 * p) * 97.0
                    oz = _hashf(bi, cyc, 41 + 2 * p) * 97.0
                    out += wb * wp * self._casc(
                        2.0 * r + ox, phi_rigid / (2.0 * math.pi) * 9.0, 2.0 * z + oz, 9, 0.0, 6.0, 80.0, 1.0)
        return out

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

    # ---- 烟雾层 ----

    @ti.func
    def _eval_cloud(self, lnr, phi0, zeta, ox, oz, layer_off):
        """烟雾层 fBm（域扭曲），φ 周期 NPHI_C。"""
        x = lnr * self._fr_c + ox + 57.3 + layer_off
        y = phi0 / (2.0 * math.pi) * self._nphi_c
        z = zeta * self._fz_c + oz + 41.9 + 0.37 * layer_off
        wx = self._gn(x * 0.37 + 2.1, y * 0.5, z * 0.5 + 1.3, self._nphi_c // 2)
        return self._fbm(x + 0.9 * wx, y, z, self._nphi_c, 5, 0.5)

    @ti.func
    def _turb_pair_smoke(self, r, phi, zeta, t_delay, loff):
        """烟雾层刚体环噪声（与主云共用 adv_core 表，不同噪声函数，单位方差）。

        Args:
            r, phi: 盘局部柱坐标。
            zeta: 层内竖直坐标 `(z − z_k)/σ_k`（无量纲）。
            t_delay: 光行时间延迟 `Δt ≥ 0`（同 `_flow_I`）。
            loff: 层偏移（使各层噪声独立）。

        Returns:
            标量，零均值、约单位方差（除以 `smoke_norm`）。
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
                    n = self._eval_cloud(lnr, phi_rigid, zeta, ox, oz, loff)
                    w = wb * wp
                    acc += w * n
                    wsq += w * w
        return acc / ti.sqrt(ti.max(wsq, 1e-6)) / self._smoke_norm

    # ---- 体积密度场 ----

    @ti.func
    def density_I(self, r, z, phi, t_delay, dir_z):
        """体积密度场：返回 (核心发射, 核心温度倍率, 核心吸收, 其他发射, 其他吸收, 烟雾发射)。

        核心发射与核心吸收在渲染核中**同时**再乘 CORE_OPAC（源函数不变）；
        "其他" = 尘埃，温度取当地 T(r)；烟雾发射单列，温度取 SMOKE_TR·T(r)。
        PHYS_STRUCT = 1：SS 外区 H(r)、Σ(r) + 竖直高斯。

        Args:
            r, z, phi: 盘局部柱坐标（r_s, r_s, rad）。
            t_delay: 光行时间延迟 `Δt ≥ 0`（r_s/c）；采样时刻 = 帧时刻 − Δt，
                标定与无光行时间渲染时传 0。
            dir_z: 光线方向 z 分量（保留接口，当前 DUST_KEPLER 路径不使用）。

        Returns:
            6 元组标量：`em_c ≥ 0`、`tf_c ∈ [0.7, 1.3·GREY_CAP]`、`ab_c ≥ 0`、
            `em_o ≥ 0`、`ab_o ≥ 0`（烟雾 + 尘埃）、`em_s ≥ 0`；盘外为 `(0, 1, 0, 0, 0, 0)`。
        """
        t = t_delay
        em = 0.0
        ab = 0.0
        em_c = 0.0
        em_s = 0.0
        ab_c_out = 0.0
        tf_c = 1.0
        if r > self._r_in and r < self._r_out:
            # SS 结构
            h_geo = self._ss_half_thickness(r)
            h_cap = h_geo
            sig = self._ss_surface_density(r)
            zc = 3.0 * h_cap
            # 大尺度低频
            if ti.static(True):
                nl = self._turb_low(r, phi, t)
                sig *= ti.exp(self._lowf_sigma * nl - 0.5 * self._lowf_sigma * self._lowf_sigma)
            # 尘埃竖直包络
            xi = (r - self._r_in) / ti.min(self._r_out - self._r_in, 6.0)
            dust_bound = h_geo * ti.max(0.0, 1.0 - 5.0 * xi * xi)
            az = ti.abs(z)
            # 优化级别 ≥ 1：核心层与各烟雾层共用的带信息每采样点只算一次
            lnr = ti.log(r)
            bw = ti.Vector([0.0, 0.0, 0.0, 0.0])
            bph = ti.Vector([0.0, 0.0])
            box = ti.Vector([0.0, 0.0, 0.0, 0.0])
            boz = ti.Vector([0.0, 0.0, 0.0, 0.0])
            bnb = ti.Vector([0.0, 0.0, 0.0, 0.0])
            if ti.static(self._opt >= 1):
                if az < ti.max(zc, self._cl_extent * r):
                    bw, bph, box, boz, bnb = self._band_info(lnr, phi, t)
            # 烟雾层
            if ti.static(self._smoke_on):
                if az < self._cl_extent * r:
                    for kk in range(2 * self._n_cl_half + 1):
                        kf = ti.cast(kk - self._n_cl_half, ti.f32)
                        dz = (z - kf * self._cl_spacing * r) / (self._cl_width * r)
                        if ti.abs(dz) < 3.0:
                            amp = self._cl_amp_norm * ti.exp(-self._cl_decay * ti.abs(kf))
                            rho_c = self._smoke_i * sig * amp * ti.exp(-0.5 * dz * dz) / (
                                2.5066283 * self._cl_width * r)
                            loff = 131.7 * ti.cast(kk + 1, ti.f32)
                            nc = 0.0
                            if ti.static(self._opt >= 1):
                                nc = self._smoke_shared(lnr, dz, loff, bw, bph, box, boz)
                            else:
                                nc = self._turb_pair_smoke(r, phi, dz, t, loff)
                            cov = 1.0 / (1.0 + ti.exp(-(nc - self._cloud_c0) / self._cloud_soft))
                            ab_sm = rho_c * ti.exp(
                                self._sigma_c * nc - 0.5 * self._sigma_c * self._sigma_c) * cov
                            ab += ab_sm
                            if ti.static(True):
                                em_s += ab_sm
            # 核心
            if az < ti.max(zc, dust_bound):
                if az < zc:
                    c = 0.0
                    tn = 0.0
                    if ti.static(self._opt >= 1):
                        c, tn = self._flow_shared(r, z, bw, bph, box, boz, bnb)
                    else:
                        c, tn = self._flow_I(r, phi, z, t)
                    softsat = 1.0 - 1.0 / (ti.max(tn, 0.0) + 1.0)
                    h_s = ti.max(h_cap * (1.0 - self._surf_noise + self._surf_noise * softsat), 1e-6)
                    zs = 3.0 * h_s
                    if az < zs:
                        # 等温静力平衡：ρ = Σ/(√(2π)·H_s)·exp(-z²/2H_s²)
                        rho_s = sig * ti.exp(-0.5 * (az / h_s) ** 2) / (2.5066283 * h_s)
                        # 温和密度起伏
                        cfac = self._core_floor + (1.0 - self._core_floor) * c / self._c_mean
                        ab_b = cfac * rho_s
                        ab_c_out = ab_b
                        em_c = ab_b * (self._surf_lo + self._surf_k * az / h_s)
                        # 灰大气温度倍率
                        if ti.static(True):
                            tau_z = self._kappa_vol * self._core_opac * cfac * sig * 0.5 * self._erfc_pos(
                                az / (1.4142136 * h_s))
                            tf_c = 1.0 + self._grey_mix * (
                                ti.min(ti.pow(0.75 * (tau_z + 2.0 / 3.0), 0.25), self._grey_cap) - 1.0)
                        # 温度起伏
                        if ti.static(True):
                            tf_c *= ti.min(ti.max(
                                1.0 + self._dt_i * (c / self._c_mean - 1.0), 0.7), 1.3)
            # 尘埃
            if ti.static(self._dust_on):
                if az < dust_bound:
                    di = ti.max(1.0 - (z / ti.max(dust_bound, 1e-6)) ** 2, 0.0)
                    if ti.static(self._dust_kepler):
                        dn = self._dust_flow(r, phi, z, t)
                        ab_d = self._dust_em * di * dn
                        ab += ab_d
                        em += ab_d * self._dust_s
        return em_c, tf_c, ab_c_out, em, ab, em_s

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
            i0 = ti.min(ti.cast(ti.floor(f), ti.i32), _BB_LUT_N - 2)
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
            i0 = ti.min(ti.cast(ti.floor(f), ti.i32), _BB_LUT_N - 2)
            w = f - ti.cast(i0, ti.f32)
            out = self._lny_lut[i0] * (1.0 - w) + self._lny_lut[i0 + 1] * w
        return out
