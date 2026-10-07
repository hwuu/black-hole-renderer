"""Disk V2 Taichi 渲染器：Schwarzschild 测地线 + 体积发射-吸收积分 + 物理后处理。

`DiskV2Renderer`（光线、步长与段内积分沿用参考实现预设 M）：

- 光线：笛卡尔等效势 `d²x/dλ² = −1.5 L² x / r⁵`，RK4，步长为位置的连续函数。
- 盘：`DiskV2Taichi.density_I` 统一气体模型（SS 外区结构 + 刚体环平流剪切级联；高斯核心 + 指数大气，
  大气含热发射与下方盘面的单次散射）。
- 辐射转移：`I = Σ T·ΔI_disk + T_end·I_sky`，段内精确均匀解 `ΔI = (j/α)(1 − e^{−αΔs})`。
- 后处理：`postfx`（白平衡 → bloom → 镶边 → 色散 → 保色度 ACES → sRGB）。

调用方式（由 `render.py --disk_model v2` 触发）：

```python
renderer = DiskV2Renderer(width=1920, height=1080, params=DiskV2Params(r_out=30.0),
                          volume_params=DiskV2VolumeParams(), skybox=sky)
img = renderer.render(cam_pos=[0, -39.7, 4.87], fov=38.0, t=2000.0)
```
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import taichi as ti

from .camera import build_camera_v1_compatible
from .noise_ti import hashf as _hashf
from .palette import doppler_lum_compensation
from .params import DiskV2Params, DiskV2VolumeParams
from .postfx import hdr_luminance, postfx, postfx_params_defaults, srgb_decode
from .taichi_impl import DiskV2Taichi, disk_g_factor_ti
from .temperature_turbulence import MIN_PATH_LENGTH as _TT_MIN_PATH


# Schwarzschild 半径，与 render.py 保持一致（无量纲单位）。
_RS: float = 1.0
# 迭代预算：最大步数 = r_escape·40 / _H_BUDGET（不影响步长本身，只限制总步数）。
_H_BUDGET: float = 0.1


@ti.data_oriented
class DiskV2Renderer:
    """V2 体积吸积盘 Taichi 渲染器。

    Args:
        width: 输出图像宽度（像素）。
        height: 输出图像高度（像素）。
        params: `DiskV2Params`（盘内外半径，r_s）。
        skybox: 天空盒 `(tex_h, tex_w, 3)` float32 数组，`[0, 1]`，等距柱状投影。
        volume_params: `DiskV2VolumeParams`（体积密度场与物理模型参数，默认预设 M）。
        r_max: 逃逸半径下限；实际 `r_escape = max(r_max, 2·cam_distance, 1.6·r_out)`。
        disk_tilt_deg: 盘倾角（度），盘面绕世界 x 轴旋转（俯仰）。
        disk_roll_deg: 盘滚转角（度），盘面绕世界 y 轴旋转；相机位于 y 轴负方向时，
            正值使盘面在画面上左低右高。注意：滚转固定在世界系，环绕过程中画面倾角
            会漂移（起始 ρ° → 环绕 90° 时变为开口角变化 → 180° 时反向）。
        camera_roll_deg: 相机滚转角 ρ（度）：相机绕自身光轴旋转，整个画面（盘与星空）
            一起倾斜；在画面系中生效，环绕过程中倾角恒定。+12.5° 为画面左低右高；0 = 不滚转。
        doppler_lum: 多普勒亮度强度 p（≥ 0）：亮度用 `Y(s·g^{p·k}·T)`；1 = 物理，0 = 无多普勒明暗；
            默认 0.25（预设 M 定稿 0.55；减弱左右明暗不对称，视频构图下左右通量比约 3.7 → 1.7）。s 为
            `volume_params.lum_temp_scale`，k = `palette.doppler_lum_compensation(T_peak, s)`
            （s = 1 时 k = 1）。
        doppler_color: 多普勒颜色强度 q（≥ 0）：色度用 `χ(min(T·g^q, T_cap))`；1 = 物理，0 = 无多普勒变色。
            默认 0.75（预设 M 定稿 1.5；随亮度指数同步减弱，避免远离侧"又亮又红"）。
        color_temp_cap_K: 颜色温度上限 T_cap（K）：色度温度超过它时取 T_cap，使最亮处止于白色、不越过
            白点变蓝（颜色序列 黑 → 暗红 → 金 → 白）。默认 None = 跟随白平衡色温 `white_balance_K`
            （白平衡下显示为白色的温度）；0 = 不封顶（参考实现）。只影响色度，不影响亮度。
        white_balance_K: 相机白平衡色温（K，1000–40000，ISP 层）：色温为该值的黑体显示为白色。None = `postfx`
            默认 4500 K（盘面主体 3000–5000 K 呈浅金、最热处为白；4000 K 偏冷发白）；参考实现为 5000 K
            （整体偏暖黄）。
            调低 → 盘面更白、更冷；调高 → 更暖、更黄。
        color_temp_floor_K: 颜色温度下限 T_floor（K，≥ 0）：色度温度低于它时取 T_floor（硬截断），
            使冷区不再显示为橙红，颜色序列变为 黑 → 暗金 → 金 → 白（暗处只靠亮度变暗）。
            0 = 不设下限（参考实现）。只影响色度，不影响亮度。
        bloom_luma_threshold: bloom 高光提取方式（仅 `lens_model="legacy"` 时生效，传给 `postfx.apply_bloom`）：True（默认）= 按亮度扣阈值、
            散射光保持像素色度；False = 逐通道扣阈值（参考实现旧行为，金色区的光晕偏橙红）。
        lens_model: 镜头模型：`"psf"`（默认）= 能量守恒镜头 PSF（眩光长尾 + 轴向色差，作用于全部
            场景光、无阈值，见 `docs/imaging_model.md`）；`"legacy"` = 参考实现的阈值 bloom + 镶边
            （会额外加光、点亮辉光后面的盘面，仅作对照）。
        lens_glare: 镜头眩光强度 ε ∈ [0, 1]（`lens_model="psf"` 时生效）：每个点散射到长尾的能量比例。
            None = `postfx` 默认 0.5（柔光镜量级）；0 = 理想镜头（无辉光）。越大辉光越明显，同时
            全画面对比度按 (1 − ε) 下降。
        exposure_ev: 曝光补偿（档，传感器层）：自动曝光 `0.9 / p99.9(盘区亮度)` 再乘 `2^EV`。
            0 = 盘区 p99.9 亮度落在 0.9（画面无过曝，最亮处只到浅金、缺少"发光"感）；默认 1.5
            （×2.83：视频构图下约 2% 的盘区像素越过白点烧白，阴影内辉光约为 EV = 0 时的 2 倍）。负值更暗。视频首帧锁定的
            曝光已包含补偿。
        film_response: 胶片响应 m ∈ [0, 1]（ISP 层，传给 `postfx.tonemap_film`）：0（默认）= 保色度 ACES，
            与引入该参数前逐位一致，高光止于浅金；1 = 逐通道 ACES（胶片各层 / 传感器各通道独立饱和），
            同一颜色越亮越淡、最亮处趋白；中间值线性混合。
        sky_gain: 天空亮度系数：天空（sRGB 解码为线性光后）× `sky_gain` 在盘曝光之后叠加，
            不参与自动曝光；0 = 黑色天空。
        ss: 超采样倍率（每像素 `ss²` 条光线；内部以 `ss·W × ss·H` 积分后盒式下采样）。
        opt_level: 优化级别（见 `docs/plans/v2_performance_plan.md` §3）：
            0 参考实现；1 精确优化（输出与 0 在浮点舍入内一致）；2 盘内步长 ×2；
            3 盘内步长 ×3（近似预览）。
        device: Taichi 未初始化时用于初始化的设备 `"cpu"` / `"gpu"`。
    """

    def __init__(
        self,
        width: int,
        height: int,
        params: DiskV2Params,
        skybox: np.ndarray,
        volume_params: Optional[DiskV2VolumeParams] = None,
        r_max: float = 10.0,
        disk_tilt_deg: float = 0.0,
        disk_roll_deg: float = 0.0,
        camera_roll_deg: float = 0.0,
        doppler_lum: float = 0.25,
        doppler_color: float = 0.75,
        color_temp_cap_K: Optional[float] = None,
        white_balance_K: Optional[float] = None,
        color_temp_floor_K: float = 0.0,
        bloom_luma_threshold: bool = True,
        lens_model: str = "psf",
        lens_glare: Optional[float] = None,
        exposure_ev: float = 1.5,
        film_response: float = 0.0,
        sky_gain: float = 0.5,
        ss: int = 1,
        opt_level: int = 0,
        device: str = "gpu",
    ) -> None:
        """初始化渲染器：上传天空盒、构造盘体 Taichi 句柄并编译主 kernel。"""
        # 调用方（CLI / 测试 / 脚本）未初始化 Taichi 时才初始化；已有程序时跳过，
        # 重复 init 会销毁已存在的所有 field
        if ti.lang.impl.get_runtime().prog is None:
            ti.init(arch=ti.cpu if device == "cpu" else ti.gpu, default_fp=ti.f32)

        self.width = width
        self.height = height
        self.params = params
        self.volume_params = volume_params if volume_params is not None else DiskV2VolumeParams()
        self.r_max = float(r_max)
        self.disk_tilt_rad = math.radians(disk_tilt_deg)
        self.disk_roll_rad = math.radians(disk_roll_deg)
        self.camera_roll_deg = float(camera_roll_deg)
        self.doppler_lum = float(doppler_lum)
        self.doppler_color = float(doppler_color)
        # 颜色温度上限（K）；0 = 不封顶。kernel 内 T_col = min(T·g^q, cap)
        # 白平衡（ISP 层）；None = postfx 默认。色温封顶默认跟随白平衡，保证最亮处止于白色
        if white_balance_K is None:
            white_balance_K = float(postfx_params_defaults()["white_balance_K"])
        if not 1000.0 <= white_balance_K <= 40000.0:
            raise ValueError("white_balance_K must be in [1000, 40000] (blackbody LUT range)")
        self.white_balance_K = float(white_balance_K)
        if color_temp_cap_K is None:
            color_temp_cap_K = self.white_balance_K
        if color_temp_cap_K < 0.0:
            raise ValueError("color_temp_cap_K must be >= 0 (0 = no cap)")
        self.color_temp_cap_K = float(color_temp_cap_K)
        # 颜色温度下限（K）；0 = 不设下限。kernel 内 T_col = max(T_col, floor)
        if color_temp_floor_K < 0.0:
            raise ValueError("color_temp_floor_K must be >= 0 (0 = no floor)")
        if self.color_temp_cap_K > 0.0 and color_temp_floor_K > self.color_temp_cap_K:
            raise ValueError("color_temp_floor_K must be <= color_temp_cap_K")
        self.color_temp_floor_K = float(color_temp_floor_K)
        self.bloom_luma_threshold = bool(bloom_luma_threshold)
        if lens_model not in ("psf", "legacy"):
            raise ValueError("lens_model must be 'psf' or 'legacy'")
        if lens_glare is not None and not 0.0 <= lens_glare <= 1.0:
            raise ValueError("lens_glare must be in [0, 1]")
        self.lens_model = lens_model
        # None = 用 postfx 默认值
        self.lens_glare = None if lens_glare is None else float(lens_glare)
        self.exposure_ev = float(exposure_ev)
        if not 0.0 <= film_response <= 1.0:
            raise ValueError("film_response must be in [0, 1]")
        self.film_response = float(film_response)
        self.sky_gain = float(sky_gain)
        self.ss = max(1, int(ss))
        # 内部渲染分辨率：ss×ss 超采样后在 NumPy 侧盒式下采样（与参考实现 render_hdr 一致）
        if opt_level not in (0, 1, 2, 3):
            raise ValueError("opt_level must be 0, 1, 2 or 3")
        self.opt_level = int(opt_level)
        # 盘内步长放大系数（级别 2 = 2，级别 3 = 3；S0 实测 ×2 误差 0.25%）
        self._step_c = {0: 1.0, 1: 1.0, 2: 2.0, 3: 3.0}[self.opt_level]
        self._iw = width * self.ss
        self._ih = height * self.ss
        # 最近一帧下采样后的盘发射 HDR `(H, W, 3)`（不含天空；曝光与对比均基于它）
        self.last_hdr: np.ndarray | None = None
        self.last_white_point: float = 1.0
        # 视频曝光锁定：非 None 时 render() 直接使用该曝光（参考实现视频首帧锁定）
        self.fixed_exposure: float | None = None
        self.disk_ti = DiskV2Taichi(params=params, volume_params=self.volume_params, opt_level=self.opt_level)
        # 亮度温度倍率 s ≠ 1 时补偿多普勒亮度指数，保持左右明暗不对称（s = 1 时补偿系数精确为 1）
        self._doppler_lum_eff = self.doppler_lum * doppler_lum_compensation(
            self.disk_ti._t_peak_vol, self.volume_params.lum_temp_scale)

        iw, ih = self._iw, self._ih
        # 线性 HDR 分两路：盘发射 Σ T·ΔI_disk 与透过的天空 T_end·I_sky（线性光）。
        # 像素值 I = exposure·hdr + sky_gain·sky：曝光只按盘计算，天空亮度独立控制。
        self.hdr_field = ti.Vector.field(3, dtype=ti.f32, shape=(iw, ih))
        self.sky_field = ti.Vector.field(3, dtype=ti.f32, shape=(iw, ih))
        # 视界命中标记（诊断 / 单测用）：1 = 光线落入视界
        self.event_horizon_field = ti.field(dtype=ti.i32, shape=(iw, ih))
        self.jitter_seed = ti.field(dtype=ti.i32, shape=())
        self.jitter_seed[None] = 0

        # 上传 skybox：sRGB 编码值 → 线性光（PNG / 程序星空均为 sRGB 编码）。
        # Taichi field 下标为 [x, y]，NumPy skybox 是 [y, x, rgb]，上传前转置。
        sky_h, sky_w = skybox.shape[:2]
        self.sky_w = int(sky_w)
        self.sky_h = int(sky_h)
        self.skybox_field = ti.Vector.field(3, dtype=ti.f32, shape=(sky_w, sky_h))
        # float32 解码：8K 等距柱状图在 float64 下需要约 800 MB 中间缓冲
        sky_lin = srgb_decode(np.clip(skybox.astype(np.float32), 0.0, 1.0)).astype(np.float32)
        self.skybox_field.from_numpy(np.transpose(sky_lin, (1, 0, 2)))

        # 相机参数 field（每帧更新）。
        self.cam_pos_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_right_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_up_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_forward_field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.pixel_width_field = ti.field(dtype=ti.f32, shape=())
        self.pixel_height_field = ti.field(dtype=ti.f32, shape=())
        self.r_escape_field = ti.field(dtype=ti.f32, shape=())

        self._compile_kernels()

    def _compile_kernels(self) -> None:
        """编译主光追 kernel（闭包捕获盘体句柄与编译期常量）。"""
        disk = self.disk_ti
        tilt = float(self.disk_tilt_rad)
        roll = float(self.disk_roll_rad)
        g_spin = float(self.params.disk_spin)
        rs = float(_RS)
        sky_w = int(self.sky_w)
        sky_h = int(self.sky_h)
        img_w = int(self._iw)
        img_h = int(self._ih)
        static_cam = bool(disk._static_cam)
        light_delay = bool(disk._light_delay)
        step_c = float(self._step_c)
        # 盘内基础步长随盘厚缩放（固定 0.03 r_s 在薄盘中会比盘本身还厚）
        thick = float(disk._thick)
        # 温度湍流（σ_T = 0 时相关代码在编译期移除）；像素足迹按输出像素计算：内部像素高度 × 每轴超采样倍率
        tt_on = bool(disk._tt_on)
        ss_f = float(self.ss)

        @ti.func
        def _compute_acceleration(pos, L2):
            """Schwarzschild 笛卡尔等效势的加速度：a = -1.5 L² x / r⁵。"""
            r2 = pos.dot(pos)
            r = ti.sqrt(r2)
            r5 = r2 * r2 * r
            return -1.5 * L2 / r5 * pos

        @ti.func
        def _sample_skybox(direction):
            """根据光线方向双线性采样天空盒（等距柱状投影，与 V1 `_sample_skybox` 同约定）。

            Formula:
                `u = φ/(2π)·W`（φ ∈ [0, 2π)），`v = θ/π·H`；u 方向周期环绕，v 方向钳制。
            """
            d = direction.normalized()
            theta = ti.acos(ti.min(ti.max(d[2], -1.0), 1.0))
            phi = ti.atan2(d[1], d[0])
            if phi < 0:
                phi += 2.0 * ti.math.pi
            u = phi / (2.0 * ti.math.pi) * sky_w
            v = theta / ti.math.pi * sky_h
            u0 = ti.cast(ti.floor(u), ti.i32)
            v0 = ti.cast(ti.floor(v), ti.i32)
            fu = u - ti.cast(u0, ti.f32)
            fv = v - ti.cast(v0, ti.f32)
            u0w = u0 % sky_w
            u1w = (u0 + 1) % sky_w
            v0h = ti.min(ti.max(v0, 0), sky_h - 1)
            v1h = ti.min(ti.max(v0 + 1, 0), sky_h - 1)
            return (self.skybox_field[u0w, v0h] * (1.0 - fu) * (1.0 - fv)
                    + self.skybox_field[u1w, v0h] * fu * (1.0 - fv)
                    + self.skybox_field[u0w, v1h] * (1.0 - fu) * fv
                    + self.skybox_field[u1w, v1h] * fu * fv)

        @ti.func
        def _world_to_local_disk(pos):
            """世界坐标 → 盘体局部坐标（先俯仰 tilt 绕 x 轴、后滚转 roll 绕 y 轴的逆变换）。

            Formula:
                ```
                (x₁, y₁, z₁) = (x, y·cos t + z·sin t, −y·sin t + z·cos t)   # 俯仰逆变换
                x' = x₁·cos r + z₁·sin r                                      # 滚转逆变换
                y' = y₁
                z' = −x₁·sin r + z₁·cos r
                ```
            """
            sin_t = ti.sin(tilt)
            cos_t = ti.cos(tilt)
            x1 = pos[0]
            y1 = pos[1] * cos_t + pos[2] * sin_t
            z1 = -pos[1] * sin_t + pos[2] * cos_t
            sin_r = ti.sin(roll)
            cos_r = ti.cos(roll)
            x_local = x1 * cos_r + z1 * sin_r
            y_local = y1
            z_local = -x1 * sin_r + z1 * cos_r
            return ti.Vector([x_local, y_local, z_local], dt=ti.f32)

        @ti.kernel
        def _ray_march_kernel():
            """主光追：Schwarzschild 测地线 + 体积发射-吸收积分，写 `hdr_field` / `sky_field`。"""
            r_esc = self.r_escape_field[None]
            max_iter = ti.cast(r_esc * 40.0 / _H_BUDGET, ti.i32)
            max_affine = r_esc * 40.0
            r_cap = 1.0 * rs

            cp = self.cam_pos_field[None]
            cr = self.cam_right_field[None]
            cu = self.cam_up_field[None]
            cf = self.cam_forward_field[None]
            pw = self.pixel_width_field[None]
            ph = self.pixel_height_field[None]
            theta_out = ph * ss_f  # 输出像素的像素角（像平面距相机 1，小角度近似）
            # 与 V1 `render._ray_march_kernel` 完全一致：垂直 FOV + aspect 像素步长。
            center = cp + cf * 1.0
            tl = center - cr * (pw * img_w / 2.0) + cu * (ph * img_h / 2.0)

            for i, j in self.hdr_field:
                px_f = ti.cast(i, ti.f32)
                py_f = ti.cast(j, ti.f32)
                pixel_pos = tl + (px_f + 0.5) * pw * cr - (py_f + 0.5) * ph * cu
                ray_dir = (pixel_pos - cp).normalized()
                if ti.static(static_cam):
                    # 静止观者本地方向 → 坐标方向：tanψ_coord = tanψ_local / sqrt(1 − r_s/r)
                    # （ψ 为与径向夹角；径向分量不变，切向分量除以 sqrt(1 − r_s/r)）
                    r0 = cp.norm()
                    rh = cp / r0
                    d_rad = ray_dir.dot(rh) * rh
                    ray_dir = (d_rad + (ray_dir - d_rad) / ti.sqrt(1.0 - rs / r0)).normalized()

                pos = cp
                step_idx = 0
                dir_ = ray_dir
                dir0 = ray_dir  # 光线离开相机时的方向（单位向量；温度湍流的累计偏折角以它为参考）

                # 角动量平方 L² = |r × dir|²。
                L_vec = pos.cross(dir_)
                L2_val = L_vec.dot(L_vec)
                # 光子环保护：冲击参数 b ≈ |x × d| 落在临界值 3√3/2 ≈ 2.6 附近（[2.4, 4.0]）的光线
                # 形成光子环与阴影边缘，它们掠过极薄的内盘；这些光线保持原步长，避免放大步长后
                # 漏采薄层、光子环断成点。该环带只占画面约 2%。
                sc = step_c
                b_imp = ti.sqrt(L2_val)
                if b_imp > 2.4 and b_imp < 4.0:
                    sc = 1.0

                escaped = False
                lam = 0.0
                escape_dir = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                event_horizon_hit = False
                hdr_accum = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                transmittance = 1.0
                step_count = 0
                affine = 0.0

                while step_count < max_iter:
                    r_cur = pos.norm()
                    # 参考实现（模型 I）步长：位置的连续函数，避免条纹。
                    #   远场 h = min(0.06·r, 2)；近视界 h ≤ 0.02 + 0.06·(r − 1)；
                    #   核心盘包络 zb = 3H(r_c) + 0.02 内 h → 0.03·s，离开后按距离 0.3·d 放大；
                    #   大气包络 atm_extent·H_a + 0.02 内 h → max(0.3·H_a, 0.03·s)（H_a = atm_height·r_c）；
                    #   s = thickness_scale（盘厚缩放），H 已含 s，H_a 不含。
                    # 盘相关距离在盘局部坐标下计算（支持倾角）。
                    h = ti.min(0.06 * r_cur, 2.0)
                    h = ti.min(h, 0.02 + 0.06 * ti.max(r_cur - 1.0, 0.0))
                    pl = _world_to_local_disk(pos)
                    rc_h = ti.sqrt(pl[0] * pl[0] + pl[1] * pl[1])
                    zb = 3.0 * disk._ss_half_thickness(ti.max(rc_h, disk._r_in)) + 0.02
                    rad_out = ti.max(disk._r_in * 0.95 - rc_h, 0.0) + ti.max(rc_h - disk._r_out * 1.02, 0.0)
                    d_slab = ti.max(ti.abs(pl[2]) - zb, 0.0) + rad_out
                    h = ti.min(h, sc * 0.03 * thick + 0.3 * d_slab)
                    d_atm = ti.max(ti.abs(pl[2]) - (disk._atm_ext * disk._atm_h * rc_h + 0.02), 0.0) + rad_out
                    h = ti.min(h, sc * ti.max(0.3 * disk._atm_h * rc_h, 0.03 * thick) + 0.3 * d_atm)
                    if ti.static(tt_on):
                        # 温度湍流：大气包络内每个可见最细格宽至少 2/sc 个采样点
                        #   h ≤ sc·½·max(c_last, K·F)，F = max(λ, λ_min)·θ（λ 取段起点）；透镜权重为 0 时不约束
                        if d_atm == 0.0:
                            if disk._temp_turb_lens_weight(dir_.normalized().dot(dir0)) > 0.0:
                                foot_h = ti.max(lam, _TT_MIN_PATH) * theta_out
                                c_last = rc_h * disk._tt_cell[disk._tt_n - 1]
                                h = ti.min(h, sc * 0.5 * ti.max(c_last, disk._tt_k * foot_h))

                    # 首步抖动：起点沿光线随机偏移 [0, h)，与超采样一起构成蒙特卡洛体积积分
                    if step_idx == 0:
                        jit = _hashf(i, j, self.jitter_seed[None])
                        h = h * jit
                    step_idx += 1

                    # RK4 主光线。
                    k1p = h * dir_
                    k1d = h * _compute_acceleration(pos, L2_val)
                    k2p = h * (dir_ + 0.5 * k1d)
                    k2d = h * _compute_acceleration(pos + 0.5 * k1p, L2_val)
                    k3p = h * (dir_ + 0.5 * k2d)
                    k3d = h * _compute_acceleration(pos + 0.5 * k2p, L2_val)
                    k4p = h * (dir_ + k3d)
                    k4d = h * _compute_acceleration(pos + k3p, L2_val)
                    new_pos = pos + (k1p + 2 * k2p + 2 * k3p + k4p) / 6.0
                    new_dir = dir_ + (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0

                    r = new_pos.norm()
                    affine += h
                    hit_horizon = r < r_cap
                    hit_escape = r > r_esc or affine > max_affine

                    # 体积密度场（density_I）+ Y(g·T) 三温度源；段中点采样。
                    # 与参考实现一致：先积分本段，再判视界 / 逃逸（落入视界前的发射保留）。
                    ds = (new_pos - pos).norm()
                    lam += ds
                    pm = 0.5 * (pos + new_pos)
                    _sl = _world_to_local_disk(pm)
                    r_local = ti.sqrt(_sl[0] ** 2 + _sl[1] ** 2)
                    if r_local > disk._r_in and r_local < disk._r_out:
                        z_local = _sl[2]
                        z_lim = 3.0 * disk._ss_half_thickness(r_local) + 0.01
                        z_lim = ti.max(z_lim, disk._atm_ext * disk._atm_h * r_local)
                        if ti.abs(z_local) < z_lim:
                            phi_local = ti.atan2(_sl[1], _sl[0])
                            # 光线方向转到盘局部坐标（g 因子与发射点位置同一坐标系）
                            dm = _world_to_local_disk(0.5 * (dir_ + new_dir))
                            # 光行时间：采样时刻 = 帧时刻 − (到段中点的光程)
                            t_delay = 0.0
                            if ti.static(light_delay):
                                t_delay = lam - 0.5 * ds
                            # 温度湍流的像素足迹（λ 取段中点）与透镜权重（段中点方向相对出射方向的偏折角）
                            foot = 1.0
                            lens_w = 0.0
                            if ti.static(tt_on):
                                foot = ti.max(lam - 0.5 * ds, _TT_MIN_PATH) * theta_out
                                lens_w = disk._temp_turb_lens_weight((0.5 * (dir_ + new_dir)).normalized().dot(dir0))
                            em_c, tf_c, ab_c, ab_a, em_a, sc_a = disk.density_I(
                                r_local, z_local, phi_local, t_delay, foot, lens_w)
                            if em_c + em_a + ab_c + ab_a > 1e-9:
                                g_phys = disk_g_factor_ti(_sl, dm, cp.norm(), rs, g_spin)
                                g_col = ti.pow(g_phys, self.doppler_color)
                                # 颜色温度截断：χ(clamp(T·g_col, T_floor, T_cap))，暗金 → 白（_col_t 内实现）
                                # 亮度温度 = s·g^{doppler_lum_eff}·T（s 为亮度温度倍率；色度不乘 s）
                                g_lum = ti.pow(g_phys, self._doppler_lum_eff) * disk._lum_ts
                                T_K = disk._page_thorne_temperature(r_local)
                                # 核心：Y(s·g·T·tf_c)/Y(s·T_peak)·χ(g·T·tf_c)
                                Tc = T_K * tf_c
                                src_c = ti.exp(disk.blackbody_luminance_ti(Tc * g_lum) - disk._ln_y_peak) * disk.blackbody_color_ti(self._col_t(Tc * g_col))
                                # 盘面有效温度源：S_disk = Y(s·g·T)/Y(s·T_peak)·χ(g·T)（散射入射场用）
                                src_d = ti.exp(disk.blackbody_luminance_ti(T_K * g_lum) - disk._ln_y_peak) * disk.blackbody_color_ti(self._col_t(T_K * g_col))
                                # 发射 j = 核心热发射 + 大气热发射（同为灰大气温度 T·tf）+ 大气散射 ω·ab_a·J，
                                # J = scatter_j·(1 − e^{−τ_c})·S_disk（下方核心层，sc_a 已含 ω·ab_a·(1 − e^{−τ_c})）；
                                # 消光 α = κ·(ab_c + ab_a)，κ 已按 tau_i 标定
                                j_total = em_c * src_c + em_a * src_c + sc_a * disk._scatter_j * src_d
                                alpha_coeff = disk._kappa_vol * (ab_a + ab_c)
                                alpha_seg = alpha_coeff * ds
                                # 精确均匀段：ΔI = T · (j/α) · (1 − exp(−α·ds))；薄极限 → T · j · ds
                                if alpha_coeff > 1e-30:
                                    hdr_accum += transmittance * j_total / alpha_coeff * (
                                        1.0 - ti.exp(-alpha_seg))
                                else:
                                    hdr_accum += transmittance * j_total * ds
                                transmittance *= ti.exp(-alpha_seg)
                    # 与参考实现一致：透射率低于 1e-3 即终止，并丢弃其后的天空贡献
                    if transmittance < 1e-3:
                        transmittance = 0.0
                        break
                    if hit_horizon:
                        event_horizon_hit = True
                        transmittance = 0.0
                        break
                    elif hit_escape:
                        escaped = True
                        escape_dir = new_dir.normalized()
                        break

                    pos = new_pos
                    dir_ = new_dir
                    step_count += 1

                bg_color = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
                if escaped:
                    bg_color = _sample_skybox(escape_dir)
                self.event_horizon_field[i, j] = 1 if event_horizon_hit else 0
                self.hdr_field[i, j] = hdr_accum
                self.sky_field[i, j] = bg_color * transmittance

        self._ray_march_kernel = _ray_march_kernel

    def _setup_camera(self, cam_pos: List[float], fov: float, forward=None,
                      roll_deg: Optional[float] = None) -> None:
        """计算相机基向量并填到 Taichi field 中（与 V1 `build_camera` 一致）。

        Args:
            cam_pos: 相机位置 `[x, y, z]`（r_s），世界 up = +z。
            fov: 竖直视野角（度）。
            forward: 相机光轴方向（世界系 3 维向量）；None = 看向原点（与旧实现逐位一致）。
            roll_deg: 本帧相机滚转角（度）；None = 构造时的 `camera_roll_deg`。运镜路径逐帧给出。
        """
        roll = self.camera_roll_deg if roll_deg is None else float(roll_deg)
        cam_pos_arr, right, up, forward, pixel_width, pixel_height, _top_left = (
            build_camera_v1_compatible(cam_pos, fov, self._iw, self._ih, roll, forward=forward)
        )
        self.cam_pos_field[None] = cam_pos_arr.astype(np.float32).tolist()
        self.cam_right_field[None] = right.astype(np.float32).tolist()
        self.cam_up_field[None] = up.astype(np.float32).tolist()
        self.cam_forward_field[None] = forward.astype(np.float32).tolist()
        self.pixel_width_field[None] = float(pixel_width)
        self.pixel_height_field[None] = float(pixel_height)

        distance = float(np.linalg.norm(cam_pos_arr))
        r_out = float(self.disk_ti._r_out)
        self.r_escape_field[None] = max(self.r_max, distance * 2.0, r_out * 1.6)

    def _downsample(self, field) -> np.ndarray:
        """Taichi `(ss·W, ss·H, 3)` field → `(H, W, 3)` NumPy，`ss×ss` 盒式平均。

        Args:
            field: `hdr_field` 或 `sky_field`。

        Returns:
            `(H, W, 3)` float32 线性 HDR。
        """
        arr = np.transpose(field.to_numpy(), (1, 0, 2))
        if self.ss > 1:
            arr = arr.reshape(self.height, self.ss, self.width, self.ss, 3).mean(axis=(1, 3))
        return arr

    @ti.func
    def _col_t(self, t_col):
        """颜色温度截断：返回 `clamp(t_col, T_floor, T_cap)`；对应参数为 0 时该侧不截断。

        Args:
            t_col: 色度温度 `T·g^q`（K，标量）。

        Returns:
            标量色度温度（K），落在 `[T_floor, T_cap]` 内（未启用的一侧不限制）。

        Formula:
            T_col' = min(max(T_col, T_floor), T_cap)

        Physical Meaning:
            白平衡色温 T_cap 处的黑体显示为白色，更高温度偏蓝；低于 T_floor 的黑体显示为橙红。
            两侧截断后颜色只在黑体轨迹的 暗金 → 金 → 白 段内变化，暗处靠亮度变暗到黑。

        Simplifications:
            纯视觉处理（物理上高温偏蓝白、低温偏红）；亮度仍按真实温度计算。
        """
        out = t_col
        if ti.static(self.color_temp_floor_K > 0.0):
            out = ti.max(out, self.color_temp_floor_K)
        if ti.static(self.color_temp_cap_K > 0.0):
            out = ti.min(out, self.color_temp_cap_K)
        return out

    def render_hdr(self, cam_pos: List[float], fov: float, t: float = 2000.0, forward=None,
                   roll_deg: Optional[float] = None):
        """GPU 阶段：积分一帧，返回 `(disk_hdr, sky)`（均为 `(H, W, 3)` 线性光；无天空时 sky 为 None）。

        Args:
            cam_pos: 相机位置 `[x, y, z]`（r_s）。
            fov: 竖直视野角（度）。
            t: 帧物理时间（r_s/c）。
            forward: 相机光轴方向（世界系 3 维向量）；None = 看向原点。
            roll_deg: 本帧相机滚转角（度）；None = 构造时的 `camera_roll_deg`。

        Returns:
            `(hdr, sky)`；同时写入 `last_hdr`。
        """
        self._setup_camera(cam_pos, fov, forward=forward, roll_deg=roll_deg)
        self.disk_ti.update_advection(float(t))
        self.jitter_seed[None] += 1
        self._ray_march_kernel()
        hdr = self._downsample(self.hdr_field)
        self.last_hdr = hdr
        sky = self._downsample(self.sky_field) if self.sky_gain > 0.0 else None
        return hdr, sky

    def finish(self, hdr: np.ndarray, sky, exposure: Optional[float] = None) -> np.ndarray:
        """CPU 阶段：曝光、叠加天空、后处理（可与下一帧的 `render_hdr` 并行）。

        Args:
            hdr: `render_hdr` 返回的盘发射 HDR。
            sky: `render_hdr` 返回的天空（线性光）或 None。
            exposure: 本帧曝光系数（> 0，乘到盘 HDR 上）；None = 按 `fixed_exposure` 或自动曝光。
                运镜视频逐帧传入 `E₀·2^{ev(t)}`；作为参数传入（不改实例状态），可在线程池中安全并行。

        Returns:
            `(H, W, 3)` float32 LDR `[0, 1]`（sRGB 编码）。

        Notes:
            曝光优先级：参数 `exposure` → `fixed_exposure`（视频首帧锁定）→ 自动曝光
            `0.9 / p99.9(L) · 2^exposure_ev`，L 为盘区亮度（`L > 1e-4`，只看盘；
            EV = 0 即参考实现预设 M 的 p99.9 → 0.9）。合成 `x = exposure·disk + sky_gain·sky`，
            再经 `postfx`：白平衡 → 镜头 PSF → 色散 → ACES（胶片响应 `film_response`）→ sRGB。
        """
        if exposure is not None:
            exposure = float(exposure)
        elif self.fixed_exposure is not None:
            exposure = float(self.fixed_exposure)
        else:
            lum = hdr_luminance(hdr)
            lum = lum[lum > 1e-4]
            exposure = 0.9 / max(float(np.percentile(lum, 99.9)), 1e-12) if lum.size else 1.0
            exposure *= 2.0 ** self.exposure_ev
        # 天空在曝光之后以独立系数叠加，再一起进入 postfx（星点参与 bloom）
        x = hdr * exposure
        if sky is not None:
            x = x + self.sky_gain * sky
        fx = {"bloom_luma_threshold": self.bloom_luma_threshold, "lens_model": self.lens_model,
              "white_balance_K": self.white_balance_K, "film_response": self.film_response}
        if self.lens_glare is not None:
            fx["lens_glare"] = self.lens_glare
        img = postfx(x, exposure=1.0, **fx)
        self.last_white_point = 1.0 / exposure if exposure > 0 else 1.0
        return img.astype(np.float32) / 255.0

    def render(self, cam_pos: List[float], fov: float, t: float = 2000.0) -> np.ndarray:
        """渲染单帧：`finish(*render_hdr(...))`。

        Args:
            cam_pos: 相机位置 `[x, y, z]`，单位为 r_s。
            fov: 竖直视野角（度）。
            t: 帧物理时间（r_s/c），决定刚体环平流相位；默认 2000（与参考实现单帧一致）。

        Returns:
            `(height, width, 3)` float32 LDR `[0, 1]`（sRGB 编码）。
        """
        return self.finish(*self.render_hdr(cam_pos, fov, t))
