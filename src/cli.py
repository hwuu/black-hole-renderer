from concurrent.futures import ThreadPoolExecutor
import argparse
import math
import imageio.v3 as iio
import numpy as np
import taichi as ti
from src.core.constants import DISK_GENERATION_SCALE_CHOICES, R_DISK_INNER_DEFAULT, R_DISK_OUTER_DEFAULT
from src.core.imaging import save_image
from src.core.skybox import load_or_generate_skybox
from src.v1.pipeline import render_image, render_interactive, render_video
from src.v1.renderer import TaichiRenderer
from src.v1.texture import compute_disk_texture_resolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schwarzschild 黑洞光线追踪渲染器")
    parser.add_argument("--pov", type=float, nargs=3, default=[6, 0, 0.5],
                        metavar=("X", "Y", "Z"),
                        help="相机位置 (default: 6 0 0.5)")
    parser.add_argument("--fov", type=float, default=90,
                        help="视野角度 0-180° (default: 90)")
    parser.add_argument("--resolution", "-r", type=str, default="fhd",
                        choices=["4k", "fhd", "hd", "sd"],
                        help="分辨率: 4k/fhd/hd/sd (default: fhd)")
    parser.add_argument("--texture", "-t", type=str, default=None,
                        help="天空盒纹理路径")
    parser.add_argument("--output", "-o", type=str, default="output/blackhole.png",
                        help="输出路径 (default: output/blackhole.png)")
    parser.add_argument("--step_size", "-s", type=float, default=0.1,
                        help="积分步长 (default: 0.1)")
    parser.add_argument("--r_max", type=float, default=10,
                        help="逃逸半径 (default: 10)")
    parser.add_argument("--n_stars", type=int, default=6000,
                        help="程序天空盒恒星数 (default: 6000)")
    parser.add_argument("--disk_texture", type=str, default=None,
                        help="吸积盘纹理路径 (default: 程序生成，仅静态单帧模式支持)")
    parser.add_argument("--disk_generation_scale", type=int, default=2, choices=DISK_GENERATION_SCALE_CHOICES,
                        help="[已废弃] 生命周期系统不使用此参数 (default: 2)")
    parser.add_argument("--force_regenerate_disk_texture", action="store_true",
                        help="[已废弃] 生命周期系统每次实时生成 (default: 关闭)")
    parser.add_argument("--disk_inner_radius", "--ar1", dest="disk_inner_radius", type=float, default=R_DISK_INNER_DEFAULT,
                        help=f"吸积盘内半径 (default: {R_DISK_INNER_DEFAULT})")
    parser.add_argument("--disk_outer_radius", "--ar2", dest="disk_outer_radius", type=float, default=R_DISK_OUTER_DEFAULT,
                        help=f"吸积盘外半径 (default: {R_DISK_OUTER_DEFAULT})")
    parser.add_argument("--disk_tilt", type=float, default=0.0,
                        help="吸积盘倾角 (度, default: 0)")
    parser.add_argument("--lens_flare", action="store_true",
                        help="启用 lens flare 效果 (default: 关闭)")
    parser.add_argument("--anti_alias", type=str, default="disabled",
                        choices=["disabled", "lod_radius"],
                        help="抗锯齿算法: disabled(关闭), lod_radius(基于半径的启发式LOD) (default: disabled)")
    parser.add_argument("--aa_strength", type=float, default=1.0,
                        help="抗锯齿强度，乘以 LOD 值。>1 更模糊(更强抗锯齿)，<1 更清晰。范围 0.5-2.0 (default: 1.0)")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                        choices=["cpu", "gpu"],
                        help="Taichi 设备: cpu 或 gpu (default: cpu)")
    parser.add_argument("--ignore_taichi_cache", action="store_true",
                        help="忽略 Taichi 离线缓存，强制重新编译 kernel (default: 关闭)")
    parser.add_argument("--video", action="store_true",
                        help="视频模式：渲染多帧并合成视频")
    parser.add_argument("--interactive", action="store_true",
                        help="交互模式：实时预览，鼠标拖拽旋转，按键切换渲染开关")
    parser.add_argument("--orbit", action="store_true",
                        help="视频模式：相机围绕原点旋转（需配合 --video）")
    parser.add_argument("--orbit_degrees", type=float, default=360.0,
                        help="轨道模式下整段视频的总旋转角度，支持负数反向旋转 (default: 360.0)")
    parser.add_argument("--n_frames", type=int, default=3600,
                        help="视频帧数 (default: 3600, 仅 --video 有效)")
    parser.add_argument("--fps", type=int, default=36,
                        help="视频帧率 (default: 36, 仅 --video 有效)")
    parser.add_argument("--resume", action="store_true",
                        help="视频模式：尝试从断点恢复（默认从头开始）")
    parser.add_argument("--disk_rotation_algorithm", type=str, default="baseline",
                        choices=["baseline", "parametric", "keyframes"],
                        help="[已废弃] 统一使用生命周期系统，此参数被忽略")
    parser.add_argument("--disk_rotation_speed", type=float, default=0.1,
                        help="吸积盘旋转速度系数 (default: 0.1)")
    parser.add_argument("--keyframes_count", type=int, default=10,
                        help="[已废弃] 统一使用生命周期系统，此参数被忽略")
    # --- Disk V2 路径开关与参数（Phase 4 接入） ---
    parser.add_argument("--disk_model", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="吸积盘模型: v1（默认，零厚度倾斜平面 + 程序纹理）或 v2（有限厚度发射-吸收积分）")
    parser.add_argument("--v2_opt", type=int, default=None, choices=[0, 1, 2, 3],
                        help="V2 优化级别：0 参考实现；1 精确优化（输出与 0 一致）；2 盘内步长 ×2；"
                             "3 盘内步长 ×3（近似预览）。默认：单帧 1，视频 2")
    parser.add_argument("--v2_supersample", type=int, default=None,
                        help="V2 超采样倍率 N：每像素 N² 条光线取平均，用于抗锯齿。默认：单帧 2，视频 1")
    parser.add_argument("--v2_sky_rot_deg_per_sec", type=float, default=0.0,
                        help="V2 视频模式：天空方位自转速度（度/视频秒），正值使星空向画面右方漂移；0 关闭 (default: 0)")
    parser.add_argument("--v2_disk_roll", type=float, default=0.0,
                        help="V2 盘滚转角（度），绕世界 y 轴；相机在 -y 方向时正值使盘面画面上左低右高 (default: 0)")
    parser.add_argument("--v2_thickness_scale", type=float, default=None,
                        help="V2 盘厚缩放：盘与烟雾层厚度同乘该值，柱密度不变；1 = 预设 M 视觉厚度"
                             "（与参考实现对齐），默认 1/9（H/r ≈ 0.003，接近物理量级）")
    parser.add_argument("--v2_lum_temp_scale", type=float, default=None,
                        help="V2 亮度温度倍率 s：亮度按 Y(s·T)/Y(s·T_peak) 计算，色度不变；1 = 物理"
                             "（与参考实现对齐），默认 1.25（外盘更亮，贴盘面视角外盘不全黑）")
    parser.add_argument("--v2_az_stretch", type=float, default=None,
                        help="V2 主云方位拉长系数 s（≥ 0）：吸积盘絮状结构沿旋转方向的拉长程度。"
                             "s > 0 时方位周期随半径增长，使各半径的结构长宽比保持一致：1 = 长宽比约 4–5"
                             "（各半径一致，开普勒剪切下的物理形状）；>1 = 更拉长、流动感更强"
                             "（长宽比约与 s 成正比，1.5 时约 6–9）；0 = 方位周期固定（旧版，外圈结构被拉成长条、细节少）。"
                             "默认 1.5")
    parser.add_argument("--v2_core_contrast", type=float, default=None,
                        help="V2 主云明暗起伏系数 α（> 0）：缩放絮状结构的明暗对比，不改变结构形状。"
                             "1 = 原始起伏（颗粒感强）；越小越柔和，过小时外圈结构会被烟雾层的条纹盖过。"
                             "默认 0.4")
    parser.add_argument("--v2_doppler_lum", type=float, default=None,
                        help="V2 多普勒亮度强度 p（≥ 0）：逼近侧变亮、远离侧变暗的程度，亮度按 Y(g^p·T) 计算"
                             "（g 为频移因子）。1 = 物理（左右亮度比很大）；0 = 无多普勒明暗；"
                             "默认 0.5（预设 M 定稿为 0.55，本版略减弱）")
    parser.add_argument("--v2_reverse_rotation", action="store_true",
                        help="V2 反转吸积盘旋转方向（平流结构与多普勒频移整体反向）")
    parser.add_argument("--v2_sky_gain", type=float, default=0.5,
                        help="V2 天空亮度系数（线性光，曝光之后叠加，不影响盘曝光；0 = 黑天空）(default: 0.5)")
    parser.add_argument("--v2_orbit_seconds", type=float, default=16.0,
                        help="V2 视频模式：内缘开普勒轨道对应视频秒数 (default: 16.0)")
    return parser.parse_args()


def resolve_v2_quality(args, video: bool):
    """解析 V2 的优化级别与超采样倍率（未显式传入时按渲染模式取默认值）。

    Args:
        args: CLI 参数（读取 `--v2_opt`、`--v2_supersample`）。
        video: 是否为视频模式。

    Returns:
        `(opt_level, supersample)`：单帧默认 `(1, 2)`，视频默认 `(2, 1)`。

    Raises:
        ValueError: `--v2_supersample < 1`。
    """
    opt = args.v2_opt if args.v2_opt is not None else (2 if video else 1)
    ss = args.v2_supersample if args.v2_supersample is not None else (1 if video else 2)
    if ss < 1:
        raise ValueError(f"--v2_supersample must be >= 1, got {ss}")
    return opt, ss


def v2_volume_overrides(args) -> dict:
    """收集 CLI 显式传入的 V2 体积参数覆盖项（未传入的保持 `DiskV2VolumeParams` 默认值）。

    Args:
        args: CLI 参数（读取 `--v2_thickness_scale`、`--v2_lum_temp_scale`、`--v2_az_stretch`、
            `--v2_core_contrast`，None = 未传入）。

    Returns:
        `DiskV2VolumeParams` 关键字参数字典，只含显式传入的字段；全部未传入时为空字典。
    """
    fields = {"thickness_scale": args.v2_thickness_scale, "lum_temp_scale": args.v2_lum_temp_scale,
              "core_az_stretch": args.v2_az_stretch, "core_contrast": args.v2_core_contrast}
    return {k: v for k, v in fields.items() if v is not None}


def validate_args(args) -> None:
    """Validate CLI arguments."""
    # FOV range check
    if not (0 < args.fov < 180):
        raise ValueError(f"FOV must be between 0 and 180 degrees, got {args.fov}")

    # Disk radius check
    if args.disk_inner_radius >= args.disk_outer_radius:
        raise ValueError(f"disk_inner_radius ({args.disk_inner_radius}) must be less than "
                        f"disk_outer_radius ({args.disk_outer_radius})")

    # Step size check
    if args.step_size <= 0:
        raise ValueError(f"step_size must be positive, got {args.step_size}")

    # AA strength range
    if not (0.5 <= args.aa_strength <= 2.0):
        raise ValueError(f"aa_strength must be between 0.5 and 2.0, got {args.aa_strength}")

    # Video parameters
    if args.n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {args.n_frames}")

    if args.fps <= 0:
        raise ValueError(f"fps must be positive, got {args.fps}")

    if not math.isfinite(args.orbit_degrees):
        raise ValueError(f"orbit_degrees must be finite, got {args.orbit_degrees}")

    if args.disk_texture and (args.video or args.interactive):
        raise ValueError("--disk_texture 仅支持静态单帧渲染，video/interactive 模式请使用生命周期系统")


def main():


    args = parse_args()
    validate_args(args)

    resolutions = {"4k": (3840, 2160), "fhd": (1920, 1080), "hd": (1280, 720), "sd": (640, 360)}
    width, height = resolutions[args.resolution]
    fov = args.fov % 180

    def _make_renderer_with_placeholder(device="cpu"):
        """Create renderer with placeholder disk texture for lifecycle system."""
        skybox, _, _ = load_or_generate_skybox(args.texture, 2048, 1024, args.n_stars)
        n_phi, n_r = compute_disk_texture_resolution(
            width, height, args.pov, fov,
            args.disk_inner_radius, args.disk_outer_radius)
        disk_tex = np.zeros((n_r, n_phi, 4), dtype=np.float32)
        return TaichiRenderer(
            width, height, skybox, disk_tex,
            step_size=args.step_size, r_max=args.r_max, device=device,
            r_disk_inner=args.disk_inner_radius, r_disk_outer=args.disk_outer_radius,
            disk_tilt=args.disk_tilt,
            lens_flare=args.lens_flare if not args.interactive else False,
            anti_alias=args.anti_alias if not args.interactive else "disabled",
            aa_strength=args.aa_strength,
            disk_rotation_speed=args.disk_rotation_speed,
            ignore_taichi_cache=args.ignore_taichi_cache
        )

    def _make_v2_renderer(args, width, height, video=False):
        """构造 V2 体积渲染器（单帧与视频共用）。

        Args:
            args: CLI 参数（读取 `--ar1/--ar2/--disk_tilt/--r_max/--texture/--n_stars/--v2_opt/
                --v2_supersample/--v2_sky_gain/--v2_doppler_lum/--device`；`--v2_doppler_lum`
                未传入时用渲染器默认值）。
            width, height: 输出分辨率（像素）。
            video: 是否为视频模式；决定 `--v2_opt`（单帧 1、视频 2）与 `--v2_supersample`
                （单帧 2、视频 1）未显式传入时的默认值。

        Returns:
            `DiskV2Renderer`，体积模型参数为预设 M（`DiskV2VolumeParams()` 默认值）。

        Notes:
            逃逸半径下限取 `max(--r_max, 50)`；渲染器内部再与 `2·相机距离`、`1.6·r_out` 取大。
        """
        from src.v2.params import DiskV2Params, DiskV2VolumeParams
        from src.v2.taichi_render import DiskV2Renderer

        opt, ss = resolve_v2_quality(args, video)
        print(f"[V2] 优化级别 {opt}，超采样倍率 {ss}（每像素 {ss * ss} 条光线）")
        ti.init(arch=ti.gpu if args.device == "gpu" else ti.cpu, default_fp=ti.f32)
        skybox, _, _ = load_or_generate_skybox(args.texture, 2048, 1024, args.n_stars)
        return DiskV2Renderer(
            width=width, height=height,
            params=DiskV2Params(r_in=args.disk_inner_radius, r_out=args.disk_outer_radius,
                                 disk_spin=-1.0 if args.v2_reverse_rotation else 1.0),
            skybox=skybox,
            volume_params=DiskV2VolumeParams(**v2_volume_overrides(args)),
            r_max=max(args.r_max, 50.0),
            disk_tilt_deg=args.disk_tilt,
            disk_roll_deg=args.v2_disk_roll,
            sky_gain=args.v2_sky_gain,
            ss=ss,
            **({} if args.v2_doppler_lum is None else {"doppler_lum": args.v2_doppler_lum}),
            opt_level=opt,
            device=args.device,
        )

    def _render_video_v2(args, width, height, fov):
        """V2 视频渲染：逐帧推进物理时间，相机可环绕，曝光首帧锁定。

        Args:
            args: CLI 参数（另读取 `--video/--orbit/--orbit_degrees/--n_frames/--fps/--v2_orbit_seconds`）。
            width, height: 输出分辨率（像素）。
            fov: 竖直视野角（度）。

        Formula:
            `dt = P_in / (v2_orbit_seconds · fps)`，`P_in = 2π / Ω(r_in)`，`Ω = sqrt(0.5 / r³)`：
            内缘转一圈对应 `v2_orbit_seconds` 秒视频。
        """
        import math as _math
        import time as _time

        renderer = _make_v2_renderer(args, width, height, video=True)

        # 物理时间步：内缘轨道周期 = v2_orbit_seconds 视频秒（r_in 取 ISCO 钳制后的值）
        r_in_m = renderer.params.r_in
        period_in = 2 * _math.pi / _math.sqrt(0.5 / r_in_m ** 3)
        dt_per_frame = period_in / (args.v2_orbit_seconds * args.fps)

        # 相机轨道
        orbit_deg = args.orbit_degrees if args.orbit else 0.0
        base_azim = _math.atan2(args.pov[1], args.pov[0])
        base_dist = _math.sqrt(args.pov[0]**2 + args.pov[1]**2 + args.pov[2]**2)
        base_elev = _math.asin(args.pov[2] / max(base_dist, 1e-9))

        print(f"[V2 video] {args.n_frames} 帧 @ {args.fps} fps，"
              f"内缘周期 {args.v2_orbit_seconds}s，dt={dt_per_frame:.4f}")

        t0 = 2000.0  # 物理起始时间（参考实现同款）
        writer = iio.imopen(args.output, "w", plugin="pyav")
        writer.init_video_stream("libx264", fps=args.fps)
        start = _time.time()
        pool = ThreadPoolExecutor(max_workers=1)
        pending = None

        def _write(frame):
            writer.write_frame((np.clip(frame, 0, 1) * 255).astype(np.uint8))

        for f in range(args.n_frames):
            t = t0 + f * dt_per_frame
            azim = base_azim + _math.radians(orbit_deg) * f / max(args.n_frames - 1, 1)
            cam_x = base_dist * _math.cos(base_elev) * _math.cos(azim)
            cam_y = base_dist * _math.cos(base_elev) * _math.sin(azim)
            cam_z = base_dist * _math.sin(base_elev)
            hdr, sky = renderer.render_hdr(cam_pos=[cam_x, cam_y, cam_z], fov=fov, t=t)
            if renderer.fixed_exposure is None:
                # 首帧同步处理并锁定曝光，避免逐帧曝光闪烁（参考实现 cmd_video 同款）
                _write(renderer.finish(hdr, sky))
                renderer.fixed_exposure = 1.0 / renderer.last_white_point
            else:
                # 后处理与写帧交给单线程池（保持帧序），与下一帧的 GPU 积分并行
                if pending is not None:
                    pending.result()
                pending = pool.submit(lambda h=hdr, s=sky: _write(renderer.finish(h, s)))
            if f % 24 == 0:
                print(f"  frame {f}/{args.n_frames}  {_time.time() - start:.1f}s", flush=True)
        if pending is not None:
            pending.result()
        pool.shutdown()
        writer.close()
        print(f"[V2 video] 完成: {args.output} ({_time.time() - start:.1f}s)")

    if args.disk_model == "v2":
        # V2 路径：独立 DiskV2Renderer（体积模型），不复用 V1 的 TaichiRenderer。
        if args.interactive:
            raise NotImplementedError("--disk_model v2 交互模式尚未接入。")
        if args.device != "gpu":
            raise ValueError(
                "--disk_model v2 当前仅支持 --device gpu（CPU 路径单帧也要数分钟）；"
                "请加上 '--device gpu' 后重试。"
            )
        if args.video:
            _render_video_v2(args, width, height, fov)
        else:
            renderer = _make_v2_renderer(args, width, height, video=False)
            img = renderer.render(cam_pos=args.pov, fov=fov)
            save_image(img, args.output)
    elif args.interactive:
        renderer = _make_renderer_with_placeholder(device="gpu")
        render_interactive(
            renderer, width, height,
            fov=fov, initial_cam_pos=args.pov,
            disk_rotation_speed=args.disk_rotation_speed,
        )
    elif args.video:
        renderer = _make_renderer_with_placeholder(device=args.device)

        print(f"Rendering video: {args.n_frames} frames at {width}x{height}")
        print(f"  orbit={args.orbit}")
        if args.orbit:
            print(f"  orbit_degrees={args.orbit_degrees}°")
        print(f"  fov={fov}°, step_size={args.step_size}, fps={args.fps}, disk_tilt={args.disk_tilt}°")
        print(f"  disk_rotation_speed={args.disk_rotation_speed}")

        render_video(
            renderer, width, height,
            n_frames=args.n_frames, fps=args.fps, output_path=args.output,
            fov=fov, static_cam_pos=args.pov,
            orbit=args.orbit,
            resume=args.resume,
            disk_rotation_speed=args.disk_rotation_speed,
            orbit_degrees=args.orbit_degrees,
        )
    else:
        img = render_image(
            width=width,
            height=height,
            cam_pos=args.pov,
            fov=fov,
            step_size=args.step_size,
            skybox_path=args.texture,
            n_stars=args.n_stars,
            r_max=args.r_max,
            device=args.device,
            disk_texture_path=args.disk_texture,
            r_disk_inner=args.disk_inner_radius,
            r_disk_outer=args.disk_outer_radius,
            disk_tilt=args.disk_tilt,
            lens_flare=args.lens_flare,
            anti_alias=args.anti_alias,
            aa_strength=args.aa_strength,
            disk_generation_scale=args.disk_generation_scale,
            force_regenerate_disk_texture=args.force_regenerate_disk_texture,
            ignore_taichi_cache=args.ignore_taichi_cache,
        )
        save_image(img, args.output)
