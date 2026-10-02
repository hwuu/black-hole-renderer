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
    parser.add_argument("--v2_ss", type=int, default=1,
                        help="V2 体积模型超采样倍率（每轴），2 = 每像素 4 条光线 (default: 1)")
    parser.add_argument("--v2_sky_gain", type=float, default=0.5,
                        help="V2 天空亮度系数（线性光，曝光之后叠加，不影响盘曝光；0 = 黑天空）(default: 0.5)")
    parser.add_argument("--v2_orbit_seconds", type=float, default=16.0,
                        help="V2 视频模式：内缘开普勒轨道对应视频秒数 (default: 16.0)")
    return parser.parse_args()


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

    def _make_v2_renderer(args, width, height):
        """构造 V2 体积渲染器（单帧与视频共用）。

        Args:
            args: CLI 参数（读取 `--ar1/--ar2/--disk_tilt/--r_max/--texture/--n_stars/--v2_ss/--v2_sky_gain/--device`）。
            width, height: 输出分辨率（像素）。

        Returns:
            `DiskV2Renderer`，体积模型参数为预设 M（`DiskV2VolumeParams()` 默认值）。

        Notes:
            逃逸半径下限取 `max(--r_max, 50)`；渲染器内部再与 `2·相机距离`、`1.6·r_out` 取大。
        """
        from src.v2.params import DiskV2Params, DiskV2VolumeParams
        from src.v2.taichi_render import DiskV2Renderer

        ti.init(arch=ti.gpu if args.device == "gpu" else ti.cpu, default_fp=ti.f32)
        skybox, _, _ = load_or_generate_skybox(args.texture, 2048, 1024, args.n_stars)
        return DiskV2Renderer(
            width=width, height=height,
            params=DiskV2Params(r_in=args.disk_inner_radius, r_out=args.disk_outer_radius),
            skybox=skybox,
            volume_params=DiskV2VolumeParams(),
            r_max=max(args.r_max, 50.0),
            disk_tilt_deg=args.disk_tilt,
            sky_gain=args.v2_sky_gain,
            ss=args.v2_ss,
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

        renderer = _make_v2_renderer(args, width, height)

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
        for f in range(args.n_frames):
            t = t0 + f * dt_per_frame
            azim = base_azim + _math.radians(orbit_deg) * f / max(args.n_frames - 1, 1)
            cam_x = base_dist * _math.cos(base_elev) * _math.cos(azim)
            cam_y = base_dist * _math.cos(base_elev) * _math.sin(azim)
            cam_z = base_dist * _math.sin(base_elev)
            frame = renderer.render(cam_pos=[cam_x, cam_y, cam_z], fov=fov, t=t)
            if renderer.fixed_exposure is None:
                # 首帧自动曝光后锁定，避免逐帧曝光闪烁（参考实现 cmd_video 同款）
                renderer.fixed_exposure = 1.0 / renderer.last_white_point
            writer.write_frame((np.clip(frame, 0, 1) * 255).astype(np.uint8))
            if f % 24 == 0:
                print(f"  frame {f}/{args.n_frames}  {_time.time() - start:.1f}s")
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
            renderer = _make_v2_renderer(args, width, height)
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
