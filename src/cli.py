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
    # 参数按成像模型分组（只影响 --help 显示，参数名与默认值不变；原理见 docs/imaging_model.md）
    g_mode = parser.add_argument_group("通用：模式与设备")
    g_struct = parser.add_argument_group("① 场景·结构：盘长什么样、怎么动（见 docs/imaging_model.md §2.2）")
    g_rad = parser.add_argument_group("② 场景·辐射：盘发什么光（温度、频移、颜色；§2.3）")
    g_sky = parser.add_argument_group("③ 场景·背景：星空（§2.4）")
    g_cam = parser.add_argument_group("④ 相机：位置、视野、姿态、运动（§3）")
    g_lens = parser.add_argument_group("⑤ 镜头：眩光与色差（§4）")
    g_isp = parser.add_argument_group("⑥ 传感器与 ISP：曝光、白平衡、色调映射（§5、§6）")
    g_solver = parser.add_argument_group("求解精度（非成像参数）")
    g_out = parser.add_argument_group("输出与视频（非成像参数）")
    g_v1 = parser.add_argument_group("V1 专用：程序纹理与旋转算法")
    g_cam.add_argument("--pov", type=float, nargs=3, default=[6, 0, 0.5],
                        metavar=("X", "Y", "Z"),
                        help="相机位置 (default: 6 0 0.5)")
    g_cam.add_argument("--fov", type=float, default=90,
                        help="视野角度 0-180° (default: 90)")
    g_cam.add_argument("--resolution", "-r", type=str, default="fhd",
                        choices=["4k", "fhd", "hd", "sd"],
                        help="分辨率: 4k/fhd/hd/sd (default: fhd)")
    g_sky.add_argument("--texture", "-t", type=str, default=None,
                        help="天空盒纹理路径")
    g_out.add_argument("--output", "-o", type=str, default="output/blackhole.png",
                        help="输出路径 (default: output/blackhole.png)")
    g_solver.add_argument("--step_size", "-s", type=float, default=0.1,
                        help="积分步长 (default: 0.1)")
    g_solver.add_argument("--r_max", type=float, default=10,
                        help="逃逸半径 (default: 10)")
    g_sky.add_argument("--n_stars", type=int, default=6000,
                        help="程序天空盒恒星数 (default: 6000)")
    g_v1.add_argument("--disk_texture", type=str, default=None,
                        help="吸积盘纹理路径 (default: 程序生成，仅静态单帧模式支持)")
    g_v1.add_argument("--disk_generation_scale", type=int, default=2, choices=DISK_GENERATION_SCALE_CHOICES,
                        help="[已废弃] 生命周期系统不使用此参数 (default: 2)")
    g_v1.add_argument("--force_regenerate_disk_texture", action="store_true",
                        help="[已废弃] 生命周期系统每次实时生成 (default: 关闭)")
    g_struct.add_argument("--disk_inner_radius", "--ar1", dest="disk_inner_radius", type=float, default=R_DISK_INNER_DEFAULT,
                        help=f"吸积盘内半径 (default: {R_DISK_INNER_DEFAULT})")
    g_struct.add_argument("--disk_outer_radius", "--ar2", dest="disk_outer_radius", type=float, default=R_DISK_OUTER_DEFAULT,
                        help=f"吸积盘外半径 (default: {R_DISK_OUTER_DEFAULT})")
    g_struct.add_argument("--disk_tilt", type=float, default=0.0,
                        help="吸积盘倾角 (度, default: 0)")
    g_lens.add_argument("--lens_flare", action="store_true",
                        help="启用 lens flare 效果 (default: 关闭)")
    g_solver.add_argument("--anti_alias", type=str, default="disabled",
                        choices=["disabled", "lod_radius"],
                        help="抗锯齿算法: disabled(关闭), lod_radius(基于半径的启发式LOD) (default: disabled)")
    g_solver.add_argument("--aa_strength", type=float, default=1.0,
                        help="抗锯齿强度，乘以 LOD 值。>1 更模糊(更强抗锯齿)，<1 更清晰。范围 0.5-2.0 (default: 1.0)")
    g_mode.add_argument("--device", "-d", type=str, default="cpu",
                        choices=["cpu", "gpu"],
                        help="Taichi 设备: cpu 或 gpu (default: cpu)")
    g_mode.add_argument("--ignore_taichi_cache", action="store_true",
                        help="忽略 Taichi 离线缓存，强制重新编译 kernel (default: 关闭)")
    g_out.add_argument("--video", action="store_true",
                        help="视频模式：渲染多帧并合成视频")
    g_mode.add_argument("--interactive", action="store_true",
                        help="交互模式：实时预览，鼠标拖拽旋转，按键切换渲染开关")
    g_cam.add_argument("--orbit", action="store_true",
                        help="视频模式：相机围绕原点旋转（需配合 --video）")
    g_cam.add_argument("--orbit_degrees", type=float, default=360.0,
                        help="轨道模式下整段视频的总旋转角度，支持负数反向旋转 (default: 360.0)")
    g_out.add_argument("--n_frames", type=int, default=3600,
                        help="视频帧数 (default: 3600, 仅 --video 有效)")
    g_out.add_argument("--fps", type=int, default=36,
                        help="视频帧率 (default: 36, 仅 --video 有效)")
    g_out.add_argument("--resume", action="store_true",
                        help="视频模式：从断点恢复（默认从头开始）。V1 逐帧存 PNG；V2 每 240 帧存一个 mp4 分段，"
                             "中断后用同样参数加 --resume 重跑，最多重渲一段，参数变化时自动从头开始")
    g_v1.add_argument("--disk_rotation_algorithm", type=str, default="baseline",
                        choices=["baseline", "parametric", "keyframes"],
                        help="[已废弃] 统一使用生命周期系统，此参数被忽略")
    g_v1.add_argument("--disk_rotation_speed", type=float, default=0.1,
                        help="吸积盘旋转速度系数 (default: 0.1)")
    g_v1.add_argument("--keyframes_count", type=int, default=10,
                        help="[已废弃] 统一使用生命周期系统，此参数被忽略")
    # --- Disk V2 路径开关与参数（Phase 4 接入） ---
    g_mode.add_argument("--disk_model", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="吸积盘模型: v1（默认，零厚度倾斜平面 + 程序纹理）或 v2（有限厚度发射-吸收积分）")
    g_solver.add_argument("--v2_opt", type=int, default=None, choices=[0, 1, 2, 3],
                        help="V2 优化级别：0 参考实现；1 精确优化（输出与 0 一致）；2 盘内步长 ×2；"
                             "3 盘内步长 ×3（近似预览）。默认：单帧 1，视频 2")
    g_solver.add_argument("--v2_supersample", type=int, default=None,
                        help="V2 超采样倍率 N：每像素 N² 条光线取平均，用于抗锯齿。默认：单帧 2，视频 1")
    g_sky.add_argument("--v2_sky_rot_deg_per_sec", type=float, default=0.0,
                        help="V2 视频模式：天空方位自转速度（度/视频秒），正值使星空向画面右方漂移；0 关闭 (default: 0)")
    g_struct.add_argument("--v2_disk_roll", type=float, default=0.0,
                        help="V2 盘滚转角（度），绕世界 y 轴；相机在 -y 方向时正值使盘面画面上左低右高 (default: 0)")
    g_struct.add_argument("--v2_thickness_scale", type=float, default=None,
                        help="V2 盘厚缩放：盘面（高斯核心）标高乘该值，柱密度不变，不影响大气标高；"
                             "1 = 预设 M 视觉厚度（与参考实现对齐），默认 1/9（H/r ≈ 0.003，接近物理量级）")
    g_rad.add_argument("--v2_lum_temp_scale", type=float, default=None,
                        help="V2 亮度温度倍率 s：亮度按 Y(s·T)/Y(s·T_peak) 计算，色度不变；1 = 物理"
                             "（与参考实现对齐），默认 1.25（外盘更亮，贴盘面视角外盘不全黑）")
    g_struct.add_argument("--v2_core_contrast", type=float, default=None,
                        help="V2 主云明暗起伏系数 α（> 0）：缩放絮状结构的明暗对比，不改变结构形状。"
                             "1 = 原始起伏；越小越柔和。默认 0.8")
    g_struct.add_argument("--v2_atm_frac", type=float, default=None,
                        help="V2 大气柱密度比 A（≥ 0）：盘面上方大气（指数尾巴）的柱密度 / 盘面核心柱密度。"
                             "大气与盘面是同一团气体、跟随同一湍流结构，稀处出现空隙；越大大气越浓、盘面越朦胧，"
                             "贴盘面或盘内视角的金色云雾越明显。0 = 无大气（只剩薄盘面）。默认 0.15"
                             "（r = 22 处大气竖直光学深度约 0.06，约一半面积是空隙）")
    g_struct.add_argument("--v2_atm_height", type=float, default=None,
                        help="V2 大气标高 H_a/r（> 0）：大气密度随高度按 exp(−|z|/H_a) 衰减，越大大气越蓬松、"
                             "伸得越高（盘的视觉厚度随之增加）。不随 --v2_thickness_scale 缩放。"
                             "默认 0.01（约为盘面核心标高的 3 倍）")
    g_struct.add_argument("--v2_atm_fine", type=float, default=None,
                        help="V2 大气小尺度起伏强度 σ_a（≥ 0）：大气密度乘保均值的对数正态起伏 exp(σ_a·n − σ_a²/2)，"
                             "使大气随高度与下方盘面逐渐脱离、形成独立的小云团。0 = 大气完全跟随盘面结构；"
                             "越大云团越碎、反差越强（≥ 0.8 时相机易落入整团浓雾）。默认 0.5")
    g_rad.add_argument("--v2_doppler_lum", type=float, default=None,
                        help="V2 多普勒亮度强度 p（≥ 0）：逼近侧变亮、远离侧变暗的程度，亮度按 Y(g^p·T) 计算"
                             "（g 为频移因子）。1 = 物理（左右亮度比很大）；0 = 无多普勒明暗；"
                             "默认 0.25（预设 M 定稿为 0.55；本版减弱，视频构图下左右通量比约 3.7 → 1.7）")
    g_rad.add_argument("--v2_doppler_color", type=float, default=None,
                        help="V2 多普勒颜色强度 q（≥ 0）：逼近侧偏白、远离侧偏红的程度，色度按 χ(T·g^q) 计算，"
                             "且色度温度封顶到白平衡色温（最亮处止于白色，不偏蓝）。1 = 物理；0 = 无多普勒变色；"
                             "默认 0.75（预设 M 定稿为 1.5；应与 --v2_doppler_lum 同步调，否则远离侧会又亮又红）")
    g_lens.add_argument("--v2_lens_glare", type=float, default=None,
                        help="V2 镜头眩光强度 ε（0–1）：镜头把每个点约 ε 的能量散射成平滑长尾（辉光），"
                             "能量守恒、作用于全部光、无阈值；辉光只在暗处（天空、黑洞阴影）显著，盘面不会被点亮。"
                             "0 = 理想镜头（无辉光）；好镜头约 0.02，柔光镜约 0.2–0.5；越大辉光越明显，"
                             "全画面对比度按 (1 − ε) 下降。原理见 docs/imaging_model.md (default: 0.5)")
    g_rad.add_argument("--v2_color_floor", type=float, default=0.0,
                        help="V2 颜色温度下限 T_floor（K，≥ 0）：色度温度低于它时取 T_floor（硬截断），冷区不再显示"
                             "为橙红，颜色序列变为 黑 → 暗金 → 金 → 白（暗处只靠亮度变暗）。只影响颜色，不影响亮度。"
                             "建议 2500–3000；0 = 不设下限 (default: 0)")
    g_isp.add_argument("--v2_white_balance", type=float, default=None,
                       help="V2 相机白平衡色温（K，1000–40000）：色温为该值的黑体显示为白色；盘面主体约 3000–5000 K。"
                            "调低 → 盘面更白更冷（4000 K 偏惨白），调高 → 更暖更黄（参考实现 5000 K 偏暖黄）。"
                            "色温封顶自动跟随，最亮处始终止于白色 (default: 4500)")
    g_isp.add_argument("--v2_exposure_ev", type=float, default=None,
                       help="V2 曝光补偿（档）：自动曝光（盘区亮度 p99.9 → 0.9，视频首帧锁定）之后再乘 2^EV。"
                            "0 = 画面无过曝，最亮处只到浅金、缺少发光感；+1.5 = 内盘约 2%% 的像素烧白、"
                            "镜头辉光随之增强（类《星际穿越》）；负值更暗 (default: 1.5)")
    g_cam.add_argument("--v2_camera_roll", type=float, default=0.0,
                        help="V2 相机滚转角（度）：相机绕自身光轴旋转，整个画面（吸积盘与星空）一起倾斜；"
                             "在画面系中生效，环绕过程中倾角恒定；+12.5 = 画面左低右高，负值反向。"
                             "与 --v2_disk_roll（盘面绕世界 y 轴、星空不随动、环绕时倾角漂移）不同 (default: 0)")
    g_struct.add_argument("--v2_reverse_rotation", action="store_true",
                        help="V2 反转吸积盘旋转方向（平流结构与多普勒频移整体反向）")
    g_sky.add_argument("--v2_sky_gain", type=float, default=0.5,
                        help="V2 天空亮度系数（线性光，曝光之后叠加，不影响盘曝光；0 = 黑天空）(default: 0.5)")
    g_struct.add_argument("--v2_orbit_seconds", type=float, default=16.0,
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
        args: CLI 参数（读取 `--v2_thickness_scale`、`--v2_lum_temp_scale`、`--v2_core_contrast`、
            `--v2_atm_frac`、`--v2_atm_height`、`--v2_atm_fine`，None = 未传入）。

    Returns:
        `DiskV2VolumeParams` 关键字参数字典，只含显式传入的字段；全部未传入时为空字典。
    """
    fields = {"thickness_scale": args.v2_thickness_scale, "lum_temp_scale": args.v2_lum_temp_scale,
              "core_contrast": args.v2_core_contrast, "atm_frac": args.v2_atm_frac,
              "atm_height": args.v2_atm_height, "atm_fine_sigma": args.v2_atm_fine}
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
                --v2_supersample/--v2_sky_gain/--v2_doppler_lum/--v2_doppler_color/--v2_color_floor/--v2_camera_roll/--v2_lens_glare/--v2_white_balance/--v2_exposure_ev/--device`；
                两个多普勒参数未传入时用渲染器默认值）。
            width, height: 输出分辨率（像素）。
            video: 是否为视频模式；决定 `--v2_opt`（单帧 1、视频 2）与 `--v2_supersample`
                （单帧 2、视频 1）未显式传入时的默认值。

        Returns:
            `DiskV2Renderer`，体积模型参数为 `DiskV2VolumeParams()` 默认值叠加 CLI 显式覆盖项（`v2_volume_overrides`）。

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
            camera_roll_deg=args.v2_camera_roll,
            sky_gain=args.v2_sky_gain,
            ss=ss,
            **({} if args.v2_doppler_lum is None else {"doppler_lum": args.v2_doppler_lum}),
            **({} if args.v2_doppler_color is None else {"doppler_color": args.v2_doppler_color}),
            color_temp_floor_K=args.v2_color_floor,
            **({} if args.v2_lens_glare is None else {"lens_glare": args.v2_lens_glare}),
            **({} if args.v2_white_balance is None else {"white_balance_K": args.v2_white_balance}),
            **({} if args.v2_exposure_ev is None else {"exposure_ev": args.v2_exposure_ev}),
            opt_level=opt,
            device=args.device,
        )

    def _render_video_v2(args, width, height, fov):
        """V2 视频渲染：逐帧推进物理时间，相机可环绕，曝光首帧锁定；分段写出、可断点续传。

        Args:
            args: CLI 参数（另读取 `--video/--orbit/--orbit_degrees/--n_frames/--fps/--v2_orbit_seconds/
                --resume`）。
            width, height: 输出分辨率（像素）。
            fov: 竖直视野角（度）。

        Formula:
            `dt = P_in / (v2_orbit_seconds · fps)`，`P_in = 2π / Ω(r_in)`，`Ω = sqrt(0.5 / r³)`：
            内缘转一圈对应 `v2_orbit_seconds` 秒视频。第 f 帧：`t = 2000 + f·dt`，相机方位
            `azim = azim_0 + orbit_degrees·f/(n_frames − 1)`，抖动种子 `f + 1`。

        Notes:
            每 240 帧写一个分段（`src/v2/video_segments.py`）。中断后用同样参数加 `--resume` 重跑，
            跳过已完成分段，最多重渲一段；输出与一次跑完逐帧一致（每帧只由 t、相机、种子、
            锁定曝光决定）。参数有变化时自动从头开始。代码版本变化不在校验范围内，换代码后
            续传请自行确认。
        """
        import json as _json
        import math as _math
        import time as _time

        from src.v2.video_segments import render_segmented

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
        start = _time.time()
        pool = ThreadPoolExecutor(max_workers=1)

        def _to_u8(frame):
            return (np.clip(frame, 0, 1) * 255).astype(np.uint8)

        def _gpu_frame(f):
            """GPU 阶段：积分第 f 帧，返回 `(hdr, sky)`。"""
            azim = base_azim + _math.radians(orbit_deg) * f / max(args.n_frames - 1, 1)
            cam = [base_dist * _math.cos(base_elev) * _math.cos(azim),
                   base_dist * _math.cos(base_elev) * _math.sin(azim),
                   base_dist * _math.sin(base_elev)]
            # render_hdr 内部先 +1：第 f 帧种子 = f + 1，与是否续传无关
            renderer.jitter_seed[None] = f
            return renderer.render_hdr(cam_pos=cam, fov=fov, t=t0 + f * dt_per_frame)

        def _render_frames(f0, f1, exposure):
            """按帧序产出 `(uint8 帧, 锁定曝光)`；后处理交给单线程池，与下一帧 GPU 积分并行。"""
            renderer.fixed_exposure = exposure
            pending = None
            for f in range(f0, f1):
                hdr, sky = _gpu_frame(f)
                if renderer.fixed_exposure is None:
                    # 首帧同步处理并锁定曝光，避免逐帧曝光闪烁（参考实现 cmd_video 同款）
                    img = _to_u8(renderer.finish(hdr, sky))
                    renderer.fixed_exposure = 1.0 / renderer.last_white_point
                    yield img, renderer.fixed_exposure
                else:
                    fut = pool.submit(lambda h=hdr, s=sky: _to_u8(renderer.finish(h, s)))
                    if pending is not None:
                        yield pending.result(), renderer.fixed_exposure
                    pending = fut
                if f % 24 == 0:
                    print(f"  frame {f}/{args.n_frames}  {_time.time() - start:.1f}s", flush=True)
            if pending is not None:
                yield pending.result(), renderer.fixed_exposure

        # 续传校验：除输出路径与 --resume 外的全部参数（含分辨率、视野）
        params = {k: v for k, v in vars(args).items() if k not in ("output", "resume")}
        params = _json.loads(_json.dumps({**params, "width": width, "height": height, "fov": fov}, default=str))
        render_segmented(args.n_frames, args.fps, args.output, params, args.resume, _render_frames)
        pool.shutdown()
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
