from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Optional
import hashlib
import json
import os
import shutil
import time
from PIL import Image
import imageio.v3 as iio
import numpy as np
import taichi as ti
from src.core.constants import R_DISK_INNER_DEFAULT, R_DISK_OUTER_DEFAULT
from src.core.skybox import load_or_generate_skybox
from src.v1.lifecycle import _advance_lifecycle_frame, _init_lifecycle_system
from src.v1.renderer import TaichiRenderer
from src.v1.texture import compute_disk_texture_resolution, load_disk_texture


def render_image(width: int, height: int, cam_pos: List[float], fov: float, step_size: float, skybox_path: Optional[str] = None,
                  n_stars: int = 6000, tex_w: int = 2048, tex_h: int = 1024, r_max: float = 10.0, device: str = "cpu",
                  disk_texture_path: Optional[str] = None, r_disk_inner: float = R_DISK_INNER_DEFAULT,
                  r_disk_outer: float = R_DISK_OUTER_DEFAULT, disk_tilt: float = 0.0,
                  lens_flare: bool = False, anti_alias: str = "disabled", aa_strength: float = 1.0,
                  disk_rotation_speed: float = 0.1, disk_generation_scale: int = 2,
                  force_regenerate_disk_texture: bool = False, ignore_taichi_cache: bool = False) -> np.ndarray:
    """
    使用 Taichi 渲染单帧图像。

    纹理通过实体生命周期系统在 t=0 生成，与 interactive 模式使用相同流程。
    若提供 disk_texture_path 则直接加载外部纹理（跳过生命周期系统）。
    """
    skybox, tex_h, tex_w = load_or_generate_skybox(skybox_path, tex_w, tex_h, n_stars)

    # 外部纹理优先；否则用占位纹理 + 生命周期系统生成
    disk_tex = load_disk_texture(disk_texture_path)
    use_lifecycle = disk_tex is None
    if use_lifecycle:
        n_phi, n_r = compute_disk_texture_resolution(
            width, height, cam_pos, fov, r_disk_inner, r_disk_outer)
        disk_tex = np.zeros((n_r, n_phi, 4), dtype=np.float32)

    renderer = TaichiRenderer(
        width, height, skybox, disk_tex,
        step_size=step_size, r_max=r_max, device=device,
        r_disk_inner=r_disk_inner, r_disk_outer=r_disk_outer,
        disk_tilt=disk_tilt,
        lens_flare=lens_flare,
        anti_alias=anti_alias,
        aa_strength=aa_strength,
        disk_rotation_speed=disk_rotation_speed,
        ignore_taichi_cache=ignore_taichi_cache
    )

    if use_lifecycle:
        factories = _init_lifecycle_system(renderer, n_r, n_phi, seed=42)
        _advance_lifecycle_frame(renderer, factories, t=0.0, dt=0.0,
                                 recompute_stats=True)

    t0 = time.time()
    print(f"Taichi: {width}x{height}, cam_pos={list(cam_pos)}, fov={fov}°, step_size={step_size}")
    img = renderer.render(cam_pos, fov, frame=0)
    print(f"Done in {time.time() - t0:.1f}s")

    return img


def render_interactive(renderer: TaichiRenderer, width: int, height: int,
                       fov: float, initial_cam_pos: List[float],
                       disk_rotation_speed: float = 0.05) -> None:
    """实时交互预览模式（实体生命周期系统）。

    使用 Taichi Legacy GUI 实时渲染黑洞场景，支持鼠标/键盘控制相机和渲染开关。
    吸积盘纹理由两层系统实时生成：
        - GPU 背景层：3D simplex noise 驱动的时间演化（宽 r 组件）
        - CPU 实体层：filament/hotspot/RT spike 的生命周期管理（窄 r 组件）

    相机控制（球坐标，始终朝向原点）:
        鼠标左键拖拽: 旋转视角 (φ, θ)
        滚轮上/下: 缩放距离
        ↑/↓: 调整 FOV

    渲染开关:
        D: 切换微分光线（抗锯齿基础，默认关）
        B: 切换 Bloom 泛光（默认关）
        L: 切换 Lens Flare（默认关）
        S: 保存当前帧截图
        ESC/Q: 退出

    Args:
        renderer: TaichiRenderer 实例
        width, height: 窗口分辨率
        fov: 初始视野角度
        initial_cam_pos: 初始相机位置 [x, y, z]
        disk_rotation_speed: 盘旋转速度
    """
    import taichi as ti

    cam_pos = np.array(initial_cam_pos, dtype=np.float64)
    r = float(np.linalg.norm(cam_pos))
    theta = float(np.arccos(np.clip(cam_pos[2] / r, -1, 1)))
    phi = float(np.arctan2(cam_pos[1], cam_pos[0]))

    toggle_diff = False
    toggle_bloom = True
    toggle_flare = False
    renderer.lens_flare = False

    # —— 初始化实体生命周期系统 ——
    factories = _init_lifecycle_system(renderer, renderer.dtex_h, renderer.dtex_w, seed=42)
    print("实体生命周期系统已启用 (filaments=200, hotspots=30, rt_spikes=15)")

    gui = ti.GUI('Black Hole Interactive', res=(width, height))
    wall_time = 0.0
    frame_count = 0
    total_frames = 0
    fps_timer = time.time()
    fps_display = 0.0
    last_frame_time = time.time()

    mouse_pressed = False
    mouse_last = (0.0, 0.0)
    solo_idx = -1

    _SOLO_NAMES = {
        0: "temp_base", 1: "spiral", 2: "spiral_temp",
        3: "turbulence", 4: "turb_temp",
        5: "filaments", 6: "filaments_temp",
        7: "rt_spikes", 8: "rt_temp",
        9: "hotspot", 10: "hotspot_temp",
        11: "az_hotspot", 12: "disturb_mod",
    }

    print(f"\n=== 交互模式 ({width}x{height}) ===")
    print(f"鼠标拖拽: 旋转 | 滚轮: 缩放 | ↑↓: FOV")
    print(f"D: 微分光线 | B: Bloom | L: Lens Flare | S: 截图 | ESC: 退出")
    print(f"1-8: solo 组件 | 0: 显示全部\n")

    while gui.running:
        # —— 事件处理 ——
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE or e.key == 'q':
                gui.running = False
            elif e.key == 'd':
                toggle_diff = not toggle_diff
                print(f"微分光线: {'开' if toggle_diff else '关'}")
            elif e.key == 'b':
                toggle_bloom = not toggle_bloom
                print(f"Bloom: {'开' if toggle_bloom else '关'}")
            elif e.key == 'l':
                toggle_flare = not toggle_flare
                renderer.lens_flare = toggle_flare
                print(f"Lens Flare: {'开' if toggle_flare else '关'}")
            elif e.key == '0':
                solo_idx = -1
                print("Solo: 全部组件")
            elif e.key == '1':
                solo_idx = 0
                print(f"Solo: {_SOLO_NAMES[0]} (idx 0)")
            elif e.key == '2':
                solo_idx = 1
                print(f"Solo: {_SOLO_NAMES[1]} (idx 1)")
            elif e.key == '3':
                solo_idx = 3
                print(f"Solo: {_SOLO_NAMES[3]} (idx 3)")
            elif e.key == '4':
                solo_idx = 11
                print(f"Solo: {_SOLO_NAMES[11]} (idx 11)")
            elif e.key == '5':
                solo_idx = 12
                print(f"Solo: {_SOLO_NAMES[12]} (idx 12)")
            elif e.key == '6':
                solo_idx = 5
                print(f"Solo: {_SOLO_NAMES[5]} (idx 5)")
            elif e.key == '7':
                solo_idx = 9
                print(f"Solo: {_SOLO_NAMES[9]} (idx 9)")
            elif e.key == '8':
                solo_idx = 7
                print(f"Solo: {_SOLO_NAMES[7]} (idx 7)")
            elif e.key == 's':
                screenshot_path = f"output/screenshot_{int(time.time())}.png"
                os.makedirs("output", exist_ok=True)
                img_save = renderer.render(cam_pos.tolist(), fov, frame=0)
                img_uint8 = (np.clip(img_save, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(img_uint8, "RGB").save(screenshot_path)
                print(f"截图已保存: {screenshot_path}")
            elif e.key == ti.GUI.UP:
                fov = max(10, fov - 5)
                print(f"FOV: {fov}°")
            elif e.key == ti.GUI.DOWN:
                fov = min(170, fov + 5)
                print(f"FOV: {fov}°")
            elif e.key == ti.GUI.LMB:
                mouse_pressed = True
                mouse_last = gui.get_cursor_pos()

        for e in gui.get_events(ti.GUI.RELEASE):
            if e.key == ti.GUI.LMB:
                mouse_pressed = False

        # 鼠标拖拽旋转
        if mouse_pressed and gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            dx = mx - mouse_last[0]
            dy = my - mouse_last[1]
            phi -= dx * 3.0
            theta = np.clip(theta - dy * 3.0, 0.05, np.pi - 0.05)
            mouse_last = (mx, my)

        # 滚轮缩放
        if gui.is_pressed(ti.GUI.UP):
            pass
        # Taichi Legacy GUI 不直接支持滚轮，用 +/- 键代替
        if gui.is_pressed('=') or gui.is_pressed('+'):
            r = max(2.0, r * 0.97)
        if gui.is_pressed('-'):
            r *= 1.03

        # 更新相机位置（球坐标 → 笛卡尔）
        cam_pos[0] = r * np.sin(theta) * np.cos(phi)
        cam_pos[1] = r * np.sin(theta) * np.sin(phi)
        cam_pos[2] = r * np.cos(theta)

        # —— 实体生命周期更新 ——
        now_real = time.time()
        dt = min(now_real - last_frame_time, 0.1)
        last_frame_time = now_real
        scaled_dt = dt * disk_rotation_speed * 20.0
        wall_time += scaled_dt

        total_frames += 1
        _advance_lifecycle_frame(renderer, factories, wall_time, scaled_dt,
                                 recompute_stats=(total_frames % 60 == 1),
                                 solo_idx=solo_idx)

        # —— 渲染到 GPU field ——
        renderer.render_to_field(
            cam_pos.tolist(), fov, frame=0,
            skip_differentials=not toggle_diff,
            skip_bloom=not toggle_bloom,
        )

        # —— 显示（直接从 GPU field，无 CPU 传输）——
        gui.set_image(renderer.final_field)

        # HUD 信息
        frame_count += 1
        fps_now = time.time()
        if fps_now - fps_timer >= 0.5:
            fps_display = frame_count / (fps_now - fps_timer)
            frame_count = 0
            fps_timer = fps_now

        n_entities = sum(len(f.entities) for f in factories.values())
        toggles = f"D:{'ON' if toggle_diff else 'off'} B:{'ON' if toggle_bloom else 'off'} L:{'ON' if toggle_flare else 'off'}"
        solo_txt = f" SOLO:{_SOLO_NAMES[solo_idx]}" if solo_idx >= 0 else ""
        gui.text(f"{fps_display:.0f} FPS | {toggles} | E:{n_entities}{solo_txt}", pos=(0.01, 0.97), color=0xFFFFFF)
        gui.text(f"r={r:.1f} fov={fov:.0f} t={wall_time:.1f}", pos=(0.01, 0.93), color=0xCCCCCC)
        gui.text("+/-: zoom | arrows: FOV | S: screenshot", pos=(0.01, 0.02), color=0x888888)

        gui.show()

    gui.close()
    print("交互模式退出")


def render_video(renderer: TaichiRenderer, width: int, height: int, n_frames: int, fps: int, output_path: str,
                 fov: float, static_cam_pos: List[float], orbit: bool = False, resume: bool = False,
                 disk_rotation_speed: float = 0.1, orbit_degrees: float = 360.0,
                 **_deprecated_kwargs) -> None:
    """
    渲染视频（多帧并合成视频），使用实体生命周期系统生成纹理。

    参数:
        renderer: TaichiRenderer 实例
        width, height: 图像尺寸
        n_frames: 帧数
        fps: 帧率
        output_path: 输出视频路径
        fov: 视野角度
        static_cam_pos: 静态模式下的相机位置
        orbit: 是否围绕原点旋转
        resume: 是否尝试从断点恢复
        disk_rotation_speed: 旋转速度系数
        orbit_degrees: 轨道模式下整段视频的总旋转角度（度）
    """
    orbit_radius = float(np.linalg.norm(static_cam_pos))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    temp_dir_name = ".frames_" + hashlib.md5(output_path.encode()).hexdigest()[:16]
    temp_dir = os.path.join(os.path.dirname(output_path), temp_dir_name)
    progress_file = os.path.join(temp_dir, "progress.json")

    params = {
        "n_frames": n_frames,
        "fov": fov,
        "orbit": orbit,
        "disk_rotation_speed": disk_rotation_speed,
        "orbit_degrees": orbit_degrees,
    }

    completed = set()
    if resume and os.path.isdir(temp_dir) and os.path.isfile(progress_file):
        with open(progress_file, "r") as f:
            saved = json.load(f)
        saved_params = saved.get("params", {})
        if saved_params != params:
            print(f"Warning: parameters changed, starting over")
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
        else:
            completed = set(saved.get("completed", []))
            print(f"Resuming: {len(completed)}/{n_frames} frames already rendered")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    total_t0 = time.time()
    angle_step = orbit_degrees / n_frames
    rendered_this_session = 0

    # —— 异步 PNG 保存 ——
    MAX_PENDING_PNGS = 4
    png_pool = ThreadPoolExecutor(max_workers=2)
    png_futures = []

    def _save_png(path, img_uint8):
        Image.fromarray(img_uint8, "RGB").save(path)

    # —— 初始化生命周期系统 ——
    n_r = renderer.dtex_h
    n_phi = renderer.dtex_w
    factories = _init_lifecycle_system(renderer, n_r, n_phi, seed=42)
    dt = disk_rotation_speed
    print(f"  生命周期系统已初始化 (n_r={n_r}, n_phi={n_phi})")

    # —— 断点恢复：快速重演模拟到 resume 点 ——
    if completed:
        max_completed = max(completed)
        print(f"  快速重演模拟到帧 {max_completed}...")
        replay_t0 = time.time()
        for f in range(max_completed + 1):
            t = f * dt
            _advance_lifecycle_frame(renderer, factories, t, dt)
        print(f"  重演完成: {time.time() - replay_t0:.1f}s")

    # —— 主渲染循环 ——
    for frame in range(n_frames):
        t = frame * dt

        if orbit:
            angle_deg = frame * angle_step
            angle_rad = np.radians(angle_deg)
            orbit_z = static_cam_pos[2]
            cx = orbit_radius * np.cos(angle_rad)
            cy = orbit_radius * np.sin(angle_rad)
            cam_pos = [cx, cy, orbit_z]
            status_str = f"{angle_deg:.1f}°"
        else:
            cam_pos = static_cam_pos
            status_str = "static"

        if frame in completed:
            continue

        t0 = time.time()
        _advance_lifecycle_frame(renderer, factories, t, dt,
                                 recompute_stats=(frame % 60 == 0))
        img = renderer.render(cam_pos, fov, frame=0)
        elapsed = time.time() - t0
        rendered_this_session += 1

        frame_path = os.path.join(temp_dir, f"frame_{frame:04d}.png")
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        if len(png_futures) >= MAX_PENDING_PNGS:
            png_futures.pop(0).result()
        png_futures.append(png_pool.submit(_save_png, frame_path, img_uint8))

        completed.add(frame)
        if rendered_this_session % 10 == 0 or frame == n_frames - 1:
            with open(progress_file, "w") as f:
                json.dump({"params": params, "completed": list(completed)}, f)

        if rendered_this_session % 100 == 0 or frame == n_frames - 1:
            eta = (time.time() - total_t0) / rendered_this_session * (n_frames - len(completed))
            print(f"  frame {frame}/{n_frames} ({status_str}) {elapsed:.1f}s, done {len(completed)}/{n_frames}, ETA {eta/60:.0f}min")

    # 等待所有 PNG 写入完成
    for f in png_futures:
        f.result()
    png_pool.shutdown(wait=False)

    if rendered_this_session > 0:
        print(f"\nSession rendered {rendered_this_session} frames in {(time.time() - total_t0)/60:.1f} min")

    if len(completed) < n_frames:
        print(f"Warning: only {len(completed)}/{n_frames} frames completed. Run again to resume.")
        return

    total_elapsed = time.time() - total_t0
    print(f"\nAll frames rendered in {total_elapsed/60:.1f} min")

    print(f"Assembling video: {output_path} ({fps} fps, {n_frames/fps:.0f}s)...")
    # 注意：如果需要更高质量的视频（减少摩尔纹），可以：
    # 1. 使用 ffmpeg 直接编码：ffmpeg -framerate {fps} -i frame_%04d.png -c:v libx264 -crf 18 -preset slow output.mp4
    # 2. 或安装 imageio-ffmpeg 并使用更高质量的编码参数
    writer = iio.imopen(output_path, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=fps)

    for frame in range(n_frames):
        frame_path = os.path.join(temp_dir, f"frame_{frame:04d}.png")
        img = iio.imread(frame_path)
        writer.write_frame(img)
        ## 逐帧写入后删除临时文件以节省空间
        #os.remove(frame_path)

    #    os.remove(progress_file)
    print(f"\n提示：如果视频有摩尔纹，可手动用 ffmpeg 重新编码更高质量：")
    print(f"  ffmpeg -framerate {fps} -i {temp_dir}/frame_%04d.png -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p {output_path}")
    #shutil.rmtree(temp_dir)
    print(f"Video saved: {output_path}")
