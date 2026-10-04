"""运镜联系表：把路径在若干时刻的画面拼成一张图，用于审构图。

每格标注时刻、镜头标签、相机位置，并画三分线（灰）、黑洞目标位置（青色十字）与实际投影位置（品红圆圈）。
默认取每个关键帧的经过时刻；曝光为单帧自动曝光 + 曝光补偿（与 `--v2_camera_path_time` 一致）。

用法::

    python scripts/contact_sheet.py configs/camera_paths/interstellar_skim.json \\
        --texture sky.jpg --v2_reverse_rotation -o output/v2_arts/contact_sheet.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import taichi as ti  # noqa: E402

from src.core.skybox import load_or_generate_skybox  # noqa: E402
from src.v2.camera_compose import project_origin  # noqa: E402
from src.v2.params import DiskV2Params, DiskV2VolumeParams  # noqa: E402
from src.v2.path_video import load_camera_plan, render_path_hdr  # noqa: E402
from src.v2.taichi_render import DiskV2Renderer  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        参数命名空间：路径文件、取样时刻、天空纹理、盘内外半径、旋转方向、内缘周期、每格分辨率、优化级别、
        每行格数、输出路径。
    """
    p = argparse.ArgumentParser(description="运镜联系表")
    p.add_argument("path", help="运镜路径文件（JSON）")
    p.add_argument("--times", type=float, nargs="+", default=None,
                   help="取样时刻（秒，0 ≤ T ≤ 路径时长）；默认为各关键帧的经过时刻")
    p.add_argument("--texture", "-t", default=None, help="天空纹理（等距柱状图）；默认程序生成星空")
    p.add_argument("--ar1", type=float, default=3.0, help="盘内半径（r_s，与 render.py 同名参数一致）(default: 3)")
    p.add_argument("--ar2", type=float, default=30.0, help="盘外半径（r_s，与 render.py 同名参数一致）(default: 30)")
    p.add_argument("--v2_reverse_rotation", action="store_true", help="反转盘旋转方向（与 render.py 同名参数一致）")
    p.add_argument("--v2_orbit_seconds", type=float, default=16.0, help="内缘转一圈对应的视频秒数 (default: 16)")
    p.add_argument("--size", type=int, nargs=2, default=[640, 360], metavar=("W", "H"), help="每格分辨率 (default: 640 360)")
    p.add_argument("--v2_opt", type=int, default=3, choices=[0, 1, 2, 3], help="优化级别 (default: 3)")
    p.add_argument("--columns", type=int, default=4, help="每行格数 (default: 4)")
    p.add_argument("--output", "-o", default="output/v2_arts/contact_sheet.png", help="输出图片路径")
    return p.parse_args()


def annotate(img: np.ndarray, title: str, footer: str, target_uv, actual_uv) -> Image.Image:
    """在一格画面上画三分线、目标 / 实际黑洞位置与文字。

    Args:
        img: `(H, W, 3)` float LDR [0, 1]。
        title: 左上角文字。
        footer: 左下角文字。
        target_uv: 黑洞目标画面位置 `(u, v)`。
        actual_uv: 黑洞实际投影位置 `(u, v)`。

    Returns:
        标注后的 PIL 图像。
    """
    h, w = img.shape[:2]
    im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for g in (1 / 3, 2 / 3):
        d.line([(g * w, 0), (g * w, h)], fill=(90, 90, 90))
        d.line([(0, g * h), (w, g * h)], fill=(90, 90, 90))
    cx, cy = target_uv[0] * w, target_uv[1] * h
    d.line([(cx - 8, cy), (cx + 8, cy)], fill=(0, 255, 255))
    d.line([(cx, cy - 8), (cx, cy + 8)], fill=(0, 255, 255))
    px, py = actual_uv[0] * w, actual_uv[1] * h
    d.ellipse([px - 3, py - 3, px + 3, py + 3], outline=(255, 0, 255))
    d.text((6, 4), title, fill=(255, 255, 0))
    d.text((6, h - 14), footer, fill=(255, 255, 0))
    return im


def main() -> None:
    """渲染各取样时刻的画面并拼成联系表。"""
    args = parse_args()
    plan = load_camera_plan(args.path)
    path = plan.path
    if args.times is None:
        times, labels = list(path.keyframe_times()), [k.label for k in path.keyframes]
    else:
        bad = [t for t in args.times if not 0.0 <= t <= path.duration]
        if bad:
            raise SystemExit(f"--times 须在 [0, {path.duration}] 内，越界：{bad}")
        times, labels = args.times, [""] * len(args.times)
    w, h = args.size

    ti.init(arch=ti.gpu, default_fp=ti.f32)
    sky, _, _ = load_or_generate_skybox(args.texture, 2048, 1024, 6000)
    renderer = DiskV2Renderer(w, h, DiskV2Params(args.ar1, args.ar2, disk_spin=-1.0 if args.v2_reverse_rotation else 1.0),
                              sky, DiskV2VolumeParams(), r_max=50.0, opt_level=args.v2_opt, ss=1)
    tiles = []
    for t, label in zip(times, labels):
        hdr, sk = render_path_hdr(renderer, path, float(t), args.v2_orbit_seconds, seed=1)
        st = path.state_at(float(t), w / h)
        actual = project_origin(st.pos[None, :], st.forward[None, :], np.array([st.roll]), np.array([st.fov]), w / h)[0]
        r_c = float(np.hypot(st.pos[0], st.pos[1]))
        title = f"t={t:.1f}s  {label}"
        footer = f"r={r_c:.1f} z={st.pos[2]:+.2f} fov={st.fov:.0f} roll={st.roll:.1f}"
        tiles.append(annotate(renderer.finish(hdr, sk), title, footer, st.subject_uv, actual))
        print(f"{title}  {footer}  BH target ({st.subject_uv[0]:.3f}, {st.subject_uv[1]:.3f}) "
              f"actual ({actual[0]:.3f}, {actual[1]:.3f})", flush=True)
    cols = args.columns
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    sheet.save(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
