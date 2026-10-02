"""V2 与参考实现（Proto）同参数对比：数值指标 + 上下拼接对比图。

用途：
    把"看图猜差距"变成客观验收。两边使用同一相机（dist = 40、仰角 7°、竖直 FOV 38°）、
    同一盘参数（r_in = 3、r_out = 30、预设 M）、同一时刻 t（默认 2000，`--t` 可改）、同一超采样，
    分别输出线性 HDR，再各自按参考实现的自动曝光（盘区亮度 p99.9 → 0.9）归一后比较。

输出：
    - 终端：覆盖率、对数亮度中位误差、R/B、左右通量比、阴影下沿（光子环下半部）通量比
    - PNG：上 = V2，下 = Proto，各 W×H，带标签

用法：
    python scripts/compare_v2_proto.py                      # 1920×1080，ss = 2
    python scripts/compare_v2_proto.py --w 960 --h 540 --ss 1

Simplifications:
    V2 用黑色天空（Proto 是稀疏程序星空），指标只统计盘区，星点对比较几乎无影响。
"""

import argparse
import glob
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PROTO = os.path.join(ROOT, "scripts", "proto_disk_reference.py")
PROTO_OUT = os.path.join(ROOT, "output", "proto")
LUM_W = np.array([0.2126, 0.7152, 0.0722])
CAM_DIST, CAM_ELEV_DEG, FOV_DEG = 40.0, 7.0, 38.0
R_IN, R_OUT, T_FRAME = 3.0, 30.0, 2000.0


def run_proto(w: int, h: int, ss: int, reuse: bool, t: float = T_FRAME) -> tuple[np.ndarray, np.ndarray]:
    """子进程运行 Proto `frame`（独立 Taichi 运行时），读回 HDR 与 LDR。

    Args:
        w, h: 输出分辨率（像素）。
        ss: 每轴超采样倍率。
        reuse: True 时跳过渲染，直接读 `output/proto/` 中上一次同分辨率的结果（迭代调试用）。
        t: 帧物理时间（r_s/c）。

    Returns:
        `(hdr, ldr)`：`(h, w, 3)` 线性 HDR float64；`(h, w, 3)` uint8 sRGB。
    """
    from PIL import Image

    if not reuse:
        subprocess.run([sys.executable, PROTO, "frame", "--w", str(w), "--h", str(h), "--ss", str(ss),
                    "--r_out", str(R_OUT), "--dist", str(CAM_DIST), "--elev", str(CAM_ELEV_DEG),
                        "--fov", str(FOV_DEG), "--t", str(t)], check=True)
    hdr = np.transpose(np.load(os.path.join(PROTO_OUT, "frame_hdr.npy")), (1, 0, 2)).astype(np.float64)
    pngs = glob.glob(os.path.join(PROTO_OUT, f"frame_M_m3_R{R_OUT:g}_*_{w}x{h}.png"))
    ldr = np.asarray(Image.open(max(pngs, key=os.path.getmtime)).convert("RGB"))
    return hdr, ldr


def run_v2(w: int, h: int, ss: int, t: float = T_FRAME) -> tuple[np.ndarray, np.ndarray]:
    """在本进程用 `DiskV2Renderer` 渲染同参数帧。

    Args:
        w, h: 输出分辨率（像素）。
        ss: 每轴超采样倍率。
        t: 帧物理时间（r_s/c）。

    Returns:
        `(hdr, ldr)`：`(h, w, 3)` 线性 HDR；`(h, w, 3)` uint8 sRGB。
    """
    import taichi as ti

    ti.init(arch=ti.gpu, default_fp=ti.f32)
    from src.v2.params import DiskV2Params, DiskV2VolumeParams
    from src.v2.taichi_render import DiskV2Renderer

    renderer = DiskV2Renderer(
        width=w, height=h,
        params=DiskV2Params(r_in=R_IN, r_out=R_OUT),
        skybox=np.zeros((64, 128, 3), np.float32),
        volume_params=DiskV2VolumeParams(thickness_scale=1.0),  # 预设 M 视觉厚度，与参考实现对齐
        r_max=90.0, sky_gain=0.0, ss=ss,
    )
    e = math.radians(CAM_ELEV_DEG)
    cam = [0.0, -CAM_DIST * math.cos(e), CAM_DIST * math.sin(e)]
    img = renderer.render(cam_pos=cam, fov=FOV_DEG, t=t)
    return renderer.last_hdr.astype(np.float64), (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def normalize(hdr: np.ndarray) -> np.ndarray:
    """按参考实现自动曝光归一：盘区亮度（L > 1e-4）p99.9 → 1。

    Args:
        hdr: `(h, w, 3)` 线性 HDR。

    Returns:
        归一化亮度图 `(h, w)`，量纲为"相对 p99.9"。
    """
    lum = hdr @ LUM_W
    v = lum[lum > 1e-4]
    return lum / max(float(np.percentile(v, 99.9)), 1e-12)


def metrics(hdr: np.ndarray) -> dict:
    """单张 HDR 的对比指标。

    Args:
        hdr: `(h, w, 3)` 线性 HDR。

    Returns:
        dict：`coverage`（归一亮度 > 0.01 的像素比例）、`rb`（盘区 R/B 均值比）、
        `lr`（左/右半幅通量比，多普勒不对称）、`lower_ring`（阴影正下方窄带通量 / 全盘通量，
        对应光子环下半部）。
    """
    h, w, _ = hdr.shape
    ln = normalize(hdr)
    disk = ln > 0.01
    # 阴影下沿：阴影角半径 ≈ 2.6 r_s / 40 r_s，换算成像素后取其下方 ±40% 的窄带
    pix_per_rad = h / (2.0 * math.tan(math.radians(FOV_DEG) / 2.0))
    r_sh = 2.6 / CAM_DIST * pix_per_rad
    yy, xx = np.mgrid[0:h, 0:w]
    dy, dx = yy - (h - 1) / 2.0, xx - (w - 1) / 2.0
    rr = np.hypot(dx, dy)
    band = (rr > 0.8 * r_sh) & (rr < 1.4 * r_sh) & (dy > 0.3 * r_sh)
    return dict(
        coverage=float(disk.mean()),
        rb=float(hdr[disk, 0].mean() / max(hdr[disk, 2].mean(), 1e-12)),
        lr=float(ln[:, : w // 2].sum() / max(ln[:, w // 2:].sum(), 1e-12)),
        lower_ring=float(ln[band].sum() / max(ln.sum(), 1e-12)),
    )


def label(img: np.ndarray, text: str) -> np.ndarray:
    """左上角叠加文字标签。

    Args:
        img: `(h, w, 3)` uint8。
        text: 标签文本。

    Returns:
        叠加标签后的 uint8 图像。
    """
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(img)
    size = max(14, img.shape[0] // 36)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(im).text((size, size // 2), text, fill=(235, 235, 235), font=font)
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser(description="V2 vs Proto 同参数对比")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "compare_v2_proto.png"))
    ap.add_argument("--t", type=float, default=T_FRAME, help="帧物理时间（r_s/c），两边相同")
    ap.add_argument("--reuse_proto", action="store_true", help="复用 output/proto 中上一次的 Proto 输出；--t 必须与那次一致（HDR 文件按时间覆盖）")
    a = ap.parse_args()

    hdr_p, ldr_p = run_proto(a.w, a.h, a.ss, a.reuse_proto, a.t)
    hdr_v, ldr_v = run_v2(a.w, a.h, a.ss, a.t)

    mp, mv = metrics(hdr_p), metrics(hdr_v)
    lp, lv = normalize(hdr_p), normalize(hdr_v)
    both = (lp > 0.01) & (lv > 0.01)
    log_err = np.abs(np.log(lv[both] / lp[both]))
    print(f"\n指标（t = {a.t:g}）     Proto      V2")
    for k, name in [("coverage", "盘覆盖率"), ("rb", "R/B"), ("lr", "左/右通量比"), ("lower_ring", "光子环下半部通量占比")]:
        print(f"  {name:<14s} {mp[k]:9.4f}  {mv[k]:9.4f}")
    print(f"  对数亮度 |ln(V2/Proto)| 中位数 {np.median(log_err):.3f}，p90 {np.percentile(log_err, 90):.3f}"
          f"（共同盘区 {both.mean() * 100:.1f}% 像素）")

    from PIL import Image

    top = label(ldr_v, "V2 (src.v2.DiskV2Renderer)")
    bottom = label(ldr_p, "Proto (scripts/proto_disk_reference.py, preset M)")
    sep = np.full((4, a.w, 3), 60, np.uint8)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    Image.fromarray(np.concatenate([top, sep, bottom], axis=0)).save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
