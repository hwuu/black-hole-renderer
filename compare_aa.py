#!/usr/bin/env python3
"""
AA 对比脚本：生成左右拼接的对比图
"""

import argparse
import subprocess
import os
from PIL import Image

def run_render(pov, fov, ar1, ar2, disk_tilt, resolution, anti_alias, aa_strength, output):
    cmd = [
        "python", "render.py",
        "--pov", str(pov[0]), str(pov[1]), str(pov[2]),
        "--fov", str(fov),
        "--ar1", str(ar1),
        "--ar2", str(ar2),
        "--disk_tilt", str(disk_tilt),
        "--resolution", resolution,
        "--anti_alias", anti_alias,
        "--aa_strength", str(aa_strength),
        "-o", output
    ]
    if anti_alias != "disabled":
        cmd.extend(["--anti_alias", anti_alias])
        if aa_strength != 1.0:
            cmd.extend(["--aa_strength", str(aa_strength)])
    else:
        cmd.extend(["--anti_alias", "disabled"])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def crop_center(img, crop_size):
    """从图像中心裁剪方块"""
    w, h = img.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    return img.crop((left, top, right, bottom))

def compare_aa(resolution="sd", aa_strength=1.0, pov="-20 0 2", fov=60, ar1=2, ar2=15, disk_tilt=20):
    """生成 AA 对比图"""
    
    resolution_map = {"sd": (640, 360), "hd": (1280, 720), "fhd": (1920, 1080), "4k": (3840, 2160)}
    w, h = resolution_map.get(resolution, (640, 360))
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 文件名
    no_aa_file = f"{output_dir}/compare_no_aa_{resolution}.png"
    aa_file = f"{output_dir}/compare_aa_{resolution}_s{aa_strength}.png"
    result_file = f"{output_dir}/compare_result_{resolution}_s{aa_strength}.png"
    
    # 渲染不带 AA
    print(f"\n=== Rendering without AA ===")
    cmd = [
        "python", "render.py",
        "--pov", *pov.split(),
        "--fov", str(fov),
        "--ar1", str(ar1),
        "--ar2", str(ar2),
        "--disk_tilt", str(disk_tilt),
        "--resolution", resolution,
        "--anti_alias", "disabled",
        "-o", no_aa_file
    ]
    subprocess.run(cmd, check=True)
    
    # 渲染带 AA
    print(f"\n=== Rendering with AA (strength={aa_strength}) ===")
    cmd = [
        "python", "render.py",
        "--pov", *pov.split(),
        "--fov", str(fov),
        "--ar1", str(ar1),
        "--ar2", str(ar2),
        "--disk_tilt", str(disk_tilt),
        "--resolution", resolution,
        "--anti_alias", "lod_radius",
        "--aa_strength", str(aa_strength),
        "-o", aa_file
    ]
    subprocess.run(cmd, check=True)
    
    # 裁剪爱因斯坦环区域（中心偏上）
    crop_size = min(w, h) // 3
    
    img_no_aa = Image.open(no_aa_file)
    img_aa = Image.open(aa_file)
    
    # 中心偏上裁剪（爱因斯坦环在黑洞上方）
    cx, cy = w // 2, h // 2
    crop_w = crop_size
    crop_h = crop_size
    
    left = cx - crop_w // 2
    top = cy - crop_h * 3 // 4  # 偏上
    right = left + crop_w
    bottom = top + crop_h
    
    crop_no_aa = img_no_aa.crop((left, top, right, bottom))
    crop_aa = img_aa.crop((left, top, right, bottom))
    
    # 左右拼接
    result = Image.new("RGB", (crop_w * 2, crop_h))
    result.paste(crop_no_aa, (0, 0))
    result.paste(crop_aa, (crop_w, 0))
    
    # 添加标签
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "No AA", fill="white", font=font)
    draw.text((crop_w + 10, 10), f"AA (s={aa_strength})", fill="white", font=font)
    
    result.save(result_file)
    print(f"\n=== Result saved: {result_file} ===")
    
    return result_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AA 对比脚本")
    parser.add_argument("--resolution", "-r", default="sd", choices=["sd", "hd", "fhd", "4k"])
    parser.add_argument("--strength", "-s", type=float, default=1.0, help="AA strength")
    parser.add_argument("--pov", default="-20 0 2")
    parser.add_argument("--fov", type=float, default=60)
    parser.add_argument("--ar1", type=float, default=2)
    parser.add_argument("--ar2", type=float, default=15)
    parser.add_argument("--disk_tilt", type=float, default=20)
    
    args = parser.parse_args()
    
    compare_aa(
        resolution=args.resolution,
        aa_strength=args.strength,
        pov=args.pov,
        fov=args.fov,
        ar1=args.ar1,
        ar2=args.ar2,
        disk_tilt=args.disk_tilt
    )