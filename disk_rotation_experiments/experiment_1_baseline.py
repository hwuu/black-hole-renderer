#!/usr/bin/env python3
"""
实验 1: Baseline - 固定纹理 + 刚体旋转

方案：
- 生成一次纹理
- 每帧按开普勒速度旋转采样点
- 这是当前 render.py 的方案
"""

import numpy as np
import time
from tqdm import tqdm
import imageio.v3 as iio
from common import (
    generate_disk_texture_topview,
    sample_disk_with_rotation,
    add_text_overlay,
)

# 参数配置
N_FRAMES = 30
FPS = 30
SIZE = 512
R_INNER = 2.0
R_OUTER = 15.0
ROTATION_SPEED = 0.1
SEED = 42
OUTPUT_PATH = "output/disk_rotation_exp1_baseline.mp4"

print("=" * 60)
print("实验 1: Baseline - 固定纹理 + 刚体旋转")
print("=" * 60)

# 生成纹理（只生成一次）
print("\n生成吸积盘纹理...")
t0_texture = time.time()
texture = generate_disk_texture_topview(
    n_phi=512,
    n_r=128,
    seed=SEED,
    r_inner=R_INNER,
    r_outer=R_OUTER,
    enable_rt=True,
    t_offset=0.0  # Baseline: 纹理不旋转
)
texture_time = time.time() - t0_texture
print(f"纹理生成完成: {texture_time:.2f}s")

# 渲染帧
print(f"\n渲染 {N_FRAMES} 帧...")
frames = []
total_frame_time = 0.0

for frame in tqdm(range(N_FRAMES), desc="Rendering"):
    t0_frame = time.time()

    # 计算旋转偏移
    t_offset = frame * ROTATION_SPEED

    # 采样纹理（应用旋转）
    image = sample_disk_with_rotation(texture, t_offset, R_INNER, R_OUTER, SIZE)

    frame_time = time.time() - t0_frame
    total_frame_time += frame_time

    # 添加文字覆盖层
    text_lines = [
        f"Experiment 1: Baseline",
        f"Method: Fixed texture + rotation",
        f"Frame: {frame}/{N_FRAMES}",
        f"t_offset: {t_offset:.2f}",
        f"Frame time: {frame_time:.3f}s",
        f"Total time: {total_frame_time:.1f}s",
    ]
    image_with_text = add_text_overlay(image, text_lines)

    frames.append(image_with_text)

# 保存视频
print(f"\n保存视频: {OUTPUT_PATH}")
frames_uint8 = [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames]

writer = iio.imopen(OUTPUT_PATH, "w", plugin="pyav")
writer.init_video_stream("libx264", fps=FPS)
for frame in frames_uint8:
    writer.write_frame(frame)
writer.close()

# 统计信息
print("\n" + "=" * 60)
print("实验 1 完成")
print("=" * 60)
print(f"纹理生成时间: {texture_time:.2f}s")
print(f"总渲染时间: {total_frame_time:.2f}s")
print(f"平均帧时间: {total_frame_time/N_FRAMES:.3f}s")
print(f"总时间: {texture_time + total_frame_time:.2f}s")
print(f"输出: {OUTPUT_PATH}")
