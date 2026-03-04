#!/usr/bin/env python3
"""
实验 3: 关键帧插值

方案：
- 预计算 N 个关键帧
- 运行时线性插值
- 平衡速度和质量
"""

import numpy as np
import time
from tqdm import tqdm
import imageio.v3 as iio
from common import (
    generate_disk_texture_topview,
    polar_to_cartesian,
    add_text_overlay,
)

# 参数配置
N_FRAMES = 30
N_KEYFRAMES = 10  # 关键帧数量
FPS = 30
SIZE = 512
R_INNER = 2.0
R_OUTER = 15.0
ROTATION_SPEED = 0.1
SEED = 42
OUTPUT_PATH = "output/disk_rotation_exp3_keyframes.mp4"

print("=" * 60)
print("实验 3: 关键帧插值")
print("=" * 60)

# 预计算关键帧
print(f"\n预计算 {N_KEYFRAMES} 个关键帧...")
keyframes = []
t0_precompute = time.time()

for i in tqdm(range(N_KEYFRAMES), desc="Precomputing"):
    # 均匀分布关键帧
    t_offset = i * (N_FRAMES * ROTATION_SPEED) / N_KEYFRAMES

    # 生成纹理
    texture = generate_disk_texture_topview(
        n_phi=512,
        n_r=128,
        seed=SEED,
        r_inner=R_INNER,
        r_outer=R_OUTER,
        enable_rt=True,
        t_offset=t_offset
    )

    # 转换为笛卡尔坐标
    image = polar_to_cartesian(texture, SIZE, R_INNER, R_OUTER)
    keyframes.append(image)

precompute_time = time.time() - t0_precompute
print(f"预计算完成: {precompute_time:.2f}s")

# 渲染帧（插值）
print(f"\n渲染 {N_FRAMES} 帧（插值）...")
frames = []
total_interp_time = 0.0

for frame in tqdm(range(N_FRAMES), desc="Rendering"):
    t0_frame = time.time()

    # 计算插值参数
    t = frame / N_FRAMES  # [0, 1)
    keyframe_float = t * N_KEYFRAMES
    keyframe_idx = int(keyframe_float) % N_KEYFRAMES
    keyframe_next = (keyframe_idx + 1) % N_KEYFRAMES
    alpha = keyframe_float - int(keyframe_float)

    # 线性插值
    image = (1 - alpha) * keyframes[keyframe_idx] + alpha * keyframes[keyframe_next]

    frame_time = time.time() - t0_frame
    total_interp_time += frame_time

    # 添加文字覆盖层
    t_offset = frame * ROTATION_SPEED
    text_lines = [
        f"Experiment 3: Keyframes",
        f"Method: {N_KEYFRAMES} keyframes + lerp",
        f"Frame: {frame}/{N_FRAMES}",
        f"Keyframe: {keyframe_idx}->{keyframe_next}",
        f"Alpha: {alpha:.3f}",
        f"Frame time: {frame_time:.4f}s",
        f"Total: {total_interp_time:.2f}s",
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
print("实验 3 完成")
print("=" * 60)
print(f"预计算时间: {precompute_time:.2f}s")
print(f"插值渲染时间: {total_interp_time:.2f}s")
print(f"平均帧时间: {total_interp_time/N_FRAMES:.4f}s")
print(f"总时间: {precompute_time + total_interp_time:.2f}s")
print(f"输出: {OUTPUT_PATH}")
print(f"\n关键帧数量: {N_KEYFRAMES}")
print(f"内存占用: ~{N_KEYFRAMES * SIZE * SIZE * 3 * 4 / 1024 / 1024:.1f} MB")
