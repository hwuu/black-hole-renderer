#!/usr/bin/env python3
"""
简化版实验 2 - 只生成 10 帧验证湍流旋转
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

# 参数配置（缩小以加快速度）
N_FRAMES = 10
FPS = 30
SIZE = 256
R_INNER = 2.0
R_OUTER = 15.0
ROTATION_SPEED = 0.1
SEED = 42
OUTPUT_PATH = "output/verify_exp2_turbulence.mp4"

print("=" * 60)
print("简化版实验 2 - 验证湍流旋转")
print("=" * 60)

print(f"\n渲染 {N_FRAMES} 帧...")
frames = []
total_time = 0.0

for frame in tqdm(range(N_FRAMES), desc="Rendering"):
    t0 = time.time()
    
    # 计算旋转偏移
    t_offset = frame * ROTATION_SPEED
    
    # 生成纹理（考虑旋转）
    texture = generate_disk_texture_topview(
        n_phi=256,
        n_r=64,
        seed=SEED,
        r_inner=R_INNER,
        r_outer=R_OUTER,
        enable_rt=True,
        t_offset=t_offset
    )
    
    # 转换为笛卡尔坐标
    image = polar_to_cartesian(texture, SIZE, R_INNER, R_OUTER)
    
    frame_time = time.time() - t0
    total_time += frame_time
    
    # 添加文字覆盖层
    text_lines = [
        f"Verify Turbulence Rotation",
        f"Frame: {frame}/{N_FRAMES}",
        f"t_offset: {t_offset:.2f}",
        f"Time: {frame_time:.2f}s",
    ]
    image_with_text = add_text_overlay(image, text_lines)
    
    frames.append(image_with_text)

# 保存视频
print(f"\n保存视频：{OUTPUT_PATH}")
frames_uint8 = [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames]

writer = iio.imopen(OUTPUT_PATH, "w", plugin="pyav")
writer.init_video_stream("libx264", fps=FPS)
for frame in frames_uint8:
    writer.write_frame(frame)
writer.close()

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
print(f"总时间：{total_time:.2f}s")
print(f"输出：{OUTPUT_PATH}")
print("\n请用播放器查看视频，确认湍流云雾是否随帧旋转")
