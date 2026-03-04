#!/usr/bin/env python3
"""
实验 2: 参数化旋转纹理

方案：
- 每帧重新生成纹理
- 在生成时考虑旋转（t_offset 参数）
- 物理正确，无累积失真
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
FPS = 30
SIZE = 512
R_INNER = 2.0
R_OUTER = 15.0
ROTATION_SPEED = 0.1
SEED = 42
OUTPUT_PATH = "output/disk_rotation_exp2_parametric.mp4"

print("=" * 60)
print("实验 2: 参数化旋转纹理")
print("=" * 60)

# 渲染帧
print(f"\n渲染 {N_FRAMES} 帧（每帧重新生成纹理）...")
frames = []
total_texture_time = 0.0
total_render_time = 0.0

for frame in tqdm(range(N_FRAMES), desc="Rendering"):
    # 计算旋转偏移
    t_offset = frame * ROTATION_SPEED

    # 生成纹理（考虑旋转）
    t0_texture = time.time()
    texture = generate_disk_texture_topview(
        n_phi=512,
        n_r=128,
        seed=SEED,
        r_inner=R_INNER,
        r_outer=R_OUTER,
        enable_rt=True,
        t_offset=t_offset  # 参数化旋转
    )
    texture_time = time.time() - t0_texture
    total_texture_time += texture_time

    # 转换为笛卡尔坐标
    t0_render = time.time()
    image = polar_to_cartesian(texture, SIZE, R_INNER, R_OUTER)
    render_time = time.time() - t0_render
    total_render_time += render_time

    frame_time = texture_time + render_time

    # 添加文字覆盖层
    text_lines = [
        f"Experiment 2: Parametric",
        f"Method: Regenerate each frame",
        f"Frame: {frame}/{N_FRAMES}",
        f"t_offset: {t_offset:.2f}",
        f"Texture: {texture_time:.2f}s",
        f"Render: {render_time:.3f}s",
        f"Total: {total_texture_time + total_render_time:.1f}s",
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
print("实验 2 完成")
print("=" * 60)
print(f"总纹理生成时间: {total_texture_time:.2f}s")
print(f"总渲染时间: {total_render_time:.2f}s")
print(f"平均帧时间: {(total_texture_time + total_render_time)/N_FRAMES:.3f}s")
print(f"总时间: {total_texture_time + total_render_time:.2f}s")
print(f"输出: {OUTPUT_PATH}")
