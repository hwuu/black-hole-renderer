#!/usr/bin/env python3
"""
快速测试 - 只生成 5 帧来验证代码是否工作
"""

import numpy as np
import time
from common import (
    generate_disk_texture_topview,
    sample_disk_with_rotation,
    polar_to_cartesian,
    add_text_overlay,
)
from PIL import Image

print("=" * 60)
print("快速测试 - 验证实验代码")
print("=" * 60)

# 参数
SIZE = 256  # 更小的尺寸以加快速度
R_INNER = 2.0
R_OUTER = 15.0
SEED = 42

# 测试 1: 生成纹理
print("\n测试 1: 生成纹理...")
t0 = time.time()
texture = generate_disk_texture_topview(
    n_phi=256,
    n_r=64,
    seed=SEED,
    r_inner=R_INNER,
    r_outer=R_OUTER,
    enable_rt=True,
    t_offset=0.0
)
print(f"  完成: {time.time() - t0:.2f}s")
print(f"  纹理形状: {texture.shape}")

# 测试 2: Baseline 采样
print("\n测试 2: Baseline 采样（旋转）...")
t0 = time.time()
image1 = sample_disk_with_rotation(texture, 0.0, R_INNER, R_OUTER, SIZE)
print(f"  完成: {time.time() - t0:.3f}s")
print(f"  图像形状: {image1.shape}")

# 测试 3: 参数化旋转
print("\n测试 3: 参数化旋转...")
t0 = time.time()
texture2 = generate_disk_texture_topview(
    n_phi=256,
    n_r=64,
    seed=SEED,
    r_inner=R_INNER,
    r_outer=R_OUTER,
    enable_rt=True,
    t_offset=1.0  # 旋转偏移
)
image2 = polar_to_cartesian(texture2, SIZE, R_INNER, R_OUTER)
print(f"  完成: {time.time() - t0:.2f}s")

# 测试 4: 添加文字
print("\n测试 4: 添加文字覆盖层...")
text_lines = [
    "Test Frame",
    "Method: Quick test",
    "Frame: 0/5",
    "Time: 0.123s",
]
image_with_text = add_text_overlay(image1, text_lines)
print(f"  完成")

# 保存测试图像
print("\n保存测试图像...")
Image.fromarray((np.clip(image1, 0, 1) * 255).astype(np.uint8)).save(
    "output/test_baseline.png"
)
Image.fromarray((np.clip(image2, 0, 1) * 255).astype(np.uint8)).save(
    "output/test_parametric.png"
)
Image.fromarray((np.clip(image_with_text, 0, 1) * 255).astype(np.uint8)).save(
    "output/test_with_text.png"
)

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
print("输出:")
print("  output/test_baseline.png")
print("  output/test_parametric.png")
print("  output/test_with_text.png")
print("\n如果这些图像看起来正常，可以运行完整实验:")
print("  python disk_rotation_experiments/experiment_1_baseline.py")
print("  python disk_rotation_experiments/experiment_2_parametric.py")
print("  python disk_rotation_experiments/experiment_3_keyframes.py")
print("\n或运行所有实验:")
print("  python disk_rotation_experiments/run_all.py")
