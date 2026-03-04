#!/usr/bin/env python3
"""
验证湍流旋转修复 - 生成多帧对比图
"""

import numpy as np
import os
from common import generate_disk_texture_topview, polar_to_cartesian
from PIL import Image

print("=" * 60)
print("验证湍流旋转修复")
print("=" * 60)

SIZE = 512
R_INNER = 2.0
R_OUTER = 15.0
SEED = 42

# 生成 4 个不同 t_offset 的帧
offsets = [0.0, 0.25, 0.5, 0.75]

# 确保输出目录存在
os.makedirs("output", exist_ok=True)

# 清理旧文件
for t_offset in offsets:
    path = f"output/verify_t{t_offset:.2f}.png"
    if os.path.exists(path):
        os.remove(path)
comparison_path = "output/verify_comparison.png"
if os.path.exists(comparison_path):
    os.remove(comparison_path)

images = []

for i, t_offset in enumerate(offsets):
    print(f"\n生成帧 {i}: t_offset = {t_offset}")
    texture = generate_disk_texture_topview(
        n_phi=512,
        n_r=128,
        seed=SEED,
        r_inner=R_INNER,
        r_outer=R_OUTER,
        enable_rt=True,
        t_offset=t_offset
    )
    image = polar_to_cartesian(texture, SIZE, R_INNER, R_OUTER)
    images.append(image)
    
    # 立即保存单帧
    path = f"output/verify_t{t_offset:.2f}.png"
    Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)
    print(f"  已保存: {path}")

# 保存对比图（4 帧横向拼接）
print("\n保存对比图...")
combined = np.hstack(images)
Image.fromarray((np.clip(combined, 0, 1) * 255).astype(np.uint8)).save(comparison_path)
print(f"  已保存: {comparison_path}")

print("\n" + "=" * 60)
print("验证完成！")
print("=" * 60)
print("\n请打开 output/verify_comparison.png 查看对比")
print("如果湍流云雾随 t_offset 旋转，说明修复成功")
