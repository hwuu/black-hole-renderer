#!/usr/bin/env python3
"""测试吸积盘旋转问题"""

import sys
import numpy as np

# 模拟吸积盘采样逻辑
def simulate_disk_sampling(frame, r):
    """
    模拟吸积盘纹理采样时的 phi 计算

    Args:
        frame: 帧编号
        r: 半径（单位：史瓦西半径）

    Returns:
        phi: 采样的角度（弧度）
    """
    # 当前代码的逻辑
    t_offset = frame * 0.1
    omega = np.sqrt(0.5 / r)  # 开普勒角速度

    # 假设初始 phi = 0
    phi_initial = 0
    phi = phi_initial + t_offset * omega

    # 计算旋转了多少圈
    rotations = phi / (2 * np.pi)

    return phi, rotations, omega

print("=" * 60)
print("吸积盘旋转测试")
print("=" * 60)

# 测试不同帧数和半径
test_cases = [
    (0, 2.0, "第 0 帧，内圈 (r=2)"),
    (0, 15.0, "第 0 帧，外圈 (r=15)"),
    (3599, 2.0, "第 3599 帧，内圈 (r=2)"),
    (3599, 15.0, "第 3599 帧，外圈 (r=15)"),
    (360, 2.0, "第 360 帧，内圈 (r=2)"),
    (360, 15.0, "第 360 帧，外圈 (r=15)"),
]

for frame, r, desc in test_cases:
    phi, rotations, omega = simulate_disk_sampling(frame, r)
    print(f"\n{desc}:")
    print(f"  t_offset = {frame * 0.1:.1f}")
    print(f"  omega = {omega:.4f} rad/unit")
    print(f"  phi = {phi:.2f} rad = {np.degrees(phi):.1f}°")
    print(f"  旋转圈数 = {rotations:.1f} 圈")

print("\n" + "=" * 60)
print("问题分析：")
print("=" * 60)
print("""
当前实现：t_offset = frame * 0.1
         phi = phi_initial + t_offset * omega

问题：
1. 第 3599 帧时，t_offset = 359.9，这个值太大了
2. 内圈（r=2）旋转约 40 圈，外圈（r=15）旋转约 15 圈
3. 不同半径旋转速度不同（开普勒旋转），导致纹理严重错位

建议修复：
1. 降低旋转速度：t_offset = frame * 0.01（或更小）
2. 或者使用归一化：t_offset = (frame / n_frames) * 2 * pi
   这样一个完整视频周期，吸积盘旋转一圈
""")
