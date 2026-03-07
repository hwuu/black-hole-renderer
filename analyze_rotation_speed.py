#!/usr/bin/env python3
"""分析不同旋转速度的失真程度"""

import numpy as np

def analyze_rotation(speed_factor, n_frames, r_inner=2.0, r_outer=15.0):
    """
    分析给定速度系数下的旋转情况

    Args:
        speed_factor: 旋转速度系数（t_offset = frame * speed_factor）
        n_frames: 总帧数
        r_inner: 内半径
        r_outer: 外半径
    """
    # 最后一帧的 t_offset
    t_offset_final = (n_frames - 1) * speed_factor

    # 内圈和外圈的角速度
    omega_inner = np.sqrt(0.5 / r_inner)
    omega_outer = np.sqrt(0.5 / r_outer)

    # 旋转角度（弧度）
    phi_inner = t_offset_final * omega_inner
    phi_outer = t_offset_final * omega_outer

    # 旋转圈数
    rotations_inner = phi_inner / (2 * np.pi)
    rotations_outer = phi_outer / (2 * np.pi)

    # 差分旋转（内外圈的差异）
    diff_rotations = rotations_inner - rotations_outer
    diff_degrees = diff_rotations * 360

    return {
        'speed_factor': speed_factor,
        'n_frames': n_frames,
        't_offset_final': t_offset_final,
        'rotations_inner': rotations_inner,
        'rotations_outer': rotations_outer,
        'diff_rotations': diff_rotations,
        'diff_degrees': diff_degrees,
    }

def assess_distortion(diff_degrees):
    """评估失真程度"""
    if diff_degrees < 10:
        return "[OK] 几乎无失真"
    elif diff_degrees < 30:
        return "[OK] 轻微失真，可接受"
    elif diff_degrees < 90:
        return "[WARN] 明显失真，短视频可接受"
    elif diff_degrees < 180:
        return "[WARN] 严重失真，长视频不推荐"
    else:
        return "[ERROR] 极度失真，不可用"

print("=" * 80)
print("吸积盘旋转速度分析")
print("=" * 80)

# 测试不同的速度系数和帧数组合
test_cases = [
    # (speed_factor, n_frames, description)
    (0.1, 3600, "当前实现（3600 帧长视频）"),
    (0.01, 3600, "降低 10 倍（3600 帧）"),
    (0.005, 3600, "降低 20 倍（3600 帧）"),
    (0.001, 3600, "降低 100 倍（3600 帧）"),
    (0.0005, 3600, "降低 200 倍（3600 帧）"),
    (0.0, 3600, "禁用旋转（3600 帧）"),
    (0.01, 360, "降低 10 倍（360 帧短视频）"),
    (0.005, 360, "降低 20 倍（360 帧）"),
]

for speed, frames, desc in test_cases:
    result = analyze_rotation(speed, frames)
    assessment = assess_distortion(result['diff_degrees'])

    print(f"\n{desc}")
    print(f"  速度系数: {speed}")
    print(f"  总帧数: {frames}")
    print(f"  最终 t_offset: {result['t_offset_final']:.2f}")
    print(f"  内圈旋转: {result['rotations_inner']:.2f} 圈 ({result['rotations_inner']*360:.0f}°)")
    print(f"  外圈旋转: {result['rotations_outer']:.2f} 圈 ({result['rotations_outer']*360:.0f}°)")
    print(f"  差分旋转: {result['diff_rotations']:.2f} 圈 ({result['diff_degrees']:.0f}°)")
    print(f"  失真评估: {assessment}")

print("\n" + "=" * 80)
print("推荐配置")
print("=" * 80)

recommendations = [
    ("短视频（360 帧，10 秒 @ 36fps）", 0.01, 360),
    ("中等视频（1800 帧，50 秒）", 0.005, 1800),
    ("长视频（3600 帧，100 秒）", 0.001, 3600),
    ("超长视频（7200 帧，200 秒）", 0.0005, 7200),
]

for desc, speed, frames in recommendations:
    result = analyze_rotation(speed, frames)
    print(f"\n{desc}:")
    print(f"  推荐速度系数: {speed}")
    print(f"  差分旋转: {result['diff_degrees']:.0f}°")
    print(f"  评估: {assess_distortion(result['diff_degrees'])}")

print("\n" + "=" * 80)
print("速度下限分析")
print("=" * 80)

print("""
理论上，速度可以无限接近 0，但：

1. **视觉效果**：
   - 速度太慢 → 吸积盘看起来像静止
   - 失去动态感
   - 建议：至少让外圈旋转 0.1-0.5 圈

2. **物理合理性**：
   - 真实吸积盘的旋转周期：数小时到数天
   - 我们的视频：通常 10-100 秒
   - 可以认为是"慢动作"观察

3. **实用建议**：
   - 差分旋转 < 30° → 失真不明显
   - 差分旋转 < 90° → 可接受
   - 差分旋转 > 180° → 不推荐

4. **极限情况**：
   - 速度系数 = 0 → 完全静止，无失真但无动态
   - 速度系数 → ∞ → 极度失真，不可用

5. **推荐范围**：
   - 最小：0.0005（几乎静止）
   - 最大：0.01（短视频）
   - 最佳：0.001-0.005（根据视频长度调整）
""")

print("\n" + "=" * 80)
print("CLI 参数建议")
print("=" * 80)

print("""
添加可配置参数：

```python
parser.add_argument("--disk_rotation_speed", type=float, default=0.005,
                    help="吸积盘旋转速度系数 (0=禁用, 推荐: 0.001-0.01)")
```

使用示例：

# 短视频（360 帧）
python render.py --video --n_frames 360 --disk_rotation_speed 0.01

# 长视频（3600 帧）
python render.py --video --n_frames 3600 --disk_rotation_speed 0.001

# 禁用旋转
python render.py --video --n_frames 3600 --disk_rotation_speed 0
""")
