#!/usr/bin/env python3
"""
运行所有实验并生成对比报告

运行顺序: 1 → 3 → 2
输出目录: 当前脚本所在目录的 output/ 子目录
"""

import subprocess
import sys
import time
import os

# 实验顺序: 1 → 3 → 2
experiments = [
    ("experiment_1_baseline.py", "Baseline - 固定纹理 + 刚体旋转", "disk_rotation_exp1_baseline.mp4"),
    ("experiment_3_keyframes.py", "关键帧插值", "disk_rotation_exp3_keyframes.mp4"),
    ("experiment_2_parametric.py", "参数化旋转纹理", "disk_rotation_exp2_parametric.mp4"),
]

def main():
    # 获取脚本所在目录和输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("吸积盘旋转方案实验 - 运行所有实验")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print("运行顺序: 1 → 3 → 2")
    print("=" * 80)
    
    results = []
    
    for script, description, output_file in experiments:
        output_path = os.path.join(output_dir, output_file)
        
        print(f"\n\n{'=' * 80}")
        print(f"实验: {description}")
        print(f"脚本: {script}")
        print(f"输出: {output_path}")
        print('=' * 80)
        
        t0 = time.time()
        try:
            subprocess.run(
                [sys.executable, script],
                cwd=script_dir,
                check=True,
                capture_output=False
            )
            elapsed = time.time() - t0
            results.append((description, elapsed, "成功"))
            print(f"\n✅ 完成: {description} ({elapsed:.1f}s)")
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            results.append((description, elapsed, "失败"))
            print(f"\n❌ 错误: {description} - {e}")
    
    # 汇总报告
    print("\n\n" + "=" * 80)
    print("实验完成 - 结果汇总")
    print("=" * 80)
    
    total_time = 0
    for desc, elapsed, status in results:
        print(f"\n{desc}:")
        print(f"  状态: {status}")
        if elapsed > 0:
            total_time += elapsed
            print(f"  时间: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    
    print(f"\n总运行时间: {total_time:.1f}s ({total_time/60:.1f}min)")
    
    print("\n" + "=" * 80)
    print("输出视频:")
    print("=" * 80)
    for script, description, output_file in experiments:
        output_path = os.path.join(output_dir, output_file)
        exists = "✓" if os.path.exists(output_path) else "✗"
        print(f"  {exists} {output_path}")

if __name__ == "__main__":
    main()