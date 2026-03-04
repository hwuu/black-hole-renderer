#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2E 渲染测试
用固定参数渲染图像，验证产出一致性（通过 image hash）
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from render import render_image

BASELINE_FILE = os.path.join(os.path.dirname(__file__), "e2e_baseline.txt")


def compute_image_hash(img: np.ndarray) -> str:
    """计算图像的 MD5 hash"""
    return hashlib.md5(img.tobytes()).hexdigest()


def render_test_image():
    """用固定参数渲染测试图像"""
    img = render_image(
        width=320,
        height=180,
        cam_pos=[6, 0, 0.5],
        fov=60,
        step_size=0.1,
        r_max=10,
        device="cpu",
        n_stars=100,
        r_disk_inner=2.0,
        r_disk_outer=3.5,
        disk_tilt=15,
        lens_flare=False,
        anti_alias="disabled",
        force_regenerate_disk_texture=True,
        ignore_taichi_cache=True,
    )
    return img


def generate_baseline():
    """生成 baseline hash"""
    print("Generating baseline...")
    img = render_test_image()
    hash_val = compute_image_hash(img)
    with open(BASELINE_FILE, "w") as f:
        f.write(hash_val)
    print(f"Baseline hash: {hash_val}")
    print(f"Saved to: {BASELINE_FILE}")
    return hash_val


def verify():
    """验证当前代码产出与 baseline 一致"""
    if not os.path.exists(BASELINE_FILE):
        print("ERROR: Baseline file not found. Run with --generate first.")
        return False

    with open(BASELINE_FILE, "r") as f:
        baseline_hash = f.read().strip()

    print("Running verification...")
    img = render_test_image()
    current_hash = compute_image_hash(img)

    print(f"Baseline hash: {baseline_hash}")
    print(f"Current hash:  {current_hash}")

    if current_hash == baseline_hash:
        print("PASS: Hash matches baseline")
        return True
    else:
        print("FAIL: Hash mismatch!")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E2E render test")
    parser.add_argument("--generate", action="store_true", help="Generate baseline")
    parser.add_argument("--verify", action="store_true", help="Verify against baseline")
    args = parser.parse_args()

    if args.generate:
        generate_baseline()
    elif args.verify:
        success = verify()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)
