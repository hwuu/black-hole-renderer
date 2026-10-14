"""HUD 遥控器叠加层：在渲染镜头角落画一个三旋钮面板，显示当前场景旋钮值。

设计见 `docs/design_film.md` §4.6。每个旋钮（引力 / 气体 / 多普勒）显示当前值：
半圆刻度盘 + 指针 + 数值。当前实现为简化版——每个镜头取动画中点处的旋钮值，
生成静态 PNG（含 alpha），由 assemble 在裁剪阶段叠加。

完整版（逐帧动画）需生成帧序列 → ProRes 4444 编码 → ffmpeg overlay，待后续迭代。
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# 面板尺寸与位置（相对画面右下角）
PANEL_W = 360
PANEL_H = 160
PANEL_MARGIN = 24
DIAL_R = 42
DIAL_SPACING = 110
BG_ALPHA = 180

# 字体（macOS 自带，Linux 退回 DejaVu）
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# 旋钮顺序与标签
_KNOB_LABELS = [
    ("gravity_scale", "引力"),
    ("disk_weight", "气体"),
    ("doppler_scale", "多普勒"),
]


def _load_font(size: int):
    """加载中文字体；找不到时退回到任意可用字体。"""
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_dial(draw: ImageDraw.ImageDraw, cx: float, cy: float, radius: float,
               value: float, label: str, font, font_small) -> None:
    """画一个半圆刻度盘 + 指针。

    Args:
        draw: PIL ImageDraw 对象。
        cx, cy: 刻度盘中心坐标。
        radius: 刻度盘半径（像素）。
        value: 旋钮值 ∈ [0, 1]。
        label: 旋钮标签（如"引力"）。
        font: 标签字体。
        font_small: 数值字体。
    """
    # 半圆弧（180° → 0°，从左到右）
    arc_bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.arc(arc_bbox, start=180, end=0, fill="#606060", width=3)

    # 刻度线（5 格）
    for i in range(6):
        angle = math.pi - i * math.pi / 5
        x1 = cx + (radius - 6) * math.cos(angle)
        y1 = cy - (radius - 6) * math.sin(angle)
        x2 = cx + (radius + 2) * math.cos(angle)
        y2 = cy - (radius + 2) * math.sin(angle)
        draw.line([(x1, y1), (x2, y2)], fill="#808080", width=1)

    # 指针（从圆心指向 value 对应的角度）
    needle_angle = math.pi * (1.0 - value)
    nx = cx + (radius - 12) * math.cos(needle_angle)
    ny = cy - (radius - 12) * math.sin(needle_angle)
    draw.line([(cx, cy), (nx, ny)], fill="#FFB347", width=3)
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill="#FFB347")

    # 标签与数值
    draw.text((cx, cy + radius + 14), label, font=font, fill="white", anchor="mm")
    draw.text((cx, cy + radius + 38), f"{value:.2f}", font=font_small, fill="#FFB347", anchor="mm")


def generate_hud_image(knob_values: Dict[str, float], canvas_w: int, canvas_h: int,
                       output_path: str) -> str:
    """生成一张带 alpha 通道的 HUD 遥控器面板图。

    Args:
        knob_values: 旋钮名 → 值（如 `{"gravity_scale": 0.5, ...}`）。
        canvas_w, canvas_h: 成片画幅（像素），用于定位面板。
        output_path: 输出 PNG 路径。

    Returns:
        同 `output_path`。
    """
    font = _load_font(20)
    font_small = _load_font(16)

    # 面板位于画面右下角
    px = canvas_w - PANEL_W - PANEL_MARGIN
    py = canvas_h - PANEL_H - PANEL_MARGIN

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 半透明圆角背景
    draw.rounded_rectangle(
        (px, py, px + PANEL_W, py + PANEL_H),
        radius=12, fill=(20, 20, 30, BG_ALPHA),
    )

    # 三个刻度盘横排
    dial_cy = py + 60
    for i, (knob_name, label) in enumerate(_KNOB_LABELS):
        dial_cx = px + 60 + i * DIAL_SPACING
        value = knob_values.get(knob_name, 1.0)
        _draw_dial(draw, dial_cx, dial_cy, DIAL_R, value, label, font, font_small)

    img.save(output_path)
    return output_path


def read_knob_values(path_json: str, shot_id: str, t: float) -> Dict[str, float]:
    """从运镜路径文件读取 t 时刻的旋钮值。

    Args:
        path_json: 运镜路径 JSON 文件路径。
        shot_id: 镜头 id（用于错误信息）。
        t: 视频时刻（秒）。

    Returns:
        `{knob_name: value}`；无 `scene_anim` 块时返回空 dict（全部取默认 1.0）。
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(path_json)))))
    from src.v2.scene_anim import load_scene_anim, knob_values_at

    with open(path_json, encoding="utf-8") as f:
        spec = json.load(f)
    anim = load_scene_anim(spec)
    return knob_values_at(anim, t, {})
