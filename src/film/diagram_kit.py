"""制片层图解工具包：Manim 场景共用的样式、时序读取与物理绘图辅助。

设计见 `docs/design_film.md` §4.8。内容层的 Manim 场景类通过 import 本模块获得
统一样式（字体、配色、字号）和 `load_timing`（读取 timeline.json 中该镜头的
时长与句子时刻，用于把动画节拍对齐旁白）。
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np

from manim import WHITE, BLACK, RED, BLUE_C, YELLOW_C, GREEN_C, ORANGE, PURPLE_C, \
    RIGHT, LEFT, UP, DOWN, ORIGIN, PI, TAU, DEGREES

from src.core.geodesic import integrate_ray

# ---------------------------------------------------------------------------
# 统一样式
# ---------------------------------------------------------------------------

FONT = "PingFang SC"
FONT_SIZE_TITLE = 56
FONT_SIZE_BODY = 44
FONT_SIZE_SMALL = 32

COLOR_BG = BLACK
COLOR_TEXT = WHITE
COLOR_ACCENT = ORANGE          # 强调色（暖橙）
COLOR_SECONDARY = BLUE_C       # 辅助色（蓝）
COLOR_DIM = "#808080"           # 弱化色（灰）
COLOR_HIGHLIGHT = YELLOW_C      # 高亮色
COLOR_NEGATIVE = RED           # 否定/危险
COLOR_POSITIVE = GREEN_C       # 肯定/安全
COLOR_GAS = PURPLE_C            # 气体/云雾

# ---------------------------------------------------------------------------
# 时序读取
# ---------------------------------------------------------------------------

_TIMELINE_CACHE = None


def load_timing(shot_id: str) -> dict:
    """从 timeline.json 读取该镜头的时长与句子时刻。

    Args:
        shot_id: 镜头 id（如 `"s02_escape"`）。

    Returns:
        `{"id": …, "start": …, "duration": …, "sentences": [{index, text, start, end}, …]}`。

    Raises:
        ValueError: 镜头不在 timeline.json 中；或文件不可读。

    Notes:
        timeline.json 的位置由环境变量 `FILM_OUTPUT_DIR` 给出（`<产物目录>`，
        由 `__main__.py` 的 `diagrams` 子命令设置），默认 `output/scenes/kids_explainer/`。
    """
    global _TIMELINE_CACHE
    output_dir = os.environ.get("FILM_OUTPUT_DIR", os.path.join("output", "scenes", "kids_explainer"))
    path = os.path.join(output_dir, "timeline.json")
    if _TIMELINE_CACHE is None:
        with open(path, encoding="utf-8") as f:
            _TIMELINE_CACHE = json.load(f)
    for shot in _TIMELINE_CACHE["shots"]:
        if shot["id"] == shot_id:
            return shot
    raise ValueError(f"镜头 {shot_id!r} 不在 timeline.json 中")


def sentence_starts(shot_id: str) -> List[float]:
    """该镜头各句旁白相对镜头起点的时刻（秒，按句序）。

    Args:
        shot_id: 镜头 id。

    Returns:
        各句 start 时刻的列表（相对镜头起点；用 `start - shot["start"]` 偏移）。
    """
    shot = load_timing(shot_id)
    base = shot["start"]
    return [s["start"] - base for s in shot["sentences"]]


# ---------------------------------------------------------------------------
# 物理绘图辅助
# ---------------------------------------------------------------------------

def light_path_lines(
    impact_parameters: List[float],
    gravity_scale: float = 1.0,
    start_r: float = 40.0,
    scale: float = 0.1,
    center: np.ndarray = ORIGIN,
    color=COLOR_ACCENT,
    stroke_width: float = 3.0,
) -> List:
    """用测地线方程积分一组光线轨迹，返回可直接加入 Manim 场景的折线列表。

    Args:
        impact_parameters: 冲击参数列表（r_s）；每条光线从 x = −start_r 处水平射入。
        gravity_scale: 引力强度 k（1 = 真实黑洞）。
        start_r: 光线起点距离（r_s）。
        scale: 世界坐标 → Manim 坐标的缩放因子（Manim 的可视范围约 ±7）。
        center: 黑洞在 Manim 画面中的位置。
        color: 线条颜色。
        stroke_width: 线条宽度。

    Returns:
        Manim `VMobject` 列表（每条光线一个 `set_points_as_corners` 折线）。
    """
    from manim import VMobject

    lines = []
    for b in impact_parameters:
        start = np.array([-start_r, b, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        positions, outcome, _ = integrate_ray(start, direction,
                                              gravity_scale=gravity_scale,
                                              r_escape=start_r * 1.5)
        # 只取有意义的段（避免轨迹过长撑爆画面）
        positions = positions[::4]  # 降采样
        max_points = 200
        if len(positions) > max_points:
            positions = positions[:max_points]
        pts = [center + np.array([p[0], p[1], 0.0]) * scale for p in positions]
        line = VMobject().set_points_as_corners(pts)
        line.set_stroke(color, stroke_width)
        lines.append(line)
    return lines
