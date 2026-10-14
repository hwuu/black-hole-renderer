"""s05_light_paths：为什么黑洞戴着"帽子"（光路图解释）。

旁白（3 句，约 20 s）：
1. 为什么黑洞会戴帽子呢？
2. 因为引力把光掰弯了：盘子背面的光，绕过黑洞的头顶，跑进了我们的眼睛。
3. 所以我们能同时看到盘子的正面、背面和底下——黑影比黑洞本身大两倍多。

动画：侧视图——黑洞在中间，吸积盘平面横穿，来自盘背面的光线被引力掰弯
绕过头顶进入左边的"眼睛"（相机）。被吞掉的光线画成暗红色。
"""

import numpy as np

from manim import *

from src.film.diagram_kit import (
    FONT, FONT_SIZE_TITLE, FONT_SIZE_BODY, FONT_SIZE_SMALL,
    COLOR_TEXT, COLOR_ACCENT, COLOR_SECONDARY, COLOR_DIM, COLOR_HIGHLIGHT,
    COLOR_NEGATIVE, COLOR_POSITIVE, load_timing, sentence_starts,
    light_path_lines,
)


class LightPaths(Scene):
    """光路图：盘背面的光绕过黑洞头顶进入眼睛。"""

    def construct(self):
        timing = load_timing("s05_light_paths")
        starts = sentence_starts("s05_light_paths")
        total = timing["duration"]
        self.camera.background_color = "#0d0d1a"

        # ---------------------------------------------------------------
        # 布局：黑洞在画面中心，眼睛在左侧
        # ---------------------------------------------------------------
        bh_pos = ORIGIN + RIGHT * 1.0
        eye_pos = ORIGIN + LEFT * 5.5
        bh_r = 0.5
        shadow_r = bh_r * 2.5

        # 黑洞
        bh = Circle(radius=bh_r, color="#1a1a1a", fill_opacity=1.0)
        bh.set_stroke(COLOR_TEXT, 1.5)
        bh.move_to(bh_pos)
        bh_label = Text("黑洞", font=FONT, font_size=FONT_SIZE_SMALL, color=COLOR_DIM)
        bh_label.next_to(bh, UP, buff=0.4)

        # 阴影（虚线圆）
        shadow = Circle(radius=shadow_r, color=COLOR_DIM, stroke_opacity=0.4)
        shadow.move_to(bh_pos)
        shadow.set_stroke(width=1.5)
        shadow_label = Text("黑影（比黑洞大 2.5 倍）", font=FONT, font_size=FONT_SIZE_SMALL,
                           color=COLOR_DIM)
        shadow_label.next_to(shadow, DOWN, buff=0.5)

        # 吸积盘（侧视：一条横线）
        disk_y = bh_pos[1]
        disk = Line(
            bh_pos + LEFT * shadow_r * 1.2 + UP * 0,
            bh_pos + RIGHT * shadow_r * 1.2 + UP * 0,
            color=COLOR_ACCENT, stroke_width=6,
        )
        disk_back = Text("盘背面", font=FONT, font_size=FONT_SIZE_SMALL, color=COLOR_ACCENT)
        disk_back.next_to(disk.get_end(), DOWN, buff=0.3)

        # 眼睛（相机）
        eye = Text("眼睛", font=FONT, font_size=FONT_SIZE_BODY, color=COLOR_TEXT)
        eye.move_to(eye_pos)
        eye_dot = Dot(eye_pos, radius=0.15, color=COLOR_TEXT)

        # ---------------------------------------------------------------
        # 段落 1：为什么戴帽子？（~3s）
        # ---------------------------------------------------------------
        title = Text("为什么黑洞戴着帽子？", font=FONT, font_size=FONT_SIZE_TITLE,
                     color=COLOR_TEXT)
        title.move_to(ORIGIN + UP * 2.5)

        self.play(Write(title), run_time=1.0)
        self.wait(max(0, starts[1] - self.renderer.time - 0.3))
        self.play(FadeOut(title), run_time=0.3)

        # ---------------------------------------------------------------
        # 段落 2：引力把光掰弯（~8s）—— 核心动画
        # ---------------------------------------------------------------
        self.play(FadeIn(bh), FadeIn(bh_label), FadeIn(shadow), FadeIn(shadow_label),
                  FadeIn(disk), FadeIn(disk_back), FadeIn(eye), FadeIn(eye_dot),
                  run_time=0.8)

        # 来自盘背面的光线——绕过黑洞上方进入眼睛
        # 使用测地线积分画真实弯曲路径
        back_rays = light_path_lines(
            [3.2, 4.0],  # 刚好在光子环附近——弯曲最大
            gravity_scale=1.0,
            start_r=10, scale=0.22,
            center=bh_pos,
            color=COLOR_ACCENT, stroke_width=3,
        )
        # 被吞掉的光线
        swallowed_rays = light_path_lines(
            [1.5, 2.0],  # 冲击参数太小——落入视界
            gravity_scale=1.0,
            start_r=10, scale=0.22,
            center=bh_pos,
            color=COLOR_NEGATIVE, stroke_width=2,
        )
        ray_labels = VGroup(
            Text("盘背面的光", font=FONT, font_size=FONT_SIZE_SMALL, color=COLOR_ACCENT),
            Text("绕过头顶", font=FONT, font_size=FONT_SIZE_SMALL, color=COLOR_ACCENT),
        )
        ray_labels[0].next_to(bh, UP, buff=1.8).shift(LEFT * 1)
        ray_labels[1].next_to(ray_labels[0], RIGHT, buff=0.5)

        explanation = Text("引力把光掰弯了", font=FONT, font_size=FONT_SIZE_BODY,
                           color=COLOR_TEXT)
        explanation.move_to(ORIGIN + UP * 2.5)

        self.play(Write(explanation), run_time=0.5)
        self.play(LaggedStart(*[Create(r) for r in back_rays], lag_ratio=0.3), run_time=1.5)
        self.play(FadeIn(ray_labels[0]), run_time=0.3)
        self.play(FadeIn(ray_labels[1]), run_time=0.3)
        self.wait(0.5)
        self.play(LaggedStart(*[Create(r) for r in swallowed_rays], lag_ratio=0.2), run_time=1.0)

        self.wait(max(0, starts[2] - self.renderer.time - 0.3))

        # ---------------------------------------------------------------
        # 段落 3：黑影比黑洞大两倍多（~9s）
        # ---------------------------------------------------------------
        result = VGroup(
            Text("正面 ✓", font=FONT, font_size=FONT_SIZE_SMALL, color=COLOR_POSITIVE),
            Text("背面 ✓", font=FONT, font_size=FONT_SIZE_SMALL, color=COLOR_POSITIVE),
            Text("底下 ✓", font=FONT, font_size=FONT_SIZE_SMALL, color=COLOR_POSITIVE),
        ).arrange(RIGHT, buff=0.8)
        result.move_to(ORIGIN + DOWN * 2.0)

        size_note = Text("黑影 ≈ 黑洞本身 × 2.5", font=FONT, font_size=FONT_SIZE_BODY,
                        color=COLOR_HIGHLIGHT)
        size_note.next_to(shadow_label, DOWN, buff=0.5)

        self.play(FadeIn(result), run_time=0.5)
        self.play(Write(size_note), run_time=0.5)
        self.wait(max(0.5, total - self.renderer.time - 0.3))
