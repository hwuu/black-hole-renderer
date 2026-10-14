"""s02_escape：逃逸速度（扔球 → 火箭 → 地球压成弹珠 → 光也逃不掉）。

旁白（5 句，约 26 s）：
1. 往天上扔一个球，它会掉下来。
2. 火箭飞得足够快，就能离开地球。
3. 这个刚好能离开的速度，叫逃逸速度。
4. 要是把整个地球，压成一颗不到两厘米的玻璃弹珠呢？
5. 那时逃逸速度会超过光速，连光也跑不出来了。

动画按句子节拍同步：每句旁白开始时切入对应的视觉段落。
"""

import numpy as np

from manim import *

from src.film.diagram_kit import (
    FONT, FONT_SIZE_TITLE, FONT_SIZE_BODY, FONT_SIZE_SMALL,
    COLOR_TEXT, COLOR_ACCENT, COLOR_SECONDARY, COLOR_DIM, COLOR_HIGHLIGHT,
    COLOR_NEGATIVE, COLOR_POSITIVE, COLOR_GAS, load_timing, sentence_starts,
)


class Escape(Scene):
    """逃逸速度：球 → 火箭 → 压缩地球 → 光也逃不掉。"""

    def construct(self):
        timing = load_timing("s02_escape")
        starts = sentence_starts("s02_escape")
        total = timing["duration"]

        # ---------------------------------------------------------------
        # 全局背景
        # ---------------------------------------------------------------
        self.camera.background_color = "#0d0d1a"

        # ---------------------------------------------------------------
        # 段落 1：扔球落回（~3s）
        # ---------------------------------------------------------------
        ground_y = -2.5
        ball_r = 0.25
        ball = Dot(ORIGIN + DOWN * 2.5 + LEFT * 0.5, radius=ball_r, color=ORANGE)
        ground = self._ground_line(ground_y)
        label_ball = Text("扔出去的球", font=FONT, font_size=FONT_SIZE_SMALL,
                           color=COLOR_DIM).next_to(ball, DOWN, buff=0.3)
        trajectory = Arc(radius=2.5, start_angle=PI * 0.2, angle=PI * 0.6,
                          arc_center=ORIGIN + DOWN * 2.0 + LEFT * 0.5)
        trajectory.set_stroke(COLOR_ACCENT, 3)

        self.play(FadeIn(ground), FadeIn(ball), FadeIn(label_ball), run_time=0.5)
        self.play(
            ball.animate.move_to(ORIGIN + UP * 1.5 + RIGHT * 0.8),
            run_time=0.8,
            rate_func=lambda t: np.sin(t * PI / 2),
        )
        self.play(
            ball.animate.move_to(ORIGIN + DOWN * 2.5 + RIGHT * 2.5),
            run_time=0.8,
            rate_func=lambda t: np.cos(t * PI / 2),
        )
        self.wait(starts[1] - self.renderer.time - 0.1 if self.renderer.time < starts[1] else 0.5)
        self.play(FadeOut(ball), FadeOut(trajectory), FadeOut(label_ball), run_time=0.3)

        # ---------------------------------------------------------------
        # 段落 2：火箭飞出（~4s）
        # ---------------------------------------------------------------
        earth = Circle(radius=1.2, color=BLUE_C, fill_opacity=0.3)
        earth_label = Text("地球", font=FONT, font_size=FONT_SIZE_BODY, color=COLOR_TEXT)
        earth_label.move_to(earth.get_center())
        rocket = self._rocket()
        rocket.next_to(earth, UP, buff=0.5)

        self.play(FadeIn(earth), FadeIn(earth_label), run_time=0.5)
        self.play(FadeIn(rocket), run_time=0.3)
        self.play(
            rocket.animate.shift(UP * 3.5),
            run_time=1.2,
            rate_func=lambda t: t ** 1.5,
        )
        self.play(FadeOut(rocket), run_time=0.2)
        self.wait(max(0, starts[2] - self.renderer.time - 0.3))
        self.play(FadeOut(earth), FadeOut(earth_label), run_time=0.3)

        # ---------------------------------------------------------------
        # 段落 3：逃逸速度概念（~5s）
        # ---------------------------------------------------------------
        title_ev = Text("逃逸速度", font=FONT, font_size=FONT_SIZE_TITLE, color=COLOR_HIGHLIGHT)
        title_ev.move_to(ORIGIN)
        sub = Text("刚好能离开星球的速度", font=FONT, font_size=FONT_SIZE_SMALL, color=COLOR_DIM)
        sub.next_to(title_ev, DOWN, buff=0.5)

        self.play(Write(title_ev), run_time=0.8)
        self.play(FadeIn(sub), run_time=0.3)
        self.wait(max(0, starts[3] - self.renderer.time - 0.3))
        self.play(FadeOut(title_ev), FadeOut(sub), run_time=0.3)

        # ---------------------------------------------------------------
        # 段落 4：地球压成弹珠（~6s）
        # ---------------------------------------------------------------
        earth2 = Circle(radius=2.0, color=BLUE_C, fill_opacity=0.3)
        earth2.move_to(ORIGIN)
        label_big = Text("地球", font=FONT, font_size=FONT_SIZE_BODY, color=COLOR_TEXT)
        label_big.move_to(earth2)

        marble = Circle(radius=0.12, color=BLUE_C, fill_opacity=0.8)
        marble.move_to(ORIGIN)
        label_small = Text("不到 2 厘米", font=FONT, font_size=FONT_SIZE_SMALL,
                           color=COLOR_HIGHLIGHT).next_to(marble, DOWN, buff=0.8)

        self.play(FadeIn(earth2), FadeIn(label_big), run_time=0.5)
        self.wait(0.5)
        self.play(
            Transform(earth2, marble),
            FadeOut(label_big),
            run_time=2.0,
        )
        self.play(FadeIn(label_small), run_time=0.3)
        self.wait(max(0, starts[4] - self.renderer.time - 0.3))

        # ---------------------------------------------------------------
        # 段落 5：光也逃不掉（~7s）
        # ---------------------------------------------------------------
        bh = Circle(radius=0.35, color=BLACK, fill_opacity=1.0)
        bh.set_stroke(WHITE, 2)
        bh_label = Text("连光也逃不掉", font=FONT, font_size=FONT_SIZE_TITLE,
                        color=COLOR_NEGATIVE)
        bh_label.next_to(bh, UP, buff=1.0)

        self.play(Transform(earth2, bh), FadeOut(label_small), run_time=0.5)
        self.play(Write(bh_label), run_time=0.8)

        # 几条光线弯回来
        from src.film.diagram_kit import light_path_lines
        ray_lines = light_path_lines(
            [2.0, 3.5, 5.0], gravity_scale=1.0,
            start_r=12, scale=0.25, center=ORIGIN,
            color=COLOR_ACCENT, stroke_width=2.5,
        )
        self.play(LaggedStart(*[Create(l) for l in ray_lines], lag_ratio=0.2), run_time=1.5)
        self.wait(max(0.5, total - self.renderer.time - 0.2))

    def _ground_line(self, y: float):
        """地面线。"""
        from manim import Line, RIGHT
        return Line(ORIGIN + LEFT * 6 + DOWN * (-y), ORIGIN + RIGHT * 6 + DOWN * (-y),
                    stroke_color=GREY_D, stroke_width=3)

    def _rocket(self):
        """简易火箭（三角形 + 矩形）。"""
        from manim import Polygon, Rectangle, VGroup
        body = Rectangle(width=0.3, height=0.8, color=GREY_A, fill_opacity=0.8)
        tip = Polygon(
            body.get_corner(UP + LEFT), body.get_corner(UP + RIGHT),
            body.get_center() + UP * 0.5,
            color=RED, fill_opacity=0.9,
        )
        return VGroup(body, tip)
