"""s06_siren：警笛多普勒（救护车驶过 → 声波前密后疏 → 光也一样）。

旁白（3 句，约 17 s）：
1. 先听一个熟悉的声音。
2. 救护车开过来的时候，声音又高又尖；开过去以后，马上变得低沉。
3. 光也一样：朝我们跑来的光被挤紧，跑开的光被拉开。

动画：救护车从左到右驶过，声波圆环在前方挤密（高频=尖）、后方拉疏（低频=钝）。
然后抽象为波浪线对比。
"""

import numpy as np

from manim import *

from src.film.diagram_kit import (
    FONT, FONT_SIZE_TITLE, FONT_SIZE_BODY, FONT_SIZE_SMALL,
    COLOR_TEXT, COLOR_ACCENT, COLOR_SECONDARY, COLOR_DIM, COLOR_HIGHLIGHT,
    COLOR_NEGATIVE, COLOR_POSITIVE, load_timing, sentence_starts,
)


class Siren(Scene):
    """警笛多普勒：救护车驶过 + 声波密度示意 + 波长对比。"""

    def construct(self):
        timing = load_timing("s06_siren")
        starts = sentence_starts("s06_siren")
        total = timing["duration"]
        self.camera.background_color = "#0d0d1a"

        # ---------------------------------------------------------------
        # 段落 1：先听一个声音（~2s）
        # ---------------------------------------------------------------
        title = Text("先听一个熟悉的声音", font=FONT, font_size=FONT_SIZE_TITLE,
                     color=COLOR_TEXT)
        title.move_to(ORIGIN)
        self.play(Write(title), run_time=0.8)
        self.wait(max(0, starts[1] - self.renderer.time - 0.3))
        self.play(FadeOut(title), run_time=0.3)

        # ---------------------------------------------------------------
        # 段落 2：救护车驶过（~6s）——声波圆环动画
        # ---------------------------------------------------------------
        car_y = 0.0
        road = Line(LEFT * 7 + DOWN * 1.5, RIGHT * 7 + DOWN * 1.5,
                     color="#333333", stroke_width=2)
        car = self._ambulance()
        car.move_to(ORIGIN + LEFT * 5 + DOWN * 0.8)

        # 声波圆环：前方挤密（间距小），后方拉疏（间距大）
        waves_front = VGroup()
        waves_back = VGroup()
        for i in range(5):
            ring_f = Circle(radius=0.5 + i * 0.35, color=COLOR_ACCENT,
                            stroke_width=2, stroke_opacity=0.7)
            ring_f.move_to(car.get_center() + RIGHT * (0.3 + i * 0.28))
            waves_front.add(ring_f)

            ring_b = Circle(radius=0.5 + i * 0.55, color=COLOR_SECONDARY,
                            stroke_width=2, stroke_opacity=0.7)
            ring_b.move_to(car.get_center() + LEFT * (0.5 + i * 0.65))
            waves_back.add(ring_b)

        label_near = Text("驶近：又高又尖", font=FONT, font_size=FONT_SIZE_SMALL,
                          color=COLOR_ACCENT)
        label_near.next_to(waves_front, UP, buff=0.5)
        label_far = Text("驶离：变得低沉", font=FONT, font_size=FONT_SIZE_SMALL,
                         color=COLOR_SECONDARY)
        label_far.next_to(waves_back, UP, buff=0.5)

        self.play(FadeIn(road), FadeIn(car), run_time=0.5)
        self.play(FadeIn(waves_front), FadeIn(waves_back), run_time=0.5)
        self.play(FadeIn(label_near), FadeIn(label_far), run_time=0.3)

        # 救护车向右移动
        self.play(
            car.animate.shift(RIGHT * 6.5),
            waves_front.animate.shift(RIGHT * 6.5),
            waves_back.animate.shift(RIGHT * 6.5),
            label_near.animate.shift(RIGHT * 6.5),
            label_far.animate.shift(RIGHT * 6.5),
            run_time=3.0,
            rate_func=lambda t: t,
        )

        self.wait(max(0, starts[2] - self.renderer.time - 0.3))
        self.play(
            FadeOut(car), FadeOut(road),
            FadeOut(waves_front), FadeOut(waves_back),
            FadeOut(label_near), FadeOut(label_far),
            run_time=0.3,
        )

        # ---------------------------------------------------------------
        # 段落 3：光也一样（~8s）——波长对比
        # ---------------------------------------------------------------
        light_title = Text("光也一样", font=FONT, font_size=FONT_SIZE_TITLE, color=COLOR_TEXT)
        light_title.move_to(ORIGIN + UP * 2.0)

        # 挤紧的波（逼近侧）
        wave_close = self._sine_wave(amplitude=0.4, wavelength=0.6, length=5.0,
                                    color=COLOR_ACCENT)
        wave_close.move_to(ORIGIN + UP * 0.5)
        label_close = Text("朝我们跑来 → 波被挤紧（更亮更蓝）", font=FONT,
                           font_size=FONT_SIZE_SMALL, color=COLOR_ACCENT)
        label_close.next_to(wave_close, UP, buff=0.3)

        # 拉开的波（远离侧）
        wave_far = self._sine_wave(amplitude=0.4, wavelength=1.4, length=5.0,
                                   color=COLOR_SECONDARY)
        wave_far.move_to(ORIGIN + DOWN * 1.2)
        label_far2 = Text("跑开的光 → 波被拉开（更暗更红）", font=FONT,
                          font_size=FONT_SIZE_SMALL, color=COLOR_SECONDARY)
        label_far2.next_to(wave_far, DOWN, buff=0.3)

        self.play(Write(light_title), run_time=0.5)
        self.play(Create(wave_close), FadeIn(label_close), run_time=0.8)
        self.play(Create(wave_far), FadeIn(label_far2), run_time=0.8)
        self.wait(max(0.5, total - self.renderer.time - 0.3))

    def _ambulance(self):
        """简易救护车（白色矩形 + 红色十字）。"""
        from manim import Rectangle, VGroup
        body = Rectangle(width=1.2, height=0.6, color=WHITE, fill_opacity=0.9)
        cross = VGroup(
            Line(ORIGIN + UP * 0.15, ORIGIN + DOWN * 0.15, color=RED, stroke_width=4),
            Line(ORIGIN + LEFT * 0.15, ORIGIN + RIGHT * 0.15, color=RED, stroke_width=4),
        )
        cross.move_to(body.get_center())
        return VGroup(body, cross)

    def _sine_wave(self, amplitude=0.4, wavelength=1.0, length=5.0, color=WHITE):
        """正弦波折线。"""
        from manim import VMobject
        n = 200
        x = np.linspace(-length / 2, length / 2, n)
        y = amplitude * np.sin(2 * PI * x / wavelength)
        pts = [np.array([xi, yi, 0.0]) for xi, yi in zip(x, y)]
        wave = VMobject().set_points_as_corners(pts)
        wave.set_stroke(color, 3)
        return wave
