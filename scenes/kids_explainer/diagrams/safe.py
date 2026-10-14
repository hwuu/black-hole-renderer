"""s08_safe：黑洞从哪来、危不危险（恒星塌缩 → 非吸尘器 → 距离尺度）。

旁白（5 句，约 36 s）：
1. 黑洞是怎么来的？比太阳重二三十倍以上的大恒星，烧到最后，会塌成一个黑洞。
2. 那黑洞会把地球吸走吗？
3. 不会。它不是吸尘器：隔得够远的时候，它的引力和同样重的普通星球一模一样。
4. 就算把太阳换成同样重的黑洞，地球照样绕着转，只是天会变黑、变冷。
5. 离我们最近的黑洞，也在一千五百光年之外；银河系中心就有一个，四百万个太阳那么重。
"""

import numpy as np

from manim import *

from src.film.diagram_kit import (
    FONT, FONT_SIZE_TITLE, FONT_SIZE_BODY, FONT_SIZE_SMALL,
    COLOR_TEXT, COLOR_ACCENT, COLOR_SECONDARY, COLOR_DIM, COLOR_HIGHLIGHT,
    COLOR_NEGATIVE, COLOR_POSITIVE, COLOR_GAS, load_timing, sentence_starts,
)


class Safe(Scene):
    """黑洞安全：恒星塌缩 → 非吸尘器 → 太阳替换思想实验 → 距离。"""

    def construct(self):
        timing = load_timing("s08_safe")
        starts = sentence_starts("s08_safe")
        total = timing["duration"]
        self.camera.background_color = "#0d0d1a"

        # ---------------------------------------------------------------
        # 段落 1：恒星塌缩（~8s）
        # ---------------------------------------------------------------
        big_star = Circle(radius=2.0, color="#FFD700", fill_opacity=0.35,
                          stroke_color=COLOR_HIGHLIGHT)
        big_star.move_to(ORIGIN + UP * 0.5)
        star_label = Text("大恒星（比太阳重 20–30 倍以上）", font=FONT,
                          font_size=FONT_SIZE_SMALL, color=COLOR_TEXT)
        star_label.next_to(big_star, DOWN, buff=0.5)

        small_bh = Circle(radius=0.25, color=BLACK, fill_opacity=1.0)
        small_bh.set_stroke(WHITE, 2)
        small_bh.move_to(big_star.get_center())
        bh_label = Text("黑洞", font=FONT, font_size=FONT_SIZE_SMALL, color=COLOR_TEXT)
        bh_label.next_to(small_bh, DOWN, buff=0.5)

        self.play(FadeIn(big_star), FadeIn(star_label), run_time=0.8)
        self.wait(0.5)
        self.play(
            Transform(big_star, small_bh),
            FadeOut(star_label),
            run_time=2.0,
        )
        self.play(FadeIn(bh_label), run_time=0.3)

        self.wait(max(0, starts[1] - self.renderer.time - 0.3))
        self.play(FadeOut(big_star), FadeOut(bh_label), run_time=0.3)

        # ---------------------------------------------------------------
        # 段落 2+3：会不会吸走地球？（~11s）
        # ---------------------------------------------------------------
        question = Text("黑洞会把地球吸走吗？", font=FONT, font_size=FONT_SIZE_TITLE,
                        color=COLOR_TEXT)
        question.move_to(ORIGIN)
        self.play(Write(question), run_time=0.8)

        answer = Text("不会！", font=FONT, font_size=FONT_SIZE_TITLE * 1.2,
                      color=COLOR_POSITIVE)
        answer.move_to(ORIGIN)
        vacuum = Text("它不是吸尘器", font=FONT, font_size=FONT_SIZE_BODY,
                      color=COLOR_NEGATIVE)
        vacuum.move_to(ORIGIN + DOWN * 1.2)

        self.wait(max(0, starts[2] - self.renderer.time - 0.3))
        self.play(Transform(question, answer), run_time=0.5)
        self.play(Write(vacuum), run_time=0.5)

        self.wait(max(0, starts[3] - self.renderer.time - 0.3))
        self.play(FadeOut(question), FadeOut(vacuum), run_time=0.3)

        # ---------------------------------------------------------------
        # 段落 4：太阳替换思想实验（~7s）
        # ---------------------------------------------------------------
        sun = Circle(radius=0.8, color="#FFD700", fill_opacity=0.5)
        sun.move_to(ORIGIN)
        sun_label = Text("太阳", font=FONT, font_size=FONT_SIZE_SMALL, color=COLOR_TEXT)
        sun_label.next_to(sun, UP, buff=0.3)

        earth = Circle(radius=0.2, color=COLOR_SECONDARY, fill_opacity=0.8)
        earth.move_to(ORIGIN + RIGHT * 3.5)
        earth_label = Text("地球", font=FONT, font_size=FONT_SIZE_SMALL, color=COLOR_TEXT)
        earth_label.next_to(earth, DOWN, buff=0.3)

        orbit = Ellipse(width=7.0, height=7.0, color=COLOR_DIM, stroke_opacity=0.3)
        orbit.move_to(sun.get_center())

        sun_as_bh = Circle(radius=0.12, color=BLACK, fill_opacity=1.0)
        sun_as_bh.set_stroke(WHITE, 2)
        sun_as_bh.move_to(sun.get_center())
        swap_note = Text("换成同样重的黑洞：\n地球照样绕着转，\n只是天变黑、变冷",
                         font=FONT, font_size=FONT_SIZE_SMALL, color=COLOR_POSITIVE,
                         line_spacing=0.6)
        swap_note.move_to(ORIGIN + UP * 2.0)

        self.play(FadeIn(sun), FadeIn(sun_label), FadeIn(earth), FadeIn(earth_label),
                  Create(orbit), run_time=0.8)
        self.wait(0.3)
        self.play(Transform(sun, sun_as_bh), FadeOut(sun_label), run_time=0.8)
        self.play(Write(swap_note), run_time=0.5)

        self.wait(max(0, starts[4] - self.renderer.time - 0.3))
        self.play(FadeOut(sun), FadeOut(sun_as_bh), FadeOut(earth), FadeOut(earth_label),
                  FadeOut(orbit), FadeOut(swap_note), run_time=0.3)

        # ---------------------------------------------------------------
        # 段落 5：距离尺度（~10s）
        # ---------------------------------------------------------------
        dist_title = Text("最近的黑洞有多远？", font=FONT, font_size=FONT_SIZE_TITLE,
                          color=COLOR_TEXT)
        dist_title.move_to(ORIGIN + UP * 2.0)

        near = Text("约 1500 光年", font=FONT, font_size=FONT_SIZE_BODY,
                    color=COLOR_POSITIVE)
        near.move_to(ORIGIN + UP * 0.3)

        center_label = Text("银河系中心有一个：人马座 A*", font=FONT,
                            font_size=FONT_SIZE_BODY, color=COLOR_TEXT)
        center_label.move_to(ORIGIN + DOWN * 1.0)

        mass = Text("400 万个太阳那么重", font=FONT, font_size=FONT_SIZE_SMALL,
                    color=COLOR_DIM)
        mass.next_to(center_label, DOWN, buff=0.3)

        self.play(Write(dist_title), run_time=0.5)
        self.play(FadeIn(near), run_time=0.3)
        self.wait(0.5)
        self.play(Write(center_label), run_time=0.5)
        self.play(FadeIn(mass), run_time=0.3)
        self.wait(max(0.5, total - self.renderer.time - 0.3))
