"""运镜路径单测：保形插值、高斯平滑、构图求解、节奏曲线、路径文件加载与校验。"""

from __future__ import annotations

import json
import os
import unittest

import numpy as np

from src.v2.camera import build_camera_v1_compatible
from src.v2.camera_compose import (UniformSpline, camera_basis, gaussian_smooth, pchip, project_origin,
                                   solve_forward)
from src.v2.camera_path import CameraPath, Keyframe, PathTiming, load_camera_path, local_speed
from src.v2.path_video import time_scale

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "scenes", "v2_arts",
                            "interstellar_skim.json")


def _simple_path(**timing) -> CameraPath:
    """三个关键帧的简单路径：远处 → 近处 → 远处，用于节奏与连续性测试。"""
    kfs = [Keyframe(60.0, -90.0, 6.0, 34.0, 12.5, (0.6, 0.45)),
           Keyframe(30.0, -60.0, 1.0, 44.0, 15.0, (0.55, 0.5)),
           Keyframe(50.0, -30.0, 5.0, 36.0, 12.5, (0.4, 0.45))]
    return CameraPath(kfs, PathTiming(**{"duration": 30.0, **timing}))


class TestPchip(unittest.TestCase):
    """保形三次插值：经过节点、不过冲、一阶导数连续、两端导数为 0。"""

    x = np.array([0.0, 1.0, 2.5, 4.0, 5.0])
    y = np.array([0.0, 2.0, 2.2, -1.0, 3.0])

    def test_passes_through_nodes(self):
        np.testing.assert_allclose(pchip(self.x, self.y, self.x), self.y, atol=1e-12)

    def test_no_overshoot(self):
        for i in range(len(self.x) - 1):
            q = np.linspace(self.x[i], self.x[i + 1], 101)
            v = pchip(self.x, self.y, q)
            lo, hi = sorted((self.y[i], self.y[i + 1]))
            self.assertTrue(np.all(v >= lo - 1e-12) and np.all(v <= hi + 1e-12))

    def test_c1_at_nodes_and_zero_end_slope(self):
        e = 1e-6
        for xi in self.x[1:-1]:
            left = (pchip(self.x, self.y, xi) - pchip(self.x, self.y, xi - e)) / e
            right = (pchip(self.x, self.y, xi + e) - pchip(self.x, self.y, xi)) / e
            self.assertAlmostEqual(left, right, delta=1e-4)
        self.assertAlmostEqual(float((pchip(self.x, self.y, e) - self.y[0]) / e), 0.0, delta=1e-4)


class TestGaussianSmooth(unittest.TestCase):
    """高斯平滑：σ = 0 原样返回；常数不变；线性函数内部不变。"""

    def test_zero_sigma_identity(self):
        a = np.random.default_rng(0).normal(size=(50, 2))
        np.testing.assert_array_equal(gaussian_smooth(a, 0.0), a)

    def test_preserves_constant_and_linear_interior(self):
        np.testing.assert_allclose(gaussian_smooth(np.full(100, 3.0), 5.0), 3.0, atol=1e-12)
        lin = np.arange(100.0)
        np.testing.assert_allclose(gaussian_smooth(lin, 5.0)[20:80], lin[20:80], atol=1e-9)


class TestUniformSpline(unittest.TestCase):
    """三次样条：经过节点；三次多项式被精确还原（含导数），端点无尖峰。"""

    def test_reproduces_cubic(self):
        x = np.linspace(-1.0, 2.0, 31)
        f = lambda t: 0.5 * t**3 - t**2 + 2 * t - 1
        sp = UniformSpline(x[0], x[1] - x[0], np.stack([f(x), 2 * f(x)], axis=1))
        q = np.linspace(-1.0, 2.0, 97)
        np.testing.assert_allclose(sp(x)[:, 0], f(x), atol=1e-12)
        np.testing.assert_allclose(sp(q)[:, 0], f(q), atol=1e-9)
        np.testing.assert_allclose(sp.derivative(q)[:, 1], 2 * (1.5 * q**2 - 2 * q + 2), atol=1e-8)


class TestCompose(unittest.TestCase):
    """构图求解：黑洞投影落在目标画面位置；基向量与渲染相机一致。"""

    def test_basis_matches_render_camera(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            pos, fwd = rng.normal(size=3) * 20, rng.normal(size=3)
            fwd /= np.linalg.norm(fwd)
            roll = float(rng.uniform(-30, 30))
            _, r, u, f, *_ = build_camera_v1_compatible(pos, 40.0, 640, 360, roll, forward=fwd)
            r2, u2 = camera_basis(fwd[None, :], np.array([roll]))
            np.testing.assert_allclose(r2[0], r, atol=1e-12)
            np.testing.assert_allclose(u2[0], u, atol=1e-12)

    def test_subject_lands_on_target(self):
        rng = np.random.default_rng(2)
        n = 50
        pos = np.stack([rng.uniform(20, 70, n) * np.cos(rng.uniform(-3, 3, n)),
                        rng.uniform(20, 70, n) * np.sin(rng.uniform(-3, 3, n)), rng.uniform(-3, 10, n)], axis=1)
        uv = rng.uniform(0.3, 0.7, (n, 2))
        fov, roll = rng.uniform(30, 60, n), rng.uniform(-20, 20, n)
        fwd = solve_forward(pos, uv, fov, 16 / 9, roll)
        np.testing.assert_allclose(project_origin(pos, fwd, roll, fov, 16 / 9), uv, atol=1e-6)

    def test_near_vertical_axis_converges(self):
        """光轴接近竖直（离 −z 约 8°）时不动点迭代不收敛，须由阻尼高斯–牛顿补救。"""
        pos = np.array([[20.0, 0.0, 45.0]])
        f0 = np.array([[-0.14578299, -0.0025818, -0.98931322]])
        f0 /= np.linalg.norm(f0)
        uv = project_origin(pos, f0, np.array([13.0]), np.array([50.0]), 16 / 9)
        f = solve_forward(pos, uv, np.array([50.0]), 16 / 9, np.array([13.0]))
        np.testing.assert_allclose(project_origin(pos, f, np.array([13.0]), np.array([50.0]), 16 / 9), uv, atol=1e-6)

    def test_unreachable_target_raises(self):
        """黑洞方向离竖直约 4° 时，远离中心线的画面位置不可达：报错，不静默返回错误光轴。"""
        pos = np.array([[-2.4987478, -2.1327776, 46.6333188]])
        with self.assertRaisesRegex(ValueError, "构图无解"):
            solve_forward(pos, np.array([[0.3639603, 0.4139306]]), np.array([44.07397]), 16 / 9, np.array([12.48697]))

    def test_centre_points_at_subject(self):
        pos = np.array([[10.0, -30.0, 4.0]])
        fwd = solve_forward(pos, np.array([[0.5, 0.5]]), np.array([40.0]), 16 / 9, np.array([12.5]))
        np.testing.assert_allclose(fwd[0], -pos[0] / np.linalg.norm(pos[0]), atol=1e-9)


class TestRhythm(unittest.TestCase):
    """节奏：首尾速度比、中段恒定、时刻单调、相机运动连续。"""

    def test_speed_profile(self):
        p = _simple_path(ramp_in=4.0, ramp_out=6.0, speed_start=0.5, speed_end=0.7)
        c = p.cruise_speed
        self.assertAlmostEqual(float(p.perceived_speed(0.0)) / c, 0.5, places=9)
        self.assertAlmostEqual(float(p.perceived_speed(30.0)) / c, 0.7, places=9)
        np.testing.assert_allclose(p.perceived_speed(np.linspace(4.0, 24.0, 50)) / c, 1.0, atol=1e-12)
        ts = np.linspace(0.0, 30.0, 30001)
        v = p.perceived_speed(ts)
        self.assertAlmostEqual(float(np.sum((v[1:] + v[:-1]) / 2 * np.diff(ts))), p.progress_total, delta=1e-6)

    def test_keyframe_times_monotone_and_span(self):
        t = _simple_path().keyframe_times()
        self.assertAlmostEqual(t[0], 0.0, delta=1e-9)
        self.assertAlmostEqual(t[-1], 30.0, delta=1e-6)
        self.assertTrue(np.all(np.diff(t) > 0))

    def test_motion_continuity(self):
        """位置的三阶差分有界且在不同帧率下一致（无数值假象）。"""
        p = _simple_path()
        jerks = []
        for fps in (30, 60):
            ts = np.arange(0.0, 30.0, 1.0 / fps)
            pos = np.array([p.state_at(float(t), 16 / 9).pos for t in ts])
            jerks.append(np.abs(np.diff(pos, n=3, axis=0)).max() * fps**3)
        self.assertLess(abs(jerks[0] - jerks[1]) / jerks[1], 0.2)


class TestLocalSpeed(unittest.TestCase):
    """局部速度：远处趋于坐标速度；径向分量按 1/A、切向按 1/√A 放大。"""

    def test_formula(self):
        pos = np.array([[1e8, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        vel = np.array([[0.3, 0.4, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        np.testing.assert_allclose(local_speed(pos, vel), [0.5, 0.2, 0.1 / np.sqrt(0.5)], rtol=1e-7)


class TestLoadAndValidate(unittest.TestCase):
    """路径文件加载、默认路径的经过时刻与路径检查、非法取值报错。"""

    @classmethod
    def setUpClass(cls):
        with open(DEFAULT_PATH) as f:
            cls.spec = json.load(f)
        cls.path = load_camera_path(cls.spec)

    def test_default_path_keyframe_times(self):
        """默认路径各关键帧的经过时刻与方案 v0.3 §4.3 一致（±0.05 s）。"""
        expected = [0.0, 5.9, 13.2, 22.4, 31.5, 38.6, 44.2, 48.5, 53.1, 58.7, 64.6, 72.9, 81.1, 87.5]
        np.testing.assert_allclose(self.path.keyframe_times(), expected, atol=0.05)

    def test_default_path_passes_check(self):
        self.path.check(16 / 9)

    def test_report_lists_every_keyframe_and_crossing(self):
        lines = self.path.report(time_scale=time_scale(3.0, 16.0), r_in=3.0, r_out=30.0, disk_spin=-1.0)
        self.assertEqual(len([ln for ln in lines if ln.startswith("  t=")]), len(self.spec["keyframes"]))
        crossings = [ln for ln in lines if "穿越盘面" in ln]
        self.assertEqual(len(crossings), 2)
        self.assertIn("穿过盘内", crossings[0])
        self.assertIn("盘外", crossings[1])
        self.assertTrue(all("逆行" in ln for ln in lines if ln.startswith("  t=")))
        self.assertTrue(any("最大局部速度" in ln for ln in lines))

    def test_invalid_values_raise(self):
        bad = [lambda s: s.update(duration=0.0),
               lambda s: s.update(keyframes=s["keyframes"][:1]),
               lambda s: s["keyframes"][0].update(fov=180.0),
               lambda s: s["keyframes"][0].update(subject_uv=[1.2, 0.5]),
               lambda s: s["rhythm"].update(speed_start=0.0),
               lambda s: s["rhythm"].update(ramp_in=80.0),
               lambda s: s.update(duration=float("inf")),
               lambda s: s.update(path_smoothing=float("nan")),
               lambda s: s["progress"].update(near_weight=float("inf")),
               lambda s: s["keyframes"][3].update(z=float("inf")),
               lambda s: s["keyframes"][3].update(phi_deg=float("nan")),
               lambda s: s["keyframes"][3].update(subject_uv=[0.5]),
               lambda s: s["keyframes"][3].pop("r"),
               lambda s: s.update(progress=None)]
        for mutate in bad:
            spec = json.loads(json.dumps(self.spec))
            mutate(spec)
            with self.assertRaises(ValueError):
                load_camera_path(spec)

    def test_check_includes_end_point(self):
        """时长不是采样步长整数倍时也检查终点：终点离黑洞不足 3 r_s 须报错。"""
        kfs = [Keyframe(20.0, 0.0, 1.0, 40.0, 0.0, (0.5, 0.5)), Keyframe(1.5, 0.0, 0.5, 40.0, 0.0, (0.5, 0.5))]
        with self.assertRaises(ValueError):
            CameraPath(kfs, PathTiming(duration=0.05, path_smoothing=0.0, ramp_in=0.0, ramp_out=0.0)).check(16 / 9)

    def test_too_short_duration_rejected(self):
        """时长短于 3 个时间步报错；恰好 3 个时间步可构造。"""
        kfs = [Keyframe(20.0, 0.0, 1.0, 40.0, 0.0, (0.5, 0.5)), Keyframe(25.0, 10.0, 1.0, 40.0, 0.0, (0.5, 0.5))]
        with self.assertRaises(ValueError):
            CameraPath(kfs, PathTiming(duration=0.005, ramp_in=0.0, ramp_out=0.0))
        CameraPath(kfs, PathTiming(duration=0.0125, ramp_in=0.0, ramp_out=0.0))

    def test_vertical_axis_fails_check(self):
        kfs = [Keyframe(0.5, 0.0, 30.0, 40.0, 0.0, (0.5, 0.5)), Keyframe(0.5, 10.0, 31.0, 40.0, 0.0, (0.5, 0.5))]
        with self.assertRaises(ValueError):
            CameraPath(kfs, PathTiming(duration=5.0, ramp_in=1.0, ramp_out=1.0)).check(16 / 9)


if __name__ == "__main__":
    unittest.main()
