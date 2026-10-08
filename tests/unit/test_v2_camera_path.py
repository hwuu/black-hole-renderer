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
LENSING_BEND_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "scenes", "lensing_bend",
                                 "lensing_bend.json")


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

    def test_off_frame_subject_lands_on_target(self):
        """黑洞目标在画面外（u、v 在 [−1, 2] 内越出 [0, 1]）时，投影仍落在目标位置。"""
        pos = np.array([[0.0, -39.9, 2.8], [0.0, -39.9, 2.8], [10.0, -30.0, 4.0]])
        uv = np.array([[-0.3, 0.6], [1.8, -0.5], [-1.0, 2.0]])
        fov, roll = np.array([6.0, 6.0, 40.0]), np.array([0.0, 5.0, 12.5])
        fwd = solve_forward(pos, uv, fov, 16 / 9, roll)
        np.testing.assert_allclose(project_origin(pos, fwd, roll, fov, 16 / 9), uv, atol=1e-6)

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
        """默认路径（120 s 版，视野全程 50°）各关键帧的经过时刻与 scenes/v2_arts/README.md 的分镜一致（±0.05 s）。"""
        expected = [0.0, 5.6, 12.4, 20.9, 29.4, 36.2, 43.2, 51.0, 57.8, 63.9, 70.2, 76.7, 83.3, 90.4, 100.7,
                    109.7, 115.0, 120.0]
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
               lambda s: s["keyframes"][0].update(subject_uv=[2.1, 0.5]),
               lambda s: s["keyframes"][0].update(subject_uv=[0.5, -1.1]),
               lambda s: s["rhythm"].update(speed_start=0.0),
               lambda s: s["rhythm"].update(ramp_in=115.0),
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

    def test_off_frame_subject_accepted(self):
        """黑洞目标画面位置可越出画面：边界 −1、2 可加载，构图落在目标位置。"""
        for uv in ([-1.0, 0.5], [2.0, 0.5], [0.5, -1.0], [0.5, 2.0], [-0.3, 0.6]):
            spec = json.loads(json.dumps(self.spec))
            spec.pop("pans", None)
            for k in spec["keyframes"]:
                k["subject_uv"] = uv
            st = load_camera_path(spec).state_at(40.0, 16 / 9)
            np.testing.assert_allclose(
                project_origin(st.pos[None, :], st.forward[None, :], np.array([st.roll]), np.array([st.fov]),
                               16 / 9)[0], uv, atol=1e-6)

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


def _uv_at(path: CameraPath, t: float) -> np.ndarray:
    """路径在时刻 t 时黑洞（原点）实际投影到的画面位置 (u, v)。"""
    st = path.state_at(t, 16 / 9)
    return project_origin(st.pos[None, :], st.forward[None, :], np.array([st.roll]), np.array([st.fov]), 16 / 9)[0]


class TestPans(unittest.TestCase):
    """`pans` 块：在时间上叠加"黑洞画面横坐标平滑移向 u、停留、再平滑移回"的摇镜，不改变节奏与相机位置。"""

    PAN = {"start": 8.0, "end": 18.0, "ramp": 4.0, "u": 0.15}

    def _spec(self, pans=None):
        spec = {"duration": 30.0, "keyframes": [
            {"r": 60.0, "phi_deg": -90.0, "z": 6.0, "fov": 40.0, "roll": 12.5, "subject_uv": [0.6, 0.45]},
            {"r": 30.0, "phi_deg": -60.0, "z": 1.0, "fov": 40.0, "roll": 15.0, "subject_uv": [0.55, 0.5]},
            {"r": 50.0, "phi_deg": -30.0, "z": 5.0, "fov": 40.0, "roll": 12.5, "subject_uv": [0.4, 0.45]}]}
        if pans is not None:
            spec["pans"] = pans
        return spec

    def test_no_pans_unchanged(self):
        a, b = load_camera_path(self._spec()), load_camera_path(self._spec([]))
        for t in np.linspace(0.0, 30.0, 31):
            np.testing.assert_array_equal(a.state_at(float(t), 16 / 9).forward, b.state_at(float(t), 16 / 9).forward)

    def test_pan_profile(self):
        """摇镜前后与原路径逐位相同；停留段（12–14 s）黑洞落在 u = 0.15；位置、视野、滚转与节奏不变；u 随时间光滑。"""
        base, pan = load_camera_path(self._spec()), load_camera_path(self._spec([self.PAN]))
        np.testing.assert_array_equal(base.keyframe_times(), pan.keyframe_times())
        for t in (0.0, 5.0, 8.0, 18.0, 22.0, 30.0):
            np.testing.assert_array_equal(base.state_at(t, 16 / 9).forward, pan.state_at(t, 16 / 9).forward)
        for t in (12.0, 13.0, 14.0):
            np.testing.assert_allclose(_uv_at(pan, t)[0], 0.15, atol=1e-6)
            np.testing.assert_allclose(_uv_at(pan, t)[1], _uv_at(base, t)[1], atol=1e-6)
        ts = np.arange(0.0, 30.0, 0.05)
        for t in ts[::20]:
            sa, sb = base.state_at(float(t), 16 / 9), pan.state_at(float(t), 16 / 9)
            np.testing.assert_array_equal(sa.pos, sb.pos)
            self.assertEqual((sa.fov, sa.roll), (sb.fov, sb.roll))
        u = np.array([pan.state_at(float(t), 16 / 9).subject_uv[0] for t in ts])
        self.assertLess(np.max(np.abs(np.diff(u, 2))), 1e-3)
        mid = pan.state_at(10.0, 16 / 9).subject_uv[0]
        self.assertTrue(0.15 < mid < base.state_at(10.0, 16 / 9).subject_uv[0])

    def test_invalid_pans_raise(self):
        bad = [{"start": -1.0, "end": 18.0, "ramp": 4.0, "u": 0.15},
               {"start": 8.0, "end": 10.0, "ramp": 4.0, "u": 0.15},
               {"start": 8.0, "end": 31.0, "ramp": 4.0, "u": 0.15},
               {"start": 8.0, "end": 18.0, "ramp": 0.0, "u": 0.15},
               {"start": 8.0, "end": 18.0, "ramp": 4.0, "u": 2.5},
               {"start": 8.0, "end": 18.0, "ramp": 4.0},
               {"start": 8.0, "end": 18.0, "ramp": 4.0, "u": float("nan")}]
        for p in bad:
            with self.assertRaises(ValueError):
                load_camera_path(self._spec([p]))
        with self.assertRaises(ValueError):
            load_camera_path(self._spec([self.PAN, {"start": 16.0, "end": 24.0, "ramp": 2.0, "u": 0.8}]))
        with self.assertRaises(ValueError):
            load_camera_path(self._spec({"start": 8.0}))
        load_camera_path(self._spec([{"start": 2.0, "end": 4.0, "ramp": 1.0, "u": 0.8}, self.PAN]))


class TestInterstellarSkimScene(unittest.TestCase):
    """interstellar_skim：视野全程 50°（无变焦）；30 s 起摇镜，黑洞移到画面左侧 u = 0.12，66 s 起摇回。"""

    @classmethod
    def setUpClass(cls):
        with open(DEFAULT_PATH) as f:
            cls.path = load_camera_path(json.load(f))

    def test_fov_constant(self):
        for t in np.linspace(0.0, self.path.duration, 61):
            self.assertAlmostEqual(self.path.state_at(float(t), 16 / 9).fov, 50.0, places=9)

    def test_pan_holds_subject_left(self):
        for t in (38.0, 50.0, 66.0):
            self.assertAlmostEqual(float(_uv_at(self.path, t)[0]), 0.12, places=6)
        for t in (29.9, 74.1):
            self.assertGreater(float(_uv_at(self.path, t)[0]), 0.45)

    def test_photon_ring_stays_in_frame(self):
        """摇镜期间光子环（视半径 asin(2.6/ρ)，临界冲击参数 3√3/2 ≈ 2.6 r_s）左缘不出画面。"""
        for t in np.arange(30.0, 74.01, 0.5):
            st = self.path.state_at(float(t), 16 / 9)
            r_cam, u_c = camera_basis(st.forward[None, :], np.array([st.roll]))
            d = -st.pos / np.linalg.norm(st.pos)
            half_w = np.degrees(np.arctan(np.tan(np.radians(st.fov) / 2) * 16 / 9))
            bh_x = np.degrees(np.arctan2(d @ r_cam[0], d @ st.forward))
            ring = np.degrees(np.arcsin(2.6 / np.linalg.norm(st.pos)))
            self.assertGreater(bh_x - ring + half_w, 0.3, f"t={t:.1f}s 光子环出画")


class TestLensingBendScene(unittest.TestCase):
    """lensing_bend 场景路径：黑洞在画面外的固定构图 + 极慢的匀速逆行环绕。"""

    @classmethod
    def setUpClass(cls):
        with open(LENSING_BEND_PATH) as f:
            cls.path = load_camera_path(json.load(f))

    def test_passes_check(self):
        self.path.check(16 / 9)

    def test_subject_fixed_off_frame(self):
        """每个时刻黑洞都投影到画面外的 (−0.571, 0.650)，视野 9°、滚转 −2°、半径 15、高度 0.4 不变。"""
        for t in np.linspace(0.0, self.path.duration, 13):
            st = self.path.state_at(float(t), 16 / 9)
            uv = project_origin(st.pos[None, :], st.forward[None, :], np.array([st.roll]), np.array([st.fov]),
                                16 / 9)[0]
            np.testing.assert_allclose(uv, [-0.571, 0.650], atol=1e-6)
            self.assertAlmostEqual(st.fov, 9.0, places=9)
            self.assertAlmostEqual(st.roll, -2.0, places=9)
            self.assertAlmostEqual(float(np.hypot(st.pos[0], st.pos[1])), 15.0, places=9)
            self.assertAlmostEqual(float(st.pos[2]), 0.4, places=9)

    def test_uniform_slow_retrograde_orbit(self):
        """方位角 30 s 内从 −91.5° 匀速转到 −88.5°（0.1°/s）；盘反转（disk_spin = −1）时为逆行。"""
        ts = np.linspace(0.0, self.path.duration, 61)
        phi = np.array([np.degrees(np.arctan2(*self.path.state_at(float(t), 16 / 9).pos[1::-1])) for t in ts])
        np.testing.assert_allclose(phi, -91.5 + 0.1 * ts, atol=1e-6)
        lines = self.path.report(time_scale=time_scale(3.0, 8.0), r_in=3.0, r_out=30.0, disk_spin=-1.0)
        self.assertTrue(all("逆行" in ln for ln in lines if ln.startswith("  t=")))


if __name__ == "__main__":
    unittest.main()
