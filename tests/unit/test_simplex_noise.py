"""Taichi 3D simplex noise / FBM 单元测试。

验证噪声函数的值域、连续性、phi 无缝性和 FBM 多倍频叠加行为。
"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def _make_renderer():
    """创建最小化 TaichiRenderer 实例，仅用于噪声评估。"""
    from render import load_or_generate_skybox, generate_disk_texture, TaichiRenderer
    skybox, _, _ = load_or_generate_skybox(None, 64, 32, 10)
    disk_tex = generate_disk_texture(n_phi=64, n_r=16, seed=42,
                                     r_inner=2.0, r_outer=15.0)
    return TaichiRenderer(64, 32, skybox, disk_tex, device="cpu",
                          r_disk_inner=2.0, r_disk_outer=15.0)


class TestSimplexNoise3D(unittest.TestCase):
    """验证 _simplex_noise_3d 的基本属性。"""

    @classmethod
    def setUpClass(cls):
        cls.renderer = _make_renderer()

    def test_value_range(self):
        """simplex noise 输出应在 [-1, 1] 范围内。"""
        rng = np.random.RandomState(123)
        coords = rng.uniform(-100, 100, size=(5000, 3)).astype(np.float32)
        vals = self.renderer.eval_noise(coords, mode="simplex")
        self.assertTrue(np.all(vals >= -1.01), f"min={vals.min()}")
        self.assertTrue(np.all(vals <= 1.01), f"max={vals.max()}")

    def test_not_constant(self):
        """不同坐标应产生不同的噪声值。"""
        rng = np.random.RandomState(456)
        coords = rng.uniform(-10, 10, size=(200, 3)).astype(np.float32)
        vals = self.renderer.eval_noise(coords, mode="simplex")
        self.assertGreater(np.std(vals), 0.05, "noise output is nearly constant")

    def test_deterministic(self):
        """相同坐标应产生完全相同的输出。"""
        coords = np.array([[1.5, 2.3, -0.7],
                           [0.0, 0.0, 0.0],
                           [-3.1, 4.2, 1.8]], dtype=np.float32)
        vals1 = self.renderer.eval_noise(coords, mode="simplex")
        vals2 = self.renderer.eval_noise(coords, mode="simplex")
        np.testing.assert_array_equal(vals1, vals2)

    def test_continuity(self):
        """相邻坐标的噪声值应接近（Lipschitz 连续性）。"""
        base = np.array([[5.0, 3.0, 1.0]], dtype=np.float32)
        epsilon = 1e-3
        offsets = np.array([
            [epsilon, 0, 0], [0, epsilon, 0], [0, 0, epsilon],
            [-epsilon, 0, 0], [0, -epsilon, 0], [0, 0, -epsilon],
        ], dtype=np.float32)
        coords = np.vstack([base, base + offsets])
        vals = self.renderer.eval_noise(coords, mode="simplex")
        base_val = vals[0]
        for i in range(1, len(vals)):
            diff = abs(vals[i] - base_val)
            self.assertLess(diff, 0.1,
                            f"discontinuity at offset {offsets[i-1]}: diff={diff}")

    def test_phi_seamless(self):
        """用 cos(phi)/sin(phi) 映射后 phi=0 与 phi=2pi 应无缝衔接。"""
        n = 50
        r_vals = np.linspace(0.1, 1.0, n)
        t_val = 0.5
        freq = 8.0

        coords_0 = np.column_stack([
            np.cos(0.0) * freq * np.ones(n),
            np.sin(0.0) * freq * np.ones(n),
            r_vals * freq + t_val,
        ]).astype(np.float32)

        coords_2pi = np.column_stack([
            np.cos(2 * np.pi) * freq * np.ones(n),
            np.sin(2 * np.pi) * freq * np.ones(n),
            r_vals * freq + t_val,
        ]).astype(np.float32)

        vals_0 = self.renderer.eval_noise(coords_0, mode="simplex")
        vals_2pi = self.renderer.eval_noise(coords_2pi, mode="simplex")
        np.testing.assert_allclose(vals_0, vals_2pi, atol=1e-5,
                                   err_msg="phi=0 and phi=2pi not seamless")


class TestFBM3D(unittest.TestCase):
    """验证 _fbm_3d 的基本属性。"""

    @classmethod
    def setUpClass(cls):
        cls.renderer = _make_renderer()

    def test_value_range(self):
        """FBM 输出应在合理范围内（不超过几何级数和的理论上界）。"""
        rng = np.random.RandomState(789)
        coords = rng.uniform(-50, 50, size=(3000, 3)).astype(np.float32)
        vals = self.renderer.eval_noise(coords, mode="fbm",
                                        octaves=4, persistence=0.5)
        max_possible = sum(0.5 ** i for i in range(4))  # ~1.875
        self.assertTrue(np.all(vals >= -(max_possible + 0.1)),
                        f"min={vals.min()}, bound={-max_possible}")
        self.assertTrue(np.all(vals <= max_possible + 0.1),
                        f"max={vals.max()}, bound={max_possible}")

    def test_more_octaves_more_detail(self):
        """更多 octave 应增加高频细节（标准差增大或至少不减小）。"""
        rng = np.random.RandomState(101)
        coords = rng.uniform(-5, 5, size=(1000, 3)).astype(np.float32)
        vals_1 = self.renderer.eval_noise(coords, mode="fbm", octaves=1,
                                          persistence=0.5)
        vals_4 = self.renderer.eval_noise(coords, mode="fbm", octaves=4,
                                          persistence=0.5)
        std_1 = np.std(vals_1)
        std_4 = np.std(vals_4)
        self.assertGreater(std_4, std_1 * 0.9,
                           f"4-octave std ({std_4:.4f}) should >= 1-octave ({std_1:.4f})")

    def test_single_octave_equals_simplex(self):
        """单个 octave、persistence=1、lacunarity=2 的 FBM 应等于原始 simplex noise。"""
        rng = np.random.RandomState(202)
        coords = rng.uniform(-10, 10, size=(500, 3)).astype(np.float32)
        vals_simplex = self.renderer.eval_noise(coords, mode="simplex")
        vals_fbm1 = self.renderer.eval_noise(coords, mode="fbm",
                                             octaves=1, persistence=1.0,
                                             lacunarity=2.0)
        np.testing.assert_allclose(vals_simplex, vals_fbm1, atol=1e-5)

    def test_deterministic(self):
        """FBM 应确定性可重复。"""
        coords = np.array([[1.0, 2.0, 3.0],
                           [-1.0, -2.0, -3.0]], dtype=np.float32)
        vals1 = self.renderer.eval_noise(coords, mode="fbm",
                                         octaves=4, persistence=0.5)
        vals2 = self.renderer.eval_noise(coords, mode="fbm",
                                         octaves=4, persistence=0.5)
        np.testing.assert_array_equal(vals1, vals2)


if __name__ == "__main__":
    unittest.main()
