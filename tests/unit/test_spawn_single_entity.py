"""单实例生成函数单元测试。

验证 _spawn_single_filament / _spawn_single_hotspot / _spawn_single_rt_spike
的输出格式、值域、稀疏性和参数统计分布。
"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from render import (
    _spawn_single_filament,
    _spawn_single_hotspot,
    _spawn_single_rt_spike,
)

N_R = 336
N_PHI = 2352
R_INNER = 2.0
R_OUTER = 15.0
R_NORM_ALL = np.linspace(0, 1, N_R)
R_VALS = R_INNER + (R_OUTER - R_INNER) * R_NORM_ALL
OMEGA_ALL = np.sqrt(0.5 / (R_VALS ** 3 + 1e-6)).astype(np.float32)


class TestSpawnSingleFilament(unittest.TestCase):
    """验证 _spawn_single_filament 输出。"""

    def _spawn(self, seed=42):
        rng = np.random.default_rng(seed)
        return _spawn_single_filament(rng, N_R, N_PHI, R_NORM_ALL, OMEGA_ALL)

    def test_return_shapes(self):
        """返回元组格式正确：(row_indices, phi_density, phi_temp, omega)。"""
        ri, pd, pt, omega = self._spawn()
        self.assertEqual(ri.ndim, 1)
        self.assertEqual(pd.ndim, 2)
        self.assertEqual(pt.ndim, 2)
        self.assertEqual(pd.shape[0], len(ri))
        self.assertEqual(pd.shape[1], N_PHI)
        self.assertEqual(pt.shape, pd.shape)
        self.assertIsInstance(omega, float)

    def test_row_indices_valid(self):
        """row_indices 应在 [0, n_r) 范围内。"""
        ri, _, _, _ = self._spawn()
        self.assertTrue(np.all(ri >= 0))
        self.assertTrue(np.all(ri < N_R))

    def test_sparse_rows(self):
        """受影响行数应远小于 n_r（稀疏表示）。"""
        ri, _, _, _ = self._spawn()
        self.assertLess(len(ri), N_R // 4,
                        f"filament affects {len(ri)} rows, too many")

    def test_density_range(self):
        """密度应在 [0, 1]。"""
        _, pd, _, _ = self._spawn()
        self.assertTrue(np.all(pd >= 0), f"min={pd.min()}")
        self.assertTrue(np.all(pd <= 1.01), f"max={pd.max()}")

    def test_density_not_all_zero(self):
        """密度不应全为零。"""
        _, pd, _, _ = self._spawn()
        self.assertGreater(pd.max(), 0.01)

    def test_temp_bounded_by_density(self):
        """温度应 <= 0.5 * 密度（按 CPU 逻辑的 clip）。"""
        _, pd, pt, _ = self._spawn()
        self.assertTrue(np.all(pt <= pd * 0.5 + 1e-6))

    def test_omega_positive(self):
        """omega 应为正值。"""
        _, _, _, omega = self._spawn()
        self.assertGreater(omega, 0)

    def test_different_seeds_different_output(self):
        """不同 seed 应产生不同结果。"""
        _, pd1, _, _ = self._spawn(seed=1)
        _, pd2, _, _ = self._spawn(seed=2)
        self.assertFalse(np.array_equal(pd1, pd2))


class TestSpawnSingleHotspot(unittest.TestCase):
    """验证 _spawn_single_hotspot 输出。"""

    def _spawn(self, seed=42):
        rng = np.random.default_rng(seed)
        return _spawn_single_hotspot(rng, N_R, N_PHI, R_NORM_ALL, OMEGA_ALL)

    def test_return_shapes(self):
        """返回元组格式正确。"""
        ri, pd, pt, omega = self._spawn()
        self.assertEqual(pd.shape[0], len(ri))
        self.assertEqual(pd.shape[1], N_PHI)
        self.assertEqual(pt.shape, pd.shape)

    def test_row_indices_valid(self):
        ri, _, _, _ = self._spawn()
        self.assertTrue(np.all(ri >= 0))
        self.assertTrue(np.all(ri < N_R))

    def test_density_range(self):
        _, pd, _, _ = self._spawn()
        self.assertTrue(np.all(pd >= 0))
        self.assertTrue(np.all(pd <= 1.01))

    def test_temp_proportional(self):
        """hotspot 温度贡献 = 0.12 * density。"""
        _, pd, pt, _ = self._spawn()
        nonzero = pd > 0.001
        if np.any(nonzero):
            ratio = pt[nonzero] / pd[nonzero]
            np.testing.assert_allclose(ratio, 0.12, atol=0.01)

    def test_sparse_rows(self):
        ri, _, _, _ = self._spawn()
        self.assertLess(len(ri), N_R // 2)


class TestSpawnSingleRtSpike(unittest.TestCase):
    """验证 _spawn_single_rt_spike 输出。"""

    def _spawn(self, seed=42):
        rng = np.random.default_rng(seed)
        return _spawn_single_rt_spike(rng, N_R, N_PHI, R_NORM_ALL, OMEGA_ALL)

    def test_return_shapes(self):
        ri, pd, pt, omega = self._spawn()
        self.assertEqual(pd.shape[0], len(ri))
        self.assertEqual(pd.shape[1], N_PHI)
        self.assertEqual(pt.shape, pd.shape)

    def test_row_indices_valid(self):
        ri, _, _, _ = self._spawn()
        self.assertTrue(np.all(ri >= 0))
        self.assertTrue(np.all(ri < N_R))

    def test_density_range(self):
        _, pd, _, _ = self._spawn()
        self.assertTrue(np.all(pd >= 0))
        self.assertTrue(np.all(pd <= 1.01))

    def test_biased_inner_disk(self):
        """RT spikes 应偏向内盘（行索引靠前）。"""
        inner_count = 0
        total = 50
        for seed in range(total):
            ri, _, _, _ = self._spawn(seed=seed)
            median_ri = np.median(ri)
            if median_ri < N_R * 0.3:
                inner_count += 1
        self.assertGreater(inner_count, total * 0.5,
                           f"only {inner_count}/{total} spikes are inner-biased")

    def test_omega_positive(self):
        _, _, _, omega = self._spawn()
        self.assertGreater(omega, 0)


class TestSpawnConsistencyBatch(unittest.TestCase):
    """批量生成多个实例，验证统计分布合理性。"""

    def test_filament_row_count_distribution(self):
        """filament 受影响行数应集中在 2-15 行。"""
        row_counts = []
        for seed in range(100):
            rng = np.random.default_rng(seed)
            ri, _, _, _ = _spawn_single_filament(rng, N_R, N_PHI,
                                                  R_NORM_ALL, OMEGA_ALL)
            row_counts.append(len(ri))
        median_rows = np.median(row_counts)
        self.assertGreaterEqual(median_rows, 1)
        self.assertLessEqual(median_rows, 20)

    def test_hotspot_row_count_distribution(self):
        """hotspot 受影响行数应集中在合理范围（h_r_width=0.02-0.05 的 3σ 截断）。"""
        row_counts = []
        for seed in range(50):
            rng = np.random.default_rng(seed)
            ri, _, _, _ = _spawn_single_hotspot(rng, N_R, N_PHI,
                                                 R_NORM_ALL, OMEGA_ALL)
            row_counts.append(len(ri))
        median_rows = np.median(row_counts)
        self.assertGreaterEqual(median_rows, 2)
        self.assertLessEqual(median_rows, 110)


if __name__ == "__main__":
    unittest.main()
