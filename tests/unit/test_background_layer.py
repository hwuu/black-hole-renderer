"""GPU 背景层生成 kernel 单元测试。

验证 _generate_background_kernel 写入的 5 个宽 r 组件的值域、
时间演化、phi 无缝性和组件独立性。
"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def _make_renderer():
    """创建最小化 TaichiRenderer 实例。"""
    from render import load_or_generate_skybox, generate_disk_texture, TaichiRenderer
    skybox, _, _ = load_or_generate_skybox(None, 64, 32, 10)
    disk_tex = generate_disk_texture(n_phi=64, n_r=16, seed=42,
                                     r_inner=2.0, r_outer=15.0)
    return TaichiRenderer(64, 32, skybox, disk_tex, device="cpu",
                          r_disk_inner=2.0, r_disk_outer=15.0)


class TestBackgroundLayerInit(unittest.TestCase):
    """验证 init_background_layer 参数初始化。"""

    @classmethod
    def setUpClass(cls):
        cls.renderer = _make_renderer()
        cls.renderer.init_background_layer(n_r=32, n_phi=64, seed=42)

    def test_comp_field_created(self):
        """comp_field 应被创建，shape=(13, n_r, n_phi)。"""
        import taichi as ti
        comp = self.renderer._comp_field
        self.assertEqual(comp.shape, (13, 32, 64))

    def test_omega_numpy_cached(self):
        """omega numpy 缓存应被创建，用于实体层差速剪切。"""
        omega_np = self.renderer._bg_omega_all_np
        self.assertEqual(omega_np.shape, (32,))
        self.assertTrue(np.all(omega_np > 0))

    def test_az_params(self):
        """方位热点参数应在合理范围内。"""
        self.assertGreaterEqual(self.renderer._bg_az_freq, 2)
        self.assertLessEqual(self.renderer._bg_az_freq, 4)
        self.assertGreaterEqual(self.renderer._bg_az_shear, 2.0)
        self.assertLessEqual(self.renderer._bg_az_shear, 4.0)

    def test_edge_and_omega_created(self):
        """edge 和 omega_rows 场应被创建。"""
        edge = self.renderer._edge_field.to_numpy()
        omega = self.renderer._omega_rows_field.to_numpy()
        self.assertEqual(edge.shape, (32,))
        self.assertEqual(omega.shape, (32,))
        self.assertTrue(np.all(edge >= 0))
        self.assertTrue(np.all(edge <= 1))
        self.assertTrue(np.all(omega > 0))


class TestGenerateBackground(unittest.TestCase):
    """验证 generate_background 的输出质量。"""

    @classmethod
    def setUpClass(cls):
        cls.n_r = 64
        cls.n_phi = 128
        cls.renderer = _make_renderer()
        cls.renderer.init_background_layer(n_r=cls.n_r, n_phi=cls.n_phi, seed=42)
        cls.renderer.generate_background(t=0.0)
        cls.comp_t0 = cls.renderer._comp_field.to_numpy()
        cls.renderer.generate_background(t=5.0)
        cls.comp_t5 = cls.renderer._comp_field.to_numpy()

    def test_temp_base_range(self):
        """temp_base (idx 0) 应在 [0, 0.25] 附近，内盘高外盘低。"""
        tb = self.comp_t0[0]
        self.assertTrue(np.all(tb >= -0.01), f"min={tb.min()}")
        self.assertTrue(np.all(tb <= 0.35), f"max={tb.max()}")
        inner_mean = tb[:self.n_r // 4].mean()
        outer_mean = tb[3 * self.n_r // 4:].mean()
        self.assertGreater(inner_mean, outer_mean,
                           "temp_base should decay radially outward")

    def test_spiral_range(self):
        """spiral (idx 1) 应非负且 <= 1。"""
        sp = self.comp_t0[1]
        self.assertTrue(np.all(sp >= -0.01), f"min={sp.min()}")
        self.assertTrue(np.all(sp <= 1.01), f"max={sp.max()}")

    def test_spiral_is_zero(self):
        """spiral (idx 1,2) 已移除，应全为零。"""
        self.assertAlmostEqual(self.comp_t0[1].sum(), 0.0, places=5)
        self.assertAlmostEqual(self.comp_t0[2].sum(), 0.0, places=5)

    def test_turbulence_range(self):
        """turbulence (idx 3) 应在 [0, 1]。"""
        turb = self.comp_t0[3]
        self.assertTrue(np.all(turb >= -0.01), f"min={turb.min()}")
        self.assertTrue(np.all(turb <= 1.01), f"max={turb.max()}")

    def test_turb_temp_proportional(self):
        """turb_temp (idx 4) = 0.05 * turbulence (idx 3)。"""
        turb = self.comp_t0[3]
        turb_t = self.comp_t0[4]
        np.testing.assert_allclose(turb_t, 0.05 * turb, atol=1e-5)

    def test_az_hotspot_range(self):
        """az_hotspot (idx 11) 应非负。"""
        az = self.comp_t0[11]
        self.assertTrue(np.all(az >= -0.01), f"min={az.min()}")

    def test_disturb_mod_range(self):
        """disturb_mod (idx 12) 应在 [0.1, 1.0]。"""
        dm = self.comp_t0[12]
        self.assertTrue(np.all(dm >= 0.09), f"min={dm.min()}")
        self.assertTrue(np.all(dm <= 1.01), f"max={dm.max()}")

    def test_temporal_evolution(self):
        """t=0 和 t=5 的输出应不同（时间演化在发生）。"""
        for idx, name in [(0, "temp_base"), (3, "turbulence"),
                          (11, "az_hotspot"), (12, "disturb_mod")]:
            diff = np.abs(self.comp_t0[idx] - self.comp_t5[idx]).mean()
            self.assertGreater(diff, 1e-4,
                               f"{name} (idx {idx}) not evolving over time")

    def test_temporal_smoothness(self):
        """极短时间间隔的变化不应爆炸（有限变化率）。"""
        self.renderer.generate_background(t=0.001)
        comp_dt = self.renderer._comp_field.to_numpy()
        for idx, name in [(0, "temp_base"), (3, "turbulence")]:
            diff = np.abs(self.comp_t0[idx] - comp_dt[idx]).mean()
            self.assertLess(diff, 0.15,
                            f"{name} (idx {idx}) changes too abruptly")

    def test_entity_indices_untouched(self):
        """实体层索引 [5-10] 不应被 background kernel 写入。"""
        self.renderer.generate_background(t=0.0)
        comp = self.renderer._comp_field.to_numpy()
        for idx in [5, 6, 7, 8, 9, 10]:
            self.assertAlmostEqual(comp[idx].sum(), 0.0, places=5,
                                   msg=f"entity idx {idx} was modified")


if __name__ == "__main__":
    unittest.main()
