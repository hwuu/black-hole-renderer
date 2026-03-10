"""实体层每帧累加逻辑单元测试。

验证 accumulate_entity_layer 的累加正确性、Keplerian 旋转、
fade 因子应用和 comp_field 写入。
"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def _make_renderer_and_factories():
    """创建 renderer + 三个工厂，用于累加测试。"""
    from render import (
        load_or_generate_skybox, generate_disk_texture, TaichiRenderer,
        EntityFactory,
        _spawn_single_filament, _spawn_single_hotspot, _spawn_single_rt_spike,
    )
    n_r, n_phi = 32, 64
    r_inner, r_outer = 2.0, 15.0

    skybox, _, _ = load_or_generate_skybox(None, 64, 32, 10)
    disk_tex = generate_disk_texture(n_phi=64, n_r=16, seed=42,
                                     r_inner=r_inner, r_outer=r_outer)
    renderer = TaichiRenderer(64, 32, skybox, disk_tex, device="cpu",
                              r_disk_inner=r_inner, r_disk_outer=r_outer)
    renderer.init_background_layer(n_r=n_r, n_phi=n_phi, seed=42)

    r_norm_all = np.linspace(0, 1, n_r)
    r_vals = r_inner + (r_outer - r_inner) * r_norm_all
    omega_all = np.sqrt(0.5 / (r_vals ** 3 + 1e-6)).astype(np.float32)

    factories = {
        'filament': EntityFactory(
            _spawn_single_filament, target_count=10,
            lifetime_range=(10, 20), fade_in=2.0, fade_out=2.0,
            n_r=n_r, n_phi=n_phi,
            r_norm_all=r_norm_all, omega_all=omega_all, seed=100),
        'hotspot': EntityFactory(
            _spawn_single_hotspot, target_count=5,
            lifetime_range=(8, 15), fade_in=2.0, fade_out=2.0,
            n_r=n_r, n_phi=n_phi,
            r_norm_all=r_norm_all, omega_all=omega_all, seed=200),
        'rt_spike': EntityFactory(
            _spawn_single_rt_spike, target_count=3,
            lifetime_range=(8, 15), fade_in=1.5, fade_out=1.5,
            n_r=n_r, n_phi=n_phi,
            r_norm_all=r_norm_all, omega_all=omega_all, seed=300),
    }
    return renderer, factories, n_r, n_phi


class TestAccumulateEntityLayer(unittest.TestCase):
    """验证 accumulate_entity_layer 方法。"""

    @classmethod
    def setUpClass(cls):
        cls.renderer, cls.factories, cls.n_r, cls.n_phi = (
            _make_renderer_and_factories())
        for f in cls.factories.values():
            f.seed_initial(now=0.0)

    def test_comp_field_entity_indices_populated(self):
        """累加后 comp[5:10] 应包含非零数据。"""
        self.renderer.accumulate_entity_layer(self.factories, now=5.0)
        comp = self.renderer._comp_field.to_numpy()
        has_nonzero = False
        for idx in [5, 6, 7, 8, 9, 10]:
            if comp[idx].sum() > 0:
                has_nonzero = True
                break
        self.assertTrue(has_nonzero,
                        "entity layer should produce non-zero values")

    def test_filament_density_nonneg(self):
        """filament density (comp[5]) 应非负。"""
        self.renderer.accumulate_entity_layer(self.factories, now=5.0)
        comp = self.renderer._comp_field.to_numpy()
        self.assertTrue(np.all(comp[5] >= -1e-6), f"min={comp[5].min()}")

    def test_hotspot_density_nonneg(self):
        """hotspot density (comp[9]) 应非负。"""
        self.renderer.accumulate_entity_layer(self.factories, now=5.0)
        comp = self.renderer._comp_field.to_numpy()
        self.assertTrue(np.all(comp[9] >= -1e-6), f"min={comp[9].min()}")

    def test_background_indices_unchanged(self):
        """累加不应影响 background 层索引 [0-4, 11, 12]。"""
        self.renderer.generate_background(t=5.0)
        comp_before = self.renderer._comp_field.to_numpy().copy()
        self.renderer.accumulate_entity_layer(self.factories, now=5.0)
        comp_after = self.renderer._comp_field.to_numpy()
        for idx in [0, 1, 2, 3, 4, 11, 12]:
            np.testing.assert_array_equal(
                comp_before[idx], comp_after[idx],
                err_msg=f"background idx {idx} was modified by entity layer")

    def test_fade_factor_applied(self):
        """刚出生的实体应有较小的 fade alpha。"""
        from render import EntityFactory, _spawn_single_filament
        n_r, n_phi = self.n_r, self.n_phi
        r_norm_all = np.linspace(0, 1, n_r)
        r_vals = 2.0 + 13.0 * r_norm_all
        omega_all = np.sqrt(0.5 / (r_vals ** 3 + 1e-6)).astype(np.float32)

        # 创建一个刚出生的实体（fade_in=10s，now=1s → alpha=0.1）
        fresh_factory = EntityFactory(
            _spawn_single_filament, target_count=50,
            lifetime_range=(100, 100), fade_in=10.0, fade_out=10.0,
            n_r=n_r, n_phi=n_phi,
            r_norm_all=r_norm_all, omega_all=omega_all, seed=42)
        # 手动 spawn，不用 seed_initial（避免 stagger）
        for _ in range(50):
            fresh_factory.entities.append(fresh_factory._spawn_one(now=0.0))

        factories_fresh = {'filament': fresh_factory}
        self.renderer.accumulate_entity_layer(factories_fresh, now=1.0)
        comp_1s = self.renderer._comp_field.to_numpy()[5].copy()

        # 完全 alive 的实体（now=15s → alpha=1.0）
        self.renderer.accumulate_entity_layer(factories_fresh, now=15.0)
        comp_15s = self.renderer._comp_field.to_numpy()[5].copy()

        # alpha=0.1 时的总量应远小于 alpha=1.0
        ratio = comp_1s.sum() / (comp_15s.sum() + 1e-10)
        self.assertLess(ratio, 0.3,
                        f"fade ratio {ratio:.3f} too large, fade not working")

    def test_keplerian_rotation_shifts(self):
        """不同时间的累加应产生 phi 方向位移。"""
        self.renderer.accumulate_entity_layer(self.factories, now=5.0)
        comp_t5 = self.renderer._comp_field.to_numpy()[5].copy()

        self.renderer.accumulate_entity_layer(self.factories, now=5.5)
        comp_t55 = self.renderer._comp_field.to_numpy()[5].copy()

        diff = np.abs(comp_t5 - comp_t55).sum()
        self.assertGreater(diff, 0, "rotation should shift patterns over time")

    def test_empty_factories(self):
        """空工厂字典应将 comp[5:10] 清零。"""
        self.renderer.accumulate_entity_layer({}, now=5.0)
        comp = self.renderer._comp_field.to_numpy()
        for idx in [5, 6, 7, 8, 9, 10]:
            self.assertAlmostEqual(comp[idx].sum(), 0.0, places=5,
                                   msg=f"idx {idx} not zeroed with empty factory")


if __name__ == "__main__":
    unittest.main()
