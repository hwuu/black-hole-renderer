"""EntityInstance 和 EntityFactory 单元测试。

验证实体生命周期（fade_factor）、工厂的初始填充、tick 调度和死亡回收。
"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from render import (
    EntityInstance,
    EntityFactory,
    _spawn_single_filament,
    _spawn_single_hotspot,
    _spawn_single_rt_spike,
    FILAMENT_DEATH_THRESHOLD,
    FILAMENT_MAX_LIFETIME,
    FILAMENT_SHEAR_ALPHA,
    FILAMENT_TAU_COOL,
)

N_R = 64
N_PHI = 128
R_INNER = 2.0
R_OUTER = 15.0
R_NORM_ALL = np.linspace(0, 1, N_R)
R_VALS = R_INNER + (R_OUTER - R_INNER) * R_NORM_ALL
OMEGA_ALL = np.sqrt(0.5 / (R_VALS ** 3 + 1e-6)).astype(np.float32)


def _make_entity(birth=0.0, lifetime=10.0, fade_in=2.0, fade_out=2.0):
    """创建一个最小化测试实体。"""
    return EntityInstance(
        row_indices=np.array([5, 6, 7]),
        phi_density=np.ones((3, 32), dtype=np.float32) * 0.5,
        phi_temp=np.ones((3, 32), dtype=np.float32) * 0.1,
        omega=0.5,
        birth_time=birth,
        lifetime=lifetime,
        fade_in=fade_in,
        fade_out=fade_out,
        fade_noise=np.random.default_rng(0).random(32).astype(np.float32),
    )


class TestEntityInstance(unittest.TestCase):
    """验证 EntityInstance 的生命周期方法。"""

    def test_total_duration(self):
        e = _make_entity(lifetime=10, fade_in=2, fade_out=3)
        self.assertAlmostEqual(e.total_duration, 15.0)

    def test_fade_factor_before_birth(self):
        e = _make_entity(birth=5.0)
        self.assertAlmostEqual(e.fade_factor(now=3.0), 0.0)

    def test_fade_factor_during_fade_in(self):
        e = _make_entity(birth=0.0, fade_in=2.0)
        self.assertAlmostEqual(e.fade_factor(now=1.0), 0.5)
        self.assertAlmostEqual(e.fade_factor(now=0.0), 0.0)

    def test_fade_factor_fully_alive(self):
        e = _make_entity(birth=0.0, fade_in=2.0, lifetime=10.0)
        self.assertAlmostEqual(e.fade_factor(now=2.0), 1.0)
        self.assertAlmostEqual(e.fade_factor(now=5.0), 1.0)
        self.assertAlmostEqual(e.fade_factor(now=12.0), 1.0)

    def test_fade_factor_during_fade_out(self):
        e = _make_entity(birth=0.0, fade_in=2.0, lifetime=10.0, fade_out=2.0)
        self.assertAlmostEqual(e.fade_factor(now=13.0), 0.5)

    def test_fade_factor_after_death(self):
        e = _make_entity(birth=0.0, fade_in=2.0, lifetime=10.0, fade_out=2.0)
        self.assertAlmostEqual(e.fade_factor(now=14.0), 0.0)
        self.assertAlmostEqual(e.fade_factor(now=20.0), 0.0)

    def test_is_dead(self):
        e = _make_entity(birth=0.0, fade_in=2.0, lifetime=10.0, fade_out=2.0)
        self.assertFalse(e.is_dead(now=5.0))
        self.assertFalse(e.is_dead(now=13.5))
        self.assertTrue(e.is_dead(now=14.0))
        self.assertTrue(e.is_dead(now=100.0))

    def test_zero_fade_in(self):
        """fade_in=0 时应立即进入 alive。"""
        e = _make_entity(birth=0.0, fade_in=0.0, lifetime=5.0, fade_out=1.0)
        self.assertAlmostEqual(e.fade_factor(now=0.0), 1.0)
        self.assertAlmostEqual(e.fade_factor(now=3.0), 1.0)

    def test_zero_fade_out(self):
        """fade_out=0 时应在 lifetime 结束后立即死亡。"""
        e = _make_entity(birth=0.0, fade_in=1.0, lifetime=5.0, fade_out=0.0)
        self.assertAlmostEqual(e.fade_factor(now=5.5), 1.0)
        self.assertAlmostEqual(e.fade_factor(now=6.0), 0.0)


class TestEntityFactory(unittest.TestCase):
    """验证 EntityFactory 的调度行为。"""

    def _make_factory(self, spawn_fn=_spawn_single_filament,
                      target=20, lifetime_range=(10, 20),
                      fade_in=2.0, fade_out=2.0, seed=42,
                      entity_type=None):
        if entity_type is None:
            entity_type = 'filament' if spawn_fn is _spawn_single_filament else 'generic'
        return EntityFactory(
            spawn_fn=spawn_fn,
            target_count=target,
            lifetime_range=lifetime_range,
            fade_in=fade_in,
            fade_out=fade_out,
            n_r=N_R, n_phi=N_PHI,
            r_norm_all=R_NORM_ALL,
            omega_all=OMEGA_ALL,
            seed=seed,
            entity_type=entity_type,
        )

    def test_seed_initial_populates(self):
        """seed_initial 应创建 target_count 个实体。"""
        f = self._make_factory(target=20)
        f.seed_initial(now=0.0)
        self.assertEqual(len(f.entities), 20)

    def test_seed_initial_staggered_ages(self):
        """seed_initial 的实体应有不同的 birth_time（分散在生命周期内）。"""
        f = self._make_factory(target=20)
        f.seed_initial(now=10.0)
        birth_times = [e.birth_time for e in f.entities]
        self.assertGreater(max(birth_times) - min(birth_times), 1.0,
                           "entities should have staggered birth times")
        for e in f.entities:
            self.assertLessEqual(e.birth_time, 10.0,
                                 "staggered entities should be born before 'now'")

    def test_seed_initial_entities_are_valid(self):
        """seed_initial 创建的实体应有正确数据。"""
        f = self._make_factory(target=5)
        f.seed_initial(now=0.0)
        for e in f.entities:
            self.assertGreater(len(e.row_indices), 0)
            self.assertGreater(e.omega, 0)
            if e.entity_type == 'filament':
                self.assertGreater(e.blob_sigma_r, 0)
                self.assertGreater(e.blob_peak_density, 0)
            else:
                self.assertEqual(e.phi_density.shape[0], len(e.row_indices))
                self.assertEqual(e.phi_density.shape[1], N_PHI)

    def test_tick_removes_dead(self):
        """tick 应移除已死亡的实体。"""
        f = self._make_factory(target=10, lifetime_range=(5, 5),
                               fade_in=1.0, fade_out=1.0)
        f.seed_initial(now=0.0)
        initial_count = len(f.entities)
        self.assertEqual(initial_count, 10)
        # 时间推进到所有实体都死亡之后
        f.tick(now=100.0, dt=0.1)
        self.assertLess(len(f.entities), initial_count)

    def test_tick_spawns_replacements(self):
        """tick 应在有缺口时产生新实体。"""
        f = self._make_factory(target=10, lifetime_range=(2, 2),
                               fade_in=0.5, fade_out=0.5)
        f.seed_initial(now=0.0)
        # 推进足够时间让所有初始实体死亡
        now = 10.0
        f.tick(now=now, dt=1.0)
        # 再多次 tick 积累足够 spawn debt
        for _ in range(20):
            now += 0.5
            f.tick(now=now, dt=0.5)
        self.assertGreater(len(f.entities), 0, "factory should spawn replacements")

    def test_factory_with_hotspots(self):
        """工厂应能与 hotspot 生成函数协作。"""
        f = self._make_factory(spawn_fn=_spawn_single_hotspot, target=5)
        f.seed_initial(now=0.0)
        self.assertEqual(len(f.entities), 5)
        for e in f.entities:
            self.assertGreater(len(e.row_indices), 0)

    def test_factory_with_rt_spikes(self):
        """工厂应能与 RT spike 生成函数协作。"""
        f = self._make_factory(spawn_fn=_spawn_single_rt_spike, target=5)
        f.seed_initial(now=0.0)
        self.assertEqual(len(f.entities), 5)
        for e in f.entities:
            self.assertGreater(len(e.row_indices), 0)

    def test_alive_entities_property(self):
        """alive_entities 属性应返回当前实体列表。"""
        f = self._make_factory(target=5)
        f.seed_initial(now=0.0)
        self.assertEqual(len(f.alive_entities), 5)
        self.assertIs(f.alive_entities, f.entities)

    def test_steady_state_count(self):
        """经过多轮 tick 后实体数量应趋近 target_count。"""
        f = self._make_factory(target=30, lifetime_range=(5, 10),
                               fade_in=1.0, fade_out=1.0)
        f.seed_initial(now=0.0)
        now = 0.0
        dt = 0.1
        for _ in range(500):
            now += dt
            f.tick(now=now, dt=dt)
        count = len(f.entities)
        self.assertGreaterEqual(count, 15, f"count={count} too low")
        self.assertLessEqual(count, 45, f"count={count} too high")


class TestFilamentPhysicsLifecycle(unittest.TestCase):
    """验证 filament 物理驱动的生命周期。"""

    def _make_filament_entity(self, omega=0.1, sigma_phi0=0.008):
        return EntityInstance(
            row_indices=np.array([5, 6, 7]),
            phi_density=np.empty((0, 0), dtype=np.float32),
            phi_temp=np.empty((0, 0), dtype=np.float32),
            omega=omega,
            birth_time=0.0,
            lifetime=60.0,
            fade_in=0.0,
            fade_out=0.0,
            fade_noise=np.random.default_rng(0).random(32).astype(np.float32),
            entity_type='filament',
            source_phi=1.0,
            total_extent=2 * np.pi,
            alpha_shear=FILAMENT_SHEAR_ALPHA * omega,
            tau_cool=FILAMENT_TAU_COOL,
            blob_base_r=0.5,
            blob_sigma_r=0.004,
            blob_sigma_phi0=sigma_phi0,
            blob_peak_density=0.8,
            blob_peak_temp=0.2,
        )

    def test_density_factor_at_birth(self):
        e = self._make_filament_entity()
        self.assertAlmostEqual(e.density_factor(0.0), 1.0)

    def test_density_factor_decreases(self):
        e = self._make_filament_entity()
        d10 = e.density_factor(10.0)
        d30 = e.density_factor(30.0)
        self.assertLess(d30, d10)
        self.assertLess(d10, 1.0)

    def test_density_factor_inner_disk_faster(self):
        """内盘 filament (大 omega) 衰减更快。"""
        e_inner = self._make_filament_entity(omega=0.118)
        e_outer = self._make_filament_entity(omega=0.011)
        self.assertLess(e_inner.density_factor(20.0),
                        e_outer.density_factor(20.0))

    def test_filament_is_dead_by_threshold(self):
        e = self._make_filament_entity(omega=0.1)
        self.assertFalse(e.is_dead(now=5.0))
        for t in np.arange(0, FILAMENT_MAX_LIFETIME, 1.0):
            if e.density_factor(t) < FILAMENT_DEATH_THRESHOLD:
                self.assertTrue(e.is_dead(now=t))
                break

    def test_filament_is_dead_max_lifetime(self):
        e = self._make_filament_entity(omega=0.001)
        self.assertTrue(e.is_dead(now=FILAMENT_MAX_LIFETIME))

    def test_filament_factory_spawns_with_type(self):
        f = EntityFactory(
            spawn_fn=_spawn_single_filament,
            target_count=5,
            lifetime_range=(60.0, 120.0),
            fade_in=0.0, fade_out=0.0,
            n_r=N_R, n_phi=N_PHI,
            r_norm_all=R_NORM_ALL,
            omega_all=OMEGA_ALL,
            seed=42,
            entity_type='filament',
        )
        f.seed_initial(now=0.0)
        for e in f.entities:
            self.assertEqual(e.entity_type, 'filament')
            self.assertGreater(e.total_extent, 0)
            self.assertGreater(e.alpha_shear, 0)

    def test_non_filament_unchanged(self):
        """hotspot/rt_spike 应继续使用固定计时器。"""
        e = _make_entity(birth=0.0, lifetime=10.0, fade_in=2.0, fade_out=2.0)
        self.assertEqual(e.entity_type, 'generic')
        self.assertFalse(e.is_dead(now=5.0))
        self.assertTrue(e.is_dead(now=14.0))


if __name__ == "__main__":
    unittest.main()
