"""实体生命周期系统性能与视觉验证测试。

验证各阶段耗时在 30 FPS 预算内，并生成测试帧验证视觉输出。
"""
import unittest
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def _make_full_setup():
    """创建接近生产分辨率的 renderer + factories。"""
    from render import (
        load_or_generate_skybox, generate_disk_texture, TaichiRenderer,
        EntityFactory,
        _spawn_single_filament, _spawn_single_hotspot, _spawn_single_rt_spike,
        compute_disk_texture_resolution,
    )
    width, height = 640, 360
    cam_pos = [20, 0, 2]
    fov = 60
    r_inner, r_outer = 2.0, 15.0

    n_phi, n_r = compute_disk_texture_resolution(
        width, height, cam_pos, fov, r_inner, r_outer)

    skybox, _, _ = load_or_generate_skybox(None, 512, 256, 100)
    disk_tex = generate_disk_texture(
        n_phi=n_phi, n_r=n_r, seed=42,
        r_inner=r_inner, r_outer=r_outer)

    renderer = TaichiRenderer(
        width, height, skybox, disk_tex, device="cpu",
        r_disk_inner=r_inner, r_disk_outer=r_outer)
    renderer.init_background_layer(n_r=n_r, n_phi=n_phi, seed=42)

    r_norm_all = np.linspace(0, 1, n_r)
    r_vals = r_inner + (r_outer - r_inner) * r_norm_all
    omega_all = np.sqrt(0.5 / (r_vals ** 3 + 1e-6)).astype(np.float32)

    factories = {
        'filament': EntityFactory(
            _spawn_single_filament, target_count=200,
            lifetime_range=(10.0, 20.0), fade_in=2.0, fade_out=2.0,
            n_r=n_r, n_phi=n_phi,
            r_norm_all=r_norm_all, omega_all=omega_all, seed=100),
        'hotspot': EntityFactory(
            _spawn_single_hotspot, target_count=30,
            lifetime_range=(8.0, 15.0), fade_in=2.0, fade_out=2.0,
            n_r=n_r, n_phi=n_phi,
            r_norm_all=r_norm_all, omega_all=omega_all, seed=200),
        'rt_spike': EntityFactory(
            _spawn_single_rt_spike, target_count=15,
            lifetime_range=(8.0, 15.0), fade_in=1.5, fade_out=1.5,
            n_r=n_r, n_phi=n_phi,
            r_norm_all=r_norm_all, omega_all=omega_all, seed=300),
    }
    for f in factories.values():
        f.seed_initial(now=0.0)

    return renderer, factories, n_r, n_phi, cam_pos, fov


class TestLifecyclePerformance(unittest.TestCase):
    """性能基准测试：验证各阶段耗时在预算内。"""

    @classmethod
    def setUpClass(cls):
        cls.renderer, cls.factories, cls.n_r, cls.n_phi, cls.cam_pos, cls.fov = (
            _make_full_setup())
        # 预热（首次 kernel 编译）
        cls.renderer.generate_background(t=0.0)
        cls.renderer.accumulate_entity_layer(cls.factories, now=0.0)
        cls.renderer.recompute_interactive_stats()
        cls.renderer.compose_interactive_texture()

    def _benchmark(self, fn, n_iter=5):
        """多次运行取中位数（排除首次编译）。"""
        times = []
        for i in range(n_iter):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return np.median(times) * 1000  # ms

    def test_background_kernel_time(self):
        """GPU 背景层生成应 < 500ms（CPU 模式上限）。"""
        ms = self._benchmark(
            lambda: self.renderer.generate_background(t=1.0))
        print(f"\n  background kernel: {ms:.1f} ms "
              f"({self.n_r}x{self.n_phi})")
        self.assertLess(ms, 500, f"background too slow: {ms:.1f}ms")

    def test_entity_accumulation_time(self):
        """实体累加应 < 200ms（CPU numpy 模式上限）。"""
        n_entities = sum(len(f.entities) for f in self.factories.values())
        ms = self._benchmark(
            lambda: self.renderer.accumulate_entity_layer(
                self.factories, now=5.0))
        print(f"\n  entity accumulation: {ms:.1f} ms "
              f"({n_entities} entities)")
        self.assertLess(ms, 200, f"entity accumulation too slow: {ms:.1f}ms")

    def test_compose_texture_time(self):
        """纹理合成 + mipmap 应 < 50ms。"""
        ms = self._benchmark(
            lambda: self.renderer.compose_interactive_texture())
        print(f"\n  compose + mipmap: {ms:.1f} ms")
        self.assertLess(ms, 50, f"compose too slow: {ms:.1f}ms")

    def test_stats_recompute_time(self):
        """统计量重算应 < 100ms（包含 GPU→CPU readback）。"""
        ms = self._benchmark(
            lambda: self.renderer.recompute_interactive_stats())
        print(f"\n  stats recompute: {ms:.1f} ms")
        self.assertLess(ms, 100, f"stats recompute too slow: {ms:.1f}ms")

    def test_total_frame_budget(self):
        """完整帧（背景+累加+合成，不含光追）应 < 800ms CPU。"""
        def full_frame():
            self.renderer.generate_background(t=2.0)
            self.renderer.accumulate_entity_layer(self.factories, now=2.0)
            self.renderer.compose_interactive_texture()

        ms = self._benchmark(full_frame)
        print(f"\n  full texture frame: {ms:.1f} ms "
              f"(target < 800ms for CPU)")
        self.assertLess(ms, 800, f"total frame too slow: {ms:.1f}ms")


class TestLifecycleVisual(unittest.TestCase):
    """视觉验证：确保生命周期系统产生有效纹理。"""

    @classmethod
    def setUpClass(cls):
        cls.renderer, cls.factories, cls.n_r, cls.n_phi, cls.cam_pos, cls.fov = (
            _make_full_setup())

    def test_texture_has_structure(self):
        """合成纹理应有非平凡的空间结构（std > 0）。"""
        self.renderer.generate_background(t=0.0)
        self.renderer.accumulate_entity_layer(self.factories, now=0.0)
        self.renderer.recompute_interactive_stats()
        self.renderer.compose_interactive_texture()

        tex = self.renderer.disk_texture_field.to_numpy()
        rgb = tex[:, :, :3]
        alpha = tex[:, :, 3]

        self.assertGreater(rgb.std(), 0.01,
                           "texture RGB has no spatial structure")
        self.assertGreater(alpha.max(), 0.01,
                           "texture alpha is all zero")
        self.assertGreater(alpha.std(), 0.001,
                           "texture alpha has no variation")

    def test_texture_changes_over_time(self):
        """不同时间的纹理应有差异（时间演化在工作）。"""
        self.renderer.generate_background(t=0.0)
        self.renderer.accumulate_entity_layer(self.factories, now=0.0)
        self.renderer.recompute_interactive_stats()
        self.renderer.compose_interactive_texture()
        tex_t0 = self.renderer.disk_texture_field.to_numpy().copy()

        self.renderer.generate_background(t=5.0)
        self.renderer.accumulate_entity_layer(self.factories, now=5.0)
        self.renderer.compose_interactive_texture()
        tex_t5 = self.renderer.disk_texture_field.to_numpy()

        diff = np.abs(tex_t0 - tex_t5).mean()
        self.assertGreater(diff, 0.001,
                           "texture not changing over time")

    def test_no_nan_or_inf(self):
        """纹理不应包含 NaN 或 Inf。"""
        self.renderer.generate_background(t=3.0)
        self.renderer.accumulate_entity_layer(self.factories, now=3.0)
        self.renderer.recompute_interactive_stats()
        self.renderer.compose_interactive_texture()

        tex = self.renderer.disk_texture_field.to_numpy()
        self.assertFalse(np.any(np.isnan(tex)), "texture contains NaN")
        self.assertFalse(np.any(np.isinf(tex)), "texture contains Inf")

    def test_full_render_produces_image(self):
        """完整光追渲染应产生有效图像。"""
        self.renderer.generate_background(t=0.0)
        self.renderer.accumulate_entity_layer(self.factories, now=0.0)
        self.renderer.recompute_interactive_stats()
        self.renderer.compose_interactive_texture()

        img = self.renderer.render(list(self.cam_pos), self.fov, frame=0)
        self.assertEqual(img.shape[2], 3)
        self.assertGreater(img.max(), 0.01, "rendered image is all black")
        self.assertFalse(np.any(np.isnan(img)), "rendered image contains NaN")

    def test_entity_counts_at_steady_state(self):
        """模拟多帧后实体数量应在目标附近。"""
        now = 0.0
        dt = 0.05
        for _ in range(200):
            now += dt
            for f in self.factories.values():
                f.tick(now=now, dt=dt)

        counts = {k: len(f.entities) for k, f in self.factories.items()}
        print(f"\n  entity counts at t={now:.1f}s: {counts}")
        self.assertGreaterEqual(counts['filament'], 100)
        self.assertLessEqual(counts['filament'], 300)
        self.assertGreaterEqual(counts['hotspot'], 10)
        self.assertLessEqual(counts['hotspot'], 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
