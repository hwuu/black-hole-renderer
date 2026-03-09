"""GPU 纹理合成单元测试：验证数据上传、统计量预计算和 GPU 纹理合成。"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


class TestUploadParametricState(unittest.TestCase):
    """验证 upload_parametric_state 的数据上传和统计量预计算。"""

    @classmethod
    def setUpClass(cls):
        from render import (
            build_disk_texture_rotating_state,
            compute_disk_texture_resolution,
            load_or_generate_skybox,
            generate_disk_texture,
            TaichiRenderer,
            _generate_disk_texture_rotating_from_state,
            _compose_disk_texture_from_fields,
            _compute_rotation_pixels,
            _roll_rows,
        )
        cam_pos = [20, 0, 2]
        fov = 60
        r_inner = 2.0
        r_outer = 15.0
        n_phi, n_r = compute_disk_texture_resolution(640, 360, cam_pos, fov, r_inner, r_outer)

        cls.state = build_disk_texture_rotating_state(
            n_phi=n_phi, n_r=n_r, seed=42,
            r_inner=r_inner, r_outer=r_outer,
            enable_rt=True, generation_scale=1
        )

        skybox, _, _ = load_or_generate_skybox(None, 512, 256, 100)
        disk_tex = generate_disk_texture(n_phi=n_phi, n_r=n_r, seed=42,
                                         r_inner=r_inner, r_outer=r_outer)

        cls.renderer = TaichiRenderer(
            640, 360, skybox, disk_tex,
            device="gpu", r_disk_inner=r_inner, r_disk_outer=r_outer
        )
        cls.renderer.upload_parametric_state(cls.state)

        cls._generate_from_state = _generate_disk_texture_rotating_from_state
        cls._compose = _compose_disk_texture_from_fields
        cls._compute_rot = _compute_rotation_pixels
        cls._roll_rows = _roll_rows

    def test_component_fields_uploaded_correctly(self):
        """验证 13 个组件 field 上传后可正确读回。"""
        readback = self.renderer._comp_field.to_numpy()
        components = [
            self.state.temp_base, self.state.spiral, self.state.spiral_temp,
            self.state.turbulence, self.state.turb_temp,
            self.state.arcs, self.state.arcs_temp,
            self.state.rt_spikes, self.state.rt_temp,
            self.state.hotspot, self.state.hotspot_temp,
            self.state.az_hotspot, self.state.disturb_mod,
        ]
        for idx, comp in enumerate(components):
            np.testing.assert_allclose(
                readback[idx], comp, atol=1e-6,
                err_msg=f"Component {idx} mismatch"
            )

    def test_omega_rows_uploaded(self):
        """验证 omega_rows field 上传正确。"""
        readback = self.renderer._omega_rows_field.to_numpy()
        np.testing.assert_allclose(readback, self.state.omega_rows, atol=1e-6)

    def test_edge_uploaded(self):
        """验证 edge field 上传正确。"""
        readback = self.renderer._edge_field.to_numpy()
        np.testing.assert_allclose(readback, self.state.edge, atol=1e-6)

    def test_statistics_match_cpu(self):
        """验证预计算统计量与 CPU 路径 _compose_disk_texture_from_fields 一致。"""
        state = self.state
        rt_weight = 0.20 if state.enable_rt else 0.0
        density = (0.15 + 0.10 * state.spiral + 0.30 * state.turbulence
                   + 0.20 * state.hotspot + 0.30 * state.arcs
                   + rt_weight * state.rt_spikes) * state.disturb_mod
        density *= state.edge[:, None]
        expected_p98 = float(np.percentile(density, 98))

        temp_struct = (state.spiral_temp + state.turb_temp + state.arcs_temp
                       + state.rt_temp + state.hotspot_temp) * state.disturb_mod
        pos_mask = temp_struct > 0
        expected_scale = float(np.percentile(temp_struct[pos_mask], 95))

        ts_scaled = np.clip(temp_struct / (expected_scale + 1e-6) * 0.8, 0, 1.2)
        expected_max = np.max(ts_scaled, axis=1)
        expected_p70 = np.quantile(ts_scaled, 0.7, axis=1)

        stats = self.renderer._param_stats_field.to_numpy()
        self.assertAlmostEqual(stats[0], expected_p98, places=5,
                               msg="density_p98 mismatch")
        self.assertAlmostEqual(stats[1], expected_scale, places=5,
                               msg="struct_scale mismatch")

        row_stats = self.renderer._param_row_stats_field.to_numpy()
        np.testing.assert_allclose(row_stats[:, 0], expected_max, atol=1e-5,
                                   err_msg="struct_max_per_r mismatch")
        np.testing.assert_allclose(row_stats[:, 1], expected_p70, atol=1e-5,
                                   err_msg="struct_p70_per_r mismatch")

    def test_parametric_gpu_ready_flag(self):
        """验证标志位已设置。"""
        self.assertTrue(self.renderer._parametric_gpu_ready)
        self.assertEqual(self.renderer._param_enable_rt, 1)
        self.assertIsInstance(self.renderer._param_color_temp, float)


class TestComposeDiskTextureKernel(unittest.TestCase):
    """验证 GPU _compose_disk_texture_kernel 与 CPU 路径像素级一致。"""

    @classmethod
    def setUpClass(cls):
        from render import (
            build_disk_texture_rotating_state,
            compute_disk_texture_resolution,
            load_or_generate_skybox,
            generate_disk_texture,
            TaichiRenderer,
            _generate_disk_texture_rotating_from_state,
        )
        cam_pos = [20, 0, 2]
        fov = 60
        r_inner = 2.0
        r_outer = 15.0
        n_phi, n_r = compute_disk_texture_resolution(640, 360, cam_pos, fov, r_inner, r_outer)

        cls.state = build_disk_texture_rotating_state(
            n_phi=n_phi, n_r=n_r, seed=42,
            r_inner=r_inner, r_outer=r_outer,
            enable_rt=True, generation_scale=1
        )

        skybox, _, _ = load_or_generate_skybox(None, 512, 256, 100)
        disk_tex = generate_disk_texture(n_phi=n_phi, n_r=n_r, seed=42,
                                         r_inner=r_inner, r_outer=r_outer)

        cls.renderer = TaichiRenderer(
            640, 360, skybox, disk_tex,
            device="gpu", r_disk_inner=r_inner, r_disk_outer=r_outer
        )
        cls.renderer.upload_parametric_state(cls.state)
        cls._generate_from_state = _generate_disk_texture_rotating_from_state

    def _compare_gpu_vs_cpu(self, t_offset: float, label: str):
        """对比 GPU kernel 与 CPU 路径在给定 t_offset 下的纹理输出。"""
        from render import _generate_disk_texture_rotating_from_state

        r = self.renderer
        r._compose_disk_texture_kernel(
            r.disk_texture_field, r._comp_field,
            r._omega_rows_field, r._edge_field,
            r._param_stats_field, r._param_row_stats_field,
            float(t_offset), r._param_enable_rt, r._param_color_temp
        )
        gpu_tex = r.disk_texture_field.to_numpy()

        cpu_tex = _generate_disk_texture_rotating_from_state(self.state, t_offset=t_offset)

        max_diff = np.max(np.abs(gpu_tex - cpu_tex))
        mean_diff = np.mean(np.abs(gpu_tex - cpu_tex))
        # GPU f32 vs CPU f64→f32 会有微小浮点差异，容差 1e-4
        self.assertLess(max_diff, 1e-4,
                        f"[{label}] max diff={max_diff:.2e}, mean={mean_diff:.2e}")

    def test_t_offset_0(self):
        """t_offset=0 无旋转。"""
        self._compare_gpu_vs_cpu(0.0, "t=0")

    def test_t_offset_5(self):
        """t_offset=5 小偏移。"""
        self._compare_gpu_vs_cpu(5.0, "t=5")

    def test_t_offset_50(self):
        """t_offset=50 中等偏移。"""
        self._compare_gpu_vs_cpu(50.0, "t=50")

    def test_t_offset_180(self):
        """t_offset=180 大偏移。"""
        self._compare_gpu_vs_cpu(180.0, "t=180")


class TestMipmapKernel(unittest.TestCase):
    """验证 GPU mipmap kernel 与 CPU generate_disk_mipmaps 输出一致。"""

    @classmethod
    def setUpClass(cls):
        from render import (
            build_disk_texture_rotating_state,
            compute_disk_texture_resolution,
            load_or_generate_skybox,
            generate_disk_texture,
            TaichiRenderer,
            _generate_disk_texture_rotating_from_state,
            generate_disk_mipmaps,
        )
        cam_pos = [20, 0, 2]
        fov = 60
        r_inner = 2.0
        r_outer = 15.0
        n_phi, n_r = compute_disk_texture_resolution(640, 360, cam_pos, fov, r_inner, r_outer)

        cls.state = build_disk_texture_rotating_state(
            n_phi=n_phi, n_r=n_r, seed=42,
            r_inner=r_inner, r_outer=r_outer,
            enable_rt=True, generation_scale=1
        )

        skybox, _, _ = load_or_generate_skybox(None, 512, 256, 100)
        disk_tex = generate_disk_texture(n_phi=n_phi, n_r=n_r, seed=42,
                                         r_inner=r_inner, r_outer=r_outer)

        cls.renderer = TaichiRenderer(
            640, 360, skybox, disk_tex,
            device="gpu", r_disk_inner=r_inner, r_disk_outer=r_outer
        )
        cls.renderer.upload_parametric_state(cls.state)
        cls.generate_disk_mipmaps = generate_disk_mipmaps

    def test_mipmap_matches_cpu(self):
        """GPU compose + mipmap 后，各级 mipmap 与 CPU 路径一致。"""
        from render import _generate_disk_texture_rotating_from_state, generate_disk_mipmaps

        r = self.renderer
        t_offset = 5.0

        # GPU: compose + mipmap
        r._compose_disk_texture_kernel(
            r.disk_texture_field, r._comp_field,
            r._omega_rows_field, r._edge_field,
            r._param_stats_field, r._param_row_stats_field,
            float(t_offset), r._param_enable_rt, r._param_color_temp
        )
        r._mipmap_copy_base_kernel(r.disk_mips_field, r.disk_texture_field)
        h, w = r.dtex_h, r.dtex_w
        for lev in range(1, r.num_mip_levels):
            r._mipmap_downsample_kernel(r.disk_mips_field, lev, h, w)
            h //= 2
            w //= 2

        gpu_mips = r.disk_mips_field.to_numpy()

        # CPU: generate texture + mipmap
        cpu_tex = _generate_disk_texture_rotating_from_state(
            self.state, t_offset=t_offset)
        cpu_mips = generate_disk_mipmaps(cpu_tex, levels=4)

        for lev, cpu_mip in enumerate(cpu_mips):
            mh, mw = cpu_mip.shape[:2]
            gpu_mip = gpu_mips[lev, :mh, :mw]
            max_diff = np.max(np.abs(gpu_mip - cpu_mip))
            self.assertLess(max_diff, 1e-3,
                            f"Mipmap level {lev}: max diff={max_diff:.2e}")


class TestUpdateDiskTextureGpu(unittest.TestCase):
    """验证 update_disk_texture_gpu 封装方法的完整流程。"""

    @classmethod
    def setUpClass(cls):
        from render import (
            build_disk_texture_rotating_state,
            compute_disk_texture_resolution,
            load_or_generate_skybox,
            generate_disk_texture,
            TaichiRenderer,
        )
        cam_pos = [20, 0, 2]
        fov = 60
        r_inner = 2.0
        r_outer = 15.0
        n_phi, n_r = compute_disk_texture_resolution(640, 360, cam_pos, fov, r_inner, r_outer)

        cls.state = build_disk_texture_rotating_state(
            n_phi=n_phi, n_r=n_r, seed=42,
            r_inner=r_inner, r_outer=r_outer,
            enable_rt=True, generation_scale=1
        )

        skybox, _, _ = load_or_generate_skybox(None, 512, 256, 100)
        disk_tex = generate_disk_texture(n_phi=n_phi, n_r=n_r, seed=42,
                                         r_inner=r_inner, r_outer=r_outer)

        cls.renderer = TaichiRenderer(
            640, 360, skybox, disk_tex,
            device="gpu", r_disk_inner=r_inner, r_disk_outer=r_outer
        )
        cls.renderer.upload_parametric_state(cls.state)

    def test_gpu_vs_cpu_update(self):
        """update_disk_texture_gpu 后纹理与 CPU update_disk_texture 结果一致。"""
        from render import _generate_disk_texture_rotating_from_state

        r = self.renderer
        t_offset = 25.0

        # GPU path
        r.update_disk_texture_gpu(t_offset)
        gpu_tex = r.disk_texture_field.to_numpy()

        # CPU path
        cpu_tex = _generate_disk_texture_rotating_from_state(
            self.state, t_offset=t_offset)

        max_diff = np.max(np.abs(gpu_tex - cpu_tex))
        self.assertLess(max_diff, 1e-4, f"max diff={max_diff:.2e}")

    def test_assert_without_upload(self):
        """未调用 upload_parametric_state 时应抛出 AssertionError。"""
        from render import (
            load_or_generate_skybox, generate_disk_texture,
            compute_disk_texture_resolution, TaichiRenderer,
        )
        # 注意：不能重新 ti.init，复用已有 renderer 但不调用 upload
        # 创建一个 mock 属性来测试
        import copy
        r = self.renderer
        old_flag = r._parametric_gpu_ready
        r._parametric_gpu_ready = False
        with self.assertRaises(AssertionError):
            r.update_disk_texture_gpu(0.0)
        r._parametric_gpu_ready = old_flag


if __name__ == "__main__":
    unittest.main()
