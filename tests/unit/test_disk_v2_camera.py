"""V2 相机与 V1 `build_camera` 一致性单测。"""

from __future__ import annotations

import unittest

import numpy as np

from src.v2.camera import build_camera_v1_compatible
from render import build_camera


class TestDiskV2CameraParity(unittest.TestCase):
    """V2 相机 helper 必须与 V1 build_camera 输出一致。"""

    def test_matches_v1_build_camera(self):
        cam_pos = [6.0, 0.0, 2.0]
        fov = 90.0
        width, height = 1280, 720

        v1 = build_camera(np.array(cam_pos), fov, width, height)
        v2 = build_camera_v1_compatible(cam_pos, fov, width, height)

        for a, b in zip(v1[:5], v2[:5]):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)

        cp, cr, cu, cf, pw, ph = v1
        center = cp + cf * 1.0
        tl_v1 = center - cr * (pw * width / 2.0) + cu * (ph * height / 2.0)
        np.testing.assert_allclose(tl_v1, v2[6], rtol=0, atol=1e-6)


class TestCameraRoll(unittest.TestCase):
    """相机滚转：基向量绕 forward 旋转，画面内容反向倾斜。"""

    def test_roll_zero_identity(self):
        """ρ = 0 时与不滚转完全一致。"""
        cam = [0.0, -35.0, 3.0]
        a = build_camera_v1_compatible(cam, 34.0, 640, 360)
        b = build_camera_v1_compatible(cam, 34.0, 640, 360, camera_roll_deg=0.0)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_roll_rotates_basis_around_forward(self):
        """ρ = 90° 时 right → 原 up 方向的相反量级关系：正交性与模长保持，forward 不变。"""
        cam = [0.0, -35.0, 3.0]
        _, r0, u0, f0, *_ = build_camera_v1_compatible(cam, 34.0, 640, 360)
        _, r1, u1, f1, *_ = build_camera_v1_compatible(cam, 34.0, 640, 360, camera_roll_deg=90.0)
        np.testing.assert_allclose(f1, f0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(r1), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(u1), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.dot(r1, u1), 0.0, atol=1e-12)
        # 绕 forward 右手旋转 90°：r' = -u0（f × r = -u），u' = r0
        np.testing.assert_allclose(r1, -u0, atol=1e-12)
        np.testing.assert_allclose(u1, r0, atol=1e-12)

    def test_positive_roll_gives_left_low_right_high(self):
        """+ρ 时画面右侧（世界 +x，相机在 -y）的盘面点上移 → 画面左低右高。"""
        cam = [0.0, -35.0, 3.0]
        _, r1, u1, f1, *_ = build_camera_v1_compatible(cam, 34.0, 640, 360, camera_roll_deg=12.5)
        # 盘面右侧远点 P（世界 +x 方向）：原画面坐标 (x>0, 0)，滚转后 y' = P·u' > 0 → 上移
        p = np.array([30.0, 0.0, 0.0])  # 盘外缘右侧
        y_after = np.dot(p, u1) / np.linalg.norm(p)
        self.assertGreater(y_after, 0.0, "画面右侧应上移（左低右高）")
        # 对称验证：左侧点下移
        y_left = np.dot(-p, u1) / np.linalg.norm(p)
        self.assertLess(y_left, 0.0, "画面左侧应下移")


class TestCameraForward(unittest.TestCase):
    """指定相机朝向：不传时与看向原点逐位一致，传入时光轴等于给定方向。"""

    def test_forward_none_is_bit_identical(self):
        """forward = None 与显式传入 −p/|p| 的结果逐位一致。"""
        cam = [3.0, -22.0, 0.3]
        a = build_camera_v1_compatible(cam, 50.0, 640, 360, camera_roll_deg=13.0)
        b = build_camera_v1_compatible(cam, 50.0, 640, 360, camera_roll_deg=13.0, forward=None)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_forward_sets_optical_axis(self):
        """给定任意长度的 forward：返回的光轴为其单位向量，基向量正交归一。"""
        cam = [3.0, -22.0, 0.3]
        fwd = np.array([0.4, 2.0, -0.1])
        _, r, u, f, *_ = build_camera_v1_compatible(cam, 50.0, 640, 360, camera_roll_deg=13.0, forward=fwd)
        np.testing.assert_allclose(f, fwd / np.linalg.norm(fwd), atol=1e-12)
        for v in (r, u):
            np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-12)
            np.testing.assert_allclose(np.dot(v, f), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.dot(r, u), 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
