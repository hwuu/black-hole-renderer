"""V2 相机与 V1 `build_camera` 一致性单测。"""

from __future__ import annotations

import unittest

import numpy as np

from disk_v2.camera import build_camera_v1_compatible
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


if __name__ == "__main__":
    unittest.main()
