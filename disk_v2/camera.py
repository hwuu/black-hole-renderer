"""V2 相机构建（与 `render.build_camera` 完全一致，避免构图偏差）。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def build_camera_v1_compatible(
    cam_pos: List[float] | np.ndarray,
    fov_deg: float,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    """构建与 V1 `render.build_camera` 相同的相机基向量与像素步长。

    Args:
        cam_pos: 相机位置 `[x, y, z]`（Schwarzschild 几何单位 `r_s`）。
        fov_deg: 垂直方向 FOV（度），与 V1 CLI `--fov` 一致。
        width: 图像宽度（像素）。
        height: 图像高度（像素）。

    Returns:
        `(cam_pos, cam_right, cam_up, cam_forward, pixel_width, pixel_height, top_left)`
        其中 `top_left` 为 V1 光追内核使用的图像平面左上角世界坐标。
    """
    cam_pos_arr = np.asarray(cam_pos, dtype=np.float64)
    cam_forward = -cam_pos_arr / np.linalg.norm(cam_pos_arr)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cam_right = np.cross(cam_forward, world_up)
    rn = np.linalg.norm(cam_right)
    if rn < 1e-6:
        cam_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        cam_right /= rn
    cam_up = np.cross(cam_right, cam_forward)
    cam_up /= np.linalg.norm(cam_up)

    fov_rad = np.radians(fov_deg)
    aspect = width / height
    image_plane_height = 2.0 * np.tan(fov_rad / 2.0)
    image_plane_width = image_plane_height * aspect

    pixel_width = image_plane_width / width
    pixel_height = image_plane_height / height

    center = cam_pos_arr + cam_forward * 1.0
    top_left = (
        center
        - cam_right * (pixel_width * width / 2.0)
        + cam_up * (pixel_height * height / 2.0)
    )

    return (
        cam_pos_arr,
        cam_right,
        cam_up,
        cam_forward,
        float(pixel_width),
        float(pixel_height),
        top_left,
    )
