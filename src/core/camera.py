from typing import Tuple, List, Optional
import numpy as np


def build_camera(cam_pos: np.ndarray, fov_deg: float, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    构建相机参数

    参数:
        cam_pos: 相机位置 [x, y, z]
        fov_deg: 视野角度（度）
        width: 图像宽度（像素）
        height: 图像高度（像素）

    返回:
        (cam_pos, cam_right, cam_up, cam_forward, pixel_width, pixel_height)
    """
    cam_pos = np.array(cam_pos, dtype=np.float64)
    cam_forward = -cam_pos / np.linalg.norm(cam_pos)

    world_up = np.array([0.0, 0.0, 1.0])
    cam_right = np.cross(cam_forward, world_up)
    rn = np.linalg.norm(cam_right)
    if rn < 1e-6:
        cam_right = np.array([1.0, 0.0, 0.0])
    else:
        cam_right /= rn
    cam_up = np.cross(cam_right, cam_forward)
    cam_up /= np.linalg.norm(cam_up)

    fov_rad = np.radians(fov_deg)
    aspect = width / height
    image_plane_height = 2.0 * np.tan(fov_rad / 2)
    image_plane_width = image_plane_height * aspect

    pixel_width = image_plane_width / width
    pixel_height = image_plane_height / height

    return cam_pos, cam_right, cam_up, cam_forward, pixel_width, pixel_height
