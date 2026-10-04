"""V2 相机构建（与 `render.build_camera` 完全一致，避免构图偏差）。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def build_camera_v1_compatible(
    cam_pos: List[float] | np.ndarray,
    fov_deg: float,
    width: int,
    height: int,
    camera_roll_deg: float = 0.0,
    forward: Optional[List[float] | np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    """构建与 V1 `render.build_camera` 相同的相机基向量与像素步长，可选绕光轴滚转、可选指定朝向。

    Args:
        cam_pos: 相机位置 `[x, y, z]`（Schwarzschild 几何单位 `r_s`）。
        fov_deg: 垂直方向 FOV（度），与 V1 CLI `--fov` 一致。
        width: 图像宽度（像素）。
        height: 图像高度（像素）。
        camera_roll_deg: 相机滚转角 ρ（度）：相机绕自身光轴（forward）旋转。
            在画面坐标系中生效，环绕（orbit）过程中倾斜恒定；+12.5° 为画面左低右高，
            负值反向。0 = 与 V1 `build_camera` 完全一致。
        forward: 相机光轴方向（世界系 3 维向量，任意非零长度，内部归一化）；None = 看向原点
            `−cam_pos / |cam_pos|`（与旧实现逐位一致）。运镜路径用它把黑洞放到指定画面位置
            （`src/v2/camera_path.py`）。不能与世界 up（+z）平行，否则 right 退化为 +x。

    Returns:
        `(cam_pos, cam_right, cam_up, cam_forward, pixel_width, pixel_height, top_left)`
        其中 `top_left` 为 V1 光追内核使用的图像平面左上角世界坐标；基向量均为单位向量，
        `(right, up, forward)` 构成正交基。

    Formula:
        f = forward / |forward|（未给定时 f = −p / |p|）
        r = normalize(f × ẑ)，u = normalize(r × f)
        r' = cos ρ · r − sin ρ · u
        u' = sin ρ · r + cos ρ · u
        （右手系绕 forward 轴旋转 ρ；世界点在画面中随基向量反向转动，
        +ρ 使画面右侧内容上移 → 左低右高）

    Physical Meaning:
        针孔相机的姿态：光轴 f、画面水平方向 r'、画面竖直方向 u'；"世界 up = +z"使画面水平线
        在 ρ = 0 时与盘面平行。

    Simplifications:
        静止观者标架（不含相机速度带来的光行差）；像平面在光轴前 1 个单位处。
    """
    cam_pos_arr = np.asarray(cam_pos, dtype=np.float64)
    if forward is None:
        cam_forward = -cam_pos_arr / np.linalg.norm(cam_pos_arr)
    else:
        fwd = np.asarray(forward, dtype=np.float64)
        cam_forward = fwd / np.linalg.norm(fwd)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cam_right = np.cross(cam_forward, world_up)
    rn = np.linalg.norm(cam_right)
    if rn < 1e-6:
        cam_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        cam_right /= rn
    cam_up = np.cross(cam_right, cam_forward)
    cam_up /= np.linalg.norm(cam_up)

    # 相机滚转：基向量 (right, up) 绕 forward 轴旋转（画面系固定，orbit 时倾斜恒定）
    roll = np.radians(camera_roll_deg)
    if roll != 0.0:
        cam_right, cam_up = (
            np.cos(roll) * cam_right - np.sin(roll) * cam_up,
            np.sin(roll) * cam_right + np.cos(roll) * cam_up,
        )

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
