"""场景旋钮轨道：路径文件 `scene_anim` 块的读取、校验与求值。

三个旋钮（`gravity_scale` k / `doppler_scale` d / `disk_weight` w，定义见
`docs/plans/kids_explainer_plan.md` §3）可随视频时间变化；路径文件用关键帧轨道描述。
本模块是纯 NumPy 实现、不 import Taichi——制片层的遥控器叠加层要在自己的进程里逐帧
求值同一条轨道，保证叠加层与渲染画面同步（`docs/design_film.md` §4.6）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .camera_path import _smootherstep, optional_block

KNOB_NAMES = ("gravity_scale", "doppler_scale", "disk_weight", "disk_radius_scale")
"""允许出现在 `scene_anim` 块中的旋钮名（其余一律报错，防止拼错静默失效）。"""


@dataclass(frozen=True)
class SceneAnim:
    """若干条旋钮轨道的集合。

    Attributes:
        tracks: 旋钮名 → (`times`, `values`)，时刻（秒）严格递增，值 ∈ [0, 1]。
    """

    tracks: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def values_at(self, t: float) -> Dict[str, float]:
        """求 t 时刻各条轨道的值。

        Args:
            t: 视频时刻（秒）。

        Returns:
            dict：只有定义了轨道的旋钮；首帧之前、末帧之后保持端值。

        Formula:
            相邻关键帧 (t0, v0) → (t1, v1) 之间：v(t) = v0 + (v1 − v0)·ss(u)，
            u = (t − t0)/(t1 − t0)，ss 为五次缓动（与摇镜 `Pan.weight` 相同，两端速度、加速度为 0）。
        """
        result: Dict[str, float] = {}
        for name, (times, values) in self.tracks.items():
            if t <= times[0]:
                result[name] = float(values[0])
            elif t >= times[-1]:
                result[name] = float(values[-1])
            else:
                idx = int(np.searchsorted(times, t, side="right")) - 1
                t0, t1 = times[idx], times[idx + 1]
                u = np.float64((t - t0) / (t1 - t0))
                result[name] = float(values[idx] + (values[idx + 1] - values[idx]) * _smootherstep(u))
        return result


def knob_values_at(anim: SceneAnim, t: float, defaults: Dict[str, float]) -> Dict[str, float]:
    """合并旋钮值：轨道覆盖 CLI 默认值。

    Args:
        anim: 路径文件的旋钮轨道（可为空）。
        t: 视频时刻（秒）。
        defaults: 各旋钮的默认值（来自 CLI 参数，键为 `KNOB_NAMES` 的子集）。

    Returns:
        dict：`defaults` 的拷贝，其中定义了轨道的旋钮被 `anim.values_at(t)` 的值覆盖
        （"同一旋钮同时有轨道和 CLI 参数时以轨道为准"）。
    """
    values = dict(defaults)
    values.update(anim.values_at(t))
    return values


def load_scene_anim(spec: dict) -> SceneAnim:
    """由路径文件内容构造 `SceneAnim`（无 `scene_anim` 块时返回空轨道）。

    Args:
        spec: 路径文件 JSON 解析后的字典。

    Returns:
        `SceneAnim`。

    Raises:
        ValueError: `scene_anim` 不是对象；旋钮名未知；轨道不是 `[时刻, 值]` 列表的列表；
            时刻非有限、非严格递增；值非有限或不在 [0, 1]。信息中给出旋钮名。
    """
    block = optional_block(spec, "scene_anim")
    tracks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, raw in block.items():
        if name not in KNOB_NAMES:
            raise ValueError(f"scene_anim.{name} 不是已知的旋钮（{', '.join(KNOB_NAMES)}）")
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"scene_anim.{name} 必须是非空的 [时刻, 值] 列表的列表")
        points = []
        for entry in raw:
            if (not isinstance(entry, (list, tuple)) or len(entry) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in entry)):
                raise ValueError(f"scene_anim.{name} 的每个关键帧必须是 [时刻, 值]，得到 {entry!r}")
            points.append((float(entry[0]), float(entry[1])))
        times = np.array([p[0] for p in points])
        values = np.array([p[1] for p in points])
        if not (np.all(np.isfinite(times)) and np.all(np.isfinite(values))):
            raise ValueError(f"scene_anim.{name} 的时刻与值必须是有限数")
        if not np.all(np.diff(times) > 0):
            raise ValueError(f"scene_anim.{name} 的时刻必须严格递增，得到 {times.tolist()}")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"scene_anim.{name} 的值必须落在 [0, 1]，得到 {values.tolist()}")
        tracks[name] = (times, values)
    return SceneAnim(tracks=tracks)
