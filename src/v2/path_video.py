"""运镜视频的渲染编排：加载路径文件、按路径取相机渲染单帧、沿路径测光。

把 `camera_path`（相机怎么走）、`exposure_curve`（曝光怎么变）与 `DiskV2Renderer`（怎么成像）接起来，
供 CLI（`--v2_camera_path`、`--v2_camera_path_time`）与联系表脚本（`scripts/contact_sheet.py`）共用。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .camera_path import CameraPath, load_camera_path
from .exposure_curve import (ExposureCurve, FadeSettings, MeteringSettings, auto_exposure, load_fade_settings,
                             load_metering_settings, meter_path)
from .scene_anim import SceneAnim, knob_values_at, load_scene_anim

# 视频第 0 帧对应的物理时刻（r_s/c），与环绕视频一致
T_PHYS_START = 2000.0
# 测光帧的抖动种子：固定值使测光结果与帧序无关、可复现
METERING_SEED = 8


@dataclass(frozen=True)
class CameraPlan:
    """一个路径文件的全部内容。

    Attributes:
        spec: 路径文件原始内容（JSON 解析后的字典），用于续传校验。
        path: 相机路径。
        metering: 测光与曝光曲线参数。
        fade: 淡入淡出参数。
        anim: 场景旋钮轨道（`scene_anim` 块，可为空；见 `scene_anim`）。
    """

    spec: dict
    path: CameraPath
    metering: MeteringSettings
    fade: FadeSettings
    anim: SceneAnim


def load_camera_plan(file_path: str) -> CameraPlan:
    """读取并校验路径文件（格式见 `docs/plans/v2_camera_path_plan.md` §5.1）。

    Args:
        file_path: JSON 路径文件。

    Returns:
        `CameraPlan`。

    Raises:
        OSError: 文件无法读取。
        ValueError: 缺少必填字段、结构错误或取值非法（信息中给出字段名）；
            `exposure.reference_time` 超出路径时长。
        json.JSONDecodeError: 文件不是合法 JSON。
    """
    with open(file_path) as f:
        spec = json.load(f)
    path = load_camera_path(spec)
    metering = load_metering_settings(spec)
    if metering.reference_time > path.duration:
        raise ValueError(f"exposure.reference_time ({metering.reference_time}) 不能超过路径时长 ({path.duration})")
    return CameraPlan(spec=spec, path=path, metering=metering,
                      fade=load_fade_settings(spec), anim=load_scene_anim(spec))


def inner_orbit_period(r_in: float) -> float:
    """盘内缘的开普勒轨道周期 P_in（物理时间，r_s/c）。

    Args:
        r_in: 盘内缘半径（r_s，> 0）。

    Returns:
        正标量周期。

    Formula:
        P_in = 2π / Ω(r_in)，Ω(r) = √(0.5 / r³)（史瓦西几何单位下的开普勒角速度，r 以 r_s 计）。
    """
    return 2 * math.pi / math.sqrt(0.5 / r_in**3)


def time_scale(r_in: float, orbit_seconds: float) -> float:
    """物理时间与视频时间之比 dt_phys / dt_video：内缘转一圈对应 `orbit_seconds` 视频秒。

    Args:
        r_in: 盘内缘半径（r_s，> 0）。
        orbit_seconds: 内缘转一圈对应的视频秒数（`--v2_orbit_seconds`，有限正数）。

    Returns:
        正标量（(r_s/c) / 视频秒）。

    Raises:
        ValueError: `orbit_seconds` 不是有限正数。

    Formula:
        dt_phys / dt_video = P_in / orbit_seconds。
    """
    if not (math.isfinite(orbit_seconds) and orbit_seconds > 0):
        raise ValueError(f"orbit_seconds 必须是有限正数，得到 {orbit_seconds}")
    return inner_orbit_period(r_in) / orbit_seconds


def physical_time(t_video: float, r_in: float, orbit_seconds: float) -> float:
    """视频时刻 → 物理时刻（决定盘的平流相位）。

    Args:
        t_video: 视频时刻（秒）。
        r_in: 盘内缘半径（r_s）。
        orbit_seconds: 内缘转一圈对应的视频秒数（`--v2_orbit_seconds`）。

    Returns:
        物理时刻（r_s/c）。

    Formula:
        t_phys = 2000 + t_video·P_in / orbit_seconds（见 `time_scale`）。
    """
    return T_PHYS_START + t_video * time_scale(r_in, orbit_seconds)


def render_path_hdr(renderer, path: CameraPath, t_video: float, orbit_seconds: float, seed: int,
                    anim: SceneAnim | None = None, knob_defaults: dict | None = None):
    """按路径在视频时刻 t 积分一帧（可选：先应用场景旋钮轨道）。

    Args:
        renderer: `DiskV2Renderer`。
        path: 相机路径。
        t_video: 视频时刻（秒，0 ≤ t ≤ 路径时长，由调用方保证；区间外相机截断到端点而盘相位不截断）。
        orbit_seconds: 内缘转一圈对应的视频秒数。
        seed: 本帧抖动种子（≥ 1）。
        anim: 场景旋钮轨道；None = 不改旋钮（保持渲染器当前值）。
        knob_defaults: 旋钮默认值（CLI 参数；定义了轨道的旋钮以轨道为准，其余取这里的值，
            缺省的旋钮取 1）。仅在 `anim` 非 None 时使用。

    Returns:
        `renderer.render_hdr` 的返回值 `(hdr, sky)`。
    """
    if anim is not None:
        renderer.set_scene_knobs(**knob_values_at(anim, t_video, knob_defaults or {}))
    st = path.state_at(t_video, renderer.width / renderer.height)
    renderer.jitter_seed[None] = seed - 1  # render_hdr 内部先 +1
    return renderer.render_hdr(cam_pos=st.pos.tolist(), fov=st.fov,
                               t=physical_time(t_video, renderer.params.r_in, orbit_seconds),
                               forward=st.forward, roll_deg=st.roll)


def reference_exposure(renderer, path: CameraPath, orbit_seconds: float, t_ref: float,
                       seed: int = 1, anim: SceneAnim | None = None,
                       knob_defaults: dict | None = None) -> float:
    """锁定曝光 E₀ 的参考帧测光（`exposure.reference_time > 0` 时使用）。

    Args:
        renderer: 正片渲染器（与成片同分辨率、质量与超采样）。
        path: 相机路径。
        orbit_seconds: 内缘转一圈对应的视频秒数。
        t_ref: 参考时刻（秒，`exposure.reference_time`，已由 `load_camera_plan` 校验不超时长）。
        seed: 抖动种子；1 与视频第 0 帧相同，保证可复现。
        anim: 场景旋钮轨道；参考帧必须与正片帧在同一旋钮状态下测光（如 `disk_weight`
            从 0 渐起的镜头，参考时刻盘已亮起）。
        knob_defaults: 旋钮默认值（CLI 参数；轨道覆盖同名项）。

    Returns:
        自动曝光系数 E₀ = `auto_exposure(参考帧 HDR)`；参考帧没有盘区像素时为 1.0
        （与 `finish` / `auto_exposure` 的回退一致）。
    """
    hdr, _ = render_path_hdr(renderer, path, t_ref, orbit_seconds, seed,
                             anim=anim, knob_defaults=knob_defaults)
    return auto_exposure(hdr)


def meter_plan(meter_renderer, plan: CameraPlan, base_ev: float, orbit_seconds: float,
               log: Callable[[str], None] = print, knob_defaults: dict | None = None) -> ExposureCurve:
    """用低分辨率渲染器沿路径测光，返回曝光曲线（测光帧同样应用场景旋钮轨道）。

    Args:
        meter_renderer: 分辨率为 `plan.metering.size` 的 `DiskV2Renderer`（建议优化级别 3、超采样倍率 1）。
        plan: 路径文件内容。
        base_ev: 曝光补偿（档，`--v2_exposure_ev`）。
        orbit_seconds: 内缘转一圈对应的视频秒数。
        log: 进度输出函数。
        knob_defaults: 场景旋钮的 CLI 默认值（轨道覆盖同名项；未定义轨道且缺省时取 1）。

    Returns:
        `ExposureCurve`。
    """
    def hdr_at(t: float) -> np.ndarray:
        """测光帧：时刻 t 的盘发射 HDR，形状 `(H, W, 3)`（分辨率为 `plan.metering.size`）。"""
        return render_path_hdr(meter_renderer, plan.path, t, orbit_seconds, METERING_SEED,
                               anim=plan.anim, knob_defaults=knob_defaults)[0]

    return meter_path(hdr_at, plan.path.duration, base_ev, plan.metering, log=log)
