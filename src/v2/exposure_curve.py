"""运镜视频的逐帧曝光（测光 + 平滑）与淡入淡出。

设计见 `docs/plans/v2_camera_path_plan.md` §5.5–§5.6。运镜路径中画面亮度跨约 7 档（远景 → 云中 → 盘底），
首帧锁定曝光无法兼顾；逐帧自动曝光又会闪烁。做法是正式渲染前沿路径低分辨率测光，
把"自动曝光相对首帧的档数"平滑后按比例补偿，得到一条缓变的曝光曲线。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .camera_compose import gaussian_smooth
from .camera_path import optional_block
from .postfx import hdr_luminance

# 自动曝光：盘区亮度 p99.9 映射到该值（与 `DiskV2Renderer.finish` 一致）
AUTO_EXPOSURE_TARGET = 0.9
# 盘区像素：线性亮度高于该值（与 `DiskV2Renderer.finish` 一致）
DISK_LUMINANCE_FLOOR = 1e-4


@dataclass(frozen=True)
class MeteringSettings:
    """测光与曝光曲线参数（默认值为方案 v0.3 定稿值）。

    Attributes:
        step: 测光间隔（视频秒，> 0）。
        size: 测光渲染分辨率 `(宽, 高)`（像素）；只用亮度相对首帧的变化，低分辨率即可。
        smoothing: 测光档数的高斯平滑标准差（视频秒，≥ 0）。
        gain: 补偿比例 k（≥ 0）：1 = 完全抵消亮度变化（每帧亮度相同），0 = 不补偿（首帧锁定）；
            小于 1 时亮处仍更亮、暗处仍更暗。
        ev_range: 补偿量的夹取范围 `(下限, 上限)`（档），叠加在曝光补偿 `--v2_exposure_ev` 之上。
    """

    step: float = 0.5
    size: Tuple[int, int] = (128, 72)
    smoothing: float = 1.5
    gain: float = 0.6
    ev_range: Tuple[float, float] = (-0.9, 1.1)


@dataclass(frozen=True)
class FadeSettings:
    """淡入淡出参数。

    Attributes:
        fade_in: 淡入时长（秒，≥ 0）。
        fade_out: 淡出时长（秒，≥ 0）。
    """

    fade_in: float = 2.0
    fade_out: float = 3.0


def disk_level(hdr: np.ndarray) -> Optional[float]:
    """盘区亮度水平：盘区像素线性亮度的 99.9 百分位（自动曝光所依据的量）。

    Args:
        hdr: 盘发射线性 HDR，形状 `(H, W, 3)`。

    Returns:
        正标量（线性亮度）；画面中没有盘区像素时返回 None。

    Formula:
        L = 0.2126·R + 0.7152·G + 0.0722·B（Rec.709 亮度），取 L > 1e-4 的像素的 p99.9。
    """
    lum = hdr_luminance(hdr)
    lum = lum[lum > DISK_LUMINANCE_FLOOR]
    return float(np.percentile(lum, 99.9)) if lum.size else None


def auto_exposure(hdr: np.ndarray) -> float:
    """自动曝光系数（不含曝光补偿），与 `DiskV2Renderer.finish` 的自动曝光一致。

    Args:
        hdr: 盘发射线性 HDR，形状 `(H, W, 3)`。

    Returns:
        正标量曝光系数：有盘区像素时为 `0.9 / p99.9(L)`，乘到 HDR 上使盘区 p99.9 等于 0.9；
        画面中没有盘区像素时为 1.0（与 `finish` 的回退值相同）。
    """
    level = disk_level(hdr)
    return AUTO_EXPOSURE_TARGET / level if level is not None else 1.0


class ExposureCurve:
    """曝光补偿曲线 ev(t)（档），测光点之间线性插值。

    Args:
        times: 测光时刻（秒），形状 `(M,)`，严格递增。
        ev: 各测光时刻的曝光补偿（档），形状 `(M,)`。
    """

    def __init__(self, times: np.ndarray, ev: np.ndarray):
        """保存测光时刻与曝光补偿。

        Args:
            times: 测光时刻（秒），形状 `(M,)`，严格递增。
            ev: 各测光时刻的曝光补偿（档），形状 `(M,)`。
        """
        self.times = np.asarray(times, dtype=np.float64)
        self.ev = np.asarray(ev, dtype=np.float64)

    def ev_at(self, t: float) -> float:
        """时刻 t 的曝光补偿。

        Args:
            t: 视频时刻（秒）。

        Returns:
            标量曝光补偿（档）；测光点之间线性插值，超出测光范围时取端点值。曝光倍率为 `2^ev`。
        """
        return float(np.interp(t, self.times, self.ev))

    @classmethod
    def from_levels(cls, times: np.ndarray, levels: np.ndarray, base_ev: float,
                    settings: MeteringSettings) -> "ExposureCurve":
        """由各测光时刻的盘区亮度水平构造曝光曲线。

        Args:
            times: 测光时刻（秒），形状 `(M,)`，间隔为 `settings.step`。
            levels: 各时刻的盘区亮度水平 `disk_level(hdr)`，形状 `(M,)`；正值，NaN 表示该时刻画面中没有盘区像素。
            base_ev: 曝光补偿 E_c（档，`--v2_exposure_ev`）。
            settings: 测光参数。

        Returns:
            `ExposureCurve`；亮度水平恒定时曝光补偿恒等于 `base_ev`。

        Formula:
            a(t) = −(log₂ L(t) − log₂ L(0))：让该时刻盘区 p99.9 回到首帧水平需要补偿的档数（正 = 需提亮）；
            ā = a ∗ G_σ（σ = smoothing / step 个测光点）；
            ev(t) = E_c + clip(k·ā(t), ev_lo, ev_hi)；
            缺失的测光点（NaN）先用相邻有效点的 log₂ L 线性插值补齐（两端取最近有效值），全部缺失时 a ≡ 0。

        Physical Meaning:
            模拟摄影师随场景缓慢调整曝光：亮度变化被部分补偿（k < 1），保留"进入云中变亮、
            盘底变暗"的明暗叙事，又不至于全白或全黑。

        Simplifications:
            只看盘区 p99.9（与自动曝光一致），不看天空与平均亮度；平滑窗口固定，不区分亮度变快变慢。
        """
        log_level = np.log2(np.asarray(levels, dtype=np.float64))
        valid = np.isfinite(log_level)
        if not valid.any():
            log_level = np.zeros(len(times))
        elif not valid.all():
            log_level = np.interp(times, np.asarray(times)[valid], log_level[valid])
        a = -(log_level - log_level[0])
        a_smooth = gaussian_smooth(a, settings.smoothing / settings.step)
        lo, hi = settings.ev_range
        return cls(times, base_ev + np.clip(settings.gain * a_smooth, lo, hi))


def meter_path(render_hdr_at: Callable[[float], np.ndarray], duration: float, base_ev: float,
               settings: MeteringSettings, log: Callable[[str], None] = print) -> ExposureCurve:
    """沿运镜路径测光，返回曝光曲线。

    Args:
        render_hdr_at: `render_hdr_at(t)` 返回视频时刻 t 的盘发射 HDR（`(H, W, 3)`，分辨率为 `settings.size`）。
        duration: 视频时长（秒）。
        base_ev: 曝光补偿（档，`--v2_exposure_ev`）。
        settings: 测光参数。
        log: 进度输出函数。

    Returns:
        `ExposureCurve`，测光时刻为 `0, step, 2·step, …`（不超过 duration）。
    """
    times = np.arange(0.0, duration + 1e-9, settings.step)
    levels = np.array([np.nan if (lv := disk_level(render_hdr_at(float(t)))) is None else lv for t in times])
    curve = ExposureCurve.from_levels(times, levels, base_ev, settings)
    log(f"[运镜] 测光 {len(times)} 点：曝光补偿 {curve.ev.min():+.2f} … {curve.ev.max():+.2f} 档")
    return curve


def fade_factor(t: float, duration: float, settings: FadeSettings) -> float:
    """淡入淡出系数 α(t)，乘到 sRGB 编码后的画面上。

    Args:
        t: 视频时刻（秒）。
        duration: 视频时长 T（秒）。
        settings: 淡入淡出参数；时长为 0 时对应一侧不淡变。

    Returns:
        标量，值域 [0, 1]；t = 0 与 t = T 处为 0（对应时长 > 0 时），中段为 1。视频末帧
        t_last = (round(T·fps) − 1) / fps，系数接近 0（60 fps、淡出 3 s 时约 9e-5，与时长无关）。

    Formula:
        α = sstep(min(t / T_in, 1, (T − t) / T_out))，sstep(s) = 3s² − 2s³（s 截断到 [0, 1]）。

    Physical Meaning:
        视频剪辑中的淡入 / 淡出黑场，作用在编码值上（与剪辑软件一致），场景亮度与曝光保持不变。
    """
    s = 1.0
    if settings.fade_in > 0:
        s = min(s, t / settings.fade_in)
    if settings.fade_out > 0:
        s = min(s, (duration - t) / settings.fade_out)
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def _sequence2(value, field: str) -> list:
    """校验路径文件中的二元数组字段。

    Args:
        value: 字段值。
        field: 字段名（用于错误信息）。

    Returns:
        长度为 2 的列表。

    Raises:
        ValueError: 不是长度为 2 的数组。
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} 必须是两个元素的数组，得到 {value!r}")
    return list(value)


def load_metering_settings(spec: dict) -> MeteringSettings:
    """由路径文件的 `exposure` 块构造测光参数（省略的字段取默认值）。

    Args:
        spec: 路径文件内容（JSON 解析后的字典）。

    Returns:
        `MeteringSettings`。

    Raises:
        ValueError: 字段取值非法，信息中给出字段名：`exposure` 不是对象；metering_step 不是有限正数；
            metering_size 不是两个正整数；smoothing 或 gain 不是有限非负数；ev_range 不是两个有限数或下限大于上限。
    """
    block = optional_block(spec, "exposure")
    d = MeteringSettings()
    try:
        step = float(block.get("metering_step", d.step))
        size = _sequence2(block.get("metering_size", d.size), "exposure.metering_size")
        smoothing = float(block.get("smoothing", d.smoothing))
        gain = float(block.get("gain", d.gain))
        ev_range = [float(v) for v in _sequence2(block.get("ev_range", d.ev_range), "exposure.ev_range")]
    except TypeError as exc:
        raise ValueError(f"exposure 块字段类型错误：{exc}") from exc
    if not (np.isfinite(step) and step > 0):
        raise ValueError(f"exposure.metering_step 必须是有限正数，得到 {step}")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size):
        raise ValueError(f"exposure.metering_size 必须是两个正整数 [宽, 高]，得到 {size}")
    for name, v in (("smoothing", smoothing), ("gain", gain)):
        if not (np.isfinite(v) and v >= 0):
            raise ValueError(f"exposure.{name} 必须是有限非负数，得到 {v}")
    if not all(np.isfinite(ev_range)) or ev_range[0] > ev_range[1]:
        raise ValueError(f"exposure.ev_range 必须是两个有限数 [下限, 上限] 且下限 ≤ 上限，得到 {ev_range}")
    return MeteringSettings(step=step, size=(size[0], size[1]), smoothing=smoothing, gain=gain,
                            ev_range=(ev_range[0], ev_range[1]))


def load_fade_settings(spec: dict) -> FadeSettings:
    """由路径文件的 `fade` 块构造淡入淡出参数（省略的字段取默认值）。

    Args:
        spec: 路径文件内容（JSON 解析后的字典），须含 `duration`。

    Returns:
        `FadeSettings`。

    Raises:
        ValueError: `duration` 缺失或不是有限正数；`fade` 不是对象；时长不是有限非负数；两者之和超过 duration。
    """
    try:
        duration = float(spec["duration"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"路径文件的 duration 缺失或类型错误：{exc}") from exc
    if not (np.isfinite(duration) and duration > 0):
        raise ValueError(f"duration 必须是有限正数，得到 {duration}")
    block = optional_block(spec, "fade")
    d = FadeSettings()
    try:
        s = FadeSettings(fade_in=float(block.get("fade_in", d.fade_in)), fade_out=float(block.get("fade_out", d.fade_out)))
    except TypeError as exc:
        raise ValueError(f"fade 块字段类型错误：{exc}") from exc
    if not (np.isfinite(s.fade_in) and np.isfinite(s.fade_out)) or min(s.fade_in, s.fade_out) < 0:
        raise ValueError(f"fade.fade_in / fade_out 必须是有限非负数，得到 {block}")
    if s.fade_in + s.fade_out > duration:
        raise ValueError(f"fade.fade_in + fade_out 不能超过 duration，得到 {block}")
    return s
