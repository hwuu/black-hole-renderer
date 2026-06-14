"""Disk V2 渲染统计工具（HDR / LDR 诊断）。

用于视觉恢复阶段定位饱和来源：palette 白化、曝光不足、还是 HDR 积分量级过大。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def hdr_luminance(rgb_hdr: np.ndarray) -> np.ndarray:
    """计算 HDR RGB 的 BT.709 亮度。

    Args:
        rgb_hdr: 形状 `(..., 3)` 或 `(H, W, 3)` 的非负 HDR RGB。

    Returns:
        与输入前两维同形状的亮度标量场。
    """
    rgb = np.asarray(rgb_hdr, dtype=np.float64)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def ldr_luminance(rgb_ldr: np.ndarray) -> np.ndarray:
    """计算 LDR RGB 的 BT.709 亮度（输入可在 `[0, 1]` 或 `[0, 255]`）。

    Args:
        rgb_ldr: RGB 数组，最后一维为 3。

    Returns:
        亮度标量场。
    """
    rgb = np.asarray(rgb_ldr, dtype=np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    return hdr_luminance(rgb)


@dataclass(frozen=True)
class RenderStats:
    """单帧 V2 渲染的 HDR / LDR 统计摘要。

    Attributes:
        hdr_min, hdr_p50, hdr_p90, hdr_p95, hdr_p99, hdr_max: HDR 亮度分位数。
        ldr_black_ratio: LDR 像素亮度 `< 1/255` 的比例。
        ldr_near_white_ratio: LDR 像素三通道均 `>= 250/255` 的比例。
        ldr_white_ratio: LDR 像素三通道均 `== 255` 的比例。
        white_point: 实际用于 tonemap 的 white point（fallback 判定后的结果）。
        reference_white_point: v2.2 物理场域 reference 推导的 white point。
        actual_hdr_white_point: 当前帧 HDR 亮度按 `white_point_percentile` 取的
            原始候选值（fallback 判定前）。与 `reference_white_point` 之比可以
            直接读出 D3 的 `ratio = actual / reference`，用于检查曝光基线漂移。
        white_point_percentile: 用于计算 `actual_hdr_white_point` 的分位数
            （v2.2 主验收用 96，单独 CLI 调用通常用 99）。
    """

    hdr_min: float
    hdr_p50: float
    hdr_p90: float
    hdr_p95: float
    hdr_p99: float
    hdr_max: float
    ldr_black_ratio: float
    ldr_near_white_ratio: float
    ldr_white_ratio: float
    white_point: float | None = None
    reference_white_point: float | None = None
    actual_hdr_white_point: float | None = None
    white_point_percentile: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """转为可打印 / 可序列化的字典。"""
        return {
            "hdr_min": self.hdr_min,
            "hdr_p50": self.hdr_p50,
            "hdr_p90": self.hdr_p90,
            "hdr_p95": self.hdr_p95,
            "hdr_p99": self.hdr_p99,
            "hdr_max": self.hdr_max,
            "ldr_black_ratio": self.ldr_black_ratio,
            "ldr_near_white_ratio": self.ldr_near_white_ratio,
            "ldr_white_ratio": self.ldr_white_ratio,
            "white_point": self.white_point,
            "reference_white_point": self.reference_white_point,
            "actual_hdr_white_point": self.actual_hdr_white_point,
            "white_point_percentile": self.white_point_percentile,
        }

    def format_summary(self) -> str:
        """格式化为单行摘要，便于 CLI 打印。"""
        wp = f", white_point={self.white_point:.6g}" if self.white_point is not None else ""
        rwp = (
            f", reference_white_point={self.reference_white_point:.6g}"
            if self.reference_white_point is not None
            else ""
        )
        actual = (
            f", actual_hdr_wp={self.actual_hdr_white_point:.6g}"
            if self.actual_hdr_white_point is not None
            else ""
        )
        pct = (
            f" (p{self.white_point_percentile:g})"
            if self.white_point_percentile is not None
            else ""
        )
        return (
            f"[V2 stats] HDR luma: min={self.hdr_min:.6g} p50={self.hdr_p50:.6g} "
            f"p90={self.hdr_p90:.6g} p95={self.hdr_p95:.6g} p99={self.hdr_p99:.6g} "
            f"max={self.hdr_max:.6g}{wp}{rwp}{actual}{pct} | "
            f"LDR: black={self.ldr_black_ratio:.4f} near_white={self.ldr_near_white_ratio:.4f} "
            f"white={self.ldr_white_ratio:.4f}"
        )


def compute_render_stats(
    hdr_rgb: np.ndarray,
    ldr_rgb: np.ndarray,
    *,
    white_point: float | None = None,
    reference_white_point: float | None = None,
    actual_hdr_white_point: float | None = None,
    white_point_percentile: float | None = None,
) -> RenderStats:
    """从 HDR buffer 与最终 LDR 图像计算诊断统计。

    Args:
        hdr_rgb: Taichi `hdr_field.to_numpy()` 结果，形状 `(W, H, 3)` 或 `(H, W, 3)`。
        ldr_rgb: 最终 uint8 或 float LDR，形状 `(H, W, 3)`。
        white_point: 可选，记录 auto exposure 实际使用的 white point。
        reference_white_point: 可选，记录物理场域 reference white point。
        actual_hdr_white_point: 可选，记录原始 HDR p{n} 候选值（fallback 判定前）。
        white_point_percentile: 可选，用于计算 `actual_hdr_white_point` 的分位数。

    Returns:
        `RenderStats` 对象。
    """
    luma_hdr = hdr_luminance(hdr_rgb).ravel()
    luma_hdr = luma_hdr[np.isfinite(luma_hdr)]
    if luma_hdr.size == 0:
        luma_hdr = np.array([0.0], dtype=np.float64)

    ldr = np.asarray(ldr_rgb)
    if ldr.dtype == np.uint8:
        ldr_f = ldr.astype(np.float64) / 255.0
    else:
        ldr_f = np.clip(ldr.astype(np.float64), 0.0, 1.0)

    black_mask = ldr_luminance(ldr_f) < (1.0 / 255.0)
    near_white_mask = np.all(ldr_f >= (250.0 / 255.0), axis=-1)
    white_mask = np.all(ldr == 255, axis=-1) if ldr.dtype == np.uint8 else np.all(ldr_f >= 0.999, axis=-1)

    n = max(int(ldr.shape[0] * ldr.shape[1]), 1)
    return RenderStats(
        hdr_min=float(np.min(luma_hdr)),
        hdr_p50=float(np.percentile(luma_hdr, 50)),
        hdr_p90=float(np.percentile(luma_hdr, 90)),
        hdr_p95=float(np.percentile(luma_hdr, 95)),
        hdr_p99=float(np.percentile(luma_hdr, 99)),
        hdr_max=float(np.max(luma_hdr)),
        ldr_black_ratio=float(np.sum(black_mask)) / n,
        ldr_near_white_ratio=float(np.sum(near_white_mask)) / n,
        ldr_white_ratio=float(np.sum(white_mask)) / n,
        white_point=white_point,
        reference_white_point=reference_white_point,
        actual_hdr_white_point=actual_hdr_white_point,
        white_point_percentile=white_point_percentile,
    )
