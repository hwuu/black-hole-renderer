"""Disk V2 内部共享的数组辅助函数。

本模块只服务于 `disk_v2` 包内部，用于统一处理：

- 标量/数组输入到 `float64` 数组的转换
- 数组结果恢复为与原始输入一致的标量或数组形状
- 布尔数组结果恢复为与原始输入一致的布尔标量或布尔数组

这些函数不承载任何吸积盘物理意义，只是为了让几何层、基础物理场层和结构调制层
在标量输入、数组输入和广播场景下保持一致、稳定、可测试的返回行为。
"""

from __future__ import annotations

import numpy as np


def _to_array(value: float | np.ndarray) -> np.ndarray:
    """将输入统一转换为 `float64` 数组。

    Args:
        value: 标量或 `np.ndarray`。

    Returns:
        `float64` 类型的 `np.ndarray`。

    Notes:
        统一内部计算类型，避免标量/数组混用时出现广播或精度行为不一致。
    """

    return np.asarray(value, dtype=np.float64)


def _restore_shape(value: np.ndarray, original: float | np.ndarray) -> float | np.ndarray:
    """把内部数组结果还原成与输入一致的标量或数组形状。

    Args:
        value: 内部计算得到的数组结果。
        original: 原始输入，用于判断应该返回标量还是数组。

    Returns:
        若 `original` 是标量则返回标量，否则返回数组。
    """

    if np.ndim(original) == 0:
        return float(value)
    return value


def _restore_bool(value: np.ndarray, original: float | np.ndarray) -> bool | np.ndarray:
    """把布尔数组结果还原成与输入一致的布尔标量或布尔数组。

    Args:
        value: 内部计算得到的布尔数组结果。
        original: 原始输入，用于判断应该返回布尔标量还是布尔数组。

    Returns:
        若 `original` 是标量则返回布尔标量，否则返回布尔数组。
    """

    if np.ndim(original) == 0:
        return bool(value)
    return value
