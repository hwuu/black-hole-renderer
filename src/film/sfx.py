"""音效合成：NumPy 生成、标准库 wave 写出。首个类型 `siren_doppler`（多普勒警笛）。

设计见 `docs/design_film.md` §4.5：频率与响度按真实多普勒公式随时间变化，
相位按发出时刻积分，变调连续、无跳变；响度按 1/距离 衰减并限幅。
"""

from __future__ import annotations

import os
import wave
from typing import Callable, Dict, List

import numpy as np

from .manifest import FilmManifest

SOUND_SPEED = 343.0
"""空气中的声速（m/s）。"""

_FADE_SECONDS = 0.01
"""首尾线性淡入淡出时长（秒），消除爆音。"""


def siren_doppler(f0: float = 700.0, speed: float = 25.0, closest: float = 10.0,
                  duration: float = 8.0, sample_rate: float = 44100.0,
                  amplitude: float = 0.8) -> np.ndarray:
    """多普勒警笛：声源以 `speed` 沿直线驶过听者（最近距离 `closest`）。

    Args:
        f0: 声源固有频率（Hz，> 0）。
        speed: 声源速度（m/s，0 < speed < 声速）。
        closest: 声源轨迹到听者的最近距离 d₀（m，> 0）；声源在**发出时刻** τ = 0
            路过该点，听者于 t = d₀/c 后才听到该声音。
        duration: 音频时长（秒，> 0）；以 t = 0 为中心，即总长的一半在驶近段。
        sample_rate: 采样率（Hz，> 0）。
        amplitude: 峰值幅度（(0, 1]；最近距离处最响）。

    Returns:
        float32 单声道波形，值域 [-amplitude, amplitude]，首尾各 10 ms 线性淡入淡出。

    Formula:
        ```
        声源位置 r_s(τ) = (v·τ, −d₀)（沿 x 轴匀速，听者在原点）
        接收时刻 t 与发出时刻 τ 满足 |r_s(τ)| = c·(t − τ)，解二次方程取较早的根：
        τ(t) = [c²t − sqrt(v²c²t² + (c² − v²)·d₀²)] / (c² − v²)
        信号 x(t) = sin(2π·f0·τ(t))·A(t)，A(t) = amplitude·d₀ / |r_s(τ(t))|
        ```
        瞬时接收频率 = f0·dτ/dt：声源在发出时刻 τ = 0 路过最接近点，该声音于
        t = d₀/c 到达听者——t < d₀/c 为驶近段（频率高于 f0），t > d₀/c 为驶离段
        （低于 f0），远场频率比趋于 (c + v)/(c − v)。

    Physical Meaning:
        救护车警笛从身边驶过的变调；儿童科普片中先用声音讲多普勒效应，
        再把同一个道理迁移到光（`docs/plans/kids_explainer_plan.md` §4.2 s06）。

    Simplifications:
        点声源、无地面反射与空气吸收；相位按发出时刻积分，忽略传播中的波形弥散。
    """
    c, v, d0 = SOUND_SPEED, float(speed), float(closest)
    if not (f0 > 0 and 0 < v < c and d0 > 0 and duration > 0 and sample_rate > 0
            and 0 < amplitude <= 1):
        raise ValueError(f"参数非法：f0={f0}, speed={speed}, closest={closest}, "
                         f"duration={duration}, sample_rate={sample_rate}, amplitude={amplitude}")
    t = (np.arange(int(round(duration * sample_rate))) / sample_rate) - duration / 2.0
    root = np.sqrt((v * v) * (c * c) * (t * t) + (c * c - v * v) * (d0 * d0))
    tau = ((c * c) * t - root) / (c * c - v * v)
    source_pos = np.stack([v * tau, np.full_like(tau, -d0)], axis=-1)
    distance = np.linalg.norm(source_pos, axis=-1)
    signal = np.sin(2 * np.pi * f0 * tau) * (amplitude * d0 / np.maximum(distance, d0))
    fade = min(_FADE_SECONDS, duration / 4.0)
    n_fade = max(1, int(round(fade * sample_rate)))
    ramp = np.linspace(0.0, 1.0, n_fade)
    signal[:n_fade] *= ramp
    signal[-n_fade:] *= ramp[::-1]
    return signal.astype(np.float32)


def write_wav(path: str, samples: np.ndarray, sample_rate: float = 44100.0) -> str:
    """写出 16-bit PCM 单声道 wav（标准库 wave，无第三方依赖）。

    Args:
        path: 输出路径（父目录须已存在）。
        samples: 单声道波形（任意幅度；写出前裁剪到 [-1, 1]）。
        sample_rate: 采样率（Hz）。

    Returns:
        同 `path`。
    """
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(round(sample_rate)))
        w.writeframes(pcm.tobytes())
    return path


_SFX_RENDERERS: Dict[str, Callable[..., np.ndarray]] = {"siren_doppler": siren_doppler}
"""音效类型注册表：清单 `sfx[].kind` → 合成函数（kwargs 为 `params` 的键）。"""

_PARAM_ALIASES = {"v": "speed"}
"""清单参数别名 → 合成函数形参名（`v` 是设计文档里的速度变量名，函数形参用全称 `speed`）。"""


def render_sfx(manifest: FilmManifest, out_dir: str, log: Callable[[str], None] = print) -> List[str]:
    """按清单的音效事件合成 wav。

    Args:
        manifest: 清单（取 `sfx` 事件；`params` 的键作为合成函数的关键字参数，
            如 siren_doppler 的 `f0` / `speed`（别名 `v`）/ `closest` / `duration`）。
        out_dir: 输出目录（`<产物目录>/sfx`），不存在则创建。
        log: 进度输出函数。

    Returns:
        写出的 wav 路径列表（事件顺序），文件名 `<镜头id>_<事件序号>.wav`。

    Raises:
        ValueError: 事件的 `kind` 没有对应的音效类型。
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    for index, event in enumerate(manifest.sfx):
        if event.kind not in _SFX_RENDERERS:
            raise ValueError(f"sfx[{index}].kind: 未知音效类型 {event.kind!r}"
                             f"（可用：{sorted(_SFX_RENDERERS)}）")
        params = {_PARAM_ALIASES.get(key, key): value for key, value in event.params.items()}
        samples = _SFX_RENDERERS[event.kind](**params)
        path = os.path.join(out_dir, f"{event.shot}_{index}.wav")
        write_wav(path, samples)
        log(f"  音效 {os.path.basename(path)}（{len(samples)} 样本）")
        paths.append(path)
    return paths
