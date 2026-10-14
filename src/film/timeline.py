"""时间线：由旁白时长排出每个镜头的时长、起止与句子时刻（旁白先行）。

设计见 `docs/design_film.md` §4.2。所有时刻量化到成片帧网格（1/fps）：
镜头时长向上取整到帧（保证旁白完整），句子时刻四舍五入到帧、
时长至少一帧（保证每条字幕可见）。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

from .manifest import FilmManifest, Shot


@dataclass(frozen=True)
class SentenceTiming:
    """一句旁白的时间（秒，帧网格化）。

    Attributes:
        shot_id / index: 所属镜头与句序（与清单 `narration` 的下标一致）。
        text: 旁白原文（也是一条字幕）。
        start / end: 起止秒（成片时间轴；`end − start ≥ 1/fps`）。
    """

    shot_id: str
    index: int
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class ShotTiming:
    """一个镜头的时间（秒，帧网格化）。

    Attributes:
        shot_id: 镜头 id。
        start: 镜头起点（成片时间轴，秒；帧网格的整数倍）。
        duration: 镜头时长（秒；`D = 帧数 / fps`，已向上取整到帧）。
        sentences: 该镜头各句旁白的时刻（保证都落在 `[start, start + duration]` 内）。
    """

    shot_id: str
    start: float
    duration: float
    sentences: Tuple[SentenceTiming, ...]


@dataclass(frozen=True)
class Timeline:
    """全片时间线。"""

    fps: int
    shots: Tuple[ShotTiming, ...]
    total_duration: float

    def shot(self, shot_id: str) -> ShotTiming:
        """按镜头 id 取时间；不存在时报错（信息中给出可用 id）。"""
        for shot in self.shots:
            if shot.shot_id == shot_id:
                return shot
        raise ValueError(f"时间线中没有镜头 {shot_id!r}（可用：{[s.shot_id for s in self.shots]}）")

    def to_json_dict(self) -> dict:
        """序列化为 JSON 兼容字典（`timeline.json` 的内容）。"""
        return {
            "fps": self.fps,
            "total_duration": self.total_duration,
            "shots": [
                {"id": shot.shot_id, "start": shot.start, "duration": shot.duration,
                 "sentences": [{"index": s.index, "text": s.text,
                                "start": s.start, "end": s.end} for s in shot.sentences]}
                for shot in self.shots
            ],
        }

    def save(self, path: str) -> None:
        """把时间线写为 `timeline.json`（UTF-8，中文不转义）。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, ensure_ascii=False, indent=2)


def load_timeline(path: str) -> Timeline:
    """读取 `timeline.json`（`Timeline.save` 的逆操作，供 subtitles 等步骤使用）。

    Args:
        path: `timeline.json` 路径。

    Returns:
        `Timeline`。

    Raises:
        OSError: 文件无法读取。
        ValueError: JSON 结构不完整或字段类型不匹配。
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    try:
        shots = tuple(
            ShotTiming(shot_id=s["id"], start=float(s["start"]), duration=float(s["duration"]),
                       sentences=tuple(SentenceTiming(shot_id=s["id"], index=int(x["index"]),
                                                       text=x["text"], start=float(x["start"]),
                                                       end=float(x["end"]))
                                       for x in s["sentences"]))
            for s in data["shots"])
        return Timeline(fps=int(data["fps"]), shots=shots,
                         total_duration=float(data["total_duration"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"timeline.json 结构不完整：{exc}") from exc


def _duration_frames(shot: Shot, durations: list, fps: int) -> int:
    """镜头时长（帧）：`max(min_duration, lead + Σs + gap·max(n−1, 0) + tail)` 严格向上取整到帧。

    Args:
        shot: 清单中的镜头（取 `padding` 与 `min_duration`）。
        durations: 该镜头各句旁白的时长（秒，按句序）。
        fps: 成片帧率。

    Returns:
        帧数 ≥ 1。先把 `秒 × fps` 舍入到 1e-9 帧以内再 `ceil`：吸收浮点表示误差
        （约 1e-13 帧量级，避免 6.0 s 恰好被取到下一帧），同时保持严格向上取整
        （任何真实大于整帧边界的时长都不会被缩短）。
    """
    spoken = sum(durations)
    gaps = shot.padding.gap * max(len(durations) - 1, 0)
    seconds = max(shot.padding.lead + spoken + gaps + shot.padding.tail, shot.min_duration)
    return max(1, math.ceil(round(seconds * fps, 9)))


def _fade_frames(manifest: FilmManifest, index: int, duration_frames: Dict[str, int]) -> int:
    """镜头 `index` 与上一镜头的交叉淡化重叠帧数 `x_i`（cut 为 0）。

    Raises:
        ValueError: `fade` 时长不足一帧，或超过相邻两个镜头中较短者的时长
            （信息中给出镜头 id 与允许的上限）。
    """
    shot = manifest.shots[index]
    if shot.transition is None or shot.transition.type != "fade":
        return 0
    x = round(shot.transition.duration * manifest.fps)
    if x < 1:
        raise ValueError(f"shots[{index}].transition: fade 时长 {shot.transition.duration}s "
                         f"在 {manifest.fps} fps 下不足一帧")
    limit = min(duration_frames[manifest.shots[index - 1].id], duration_frames[shot.id])
    if x > limit:
        raise ValueError(f"镜头 {shot.id} 的 fade 时长 {shot.transition.duration}s 超过相邻镜头"
                         f"中较短者的时长（≤ {limit / manifest.fps:.2f}s）")
    return x


def _check_path_frames(manifest: FilmManifest, shot: Shot, need_frames: int, fps: int) -> None:
    """校验 render 镜头的运镜路径时长是否够用（`round(duration·fps) ≥ D_i` 帧）。

    Raises:
        ValueError: 路径文件不可读、缺少 `duration`，或时长不足（信息中给出需要的秒数）。
    """
    path_file = os.path.join(manifest.base_dir, shot.path)
    try:
        with open(path_file, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise ValueError(f"镜头 {shot.id} 的路径文件不可读：{path_file}（{exc}）") from exc
    if not isinstance(data, dict) or not isinstance(data.get("duration"), (int, float)):
        raise ValueError(f"镜头 {shot.id} 的路径文件缺少 duration 字段：{path_file}")
    have_frames = round(float(data["duration"]) * fps)
    if have_frames < need_frames:
        raise ValueError(f"镜头 {shot.id} 的运镜路径时长 {float(data['duration'])}s（{have_frames} 帧）"
                         f"不足 D = {need_frames} 帧（约 {need_frames / fps:.2f}s）；"
                         f"请把路径 duration 加长到 ≥ {need_frames / fps:.2f}s")


def build_timeline(manifest: FilmManifest, sentence_durations: Dict[Tuple[str, int], float],
                   log=print) -> Timeline:
    """由清单与旁白时长排出全片时间线。

    Args:
        manifest: 清单。
        sentence_durations: `{(镜头 id, 句序): 时长秒}`，须覆盖清单中全部旁白句。
        log: 进度输出函数。

    Returns:
        `Timeline`。

    Raises:
        ValueError: 缺少某句旁白的时长；render 镜头路径时长不足；
            `fade` 不足一帧或超过相邻镜头中较短者；`sfx.at` 落在该镜头时长之外；
            某句旁白的起点落在其镜头时长之外（留白 + 前句累计挤满镜头）。

    Formula:
        ```
        D_i = max(min_duration_i, lead_i + Σ_k s_ik + gap_i·max(n_i − 1, 0) + tail_i)  （秒，向上取整到帧）
        start_0 = 0；start_i = start_{i−1} + D_{i−1} − x_i                              （帧）
        句 k 起止 = start_i + lead_i + Σ_{j<k}(s_ij + gap_i) … + s_ik                    （秒，四舍五入到帧）
        ```
        句子时刻保证落在所属镜头内：终点钳制到镜头末帧；最短句也占一帧，
        起点已在镜头外时报错（由作者提高 min_duration 或减小留白）。
    """
    fps = manifest.fps
    for shot in manifest.shots:
        for k in range(len(shot.narration)):
            if (shot.id, k) not in sentence_durations:
                raise ValueError(f"缺少镜头 {shot.id} 第 {k} 句的旁白时长")
    duration_frames = {shot.id: _duration_frames(
        shot, [sentence_durations[(shot.id, k)] for k in range(len(shot.narration))], fps)
        for shot in manifest.shots}
    for index, shot in enumerate(manifest.shots):
        if shot.kind == "render":
            _check_path_frames(manifest, shot, duration_frames[shot.id], fps)
    fades = {index: _fade_frames(manifest, index, duration_frames)
             for index in range(len(manifest.shots))}
    for event in manifest.sfx:
        limit = duration_frames[event.shot] / fps
        if event.at > limit:
            raise ValueError(f"sfx（镜头 {event.shot}）的 at = {event.at}s 超出该镜头时长 {limit:.2f}s")

    timings: list[ShotTiming] = []
    start_f = 0
    for index, shot in enumerate(manifest.shots):
        if index > 0:
            start_f += duration_frames[manifest.shots[index - 1].id] - fades[index]
        shot_end_f = start_f + duration_frames[shot.id]
        sentences = []
        prefix = 0.0
        for k, text in enumerate(shot.narration):
            spoken = sentence_durations[(shot.id, k)]
            f_start = start_f + round((shot.padding.lead + prefix) * fps)
            if f_start >= shot_end_f:
                raise ValueError(f"镜头 {shot.id} 第 {k} 句的起点落在镜头时长之外"
                                 f"（lead + 前句累计 {shot.padding.lead + prefix:.2f}s ≥ D；"
                                 f"请提高 min_duration 或减小留白）")
            f_end = min(max(start_f + round((shot.padding.lead + prefix + spoken) * fps),
                            f_start + 1), shot_end_f)
            sentences.append(SentenceTiming(shot.id, k, text, f_start / fps, f_end / fps))
            prefix += spoken + shot.padding.gap
        timings.append(ShotTiming(shot.id, start_f / fps, duration_frames[shot.id] / fps,
                                  tuple(sentences)))
        log(f"  {shot.id}: start={start_f / fps:.2f}s  D={duration_frames[shot.id] / fps:.2f}s  "
            f"句子 {len(sentences)}")
    total_f = start_f + duration_frames[manifest.shots[-1].id]
    return Timeline(fps=fps, shots=tuple(timings), total_duration=total_f / fps)
