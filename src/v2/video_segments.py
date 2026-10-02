"""V2 视频分段写出与断点续传。

长视频按固定帧数切成若干 mp4 分段，每段写完（编码器 flush、文件关闭）后才从临时名改为正式名，
因此磁盘上的正式分段文件一定完整。中断后用同样参数重跑并加 `--resume`，已完成的分段直接跳过，
最多重渲一段；全部完成后把各分段的 H.264 码流无损拼接（不重新编码）为最终文件。

V2 每一帧只由（物理时刻、相机位置、抖动种子、锁定曝光）决定，与前面的帧无关，所以续传不需要
重演前面的帧；锁定曝光与渲染参数一起存在进度文件里。

目录结构（与 V1 `.frames_*` 同样放在输出文件旁）::

    output/.v2seg_<md5(output)[:16]>/
        progress.json        {"params": {...}, "exposure": float | null}
        seg_00000.mp4        第 0 段（帧 0 .. seg_frames-1）
        seg_00001.mp4        ...
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Callable, Iterable, Iterator, Optional

import av
import numpy as np

# 每段帧数：24 fps 下 10 s；中断后最多重渲这么多帧
SEG_FRAMES_DEFAULT = 240


def segment_dir(output_path: str) -> str:
    """分段临时目录路径（输出文件同目录下的隐藏目录，按输出路径哈希区分）。

    Args:
        output_path: 最终视频路径。

    Returns:
        目录路径字符串（不保证存在）。
    """
    name = ".v2seg_" + hashlib.md5(output_path.encode()).hexdigest()[:16]
    return os.path.join(os.path.dirname(output_path) or ".", name)


def segment_path(seg_dir: str, k: int) -> str:
    """第 k 个分段的正式文件路径。

    Args:
        seg_dir: `segment_dir` 返回的目录。
        k: 分段序号（≥ 0）。

    Returns:
        `seg_dir/seg_{k:05d}.mp4`。
    """
    return os.path.join(seg_dir, f"seg_{k:05d}.mp4")


def prepare(seg_dir: str, params: dict, resume: bool) -> Optional[float]:
    """准备分段目录：续传时校验参数并取回锁定曝光，否则清空重来。

    Args:
        seg_dir: 分段目录。
        params: 决定帧内容的全部参数（可 JSON 序列化）；续传时必须与上次完全一致。
        resume: True = 尝试续传。

    Returns:
        上次锁定的曝光（续传且第 0 帧已渲染时）；否则 None（需在第 0 帧重新计算）。

    Notes:
        参数不一致时打印警告并清空目录从头开始（与 V1 `--resume` 行为一致）。
    """
    progress = os.path.join(seg_dir, "progress.json")
    if resume and os.path.isfile(progress):
        with open(progress) as f:
            saved = json.load(f)
        if saved.get("params") == params:
            return saved.get("exposure")
        print("[V2 video] 警告：参数与上次不同，从头开始")
    if os.path.isdir(seg_dir):
        shutil.rmtree(seg_dir)
    os.makedirs(seg_dir, exist_ok=True)
    save_progress(seg_dir, params, None)
    return None


def save_progress(seg_dir: str, params: dict, exposure: Optional[float]) -> None:
    """写进度文件（参数 + 锁定曝光）。

    Args:
        seg_dir: 分段目录。
        params: 渲染参数。
        exposure: 锁定曝光；None = 尚未计算。
    """
    with open(os.path.join(seg_dir, "progress.json"), "w") as f:
        json.dump({"params": params, "exposure": exposure}, f)


def write_segment(path: str, frames: Iterable[np.ndarray], fps: int) -> int:
    """把一段帧编码为 H.264 mp4；写完后才从临时名改为正式名。

    Args:
        path: 正式分段路径。
        frames: `(H, W, 3)` uint8 RGB 帧序列（按帧序产出）。
        fps: 帧率。

    Returns:
        写入的帧数。

    Notes:
        关闭 B 帧（`bf = 0`），使每段的 dts = pts 且从 0 开始，拼接时只需整体平移时间戳。
    """
    tmp = path + ".part"
    n = 0
    with av.open(tmp, "w", format="mp4") as c:
        s = None
        for img in frames:
            if s is None:
                s = c.add_stream("libx264", rate=fps)
                s.width, s.height = img.shape[1], img.shape[0]
                s.pix_fmt = "yuv420p"
                s.options = {"bf": "0"}
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
            n += 1
        if s is not None:
            for pkt in s.encode():
                c.mux(pkt)
    os.replace(tmp, path)
    return n


def concat_segments(paths: list, output_path: str) -> None:
    """把各分段的 H.264 码流无损拼接为一个 mp4（不重新编码）。

    Args:
        paths: 分段文件路径（按帧序）。
        output_path: 输出路径。

    Formula:
        第 k 段的包时间戳平移 `offset_k = Σ_{i<k} 第 i 段时长`（以流时间基计），保持 pts/dts 单调。
    """
    with av.open(output_path, "w", format="mp4") as oc:
        ost = None
        offset = 0
        for p in paths:
            with av.open(p) as ic:
                ist = ic.streams.video[0]
                if ost is None:
                    ost = oc.add_stream_from_template(ist)
                end = offset
                for pkt in ic.demux(ist):
                    if pkt.dts is None:  # demux 结束时的空包
                        continue
                    pkt.pts += offset
                    pkt.dts += offset
                    end = max(end, pkt.pts + pkt.duration)
                    pkt.stream = ost
                    oc.mux(pkt)
                offset = end


def render_segmented(
    n_frames: int,
    fps: int,
    output_path: str,
    params: dict,
    resume: bool,
    render_frames: Callable[[int, int, Optional[float]], Iterator[tuple]],
    seg_frames: int = SEG_FRAMES_DEFAULT,
    log: Callable[[str], None] = print,
) -> None:
    """分段渲染整段视频（可续传），完成后拼接并删除分段目录。

    Args:
        n_frames: 总帧数。
        fps: 帧率。
        output_path: 最终视频路径。
        params: 决定帧内容的参数（续传校验用）。
        resume: True = 跳过已完成分段。
        render_frames: `render_frames(f0, f1, exposure)` 按帧序产出 `(img_uint8, exposure)`，覆盖帧
            `[f0, f1)`；`exposure` 为 None 时由第 f0 帧计算锁定值并随后续帧一起产出（只会在 f0 = 0
            发生）。调用方可在内部流水线化（GPU 积分与后处理并行）。
        seg_frames: 每段帧数。
        log: 进度输出函数。
    """
    seg_dir = segment_dir(output_path)
    exposure = prepare(seg_dir, params, resume)
    n_seg = (n_frames + seg_frames - 1) // seg_frames
    paths = [segment_path(seg_dir, k) for k in range(n_seg)]
    done = sum(os.path.isfile(p) for p in paths)
    if done:
        log(f"[V2 video] 续传：{done}/{n_seg} 段已完成")
    for k, p in enumerate(paths):
        if os.path.isfile(p):
            continue
        f0, f1 = k * seg_frames, min((k + 1) * seg_frames, n_frames)
        if exposure is None and f0 != 0:
            raise RuntimeError("锁定曝光缺失：第 0 段未完成却要渲染后续分段")

        def frames():
            nonlocal exposure
            for img, exp in render_frames(f0, f1, exposure):
                if exposure is None:
                    exposure = exp
                    save_progress(seg_dir, params, exposure)
                yield img

        write_segment(p, frames(), fps)
        log(f"[V2 video] 分段 {k + 1}/{n_seg} 完成（帧 {f0}–{f1 - 1}）")
    concat_segments(paths, output_path)
    shutil.rmtree(seg_dir)
