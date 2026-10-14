"""成片合成：把全部镜头按时间线拼接，混入旁白与音效，烧录字幕。

设计见 `docs/design_film.md` §4.7。生成一条 ffmpeg 命令（滤镜图）：
各镜头从开头保留 `D_i`、按 `x_i` 交叉淡化拼接，旁白按句子时刻放置（`adelay`），
音效按镜头内时刻混入，最后烧录字幕输出 H.264 + AAC。

镜头视频来源：
- render：`shots/<id>.mp4`（本机渲染或远端取回）
- diagram：`diagrams/<id>.mp4`（Manim 渲染）
- image：从 `assets/` 下载的图片合成为静帧视频
"""

from __future__ import annotations

import os
import subprocess
from typing import List, Optional

from .manifest import FilmManifest, Shot
from .timeline import Timeline


def _shot_video_path(manifest: FilmManifest, shot: Shot, out_dir: str,
                     preview: bool = False) -> Optional[str]:
    """定位一个镜头的视频文件（缺失返回 None，不报错——合成时可跳过或用黑帧）。"""
    sub = "shots_preview" if preview else "shots"
    if shot.kind == "render":
        path = os.path.join(out_dir, sub, f"{shot.id}.mp4")
    elif shot.kind == "diagram":
        path = os.path.join(out_dir, "diagrams", f"{shot.id}.mp4")
    elif shot.kind == "image":
        path = os.path.join(out_dir, "diagrams", f"{shot.id}.mp4")
    else:
        return None
    return path if os.path.exists(path) else None


def build_filter_graph(manifest: FilmManifest, timeline: Timeline,
                       out_dir: str) -> List[str]:
    """构建 ffmpeg 滤镜图的各段（纯函数，方便单测）。

    Returns:
        ffmpeg `-filter_complex` 的字符串列表（每个元素是一个 `[in]…[out]` 段）。
        调用方用 `;` 连接。

    Notes:
        本实现为简化版：不做 xfade 交叉淡化（直接 concat 拼接），
        不叠加 HUD，不混入 SFX。完整版按设计文档 §4.7 逐步补齐。
    """
    parts: List[str] = []
    video_labels: List[str] = []
    audio_labels: List[str] = []
    v_idx = 0

    for i, shot in enumerate(timeline.shots):
        spec = manifest.shot(shot.shot_id)
        dur = shot.duration
        v_label = f"v{i}"
        video_labels.append(f"[{v_label}]")

        parts.append(
            f"[{v_idx}:v]trim=duration={dur:.4f},"
            f"setpts=PTS-STARTPTS,scale={manifest.size[0]}:{manifest.size[1]},"
            f"fps={manifest.fps}[{v_label}]"
        )
        v_idx += 1

    # 旁白：逐句 adelay 到起始时刻
    a_idx = v_idx
    for shot in timeline.shots:
        for s in shot.sentences:
            delay_ms = int(s.start * 1000)
            if delay_ms > 0:
                parts.append(f"[{a_idx}:a]adelay={delay_ms}|{delay_ms}[a{len(audio_labels)}]")
            else:
                parts.append(f"[{a_idx}:a]anull[a{len(audio_labels)}]")
            audio_labels.append(f"[a{len(audio_labels)}]")
            a_idx += 1

    # 视频拼接
    concat_in = "".join(video_labels)
    parts.append(f"{concat_in}concat=n={len(video_labels)}:v=1:a=0[vout]")

    # 音频混合
    if audio_labels:
        mix_in = "".join(audio_labels)
        parts.append(f"{mix_in}amix=inputs={len(audio_labels)}:normalize=0[aout]")
    else:
        parts.append(f"anullsrc[r=48000][aout]")

    return parts


def assemble_film(manifest: FilmManifest, out_dir: str, preview: bool = False) -> str:
    """合成成片：拼接镜头 + 混入旁白 + 烧录字幕 → 一条 H.264 mp4。

    实现为分步 ffmpeg（比单条 filter_complex 在多输入下更可靠）：
    1. 逐镜头裁剪到 D_i、统一分辨率/帧率 → 临时分段
    2. concat demuxer 拼接全部分段 → 无声视频
    3. 旁白逐句 adelay 到起始时刻 → amix 混合 → 音频
    4. 视频与音频封装 + 字幕烧录

    Args:
        manifest: 清单。
        out_dir: 产物目录。
        preview: True = 用 shots_preview/ 下的预览视频。

    Returns:
        成片路径。

    Raises:
        RuntimeError: 缺少必要产物。
        subprocess.CalledProcessError: ffmpeg 失败。
    """
    import tempfile

    timeline_path = os.path.join(out_dir, "timeline.json")
    if not os.path.exists(timeline_path):
        raise RuntimeError(f"缺少 {timeline_path}；先运行 timeline 子命令")
    from .timeline import load_timeline
    timeline = load_timeline(timeline_path)

    w, h = manifest.size
    fps = manifest.fps
    tmp_dir = tempfile.mkdtemp(prefix="film_assemble_")

    try:
        # ---- 步骤 1：逐镜头裁剪 + 缩放 + 统一帧率（hud 镜头叠加遥控器面板） ----
        from .hud import generate_hud_image, read_knob_values
        segments = []
        for i, shot in enumerate(timeline.shots):
            spec = manifest.shot(shot.shot_id)
            path = _shot_video_path(manifest, spec, out_dir, preview)
            if path is None:
                raise RuntimeError(f"缺少镜头 {shot.shot_id} 的视频；先运行 jobs / diagrams")
            seg = os.path.join(tmp_dir, f"seg_{i:02d}.mp4")
            if getattr(spec, "hud", False) and spec.path:
                hud_img = os.path.join(tmp_dir, f"hud_{i}.png")
                path_json = os.path.join(manifest.base_dir, spec.path)
                t_mid = shot.duration / 2.0
                knobs = read_knob_values(path_json, shot.shot_id, t_mid)
                generate_hud_image(knobs, w, h, hud_img)
                subprocess.run([
                    "ffmpeg", "-y", "-v", "error",
                    "-i", path,
                    "-loop", "1", "-i", hud_img,
                    "-t", f"{shot.duration:.4f}",
                    "-filter_complex",
                    f"[0:v]scale={w}:{h},fps={fps}[base];"
                    f"[base][1:v]overlay=0:0:shortest=1[out]",
                    "-map", "[out]",
                    "-an",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-pix_fmt", "yuv420p",
                    seg,
                ], check=True)
            else:
                subprocess.run([
                    "ffmpeg", "-y", "-v", "error",
                    "-i", path,
                    "-t", f"{shot.duration:.4f}",
                    "-vf", f"scale={w}:{h},fps={fps}",
                    "-an",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-pix_fmt", "yuv420p",
                    seg,
                ], check=True)
            segments.append(seg)
            hud_tag = "（含 HUD）" if getattr(spec, "hud", False) else ""
            print(f"  [{i + 1}/{len(timeline.shots)}] {shot.shot_id} 裁剪完成{hud_tag}")

        # ---- 步骤 2：xfade 交叉淡化拼接 ----
        # offset_N = sum(D[0..N-1]) - sum(x[1..N])
        # xfade 把上一段的最后 x_N 秒与下一段的前 x_N 秒重叠淡化
        if len(segments) == 1:
            silent_video = segments[0]
        else:
            filter_parts = []
            prev_label = "0:v"
            cum_dur = 0.0
            cum_fade = 0.0
            for i in range(1, len(segments)):
                shot_spec = manifest.shot(timeline.shots[i].shot_id)
                trans = shot_spec.transition
                fade_dur = trans.duration if trans and trans.type == "fade" else 0.0
                cum_dur += timeline.shots[i - 1].duration
                cum_fade += fade_dur
                offset = cum_dur - cum_fade
                out_label = f"vx{i}"
                if fade_dur > 0:
                    filter_parts.append(
                        f"[{prev_label}][{i}:v]xfade=transition=fade:"
                        f"duration={fade_dur:.4f}:offset={offset:.4f}[{out_label}]")
                else:
                    filter_parts.append(
                        f"[{prev_label}][{i}:v]concat=n=2:v=1:a=0[{out_label}]")
                prev_label = out_label
            filter_str = ";".join(filter_parts)
            silent_video = os.path.join(tmp_dir, "silent.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-v", "error",
                *[arg for seg in segments for arg in ("-i", seg)],
                "-filter_complex", filter_str,
                "-map", f"[{prev_label}]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-pix_fmt", "yuv420p",
                silent_video,
            ], check=True)
        print(f"  视频拼接完成（含交叉淡化）")

        # ---- 步骤 3：旁白混合 ----
        # 逐句 adelay 后 amix；全部放在一条 filter_complex 里（纯音频流，规模小得多）
        audio_inputs = []
        audio_labels = []
        filter_parts = []
        a_idx = 0
        for shot in timeline.shots:
            for s in shot.sentences:
                mp3 = os.path.join(out_dir, "narration", f"{shot.shot_id}_{s.index}.mp3")
                if not os.path.exists(mp3):
                    raise RuntimeError(f"缺少旁白 {mp3}")
                audio_inputs.extend(["-i", mp3])
                delay_ms = int(s.start * 1000)
                label = f"a{a_idx}"
                if delay_ms > 0:
                    filter_parts.append(
                        f"[{a_idx}:a]adelay={delay_ms}|{delay_ms}[{label}]")
                else:
                    filter_parts.append(f"[{a_idx}:a]anull[{label}]")
                audio_labels.append(f"[{label}]")
                a_idx += 1

        mixed_audio = os.path.join(tmp_dir, "audio.m4a")
        if audio_labels:
            mix_in = "".join(audio_labels)
            filter_str = ";".join(filter_parts) + ";" + \
                mix_in + f"amix=inputs={len(audio_labels)}:normalize=0,apad[aout]"
            subprocess.run([
                "ffmpeg", "-y", "-v", "error",
                *audio_inputs,
                "-filter_complex", filter_str,
                "-map", "[aout]",
                "-c:a", "aac", "-b:a", "192k",
                "-t", f"{timeline.total_duration:.4f}",
                mixed_audio,
            ], check=True)
            print(f"  音频混合完成（{len(audio_labels)} 句）")

        # ---- 步骤 4：封装 + 字幕 ----
        subtitle_path = os.path.join(out_dir, "subtitles.ass")
        if not os.path.exists(subtitle_path):
            from .subtitles import build_ass
            with open(subtitle_path, "w", encoding="utf-8") as f:
                f.write(build_ass(manifest, timeline))

        output = os.path.join(
            out_dir,
            f"{manifest.film_id}_{w}x{h}p{fps}.mp4",
        )

        if os.path.exists(subtitle_path):
            # 两步：先封装视频+音频，再烧字幕（比一步到位更可靠）
            muxed = os.path.join(tmp_dir, "muxed.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-v", "error",
                "-i", silent_video,
                "-i", mixed_audio,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "copy",
                "-shortest",
                muxed,
            ], check=True)
            subprocess.run([
                "ffmpeg", "-y", "-v", "error",
                "-i", muxed,
                "-vf", f"subtitles={subtitle_path}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "copy",
                output,
            ], check=True)
        else:
            subprocess.run([
                "ffmpeg", "-y", "-v", "error",
                "-i", silent_video,
                "-i", mixed_audio,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "copy",
                "-shortest",
                output,
            ], check=True)

        print(f"  成片 {timeline.total_duration:.1f}s → {output}")
        return output

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
