"""ASS 字幕生成：每句一条，超长折行；图片署名在该镜头期间显示于右下角。

设计见 `docs/design_film.md` §4.4。字体苹方（PingFang SC），主字幕白字黑边、底部居中，
署名灰色小字、右下角。折行优先在标点后断开，避免把词组拆到两行。
"""

from __future__ import annotations

from typing import List

from .manifest import FilmManifest
from .timeline import Timeline

MAX_LINE_CHARS = 18
"""单行字幕的最大字符数（面向 8–12 岁观众的大字幕，一屏一行）。"""

_BREAK_AFTER = "，。！？；：、…—,!?;:"
"""折行优先跟随的标点。"""


def wrap_sentence(text: str, max_chars: int = MAX_LINE_CHARS) -> List[str]:
    """把一句旁白折成每行 ≤ max_chars 的行。

    Args:
        text: 一句旁白（不含换行）。
        max_chars: 单行最大字符数。

    Returns:
        行列表（至少一行）；优先在标点后断行，窗口内无标点时硬折。
    """
    lines: List[str] = []
    start = 0
    while start < len(text):
        if len(text) - start <= max_chars:
            lines.append(text[start:])
            break
        window = text[start:start + max_chars]
        pos = max(window.rfind(p) for p in _BREAK_AFTER)
        cut = (pos + 1) if pos >= 0 else max_chars
        lines.append(text[start:start + cut])
        start += cut
    return lines


def ass_time(t: float) -> str:
    """秒 → ASS 时间戳 `H:MM:SS.cc`（百分秒）。

    Args:
        t: 时刻（秒，≥ 0）。

    Returns:
        形如 `0:00:01.29` 的时间戳（四舍五入到百分秒）。
    """
    cs = round(t * 100)
    hours, rem = divmod(cs, 360000)
    minutes, rem = divmod(rem, 6000)
    seconds, cents = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cents:02d}"


def build_ass(manifest: FilmManifest, timeline: Timeline) -> str:
    """由清单与时间线生成 ASS 字幕全文。

    Args:
        manifest: 清单（取画幅与图片署名）。
        timeline: 时间线（取每句与每个镜头的起止）。

    Returns:
        ASS 字幕全文（文本）。事件：每句旁白一条 Dialogue（Default 样式，折行用 ``\\N``
        连接）；image 镜头（含图层）在该镜头期间以 Credit 样式显示
        ``图片：{署名}，{许可}``（多个资源用"；"连接）。
    """
    width, height = manifest.size
    fontsize = round(height * 0.045)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,PingFang SC,{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H96000000,"
        "0,0,0,0,100,100,0,0,1,2,1,2,40,40,40,1",
        "Style: Credit,PingFang SC,16,&H00D8D8D8,&H000000FF,&H00000000,&H96000000,"
        "0,0,0,0,100,100,0,0,1,1,1,3,30,30,26,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for shot in timeline.shots:
        for sentence in shot.sentences:
            text = "\\N".join(wrap_sentence(sentence.text))
            lines.append(f"Dialogue: 0,{ass_time(sentence.start)},{ass_time(sentence.end)},"
                         f"Default,,0,0,0,,{text}")
    for shot in timeline.shots:
        spec = manifest.shot(shot.shot_id)
        if not spec.layers:
            continue
        credits = "；".join(f"{manifest.assets[layer.asset].credit}，{manifest.assets[layer.asset].license}"
                            for layer in spec.layers)
        lines.append(f"Dialogue: 0,{ass_time(shot.start)},{ass_time(shot.start + shot.duration)},"
                     f"Credit,,0,0,0,,图片：{credits}")
    return "\n".join(lines) + "\n"
