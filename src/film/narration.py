"""旁白合成：逐句合成 mp3、按 (引擎, 音色, 语速, 文本) 哈希缓存、ffprobe 测时长。

逐句合成使每句字幕的起止时刻精确已知，时间线按实际时长排（`docs/design_film.md` §4.1）。
引擎三选一：`edge-tts`（在线）、`say`（macOS 本地回退）、`voxcpm`（VoxCPM2 大模型
Voice Design，需 CUDA，在远程 GPU 实例上运行——`voice` 为人设提示而非音色名）；
合成与测量函数均可注入替身，单测不依赖网络与外部命令。
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
from typing import Callable, Dict, Tuple

from .manifest import FilmManifest
from .sfx import write_wav

SynthFunc = Callable[[str, str, str, str], None]
"""合成函数：`(text, voice, rate, out_path)` → 写出 mp3。"""

ProbeFunc = Callable[[str], float]
"""时长测量函数：`path` → 秒。"""


def synth_edge_tts(text: str, voice: str, rate: str, out_path: str) -> None:
    """edge-tts 在线合成（调用微软在线服务，需要联网；服务不可达时清单改用 say 引擎）。

    Args:
        text: 旁白文本（UTF-8）。
        voice: 音色名（如 `zh-CN-YunxiaNeural`）。
        rate: 语速（形如 `"+0%"`）。
        out_path: 输出 mp3 路径。
    """
    async def _run() -> None:
        import edge_tts

        await edge_tts.Communicate(text, voice, rate=rate).save(out_path)

    asyncio.run(_run())


def synth_say(text: str, voice: str, rate: str, out_path: str) -> None:
    """macOS `say` 本地合成：先输出 AIFF 再经 ffmpeg 转 mp3（与 edge-tts 同产物格式）。

    Args:
        text: 旁白文本（UTF-8）。
        voice: 音色名（如 `Flo`；`say -v "?"` 列出可用音色）。
        rate: 未使用（设计：语速仅 edge-tts 使用，say 保持自然语速）。
        out_path: 输出 mp3 路径。
    """
    aiff = out_path + ".aiff"
    subprocess.run(["say", "-v", voice, text, "-o", aiff], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", aiff, "-f", "mp3", out_path], check=True)
    os.remove(aiff)


_VOXCPM_MODEL = None
"""VoxCPM2 模型单例（首次加载约 1 分钟、显存约 8 GB；全片句子共用一次加载）。"""


def _voxcpm_model():
    """惰性加载 VoxCPM2 单例；测试可通过直接给 `_VOXCPM_MODEL` 赋值注入替身。

    Args:
        无。

    Returns:
        加载好的 VoxCPM2 模型（带 `generate(text, cfg_value, inference_timesteps)`）。
    """
    global _VOXCPM_MODEL
    if _VOXCPM_MODEL is None:
        from voxcpm import VoxCPM

        _VOXCPM_MODEL = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    return _VOXCPM_MODEL


def synth_voxcpm(text: str, voice: str, rate: str, out_path: str,
                 reference_wav_path: str = None) -> None:
    """VoxCPM2 大模型合成（需 CUDA ≥ 12 与约 8 GB 显存，在远程 GPU 实例上运行）。

    支持两种模式：
    - **Voice Design**（`reference_wav_path=None`）：从 `voice` 人设描述生成声音。
      同一描述在不同句子上可能产出不同音色（模型每次重新演绎）——一致性不保证。
    - **Voice Cloning**（`reference_wav_path` 非空）：从参考音频克隆声音，
      逐句模仿**同一个人**，彻底解决一致性。推荐用于成片。

    Args:
        text: 旁白句正文。
        voice: 人设提示（**不含括号**）。Voice Design 模式下自动包成 `(人设)` 前缀；
            Voice Cloning 模式下未使用（音色由参考音频决定）。
        rate: 未使用（语速由人设/参考音频决定；保持与其它引擎同一签名）。
        out_path: 输出 mp3 路径（先写 wav 再经 ffmpeg 转换，与其它引擎同产物格式）。
        reference_wav_path: 参考音频路径（5–10 秒干净人声，wav/mp3 均可）。
            设置后进入克隆模式。

    Returns:
        无；副作用为写出 `out_path`。

    Formula:
        ```
        Voice Design:  wav = model.generate(f"({voice}){text}", cfg_value=2.0, ...)
        Voice Cloning:  wav = model.generate(text, reference_wav_path=ref, cfg_value=2.0, ...)
        48 kHz float → 16-bit PCM wav（sfx.write_wav）→ ffmpeg -f mp3 → mp3
        ```

    Simplifications:
        全片共用一个参考音频；运行环境需 `HF_ENDPOINT=https://hf-mirror.com`、
        `HF_HOME`（数据盘）与 `HF_HUB_DISABLE_XET=1`（绕过 Xet 401）。
    """
    model = _voxcpm_model()
    if reference_wav_path:
        wav = model.generate(text=text, reference_wav_path=reference_wav_path,
                             cfg_value=2.0, inference_timesteps=10)
    else:
        wav = model.generate(text=f"({voice}){text}", cfg_value=2.0, inference_timesteps=10)
    sample_rate = getattr(model.tts_model, "sample_rate", 48000)
    wav_path = out_path + ".wav"
    write_wav(wav_path, wav, sample_rate)
    # -f mp3：输出路径可能是原子替换的 .tmp（无扩展名信息），显式指定容器格式
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav_path, "-f", "mp3", out_path], check=True)
    os.remove(wav_path)


def probe_duration(path: str) -> float:
    """ffprobe 测音频时长（秒）。

    Args:
        path: 音频文件路径。

    Returns:
        时长（秒，浮点）。

    Raises:
        subprocess.CalledProcessError: 文件不是合法音频。
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def _cache_key(engine: str, voice: str, rate: str, text: str) -> str:
    """缓存键：`sha256(引擎|音色|语速|文本)`——任一项变化都会触发重新合成。

    Args:
        engine / voice / rate / text: 清单中的引擎、音色、语速与该句旁白文本。

    Returns:
        十六进制摘要字符串。
    """
    return hashlib.sha256(f"{engine}|{voice}|{rate}|{text}".encode("utf-8")).hexdigest()


def _cache_hit(mp3: str, key_file: str, key: str) -> bool:
    """缓存命中判定：mp3 与键文件都存在、且键文件内容与当前键一致。

    Args:
        mp3: 音频文件路径。
        key_file: 键文件路径（mp3 旁存，内容为写入时的缓存键）。
        key: 当前缓存键。

    Returns:
        命中为 True（跳过合成）；文件缺失或键不匹配为 False。
    """
    if not (os.path.exists(mp3) and os.path.exists(key_file)):
        return False
    with open(key_file, encoding="utf-8") as f:
        return f.read() == key


def narrate_manifest(manifest: FilmManifest, out_dir: str,
                     synth: Dict[str, SynthFunc] | None = None,
                     probe: ProbeFunc = probe_duration, log=print) -> Dict[Tuple[str, int], float]:
    """逐句合成全片旁白，返回各句时长。

    Args:
        manifest: 清单（取 `voice` 与全部镜头的 `narration`）。
        out_dir: 输出目录（`<产物目录>/narration`），不存在则创建；文件名 `<镜头id>_<句序>.mp3`。
        synth: 引擎名 → 合成函数；None = 默认实现（`edge-tts` 在线 + `say` 本地 + `voxcpm` 大模型）。
        probe: 时长测量函数（默认 ffprobe；测试注入替身）。
        log: 进度输出函数（仅在实际合成时输出；缓存命中静默）。

    Returns:
        `{(镜头 id, 句序): 时长秒}`，覆盖清单中全部旁白句。

    Raises:
        ValueError: 清单的引擎没有对应的合成函数。

    Notes:
        缓存：每个 mp3 旁存同名 `.key` 文件记录写入时的缓存键；文本 / 音色 / 语速 /
        引擎任一变化（键不匹配）都重新合成，否则跳过合成、只测时长。
        重合成先写 `<mp3>.tmp` 再原子替换正式 mp3：合成中途失败时正式产物保持
        旧内容、旧缓存键不变（改回旧文本仍正确命中），失败的临时文件被清理。
        远端合成（voxcpm）后把整个 narration 目录 rsync 回本地即可——`.key` 与
        本地同一规则计算，本地 narrate 全量命中、只测时长。
    """
    os.makedirs(out_dir, exist_ok=True)
    engines = synth if synth is not None else {"edge-tts": synth_edge_tts, "say": synth_say,
                                               "voxcpm": synth_voxcpm}
    if manifest.voice.engine not in engines:
        raise ValueError(f"没有 {manifest.voice.engine!r} 引擎的合成函数（可用：{sorted(engines)}）")
    synth_fn = engines[manifest.voice.engine]
    # Voice Cloning：设置参考音频后包装 synth_fn 并更新缓存键（避免与 Voice Design 的缓存碰撞）
    reference_path = None
    if manifest.voice.reference_wav and manifest.voice.engine == "voxcpm":
        reference_path = manifest.voice.reference_wav
        if not os.path.isabs(reference_path):
            reference_path = os.path.join(manifest.base_dir, reference_path)
        if not os.path.exists(reference_path):
            raise ValueError(f"voice.reference_wav 不存在：{reference_path}")
        base_fn = synth_fn

        def synth_fn(text, voice, rate, out_path, _ref=reference_path, _base=base_fn):
            _base(text, voice, rate, out_path, reference_wav_path=_ref)

    voice_key = (manifest.voice.name if reference_path is None
                 else f"clone:{os.path.basename(reference_path)}")
    durations: Dict[Tuple[str, int], float] = {}
    for shot in manifest.shots:
        for k, text in enumerate(shot.narration):
            key = _cache_key(manifest.voice.engine, voice_key, manifest.voice.rate, text)
            mp3 = os.path.join(out_dir, f"{shot.id}_{k}.mp3")
            if not _cache_hit(mp3, mp3 + ".key", key):
                tmp = mp3 + ".tmp"
                try:
                    synth_fn(text, manifest.voice.name, manifest.voice.rate, tmp)
                    os.replace(tmp, mp3)
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                with open(mp3 + ".key", "w", encoding="utf-8") as f:
                    f.write(key)
                log(f"  合成 {os.path.basename(mp3)}")
            durations[(shot.id, k)] = probe(mp3)
    return durations
