"""制片层清单（film.json）的数据模型与读取校验。

设计见 `docs/design_film.md` §3.1。清单是**封闭 schema**：未知字段一律报错，
让拼写错误在加载时就失败而不是静默生效；非法内容抛 `ValueError` 并在信息中给出字段名。
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# 成片画幅 → 渲染 CLI 的分辨率预设（与 src/cli.py 的 resolutions 表一致）
SIZE_PRESETS: Dict[Tuple[int, int], str] = {
    (640, 360): "sd", (1280, 720): "hd", (1920, 1080): "fhd", (3840, 2160): "4k",
}
# 由制片层生成、清单中禁止出现的 render.py 参数（docs/design_film.md §3.1）
RESERVED_RENDER_ARGS = frozenset({
    "-r", "--resolution", "--fps", "-o", "--output", "--video", "--n_frames",
    "--resume", "--v2_camera_path", "--v2_camera_path_time", "--disk_model",
})
SHOT_KINDS = frozenset({"render", "diagram", "image"})
VOICE_ENGINES = frozenset({"edge-tts", "say", "voxcpm"})
SHOT_ID_RE = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class VoiceSpec:
    """配音引擎与音色。

    Attributes:
        engine: `edge-tts`（在线）、`say`（macOS 本地回退）或 `voxcpm`（大模型）。
        name: 音色名（edge-tts/say）或人设提示（voxcpm Voice Design）。voxcpm 克隆模式下未使用。
        rate: 语速（仅 edge-tts 使用，形如 `"+0%"`）。
        reference_wav: 克隆模式参考音频路径（仅 voxcpm；相对清单目录或绝对路径）。
            设置后 `synth_voxcpm` 用 Voice Cloning 逐句模仿同一人声，彻底解决一致性。
    """

    engine: str
    name: str
    rate: str
    reference_wav: Optional[str] = None


@dataclass(frozen=True)
class Padding:
    """旁白留白（秒）：镜头开头 `lead`、句间 `gap`、结尾 `tail`，均 ≥ 0。

    Attributes:
        lead: 第一句旁白相对镜头起点的延迟（秒；钩子镜头用它做无声开场）。
        gap: 相邻两句旁白之间的间隔（秒）。
        tail: 最后一句旁白结束到镜头末尾的最少余量（秒）。
    """

    lead: float
    gap: float
    tail: float


DEFAULT_PADDING = Padding(lead=0.4, gap=0.25, tail=0.6)
"""省略 `padding` 块时的留白（docs/design_film.md §3.1 示例值）。"""


@dataclass(frozen=True)
class Transition:
    """与上一镜头的转场：`cut` 直接拼接，`fade` 交叉淡化 `duration` 秒。"""

    type: str
    duration: float


@dataclass(frozen=True)
class Layer:
    """image 镜头的一个图片图层。

    Attributes:
        asset: 资源名（须在清单 `assets` 中定义）。
        box: 归一化位置框 `[x, y, w, h]`（左上角 + 尺寸，均在 [0, 1] 且不越出画面）。
    """

    asset: str
    box: Tuple[float, float, float, float]


@dataclass(frozen=True)
class Asset:
    """外部图片资源：下载地址、署名与许可（署名在字幕中自动显示）。"""

    url: str
    credit: str
    license: str


@dataclass(frozen=True)
class SfxEvent:
    """一条音效事件：`kind` 类型（见 `sfx` 模块）、镜头内起始时刻 `at`（秒）与参数。"""

    shot: str
    kind: str
    at: float
    params: Dict[str, object]


@dataclass(frozen=True)
class Shot:
    """一个镜头（render / diagram / image 三选一）。

    Attributes:
        id: 镜头 id（唯一，`[a-z0-9_]+`），也是产物文件名。
        kind: `render` / `diagram` / `image`。
        path: render：运镜路径文件（相对清单目录）。
        render_args: render：本镜头追加的 render.py 参数。
        hud: render：是否叠加宇宙遥控器。
        padding: 解析后的完整留白（全局 `padding` 被镜头级子字段覆盖）。
        scene: diagram：`文件:类名`。
        layers: image：图片图层（非空）。
        narration: 旁白句子列表（每句一条字幕，可为空）。
        min_duration: 最短时长（秒）。
        transition: 与上一镜头的转场；首个镜头为 None。
    """

    id: str
    kind: str
    path: Optional[str] = None
    render_args: Tuple[str, ...] = ()
    hud: bool = False
    padding: Padding = DEFAULT_PADDING
    scene: Optional[str] = None
    layers: Tuple[Layer, ...] = ()
    narration: Tuple[str, ...] = ()
    min_duration: float = 0.0
    transition: Optional[Transition] = None


@dataclass(frozen=True)
class FilmManifest:
    """一部成片的全部编排数据（清单 schema 见 docs/design_film.md §3.1）。

    Attributes:
        title: 片名。
        size: 成片画幅 `(宽, 高)`，必须对应 `SIZE_PRESETS` 之一。
        fps: 成片帧率。
        voice: 配音引擎与音色。
        music: 背景音乐文件路径；None = 无音乐。
        padding: 全局旁白留白。
        render_defaults: 所有渲染镜头共用的 render.py 参数。
        shots: 镜头列表（至少一个）。
        assets: 外部图片资源（名 → 资源）。
        sfx: 音效事件列表。
        base_dir: 清单文件所在目录（镜头路径相对它解析）。
    """

    title: str
    size: Tuple[int, int]
    fps: int
    voice: VoiceSpec
    music: Optional[str]
    padding: Padding
    render_defaults: Tuple[str, ...]
    shots: Tuple[Shot, ...]
    assets: Dict[str, Asset]
    sfx: Tuple[SfxEvent, ...]
    base_dir: str

    @property
    def resolution_key(self) -> str:
        """成片画幅对应的渲染 CLI 分辨率预设（如 `[1920, 1080] → "fhd"`）。"""
        return SIZE_PRESETS[self.size]

    @property
    def film_id(self) -> str:
        """片子标识 = 清单所在目录名（产物写到 `output/scenes/<film_id>/`）。"""
        return os.path.basename(os.path.normpath(self.base_dir))

    @property
    def output_dir(self) -> str:
        """产物目录（仓库惯例：`output/scenes/<film_id>/`，不入库）。"""
        return os.path.join("output", "scenes", self.film_id)

    def shot(self, shot_id: str) -> Shot:
        """按 id 取镜头；不存在时报错（信息中给出可用 id）。"""
        for shot in self.shots:
            if shot.id == shot_id:
                return shot
        raise ValueError(f"未知镜头 {shot_id!r}（可用：{[s.id for s in self.shots]}）")


def _fail(field: str, message: str) -> None:
    """抛出带字段名的 `ValueError`（设计约定：校验错误必须给出字段名）。

    Args:
        field: 字段名（如 `shots[0].path`）。
        message: 错误内容。

    Raises:
        ValueError: 总是。
    """
    raise ValueError(f"{field}: {message}")


def _check_keys(block: dict, allowed: set, field: str) -> None:
    """封闭 schema 检查：块中只允许出现 `allowed` 里的字段。

    Args:
        block: 清单中的 JSON 块。
        allowed: 允许的字段名集合。
        field: 报错时使用的字段名。

    Raises:
        ValueError: 块中存在未知字段（信息中列出字段名与 `field`）。
    """
    unknown = sorted(set(block) - allowed)
    if unknown:
        _fail(field, f"未知字段 {unknown}（封闭 schema）")


def _nonempty_str(value, field: str) -> str:
    """校验并返回非空字符串。

    Args:
        value: 待校验值。
        field: 报错时使用的字段名。

    Returns:
        原字符串（保证非空）。

    Raises:
        ValueError: 不是字符串或为空串。
    """
    if not isinstance(value, str) or not value:
        _fail(field, f"必须是非空字符串，得到 {value!r}")
    return value


def _finite(value, field: str, minimum: float = 0.0, strict: bool = False) -> float:
    """校验并返回有限数；默认要求 ≥ `minimum`（`strict = True` 时为 >）。

    Args:
        value: 待校验值（布尔值按非法处理）。
        field: 报错时使用的字段名。
        minimum: 下限（默认 0）。
        strict: True 时要求严格大于 `minimum`。

    Returns:
        转换后的浮点数。

    Raises:
        ValueError: 不是数字 / 非有限数 / 越过下限。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(field, f"必须是数字，得到 {value!r}")
    v = float(value)
    if not math.isfinite(v):
        _fail(field, f"必须是有限数，得到 {value!r}")
    if v < minimum or (strict and v == minimum):
        _fail(field, f"必须 {'>' if strict else '≥'} {minimum}，得到 {v}")
    return v


def _padding_from(block, field: str, base: Padding) -> Padding:
    """由 `block`（可为 None）构造留白：缺省字段取 `base` 的值。

    Args:
        block: 清单中的 padding 块（`{"lead": .., "gap": .., "tail": ..}` 的任意子集）。
        field: 报错时使用的字段名。
        base: 基准留白（全局值或默认值）。

    Returns:
        解析后的完整 `Padding`。
    """
    if block is None:
        return base
    if not isinstance(block, dict):
        _fail(field, "必须是对象")
    _check_keys(block, {"lead", "gap", "tail"}, field)
    values = {name: _finite(block.get(name, getattr(base, name)), f"{field}.{name}")
              for name in ("lead", "gap", "tail")}
    return Padding(**values)


def _render_args_from(value, field: str) -> Tuple[str, ...]:
    """校验 render.py 参数列表：全部是非空字符串，且不含制片层保留参数。

    保留参数按**选项名**比较：`--n_frames=1` 这类 `--参数=值` 写法先取 `=` 前的
    选项名再比对（argparse 同样接受该写法，只比对整串会漏防）。参数值不可能恰好
    等于保留旗标名，因此非选项位置的字符串无须比较。

    Args:
        value: 清单中的参数列表（`render_defaults` 或镜头 `render_args`）。
        field: 报错时使用的字段名。

    Returns:
        校验后的参数元组。

    Raises:
        ValueError: 列表元素不是非空字符串，或包含保留参数（信息中给出该参数）。
    """
    if not isinstance(value, list):
        _fail(field, "必须是字符串列表")
    for token in value:
        if not isinstance(token, str) or not token:
            _fail(field, f"必须是非空字符串，得到 {token!r}")
    for token in value:
        if token.startswith("-"):
            name = token.split("=", 1)[0]
            if name in RESERVED_RENDER_ARGS:
                _fail(field, f"不得包含由制片层生成的参数 {token!r}")
    return tuple(value)


def _layers_from(value, field: str) -> Tuple[Layer, ...]:
    """校验 image 镜头的图层列表（非空；资源引用与位置框合法且有限）。

    Args:
        value: 清单中的 `layers` 列表。
        field: 报错时使用的字段名。

    Returns:
        `Layer` 元组（非空）。

    Raises:
        ValueError: 列表为空 / 元素不是对象 / `asset` 非空字符串缺失 /
            `box` 不是四个有限数、为负或越出画面（x+w > 1 或 y+h > 1）。
    """
    if not isinstance(value, list) or not value:
        _fail(field, "必须是非空列表")
    layers = []
    for i, raw in enumerate(value):
        f = f"{field}[{i}]"
        if not isinstance(raw, dict):
            _fail(f, "必须是对象")
        _check_keys(raw, {"asset", "box"}, f)
        box = raw.get("box")
        if (not isinstance(box, list) or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in box)):
            _fail(f"{f}.box", f"必须是四个数字 [x, y, w, h]，得到 {box!r}")
        x, y, w, h = (float(v) for v in box)
        if (any(not math.isfinite(v) for v in (x, y, w, h))
                or min(x, y, w, h) < 0.0 or x + w > 1.0 or y + h > 1.0):
            _fail(f"{f}.box", f"必须是四个有限数且落在画面内（x+w ≤ 1、y+h ≤ 1、均 ≥ 0），得到 {box}")
        layers.append(Layer(asset=_nonempty_str(raw.get("asset"), f"{f}.asset"), box=(x, y, w, h)))
    return tuple(layers)


def _transition_from(raw, field: str, is_first: bool) -> Optional[Transition]:
    """校验与上一镜头的转场；`fade` 要求时长为有限正数（帧数化与边界在 timeline 做）。

    Args:
        raw: 清单中的 `transition` 块（None = 省略，默认 cut）。
        field: 报错时使用的字段名。
        is_first: 是否首个镜头（没有上一镜头，不允许 fade）。

    Returns:
        `Transition`；省略时返回 None。

    Raises:
        ValueError: 块不是对象 / 含未知字段 / type 不是 cut 或 fade /
            fade 时长非正数 / 首个镜头声明 fade。
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        _fail(field, "必须是对象")
    _check_keys(raw, {"type", "duration"}, field)
    kind = raw.get("type")
    if kind == "cut":
        return Transition(type="cut", duration=0.0)
    if kind == "fade":
        if is_first:
            _fail(field, "首个镜头没有上一镜头，不能是 fade")
        duration = _finite(raw.get("duration"), f"{field}.duration", strict=True)
        return Transition(type="fade", duration=duration)
    _fail(field, f"type 必须是 cut 或 fade，得到 {kind!r}")


def _shots_from(value, field: str, base_padding: Padding) -> Tuple[Shot, ...]:
    """校验镜头列表：封闭字段、id 唯一合法、按 kind 校验必填与互斥字段。

    Args:
        value: 清单中的 `shots` 列表（须非空）。
        field: 报错时使用的字段名。
        base_padding: 全局留白（镜头级 `padding` 子字段覆盖对应项后存入每个 Shot）。

    Returns:
        `Shot` 元组（至少一个）。

    Raises:
        ValueError: 各类结构与取值错误（信息中给出 `shots[i].字段`）：id 重复或非法、
            kind 未知、必填字段缺失、互斥字段越 kind 出现、`hud` 非布尔、
            `narration` 含非字符串、首镜头声明 fade 等。
    """
    if not isinstance(value, list) or not value:
        _fail(field, "必须是非空列表")
    allowed = {"id", "kind", "path", "render_args", "hud", "padding", "scene",
               "layers", "narration", "min_duration", "transition"}
    kind_fields = {"render": {"path", "render_args", "hud"},
                   "diagram": {"scene"}, "image": {"layers"}}
    shots: list[Shot] = []
    ids: set[str] = set()
    for i, raw in enumerate(value):
        f = f"{field}[{i}]"
        if not isinstance(raw, dict):
            _fail(f, "必须是对象")
        _check_keys(raw, allowed, f)
        shot_id = _nonempty_str(raw.get("id"), f"{f}.id")
        if not SHOT_ID_RE.match(shot_id):
            _fail(f"{f}.id", f"必须匹配 [a-z0-9_]+，得到 {shot_id!r}")
        if shot_id in ids:
            _fail(f"{f}.id", f"镜头 id 重复：{shot_id}")
        ids.add(shot_id)
        kind = raw.get("kind")
        if kind not in SHOT_KINDS:
            _fail(f"{f}.kind", f"必须是 {sorted(SHOT_KINDS)} 之一，得到 {kind!r}")
        for name in ("path", "render_args", "hud", "scene", "layers"):
            if name in raw and name not in kind_fields[kind]:
                _fail(f"{f}.{name}", f"kind={kind} 的镜头不允许字段 {name}")
        path = raw.get("path")
        scene = raw.get("scene")
        if kind == "render":
            path = _nonempty_str(path, f"{f}.path")
        elif kind == "diagram":
            scene = _nonempty_str(scene, f"{f}.scene")
            parts = scene.split(":")
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                _fail(f"{f}.scene", f"必须是 文件路径:类名，得到 {scene!r}")
        layers = _layers_from(raw.get("layers"), f"{f}.layers") if kind == "image" else ()
        hud = raw.get("hud", False)
        if not isinstance(hud, bool):
            _fail(f"{f}.hud", f"必须是布尔值，得到 {hud!r}")
        narration = raw.get("narration", [])
        if not isinstance(narration, list):
            _fail(f"{f}.narration", "必须是字符串列表")
        for k, text in enumerate(narration):
            if not isinstance(text, str) or not text:
                _fail(f"{f}.narration[{k}]", "必须是非空字符串")
        shots.append(Shot(
            id=shot_id, kind=kind, path=path,
            render_args=_render_args_from(raw.get("render_args", []), f"{f}.render_args"),
            hud=hud,
            padding=_padding_from(raw.get("padding"), f"{f}.padding", base_padding),
            scene=scene, layers=layers, narration=tuple(narration),
            min_duration=_finite(raw.get("min_duration", 0.0), f"{f}.min_duration"),
            transition=_transition_from(raw.get("transition"), f"{f}.transition", is_first=(i == 0)),
        ))
    return tuple(shots)


def load_manifest(path: str) -> FilmManifest:
    """读取并校验清单 `film.json`（封闭 schema，见 docs/design_film.md §3.1）。

    Args:
        path: 清单文件路径。

    Returns:
        `FilmManifest`（`base_dir` 为清单所在目录）。

    Raises:
        OSError: 文件无法读取。
        json.JSONDecodeError: 不是合法 JSON。
        ValueError: 结构或取值非法（信息中给出字段名）：未知字段、缺必填字段、
            画幅不对应分辨率预设、参数列表含保留参数、镜头 id 重复或非法、
            kind 与字段不匹配、`sfx[].shot` / `layers[].asset` 引用不存在等。
    """
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    if not isinstance(spec, dict):
        raise ValueError("film.json 必须是 JSON 对象")
    _check_keys(spec, {"title", "size", "fps", "voice", "music", "padding",
                       "render_defaults", "shots", "assets", "sfx"}, "film.json")
    title = _nonempty_str(spec.get("title"), "title")
    size = spec.get("size")
    if (not isinstance(size, list) or len(size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in size)
            or (int(size[0]), int(size[1])) not in SIZE_PRESETS):
        _fail("size", f"必须对应分辨率预设 {sorted(SIZE_PRESETS)} 之一，得到 {size!r}")
    fps = spec.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        _fail("fps", f"必须是正整数，得到 {fps!r}")
    voice_raw = spec.get("voice")
    if not isinstance(voice_raw, dict):
        _fail("voice", "必须是对象")
    _check_keys(voice_raw, {"engine", "name", "rate", "reference_wav"}, "voice")
    engine = voice_raw.get("engine")
    if engine not in VOICE_ENGINES:
        _fail("voice.engine", f"必须是 {sorted(VOICE_ENGINES)} 之一，得到 {engine!r}")
    reference_wav = voice_raw.get("reference_wav")
    if reference_wav is not None and (not isinstance(reference_wav, str) or not reference_wav):
        _fail("voice.reference_wav", "必须是文件路径字符串或省略")
    voice = VoiceSpec(engine=engine, name=_nonempty_str(voice_raw.get("name"), "voice.name"),
                      rate=_nonempty_str(voice_raw.get("rate", "+0%"), "voice.rate"),
                      reference_wav=reference_wav)
    music = spec.get("music")
    if music is not None and (not isinstance(music, str) or not music):
        _fail("music", "必须是文件路径字符串或 null")
    padding = _padding_from(spec.get("padding"), "padding", DEFAULT_PADDING)
    shots = _shots_from(spec.get("shots"), "shots", padding)
    assets_raw = spec.get("assets", {})
    if not isinstance(assets_raw, dict):
        _fail("assets", "必须是对象")
    assets: Dict[str, Asset] = {}
    for name, raw in assets_raw.items():
        f = f"assets.{name}"
        if not isinstance(raw, dict):
            _fail(f, "必须是对象")
        _check_keys(raw, {"url", "credit", "license"}, f)
        assets[name] = Asset(url=_nonempty_str(raw.get("url"), f"{f}.url"),
                             credit=_nonempty_str(raw.get("credit"), f"{f}.credit"),
                             license=_nonempty_str(raw.get("license"), f"{f}.license"))
    sfx_raw = spec.get("sfx", [])
    if not isinstance(sfx_raw, list):
        _fail("sfx", "必须是列表")
    sfx: list[SfxEvent] = []
    for i, raw in enumerate(sfx_raw):
        f = f"sfx[{i}]"
        if not isinstance(raw, dict):
            _fail(f, "必须是对象")
        _check_keys(raw, {"shot", "kind", "at", "params"}, f)
        params = raw.get("params", {})
        if not isinstance(params, dict):
            _fail(f"{f}.params", "必须是对象")
        sfx.append(SfxEvent(shot=_nonempty_str(raw.get("shot"), f"{f}.shot"),
                            kind=_nonempty_str(raw.get("kind"), f"{f}.kind"),
                            at=_finite(raw.get("at"), f"{f}.at"),
                            params=params))
    shot_ids = {s.id for s in shots}
    for event in sfx:
        if event.shot not in shot_ids:
            _fail("sfx", f"shot 必须指向存在的镜头，得到 {event.shot!r}（可用：{sorted(shot_ids)}）")
    for shot in shots:
        for layer in shot.layers:
            if layer.asset not in assets:
                _fail(f"shots[{shot.id}].layers", f"asset {layer.asset!r} 未在 assets 中定义")
    return FilmManifest(
        title=title, size=(int(size[0]), int(size[1])), fps=fps, voice=voice, music=music,
        padding=padding, render_defaults=_render_args_from(spec.get("render_defaults", []),
                                                            "render_defaults"),
        shots=shots, assets=assets, sfx=tuple(sfx),
        base_dir=os.path.dirname(os.path.abspath(path)),
    )
