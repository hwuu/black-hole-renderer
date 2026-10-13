# 制片层设计：从渲染器到成片

> **状态**：v0.3（2026-10-13），方案已确认，未实施。首个使用者是儿童科普片，执行方案见
> [`plans/kids_explainer_plan.md`](plans/kids_explainer_plan.md)。
>
> **关联文档**：渲染引擎见 [`design_ad_v2.md`](design_ad_v2.md)、[`imaging_model.md`](imaging_model.md)；
> 运镜路径文件格式见 [`plans/v2_camera_path_plan.md`](plans/v2_camera_path_plan.md) §5.1。
>
> **术语**：
> - **成片**：一部完整的影片，由若干镜头按时间线拼接，配旁白、字幕、音效与音乐。
> - **镜头（shot）**：成片中一段连续画面，类型为渲染（render）、图解（diagram）或图片（image）。
> - **清单（manifest）**：描述一部成片的 JSON 文件 `film.json`：画幅、帧率、配音、镜头列表。
> - **时间线（timeline）**：每个镜头的起止时刻与时长，由旁白时长推出（§4.2）。
> - **场景旋钮**：渲染引擎中可逐帧变化的三个场景参数（引力强度、多普勒强度、盘的气体量），
>   定义见执行方案 §3。

---

## 1. 背景

渲染器（物理引擎、相机路径、光线追踪、成像链）已经成熟，能单独产出镜头。做一部有旁白、图解、字幕的成片，
还需要编排旁白与画面的时间、生成图解动画、合成音视频。这些工作与"黑洞怎么成像"无关，不应放进渲染引擎。

## 2. 分层与依赖

```
+--------------------------------------------------------------------+
| Content      scenes/<film>/                                        |
|              film.json  script.md  shots/*.json  diagrams/*.py     |
+--------------------------------------------------------------------+
| Production   src/film/                 (python -m src.film ...)    |
|              manifest  timeline  narration  subtitles  sfx  hud    |
|              render_jobs  diagrams  diagram_kit  assemble          |
+--------------------------------------------------------------------+
| Engine       src/core/  src/v2/  src/cli.py  (render.py)           |
|              physics / camera path / ray tracing / imaging         |
+--------------------------------------------------------------------+
```

| 层 | 职责 | 依赖规则 |
|----|------|----------|
| 内容层 `scenes/<film>/` | 一部片子的数据：清单、脚本、运镜路径，以及本片专属的 Manim 图解脚本 | 只依赖制片层 `diagram_kit` 与引擎层的纯 NumPy 模块 |
| 制片层 `src/film/` | 通用的成片流水线：读清单、排时间线、配音、字幕、音效、遥控器叠加层、渲染任务、图解、合成 | 依赖引擎层的公开接口；不认识任何具体片子 |
| 引擎层 `src/core/`、`src/v2/`、`src/cli.py` | 物理、相机、光追、成像 | **不认识"成片"概念**，不 import `src/film` |

**引擎调用方式（已确认）**：制片层通过 `render.py` 命令行调用渲染，不直接 import 视频管线。
- 命令行是引擎现有的稳定接口，有文档与单测；
- 每个渲染镜头对应一条可复现的命令，可导出为脚本在远端 4090 上执行（见 AGENTS.md "远程渲染"）；
- 引擎层不需要重构（运镜视频管线目前是 `src/cli.py:main` 内的闭包）。

制片层可以 import 引擎层的**纯函数模块**（不触发 Taichi 编译），目前用到两个：
`src/v2/scene_anim.py`（旋钮轨道求值，遥控器叠加层用它保证与画面同步）、
`src/core/geodesic.py`（NumPy 光线积分，图解用它画光路）。
两个模块都不得 import Taichi——它们要在制片层与图解的进程内使用。

## 3. 数据模型

### 3.1 清单 `film.json`

```json
{
  "title": "黑洞是什么",
  "size": [1920, 1080],
  "fps": 30,
  "voice": {"engine": "edge-tts", "name": "zh-CN-YunxiaNeural", "rate": "+0%"},
  "music": null,
  "padding": {"lead": 0.4, "gap": 0.25, "tail": 0.6},
  "render_defaults": ["--ar1", "3", "--ar2", "30", "--device", "gpu"],
  "shots": [
    {"id": "s04_gravity", "kind": "render", "path": "shots/s04_gravity.json",
     "render_args": ["--v2_doppler_lum", "1"], "hud": true,
     "narration": ["想象一下，我们有一个宇宙遥控器。", "把引力调小……"],
     "min_duration": 20, "transition": {"type": "fade", "duration": 0.5}},
    {"id": "s05_light_paths", "kind": "diagram", "scene": "diagrams/light_paths.py:LightPaths",
     "narration": ["……"]},
    {"id": "s06_siren", "kind": "diagram", "scene": "diagrams/siren.py:Siren",
     "narration": ["……"]},
    {"id": "s09_photo", "kind": "image", "layers": [{"asset": "eht_m87", "box": [0.05, 0.1, 0.45, 0.8]}],
     "narration": ["……"]}
  ],
  "assets": {"eht_m87": {"url": "https://…", "credit": "EHT Collaboration", "license": "CC BY 4.0"}},
  "sfx": [{"shot": "s06_siren", "kind": "siren_doppler", "at": 1.0, "params": {"f0": 700, "v": 25}}]
}
```

| 字段 | 含义 | 约束 |
|------|------|------|
| `title` | 片名，用于产物文件名与字幕 | 必填 |
| `size` / `fps` | 成片画幅与帧率；`size` 必须对应 CLI 预设之一（`sd`/`hd`/`fhd`/`4k`，如 `[1920,1080] → fhd`），由它生成 `-r` | 预设之外的尺寸报错 |
| `voice` | 配音引擎与音色、语速 | 当前只支持 `edge-tts`；`rate` 形如 `"+0%"` |
| `music` | 背景音乐文件路径或 null | 可省略；说话时自动压低（§4.7） |
| `padding` | 旁白留白（秒）：镜头开头 `lead`、句间 `gap`、结尾 `tail` | 均 ≥ 0 |
| `render_defaults` | 所有渲染镜头共用的 `render.py` 参数 | 不得包含 `-r`/`--resolution`、`--fps`、`-o`/`--output`、`--video`、`--n_frames`、`--resume`、`--v2_camera_path`、`--v2_camera_path_time`、`--disk_model`（由制片层生成；制片层只支持 V2 路径模式，`--disk_model v2` 由 jobs 固定给出） |
| `shots[].id` | 镜头 id，也是产物文件名 | 唯一，`[a-z0-9_]+` |
| `shots[].kind` | `render` / `diagram` / `image` | 必填 |
| `shots[].path` | render：运镜路径文件（相对清单目录） | render 必填 |
| `shots[].render_args` | render：本镜头追加的 `render.py` 参数 | 同 `render_defaults` 的限制 |
| `shots[].hud` | render：是否叠加宇宙遥控器（§4.6） | 默认 false；仅 render |
| `shots[].padding` | 镜头级留白，覆盖全局 `padding` 的任意子字段 | 可省略（如 s01 钩子前 6 s 无旁白：`{"lead": 6.0}`） |
| `shots[].scene` | diagram：`文件:类名` | diagram 必填 |
| `shots[].layers` | image：图片图层（资源名 + 归一化位置框） | image 必填；`asset` 须在 `assets` 中定义；`box` 为 `[x, y, w, h]`，均在 `[0, 1]` 且 `x+w ≤ 1`、`y+h ≤ 1` |
| `shots[].narration` | 旁白句子列表（每句一条字幕） | 可为空 |
| `shots[].min_duration` | 最短时长（秒） | 默认 0 |
| `shots[].transition` | 与上一镜头的转场：`type` 为 `cut` / `fade`，`fade` 带 `duration` | 默认 cut；`0 < duration ≤ min(D_{i−1}, D_i)`，清单加载时校验 |
| `assets` | 外部图片：下载地址、署名、许可 | 每项含 `url`、`credit`、`license`（必填字符串）；字幕区末尾自动列出署名 |
| `sfx` | 音效：所属镜头、镜头内起始时刻、类型与参数 | `shot` 须指向存在的镜头 id；`at ≥ 0` 且落在该镜头时长内；类型见 §4.5 |

读取与校验沿用引擎层风格：frozen dataclass + 读取函数，非法字段抛 `ValueError` 并给出字段名。
清单是**封闭 schema**：未知字段一律报错。跨字段校验包括：`sfx[].shot` 指向存在的镜头、
`layers[].asset` 在 `assets` 中定义、`hud` 只出现在 render 镜头。

### 3.2 产物目录

产物全部写入 `output/scenes/<film>/`（不入库）：

```
narration/<shot>_<k>.mp3   per-sentence speech (cached by text hash)
timeline.json              shot start/duration + sentence timing
shots/<shot>.mp4           rendered shots (local or rsync'd back from 4090)
diagrams/<shot>.mp4        Manim output
hud/<shot>.mov             remote-control overlay (with alpha)
sfx/<shot>.wav             synthesized sound effects
assets/                    downloaded images
subtitles.ass              burned-in subtitles
render_jobs.sh             render commands for the 4090
<film>_<W>x<H>p<fps>.mp4   final film
```

## 4. 流水线

```
film.json
   |
   v
narrate ----------> narration/*.mp3
   |
   v
timeline ----------> timeline.json          (reads narration/*.mp3; checks path durations)
   |
   +--> diagrams ----> diagrams/*.mp4       (reads timeline.json for sentence timings)
   |
   +--> subtitles --> subtitles.ass          (reads timeline.json)
   |
   +--> jobs --------> render_jobs.sh       (reads timeline.json for D_i)
                           |
                           v
                       shots/*.mp4         (run locally or on the 4090)
   |
   +--> hud ---------> hud/*.mov            (per render shot, its scene_anim track)
   +--> sfx ---------> sfx/*.wav
   +--> assets ------> assets/*
   |
   v
assemble ----> <film>.mp4 (+ subtitles)     (reads all artifacts + timeline.json)
```

子命令：`python -m src.film <film.json> {narrate, timeline, jobs, diagrams, subtitles, hud, sfx, assets, assemble, all}`。
`all` 依次执行 narrate → timeline → jobs → diagrams → subtitles → hud → sfx → assets，最后执行 assemble
（渲染产物缺失时报错并列出缺失项）；渲染镜头由 `render_jobs.sh` 单独执行（本机或远端）。

### 4.1 旁白（narration）

逐句调用 edge-tts 合成 mp3，用 ffprobe 测量时长；缓存键为 `(音色, 语速, 句子文本)` 的哈希，
文本不变不重新合成。**逐句合成**使每句字幕的起止时刻精确已知，不需要按字数估计。

### 4.2 时间线（timeline）

**旁白先行**：镜头时长由旁白决定，画面按时长制作。`lead_i`、`gap_i`、`tail_i` 为镜头 i 的留白
（镜头级 `padding` 覆盖全局值）。

```
D_i = max(min_duration_i, lead_i + Σ_k s_ik + gap_i·max(n_i − 1, 0) + tail_i)
start_0 = 0
start_i = start_{i−1} + D_{i−1} − x_i        x_i = fade duration of shot i (0 for cut)
sentence k of shot i: [start_i + lead_i + Σ_{j<k}(s_ij + gap_i),  … + s_ik]
```

- `s_ik`：第 i 镜头第 k 句的配音时长；`n_i`：句数；`x_i`：与上一镜头的交叉淡化重叠时长。
  无旁白的镜头（n_i = 0）时长为 `max(min_duration_i, lead_i + tail_i)`。
- **帧网格量化**：所有镜头时长、起止与句子时刻量化到成片帧网格（1/fps）；渲染端按
  `round(路径 duration × fps)` 生成帧（`cli.py:resolve_n_frames`），合成端以同一规则对齐。
- 渲染镜头：帧数检查 `round(路径 duration × fps) ≥ ceil(D_i × fps)`。不满足时报错并给出需要的时长，
  由作者修改路径文件；制片层不自动拉伸路径，以免改变运镜节奏。冗余的尾部帧在合成时裁掉（§4.7）。
- 转场约束：`fade` 时长 `x_i` 满足 `0 < x_i ≤ min(D_{i−1}, D_i)`（清单加载时校验，见 §3.1）；
  `cut` 为直接拼接（x_i = 0）。
- 成片内渲染镜头的路径文件应设 `fade: {"fade_in": 0, "fade_out": 0}`：转场统一由清单的 `transition`
  负责，否则路径自带的淡入淡出会与交叉淡化叠加。`jobs` 生成命令时校验，非 0 给出警告。
- 图解镜头：时长与句子时刻写入 `timeline.json`，Manim 脚本读取后按句子节拍安排动画（§4.8）。

### 4.3 渲染任务（render_jobs）

为每个 render 镜头生成一条 `render.py` 命令：`--disk_model v2 --video` + `render_defaults` + 镜头
`render_args` + `--v2_camera_path <path> -r <size 对应预设> --fps <fps> -o output/scenes/<film>/shots/<id>.mp4`。
生成两种目标：**本机**直接可执行；**远端**替换 Python 解释器路径与输出根目录
（配合 AGENTS.md "远程渲染"，产物用 rsync 取回）。命令写入 `render_jobs.sh`，逐条执行、可断点续传
（`--resume`）。执行前确保输出目录已存在——渲染分段临时目录建在输出文件的同目录
（`video_segments.segment_dir`），其父目录须先建好；跨机器续传需一并同步隐藏分段目录。

### 4.4 字幕（subtitles）

每句一条，单行不超过 18 个汉字，超长时在标点处折行；生成 ASS 字幕（苹方，白字黑边，底部居中），合成时烧进画面。
外部图片的署名（如 "图片：EHT Collaboration，CC BY 4.0"）在该镜头期间显示于右下角。

### 4.5 音效（sfx）

NumPy 合成，输出 wav。首个类型 `siren_doppler`：一个声源以速度 v 沿直线驶过听者，频率与响度随时间变化：

```
f_obs(t) = f_src · c_s / (c_s − v·cosθ(t))         c_s = 343 m/s
```

θ(t) 为声源速度与"声源 → 听者"方向的夹角；按发出时刻（考虑声音传播延迟）积分相位，避免变调不连续；
响度按 1/距离 衰减并限幅。

### 4.6 宇宙遥控器叠加层（hud）

渲染镜头可在画面角落叠加一个遥控器：三个旋钮"引力 / 气体 / 多普勒"，指针角度与刻度显示当前值。
逐帧数值直接调用引擎层 `src/v2/scene_anim.py`，用渲染时同一条轨道求值，**同步由构造保证**。
用 PIL 绘制带透明通道的帧序列，编码为 `hud/<shot>.mov`（ProRes 4444 或 PNG 编码，保留透明通道）。

### 4.7 合成（assemble）

生成一条 ffmpeg 命令（滤镜图）：
1. 各镜头从开头保留 `D_i`、统一画幅与帧率；图片镜头由图层合成静帧；
2. 渲染镜头叠加遥控器；
3. 按 `x_i` 交叉淡化（`xfade`）拼接；
4. 旁白按句子时刻放置（`adelay`）并混音；音效按镜头内时刻放置；背景音乐随旁白压低（`sidechaincompress`）；
5. 烧录字幕；输出 H.264 + AAC。

滤镜图由纯函数生成，单测只校验命令文本；另有一个冒烟测试，用两段合成小视频实际运行 ffmpeg（无 ffmpeg 时跳过）。

### 4.8 图解（diagrams）

内容层的 Manim 场景类负责具体画面；制片层 `diagram_kit` 提供：
- `load_timing(shot_id)`：读取 `timeline.json` 中该镜头的时长与句子时刻；
- 统一样式：配色、字体（苹方）、字号、线宽；
- 物理绘图辅助：用 `src/core/geodesic.py` 积分光线，返回可直接画的折线。

`diagrams` 子命令以固定参数调用 Manim（`-r W,H --fps fps`，媒体目录指向产物目录），并通过环境变量
传入清单路径与镜头 id；`load_timing` 由清单路径推出产物目录（`output/scenes/<清单目录名>/`），
从 `timeline.json` 取该镜头的时长与句子时刻。中文只用 `Text`，不用 LaTeX。

## 5. 依赖

制片层依赖单独列在 `requirements-film.txt`（`manim`、`edge-tts`），不进 `requirements.txt`，引擎环境保持精简。
系统依赖：ffmpeg（含 libass）；Manim 的 cairo / pango 优先由 pip 轮子提供，缺库时用 brew 补装。
edge-tts 调用微软在线服务，需要联网。

## 6. 测试

| 模块 | 测试要点 |
|------|----------|
| `manifest` | 合法清单读取；缺字段、类型错、id 重复、未知字段、`render_args` 含保留参数、`size` 非预设、`transition` 超出相邻镜头时长时报错并给出字段名；`sfx[].shot` / `layers[].asset` / `hud` / `sfx[].at` 引用校验 |
| `timeline` | 时长公式（含无旁白镜头 n = 0）、最短时长、交叉淡化重叠、句子时刻、帧网格量化、镜头级 padding 覆盖；路径时长不足时报错并给出需要值 |
| `narration` | 缓存键、缓存命中不调用合成（合成函数注入替身）、ffprobe 时长解析 |
| `subtitles` | 折行规则、ASS 时间格式、署名显示区间 |
| `sfx` | 警笛瞬时频率曲线与公式一致（过零点法测频）、驶近高于驶离、首尾无爆音 |
| `hud` | 旋钮值 → 指针角度映射；与 `scene_anim` 求值逐帧一致 |
| `render_jobs` | 命令生成（参数拼接、`--disk_model v2` 固定、输出路径、fade 非 0 警告、远端 Python 与输出根目录替换） |
| `diagrams` | Manim 命令拼接与环境变量、产物路径定位 |
| `diagram_kit` | `load_timing` 读取句时刻、镜头缺失时报错 |
| `assets` | 下载、`layers` 引用校验、缺文件报错 |
| `assemble` | 滤镜图文本；冒烟测试（无 ffmpeg 时跳过） |

## 7. 非目标

- 不做通用剪辑软件：没有时间线编辑界面，所有编排写在清单里。
- 不做自动运镜：每个渲染镜头的路径由作者手写，制片层只校验时长。
- 制片层不直接调用 Taichi，不在进程内渲染。

## 变更记录

- v0.3（2026-10-13）：Oracle 复审修订。`size` 限定 CLI 四档预设、排除表补参数别名并固定
  `--disk_model v2`；嵌套字段约束补全（transition 上限、box 边界、sfx.at、封闭 schema）；
  转场时长约束与帧网格量化写明；产物目录与流程图补 `subtitles.ass`；render_jobs 补输出目录前提；
  `load_timing` 定位约定；测试表补 `diagrams` / `diagram_kit` / `assets`；框图对齐修正。
- v0.2（2026-10-13）：review 修订。新增 `render_jobs` 小节并重排序号；时间线公式修正无旁白镜头
  （n = 0）的边界、帧数化时长检查、镜头级 `padding`；成片镜头要求路径 `fade` 为 0（避免与转场叠加）；
  `render_defaults` 排除参数补全；流程框图重画对齐；测光、依赖与测试要点随之更新。
- v0.1（2026-10-13）：首版。
