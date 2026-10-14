# 《黑洞是什么》脚本与分镜

> **状态**：v2（2026-10-14），成片已产出（1080p30 / 3:41 / 47.7MB），等用户验收。
> **成片**：`output/scenes/kids_explainer/kids_explainer_1920x1080p30.mp4`
> **清单**：[film.json](film.json)；**方案**：[docs/plans/kids_explainer_plan.md](../../docs/plans/kids_explainer_plan.md)；
> **制片层用法**：[docs/design_film.md](../../docs/design_film.md)。
> 目标观众 8–12 岁；横屏 1080p30；VoxCPM2 中文旁白 + 烧录字幕。

## 分镜表

时长为按旁白字数（约 4.3 字/秒）的预估；`min_duration` 保证画面展示时间。
HUD = 宇宙遥控器叠加层。旋钮 (k, d, w) 见方案 §3。

| 镜头 | 类型 | 时长（约） | 画面 | 旋钮 |
|------|------|-----------|------|------|
| s01_hook | render | 12 s | 远景缓推，前 6 s 只有音乐般的画面没有旁白 | 正片外观，无 HUD |
| s02_escape | diagram | 26 s | 扔球落回 → 火箭飞出地球 → 地球压成弹珠（不到 2 cm）→ 逃逸要超过光速 | — |
| s03_dark | render + HUD | 20 s | 只有黑影和被扭曲的星空；然后气体聚拢、盘亮起来 | (1, 0, 0→1) |
| s04_gravity | render + HUD | 26 s | "想象一下"：引力调小，帽子放下、黑影缩成一点；关掉，黑洞消失、透出星光；调回来，黑洞重新长出来 | (1→0→1, 0, 1) |
| s05_light_paths | diagram | 20 s | 侧视光路：盘背面的光绕过头顶进入眼睛；光子环 | — |
| s06_siren | diagram + 音效 | 17 s | 救护车驶过（合成警笛，真实多普勒变调），声波前密后疏 → 换成光波 | — |
| s07_doppler | render + HUD | 19 s | 多普勒开关拧开：朝我们一侧更亮、更蓝，另一侧更暗、更红 | (1, 0→1, 1) |
| s08_safe | diagram | 36 s | 大恒星塌缩；太阳换成同样重的黑洞地球照样转；最近的黑洞约 1500 光年；银河系中心人马座 A* | — |
| s09_photo | image | 26 s | 左：M87*（2019）；中：人马座 A*（2022）；右：我们的俯视模拟图（标注"模拟"） | — |
| s10_recap | render + HUD | 24 s | 远景拉出；三句回顾 + 结尾 | (1, 1, 1) |

## 旁白全文

### s01_hook（镜头级 padding：前 6 s 无旁白）

> 你有没有想过，宇宙里有一种东西，连光都逃不掉？

### s02_escape

> 往天上扔一个球，它会掉下来。
> 火箭飞得足够快，就能离开地球。
> 这个刚好能离开的速度，叫逃逸速度。
> 要是把整个地球，压成一颗不到两厘米的玻璃弹珠呢？
> 那时逃逸速度会超过光速，连光也跑不出来了。

### s03_dark

> 看，宇宙里真的有这样的地方。
> 中间的黑影，就是黑洞的影子。
> 周围的气体被它抓住，转得飞快，热得发光，变成了一个亮亮的圆盘。

### s04_gravity

> 想象一下，我们手里有一个宇宙遥控器。
> 把引力调小——盘子背面的光不绕弯了，帽子放了下来，黑影也缩成了一点点。
> 把引力关掉，黑洞就消失了，中间能看见后面的星星。
> 把引力调回来，黑洞又回来了，帽子也重新戴上了。

### s05_light_paths

> 为什么黑洞会戴帽子呢？
> 因为引力把光掰弯了：盘子背面的光，绕过黑洞的头顶，跑进了我们的眼睛。
> 所以我们能同时看到盘子的正面、背面和底下——黑影比黑洞本身大两倍多。

### s06_siren

> 先听一个熟悉的声音。
> 救护车开过来的时候，声音又高又尖；开过去以后，马上变得低沉。
> 光也一样：朝我们跑来的光被挤紧，跑开的光被拉开。

### s07_doppler

> 黑洞旁边的气体转得有多快？最快能达到光速的一半！
> 现在，打开遥控器上的多普勒开关。
> 朝我们转过来的一侧，变得更亮、更蓝；转走的一侧，变得更暗、更红。

### s08_safe

> 黑洞是怎么来的？比太阳重二三十倍以上的大恒星，烧到最后，会塌成一个黑洞。
> 那黑洞会把地球吸走吗？
> 不会。它不是吸尘器：隔得够远的时候，它的引力和同样重的普通星球一模一样。
> 就算把太阳换成同样重的黑洞，地球照样绕着转，只是天会变黑、变冷。
> 离我们最近的黑洞，也在一千五百光年之外；银河系中心就有一个，四百万个太阳那么重。

### s09_photo

> 科学家真的拍到过黑洞。
> 左边这张，是五千五百万光年外的M87星系的黑洞；中间这张，是我们银河系中心的人马座A。
> 右边，是我们用物理规律算出来的样子。
> 照片里那一圈亮环，是发光的气体；中间的洞，是黑洞的影子。

### s10_recap

> 记住今天的三件事。
> 黑洞的引力大到连光都跑不出来，所以它是黑的。
> 引力把光掰弯，让它戴上了奇怪的帽子。
> 朝我们转来的气体更亮更蓝，黑洞离我们很远，不危险。
> 宇宙里还有很多秘密，等你长大去发现。

## 措辞与事实依据

| 内容 | 依据 |
|------|------|
| 地球压成"不到两厘米的玻璃弹珠"（2r_s = 17.7 mm） | Chandra 黑洞入门；中国科普博览 18 mm |
| 照片里是"黑洞的影子"（约为视界 2.5 倍） | ESO eso1907 |
| "它不是吸尘器"（远处引力与同质量星球相同；太阳换成黑洞地球照样转） | NASA 黑洞总览页 |
| 恒星阈值"比太阳重二三十倍以上"（不给单一值） | NASA 约 20 倍、Chandra 约 25 倍、王松 2024 为 25–40 倍 |
| 最近的黑洞"约一千五百光年"（Gaia BH1） | NASA 黑洞总览页 |
| 银河系中心"四百万个太阳"（人马座 A*，2022 年拍到） | NASA、ESO |
| M87*"五千五百万光年"（不用旧值 5300 万） | ESO eso1907 |
| 气体最快"光速的一半"（ISCO 处 β = 0.5） | `relativity.orbital_beta_local` |
| 引力旋钮口吻为想象实验（画面标注"想象一下"） | 防止误以为引力可调 |
| 颜色说明：真实吸积盘主要发射射电/X 射线，金色是色调选择 | 方案 §4.3（S6 图解或字幕可补一句） |

## 制作说明

- **参数组**：s01 用正片外观（白平衡 10000 K、胶片响应 1、EV +3）；s03/s04/s07/s10 用教学参数组
  （多普勒亮度与颜色取物理值 1、`--v2_color_cap 0` 取消色温封顶使迎面侧真实偏蓝、EV +3）。
  具体观感 S5 小样确认。
- **旋钮轨道**（`shots/*.json` 的 `scene_anim` 块，时刻为草稿、S5 随小样微调）：
  - s03：w 0→1（5–14 s），曝光锁定（gain 0，reference_time 20）；
  - s04：k 1→0→1（保持 5 s / 6 s / 渐回 7 s），d 恒为 0（"k 旋钮 + d = 1"物理上不自洽，影片中不出现）；
  - s07：d 0→1（7–14 s），reference_time 18（多普勒全开的目标外观）；
  - s10：三旋钮全 1。
- **s09 模拟图**：S5 用 `shots/s09_sim.json`（近盘轴俯视，约 11°）单帧渲染
  （`--v2_camera_path_time`），按 EHT 分辨率高斯模糊后输出到
  `output/scenes/kids_explainer/shots/s09_sim_face_on.png`，画面标注"模拟"。
  M87* 倾角（喷流约 17°）在 S5 前查 EHT 论文核实后微调。
- **EHT 照片素材**（CC BY 4.0，署名 EHT Collaboration，字幕右下角自动显示）：
  - M87*：`https://www.eso.org/public/archives/images/large/eso1907a.jpg`（已核实，ESO eso1907a）；
  - 人马座 A*：`https://www.eso.org/public/archives/images/large/eso2208-eht-mw.jpg`
    **（未核实——图页 ID 多次 404，S7 下载前在 ESO 图片库确认实际地址）**。
- **音效**：s06 合成警笛（多普勒变调，`sfx.py`），镜头内 2.5 s 处开始、持续 6 s，与第 2 句旁白同步。
- **音色**：macOS 神经声音（Flo / Eddy 等）在 `say -v "?"` 中列出但**未下载**，合成时静默输出
  空音频（S4 实测 9 个 zh_CN 声音中 8 个如此）；系统预装经典声**婷婷 Tingting**（大陆腔）为
  正片草稿音色。对比小样：`output/scenes/kids_explainer/narration/voice_samples/`
  （Tingting 大陆腔 / Meijia 台湾腔 / Sinji 粤语腔）。如需更自然的神经声音：系统设置 →
  辅助功能 → 朗读内容 → 系统声音 → 管理声音 → 下载 Flo / Eddy，改 film.json 的 `voice.name`
  后重跑 narrate + timeline 即可（缓存键含音色名，自动重合成）。edge-tts 在线引擎在本机
  网络不可达（S0 实测），服务恢复后把 `voice.engine` 改回 `edge-tts` 同理切换。
- **音色升级 TODO（Tingting 观感不佳，待换）**：VoxCPM2（`openbmb/VoxCPM2`，Apache-2.0，
  2B 参数，48 kHz，中文含方言；Voice Design 用自然语言描述音色即可，如"温柔的年轻女声"，
  无需参考音频）。约束：CUDA ≥ 12 + 约 8 GB 显存 → 本机 M5 不可用，放 4090 实例合成
  （RTF ≈ 0.3；模型约 4 GB 经 `HF_ENDPOINT=https://hf-mirror.com` 下载，`pip install voxcpm`
  含 torch ≥ 2.5），rsync 回 `output/scenes/kids_explainer/narration/`。集成要点：
  narration.py 引擎注册表加 `voxcpm`、manifest `VOICE_ENGINES` 白名单扩展；**远端合成脚本
  须用 `narration._cache_key` 同一规则写 `.key` 伴档**，本地 narrate 才能全量命中、只测时长；
  换后重跑 narrate + timeline（时间线随新时长重排）。
- **分辨率口径**：本片教学口径"望远镜分辨率约 20 微角秒 ≈ 在上海看清北京的一根头发丝"
  （ESO 官方类比：在月球表面看一张信用卡，两者等价）。

## 构建命令

```bash
# 旁白（缓存：文本/音色不变不重合成）
python -m src.film scenes/kids_explainer/film.json narrate

# 时间线（含旁白合成与路径时长校验）→ output/scenes/kids_explainer/timeline.json
python -m src.film scenes/kids_explainer/film.json timeline

# 渲染任务（本机 / 远端 4090）
python -m src.film scenes/kids_explainer/film.json jobs
python -m src.film scenes/kids_explainer/film.json jobs --remote

# 音效 / 字幕
python -m src.film scenes/kids_explainer/film.json sfx
python -m src.film scenes/kids_explainer/film.json subtitles
```
