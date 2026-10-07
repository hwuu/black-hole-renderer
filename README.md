# Black Hole Renderer

基于广义相对论的史瓦西黑洞光线追踪渲染器。

## 特性

- **物理正确**：基于史瓦西度规的零测地线方程，正确模拟引力透镜
- **吸积盘渲染**：温度剖面纹理、多普勒效应（亮度+颜色偏移）、FBM噪声絮状结构、边缘软化
- **镜头效果**：分离式 Bloom + 色散（RGB不同模糊半径）、镜头光晕（可选）
- **可调倾角**：支持吸积盘倾斜角度
- **抗锯齿**：Ray differentials + Mipmap LOD，减少摩尔纹
- **高性能**：Taichi 并行框架，1080p 渲染 < 2s
- **视频生成**：支持环绕视频、断点续传
- **Disk V2（体积吸积盘）**：统一气体模型（盘面与上方大气是同一团气体、同一湍流场）+ 发射-吸收-散射积分、刚体环平流（视频不卷绕）、Page–Thorne 温度、严格频移、按真实相机建模的成像链（镜头 PSF / 曝光 / ISP，见 [`docs/imaging_model.md`](docs/imaging_model.md)）。详见 [`docs/design_ad_v2.md`](docs/design_ad_v2.md) 与 [`docs/plans/v2_unified_gas_plan.md`](docs/plans/v2_unified_gas_plan.md)。

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### 单帧渲染

```bash
# 基本用法
python render.py -o output/blackhole.png

# 自定义相机位置和视野
python render.py --pov 6 0 2 --fov 120 -o output/custom.png

# 指定吸积盘半径
python render.py --ar1 2.0 --ar2 5.0 -o output/disk.png

# 高分辨率
python render.py -r 4k -o output/4k.png

# 使用 GPU 加速
python render.py --device gpu -o output/gpu.png
```

### Disk V2 模式（体积吸积盘）

V2 用统一气体模型的 3D 体积密度场（SS 外区结构 + 剪切级联湍流 + 高斯核心盘面 + 指数大气 +
Page–Thorne 温度 + 灰大气温度 + 刚体环平流）+ 体积发射-吸收-散射积分 + 按真实相机建模的成像链
（CIE 黑体 + Y(g·T) 亮度 + 自动曝光与曝光补偿 + 能量守恒镜头 PSF（眩光 + 色差）+ von Kries 白平衡 + 保色度 ACES）。
成像原理（每个像素的颜色怎么来的、参数分哪几层）见 [`docs/imaging_model.md`](docs/imaging_model.md)。
设计见 [`docs/design_ad_v2.md`](docs/design_ad_v2.md)，统一气体模型的推导、原型实测与参数定稿见
[`docs/plans/v2_unified_gas_plan.md`](docs/plans/v2_unified_gas_plan.md)。

```bash
# 单帧（预设 M 构图；默认优化级别 1、超采样倍率 2）
python render.py --disk_model v2 --pov 0 -39.7 4.87 --fov 38 \
                 --ar1 3 --ar2 30 -r fhd --device gpu -o output/v2.png

# 视频（内缘一圈 16 s，可加 --orbit 环绕）
python render.py --disk_model v2 --video --pov 0 -39.7 4.87 --fov 38 \
                 --ar1 3 --ar2 30 -r hd --device gpu --n_frames 240 --fps 24 -o output/v2.mp4

# 运镜视频（示例路径 120 s：远景 → 掠云 → 穿过盘面 → 盘底 → 绕盘边回升 → 远景；帧数 = round(路径时长 × --fps)）
python render.py --disk_model v2 --video --v2_camera_path scenes/v2_arts/interstellar_skim.json \
                 --ar1 3 --ar2 30 --v2_reverse_rotation --v2_orbit_seconds 8 --v2_temp_turb 0.1 \
                 -r fhd --fps 60 --device gpu -o output/v2_path.mp4

# 运镜路径的单帧（检查构图）与联系表（各关键帧经过时刻各一格，画三分线与黑洞目标位置）
python render.py --disk_model v2 --v2_camera_path scenes/v2_arts/interstellar_skim.json --v2_camera_path_time 63.9 \
                 --ar1 3 --ar2 30 --v2_reverse_rotation --v2_orbit_seconds 8 --v2_temp_turb 0.1 \
                 -r sd --device gpu -o output/v2_path_t64.png
python scripts/contact_sheet.py scenes/v2_arts/interstellar_skim.json --v2_reverse_rotation --v2_orbit_seconds 8 \
                 -o output/v2_arts/contact_sheet.png   # 联系表只用于检查构图，不含温度湍流
```

### 视频生成

```bash
# 环绕视频（默认 3600 帧，36 fps）
python render.py --video --orbit -o output/demo.mp4

# 自定义轨道总角度（半圈）
python render.py --video --orbit --orbit_degrees 180 --n_frames 1800 --fps 30 -o output/demo.mp4

# 程序生成吸积盘纹理时使用 1x 原分辨率
python render.py --video --orbit --disk_generation_scale 1 -o output/demo.mp4

# 断点续传
python render.py --video --orbit --resume -o output/demo.mp4
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pov` | 相机位置 (x, y, z) | 6 0 0.5 |
| `--fov` | 视野角度 (0-180°) | 90 |
| `--resolution`, `-r` | 分辨率: 4k/fhd/hd/sd | fhd |
| `--texture`, `-t` | 天空盒纹理路径 | 程序生成 |
| `--disk_texture` | 吸积盘纹理路径 | 程序生成 |
| `--disk_generation_scale` | 程序生成吸积盘纹理时的降采样倍率：1/2/4 | 2 |
| `--ar1` | 吸积盘内半径 | 2.0 rs |
| `--ar2` | 吸积盘外半径 | 15 rs |
| `--disk_tilt` | 吸积盘倾角（度） | 0 |
| `--step_size`, `-s` | 积分步长 | 0.1 |
| `--r_max` | 逃逸半径 | 10 |
| `--n_stars` | 天空盒恒星数量 | 6000 |
| `--anti_alias` | 抗锯齿模式: disabled/lod_radius | disabled |
| `--aa_strength` | 抗锯齿强度 | 1.0 |
| `--lens_flare` | 开启镜头光晕效果 | - |
| `--output`, `-o` | 输出文件路径 | output/blackhole.png |
| `--device`, `-d` | Taichi 设备: cpu/gpu | cpu |

### 视频参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video` | 开启视频模式 | - |
| `--orbit` | 相机围绕原点旋转 | - |
| `--orbit_degrees` | 轨道模式下整段视频的总旋转角度，支持负数反向旋转 | 360 |
| `--n_frames` | 视频帧数 | 3600 |
| `--fps` | 视频帧率 | 36 |
| `--resume` | 从断点恢复 | - |

### Disk V2 参数（仅 `--disk_model v2`）

参数按成像模型分层（原理与每层的物理含义见 [`docs/imaging_model.md`](docs/imaging_model.md)；`--help` 按同样的层分组）：

| 层 | 参数 | 说明 | 默认值 |
|----|------|------|--------|
| 模式 | `--disk_model` | 吸积盘模型: `v1` / `v2` | v1 |
| ① 场景·结构 | `--v2_thickness_scale` | 盘厚缩放：盘面（高斯核心）标高乘该值，柱密度不变，不影响大气标高；1 = 预设 M 视觉厚度 | 1/9（H/r ≈ 0.003） |
| ① 场景·结构 | `--v2_core_contrast` | 主云明暗起伏系数 α（> 0）：缩放絮状结构的明暗对比，不改变形状。1 = 原始起伏；越小越柔和 | 0.8 |
| ① 场景·结构 | `--v2_core_floor` | 主云密度下限 f（0–1）：盘面密度因子 = f + (1 − f)·c/⟨c⟩。越小稀处越空，斜看（如透镜瀑布）时团块之间透出缝；0 = 稀处可完全透空 | 0.15 |
| ① 场景·结构 | `--v2_atm_frac` | 大气柱密度比 A（≥ 0）：盘面上方大气的柱密度 / 盘面核心柱密度。大气与盘面是同一团气体、跟随同一湍流结构，稀处出现空隙；越大越朦胧，盘内视角金色云雾越明显；0 = 无大气 | 0.15 |
| ① 场景·结构 | `--v2_atm_height` | 大气标高 H_a/r（> 0）：大气密度按 exp(−\|z\|/H_a) 随高度衰减，越大越蓬松、伸得越高；不随盘厚缩放 | 0.01 |
| ① 场景·结构 | `--v2_atm_fine` | 大气小尺度起伏强度 σ_a（≥ 0）：保均值对数正态起伏，使大气随高度与盘面脱离、形成独立小云团；0 = 完全跟随盘面；≥ 0.8 时相机易落入整团浓雾 | 0.5 |
| ① 场景·结构 | `--v2_disk_roll` | 盘滚转角（度），绕世界 y 轴；相机在 -y 方向时正值使盘面在画面上左低右高。注意滚转固定在世界系，环绕过程中画面倾角会漂移（起始 ρ° → 环绕 90° 时变为开口角变化 → 180° 时反向） | 0 |
| ① 场景·结构 | `--v2_reverse_rotation` | 反转吸积盘旋转方向（平流结构与多普勒频移整体反向） | 关闭 |
| ① 场景·结构 | `--v2_orbit_seconds` | 视频模式：内缘开普勒轨道一圈对应的视频秒数（盘的转速） | 16.0 |
| ② 场景·辐射 | `--v2_temp_density_coupling` | 密度–温度耦合 β（0–1）：温度 × clamp(1 + β·(c/⟨c⟩ − 1), 0.7, 1.3)，浓处更热。盘光学厚时亮度只取决于温度（基尔霍夫：源函数 = B(T)），团块明暗主要靠它；局部耗散 ∝ 柱密度、T_eff ∝ Σ^{1/4} 推得物理值 0.25 | 0.05 |
| ② 场景·辐射 | `--v2_lum_temp_scale` | 亮度温度倍率 s：亮度按 Y(s·T)/Y(s·T_peak) 计算，色度不变；1 = 物理（与参考实现对齐），>1 时外盘更亮、纹理对比度略降（艺术夸张），多普勒明暗不对称自动补偿 | 1.25 |
| ② 场景·辐射 | `--v2_doppler_lum` | 多普勒亮度强度 p（≥ 0）：逼近侧变亮、远离侧变暗的程度，亮度按 Y(g^p·T) 计算。1 = 物理（左右亮度比很大）；0 = 无多普勒明暗 | 0.25 |
| ② 场景·辐射 | `--v2_doppler_color` | 多普勒颜色强度 q（≥ 0）：逼近侧偏白、远离侧偏红的程度，色度按 χ(T·g^q) 计算，色度温度封顶到白平衡色温（最亮处止于白色，不偏蓝）。1 = 物理；0 = 无多普勒变色；应与 `--v2_doppler_lum` 同步调 | 0.75 |
| ② 场景·辐射 | `--v2_color_floor` | 颜色温度下限 T_floor（K，≥ 0）：色度温度低于它时取 T_floor（硬截断），冷区固定为暗金、只靠亮度变暗，颜色序列为 黑 → 暗金 → 金 → 白。只影响颜色不影响亮度；建议 2500–3000；0 = 不设下限 | 0 |
| ② 场景·辐射 | `--v2_temp_turb` | 小尺度温度湍流强度 σ_T（0–0.5）：温度乘对数正态起伏 exp(σ_l·n − 2σ_l²·V)（平均热辐射通量 ⟨T⁴⟩ 守恒），n 为主云剪切级联向更小尺度延伸 2 个八度的起伏，比像素小的尺度自动淡出、经过强透镜的光线不加；局部强度 σ_l 随主云浓淡成片变化（σ_T 为全盘均方根），噪声坐标经过扭曲、不呈行列排布。只改温度、不改密度与遮挡，使贴盘面视角的近处云层出现明暗与冷暖细节，远景基本不变；视频贴盘帧耗时约 2.4 倍。0.1 = 温度起伏约 10%。见 [`docs/plans/v2_temperature_turbulence_plan.md`](docs/plans/v2_temperature_turbulence_plan.md) | 0（关闭） |
| ③ 场景·背景 | `--v2_sky_gain` | 天空亮度系数（sRGB 解码为线性光后，曝光之后叠加；不影响盘曝光；0 = 黑天空） | 0.5 |
| ④ 相机 | `--v2_camera_roll` | 相机滚转角（度），相机绕自身光轴旋转，整个画面（盘与星空）一起倾斜；在画面系中生效，环绕过程中倾角恒定；+12.5 = 画面左低右高，负值反向 | 0 |
| ④ 相机 | `--v2_camera_path` | 运镜路径文件（JSON）。视频模式下每帧的相机位置、朝向、视野、滚转由路径给出，黑洞落在路径指定的画面位置；曝光改为沿路径测光 + 平滑，首尾淡入淡出；帧数 = round(路径时长 × `--fps`)。此时 `--pov`、`--fov`、`--v2_camera_roll` 不生效，不能与 `--orbit`、`--interactive` 同用。示例路径为 `scenes/v2_arts/interstellar_skim.json`。格式见 [`docs/plans/v2_camera_path_plan.md`](docs/plans/v2_camera_path_plan.md) §5.1 | 不使用 |
| ④ 相机 | `--v2_camera_path_time` | 单帧模式：渲染运镜路径在视频时刻 T（秒，0 ≤ T ≤ 路径时长）的画面，用于检查构图；曝光为单帧自动曝光 + `--v2_exposure_ev`；需配合 `--v2_camera_path` | 不使用 |
| ⑤ 镜头 | `--v2_lens_glare` | 镜头眩光强度 ε（0–1）：镜头把每个点约 ε 的能量散射成平滑长尾（辉光）；能量守恒、作用于全部光、无阈值，辉光只在暗处（天空、黑洞阴影）显著，盘面不会被点亮。0 = 理想镜头；好镜头约 0.02，柔光镜约 0.2–0.5；越大辉光越明显，全画面对比度按 (1 − ε) 下降 | 0.5 |
| ⑥ 传感器与 ISP | `--v2_exposure_ev` | 曝光补偿（档）：自动曝光（盘区亮度 p99.9 → 0.9，视频首帧锁定）之后再乘 2^EV。0 = 无过曝、最亮处只到浅金、缺少发光感；+1.5 = 内盘约 2% 的像素烧白、阴影内辉光约翻倍；负值更暗 | +1.5 |
| ⑥ 传感器与 ISP | `--v2_white_balance` | 白平衡色温（K）：色温为该值的黑体显示为白色；盘面主体约 3000–5000 K。调低更白更冷（4000 K 偏惨白）、调高更暖更黄（参考实现 5000 K 偏暖黄）；色温封顶自动跟随 | 4500 |
| ⑥ 传感器与 ISP | `--v2_film_response` | 胶片响应 m（0–1）：色调映射在保色度 ACES 与逐通道 ACES 之间线性混合。0 = 保色度（只压亮度，最亮处止于浅金、不烧白）；1 = 逐通道（胶片各层 / 传感器各通道独立饱和，同一颜色越亮越淡、最亮处趋白，暗部饱和度略升）。见 [`docs/imaging_model.md`](docs/imaging_model.md) §6 | 0 |
| 求解精度 | `--v2_opt` | 优化级别：0 基准实现；1 精确优化（输出与 0 一致）；2 盘内步长 ×2（视觉等价）；3 盘内步长 ×3（近似预览）。见 `docs/plans/v2_performance_plan.md` | 单帧 1，视频 2 |
| 求解精度 | `--v2_supersample` | 超采样倍率 N：每像素 N² 条光线取平均，用于抗锯齿 | 单帧 2，视频 1 |

V2 同时使用通用参数 `--pov`、`--fov`、`--ar1`、`--ar2`、`--disk_tilt`、`--r_max`（V2 下限 50）、
`--texture`、`--video`、`--orbit`、`--orbit_degrees`、`--n_frames`、`--fps`。
其余物理模型与视觉参数取 `src.v2.params.DiskV2VolumeParams` 默认值（每个字段的含义、公式与取值范围见其
docstring），不开放 CLI。其中主云剪切级联的 `shear_*`（长宽比 10 → 2.5、拖尾倾角系数 0.01）、
`core_oct_gain`（逐八度增益 0.6，小尺度条纹较弱）、大气的 `atm_extent` / `atm_cov_*` / `atm_fine_fz`、
散射的 `abs_scatter_ratio` / `scatter_j` 说明见 [`docs/plans/v2_unified_gas_plan.md`](docs/plans/v2_unified_gas_plan.md) §5。

**V2 使用注意**：

- 必须显式传 `--ar1 3 --ar2 30`：`--ar1/--ar2` 的默认值（2 / 15）是 V1 的；V2 会把 `r_in < 3` 钳制到 ISCO 并 warning。
- 仅支持 `--device gpu`（CPU 单帧也要数分钟）；不支持 `--interactive`。
- 辉光由镜头 PSF 产生（`--v2_lens_glare`）：能量守恒、无阈值，只在暗处显著，不会点亮盘面。"发光感"主要靠曝光补偿让最亮处烧白（`--v2_exposure_ev`）。参考实现的阈值 bloom + 镶边保留为渲染器参数 `lens_model="legacy"`（会额外加光，仅作对照），原因与实测见 `docs/imaging_model.md` §9。
- 视频断点续传：每 240 帧写一个分段（`output/` 下 `.v2seg_*` 目录），中断后用同一条命令加 `--resume` 继续，最多重渲 1 个分段；参数变化时自动从头开始；完成后无损拼接并删除分段。
- 曝光自动：盘区亮度 p99.9 映射到 0.9 再乘 2^EV（默认 +1.5 档；只看盘，天空不参与）；视频首帧计算后锁定。
- 天空：`--texture` 读等距柱状 PNG，不传则程序生成星空；双线性采样，星点参与 bloom。
- 速度（M 系列 GPU 实测，默认参数，720p、每像素 1 条光线，只计光线积分）：级别 0 4.5 s，级别 1 4.7 s，级别 2 2.7 s，级别 3 2.0 s（统一气体模型比旧烟雾模型快约 40%；级别 1 在统一模型下已无提速，只保证与级别 0 一致）；盘内视角（`--pov 0 -22 0.3`）级别 2 约 2.5 s。720p 视频（级别 2，含天空、后处理与编码）约 3.4 s/帧。超采样倍率 N 耗时约为 N² 倍。

## 物理模型

采用笛卡尔等效形式的光线方程：

```
d²x/dλ² = -1.5 · L² · x / r⁵
```

其中 L² 为角动量平方（守恒量），使用 4 阶 RK4 积分器求解。

## 参考

- [JaeHyunLee94/BlackHoleRendering](https://github.com/JaeHyunLee94/BlackHoleRendering)
- [rantonels/starless](https://github.com/rantonels/starless)
- [flannelhead/blackstar](https://github.com/flannelhead/blackstar)
