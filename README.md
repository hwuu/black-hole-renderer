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
- **Disk V2（体积吸积盘）**：3D 体积密度场 + 发射-吸收积分、刚体环平流（视频不卷绕）、Page–Thorne 温度、严格频移、物理后处理链。详见 [`docs/design_ad_v2.md`](docs/design_ad_v2.md) 与 [`docs/plans/v2_volumetric_video_plan.md`](docs/plans/v2_volumetric_video_plan.md)。

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

V2 用 3D 体积密度场（SS 外区结构 + Page–Thorne 温度 + 灰大气 + 刚体环平流 +
乘性级联噪声 + 烟雾 + 尘埃）+ 体积发射-吸收积分 + 物理后处理链
（CIE 黑体 + Y(g·T) 亮度 + von Kries 白平衡 + 高光 bloom + 色散 + 保色度 ACES）。
设计见 [`docs/design_ad_v2.md`](docs/design_ad_v2.md)，参数三层定稿值见
[`docs/plans/v2_volumetric_video_plan.md`](docs/plans/v2_volumetric_video_plan.md) §2。

```bash
# 单帧（预设 M 构图；1080p 建议 --v2_ss 2）
python render.py --disk_model v2 --pov 0 -39.7 4.87 --fov 38 \
                 --ar1 3 --ar2 30 -r fhd --v2_ss 2 --device gpu -o output/v2.png

# 视频（内缘一圈 16 s，可加 --orbit 环绕）
python render.py --disk_model v2 --video --pov 0 -39.7 4.87 --fov 38 \
                 --ar1 3 --ar2 30 -r hd --device gpu --n_frames 240 --fps 24 -o output/v2.mp4
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
| `--ar2` | 吸积盘外半径 | 3.5 rs |
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

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--disk_model` | 吸积盘模型: `v1` / `v2` | v1 |
| `--v2_ss` | 体积模型超采样倍率（每轴），2 = 每像素 4 条光线 | 1 |
| `--v2_orbit_seconds` | 视频模式：内缘开普勒轨道一圈对应的视频秒数 | 16.0 |
| `--v2_sky_gain` | 天空亮度系数（sRGB 解码为线性光后，曝光之后叠加；不影响盘曝光；0 = 黑天空） | 0.5 |

V2 同时使用通用参数 `--pov`、`--fov`、`--ar1`、`--ar2`、`--disk_tilt`、`--r_max`（V2 下限 50）、
`--texture`、`--video`、`--orbit`、`--orbit_degrees`、`--n_frames`、`--fps`。
物理模型与视觉参数取预设 M（`src.v2.params.DiskV2VolumeParams` 默认值），暂不开放 CLI。

**V2 使用注意**：

- 必须显式传 `--ar1 3 --ar2 30`：`--ar1/--ar2` 的默认值（2 / 15）是 V1 的；V2 会把 `r_in < 3` 钳制到 ISCO 并 warning。
- 仅支持 `--device gpu`（CPU 单帧也要数分钟）；不支持 `--interactive`。
- 曝光自动：盘区亮度 p99.9 映射到 0.9（只看盘，天空不参与）；视频首帧计算后锁定。
- 天空：`--texture` 读等距柱状 PNG，不传则程序生成星空；双线性采样，星点参与 bloom。
- 速度（M 系列 GPU 实测，稳态每帧）：960×540 ≈ 4.8 s；1080p ≈ 15.8 s；1080p `--v2_ss 2` ≈ 52 s。
- 与参考实现对比验收：`python scripts/compare_v2_proto.py`。

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
