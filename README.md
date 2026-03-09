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
