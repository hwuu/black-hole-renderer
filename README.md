# Black Hole Renderer

基于广义相对论的史瓦西黑洞光线追踪渲染器。

## 特性

- **物理正确**：基于史瓦西度规的零测地线方程，正确模拟引力透镜
- **吸积盘渲染**：温度剖面纹理、多普勒效应、真实自转、FBM噪声絮状结构、螺旋臂
- **高性能**：Taichi 并行框架，1080p 渲染 < 1s
- **视频生成**：支持环绕视频、断点续传

## 安装

```bash
pip install numpy pillow taichi imageio
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

# 自定义轨道参数
python render.py --video --orbit --orbit_radius 10 --orbit_z 1 --n_frames 1800 --fps 30 -o output/demo.mp4

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
| `--ar1` | 吸积盘内半径 | 2.0 rs |
| `--ar2` | 吸积盘外半径 | 3.5 rs |
| `--output`, `-o` | 输出文件路径 | output/blackhole.png |
| `--framework`, `-f` | 渲染框架: numpy/taichi | taichi |
| `--device`, `-d` | Taichi 设备: cpu/gpu | cpu |

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
