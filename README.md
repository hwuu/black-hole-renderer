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
- **Disk V2（v2.1 + 视觉恢复 experimental）**：有限厚度发射-吸收积分、预烘焙 visual atlas（V1 云雾 + spiral warp）、弱 clump 自遮挡、温度量纲化、palette + HDR 链路、g-factor 相对论修正。详见 [`docs/design_ad_v2.md`](docs/design_ad_v2.md)、[`docs/plans/v2_visual_recovery_plan.md`](docs/plans/v2_visual_recovery_plan.md) 与 [`docs/plans/realism_uplift_plan.md`](docs/plans/realism_uplift_plan.md)。

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

### Disk V2 模式（v2.1，视觉恢复 experimental）

V2 用有限厚度发射-吸收积分代替 V1 的零厚度倾斜平面；主视觉结构来自预烘焙 visual atlas（V1 云雾 + spiral warp）。
当前仅支持单帧渲染（视频和交互模式将在后续 Phase 接入）。

```bash
# 固定验收（Interstellar 风格参考图；ar1=2, ar2=15，远景相机以保留盘缘）
bash scripts/v2_visual_acceptance.sh

# 或手动渲染 acceptance
python render.py --disk_model v2 --texture output/black_sky.png \
  --pov 24 0 8 --fov 90 --ar1 2 --ar2 15 --disk_tilt 20 \
  -r hd --device gpu --v2_visual_preset interstellar \
  -o output/v2_acceptance_bloom.png

# V2 大半径 demo（推荐 GPU + r_out 50 + auto exposure）
python render.py --disk_model v2 --pov 30 0 10 --fov 90 \
                 --ar1 3 --ar2 50 --disk_tilt 20 \
                 -r hd --device gpu --v2_auto_exposure \
                 -o output/v2_demo.png

# 打印 HDR/LDR 诊断统计（不改变图像）
python render.py --disk_model v2 --ar1 3 --ar2 50 --device gpu \
                 --v2_print_stats -o output/v2_stats.png

# 关闭相对论 g-factor 看纯发射率
python render.py --disk_model v2 --v2_disable_g_factor \
                 --ar1 3 --ar2 50 --device gpu \
                 -o output/v2_no_g.png

# 启用 HDR 域 Bloom（实验性，参数留待视觉验收阶段重校）
python render.py --disk_model v2 \
                 --ar1 3 --ar2 50 --device gpu \
                 --v2_bloom_intensity 0.5 \
                 --v2_bloom_threshold 0.3 \
                 --v2_bloom_radius 4 \
                 -o output/v2_bloom.png

# 手动降低 HDR 发射（高级参数；常规优先用 --v2_auto_exposure）
python render.py --disk_model v2 \
                 --ar1 3 --ar2 50 --device gpu \
                 --v2_emission_scale 0.05 \
                 -o output/v2_exposure_check.png

# physical palette + 关闭 cinematic 增强
python render.py --disk_model v2 --v2_palette_mode physical \
                 --ar1 3 --ar2 50 --device gpu \
                 -o output/v2_physical.png
```

V2 参数说明见下方 "Disk V2 参数" 小节。

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

### Disk V2 参数（v2.1，仅 `--disk_model v2`）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--disk_model` | 吸积盘模型: `v1` / `v2` | v1 |
| `--v2_T_peak_K` | 中面温度峰值（K），决定颜色基调 | 1e7 |
| `--v2_clump_count` | 显式团块数量 | 400 |
| `--v2_palette_mode` | 调色模式: `physical` / `cinematic` | cinematic |
| `--v2_volume_samples` | 盘内体积积分步数 | 16 |
| `--v2_opacity_scale` | 盘体不透明度缩放 | 0.5 |
| `--v2_lum_power` | g-factor 亮度指数（Phase 5 严格物理 = 4） | 4.0 |
| `--v2_g_cap` | g-factor 上限 | 6.0 |
| `--v2_disable_g_factor` | 关闭相对论 g-factor 修正 | False |
| `--v2_r_max` | V2 路径的逃逸半径下限，None 表示沿用 `--r_max` | None |
| `--v2_bloom_intensity` | HDR 域 Bloom 强度，0 关闭 | 0.0 |
| `--v2_bloom_threshold` | Bloom 亮度阈值（HDR） | 1.0 |
| `--v2_bloom_radius` | Bloom 高斯模糊半径（像素） | 4.0 |
| `--v2_emission_scale` | HDR 发射率整体缩放（高级参数） | 1.0 |
| `--v2_auto_exposure` | 按 HDR 亮度分位数自动设置 white point | 关闭 |
| `--v2_white_point_percentile` | auto exposure 使用的 HDR 亮度分位数 | 99.0 |
| `--v2_print_stats` | 渲染后打印 HDR/LDR 诊断统计 | 关闭 |
| `--v2_seed` | 团块/atlas 随机种子 | 42 |
| `--v2_visual_preset` | 视觉预设：`interstellar`（auto exposure + cinematic + 弱 bloom） | 无 |
| `--v2_turbulence_strength` | visual atlas 云雾强度 | 0.35 |
| `--v2_spiral_warp_strength` | 径向 spiral warp 强度 | 1.8 |
| `--v2_alpha_clip_threshold` | atlas Alpha Clip 阈值 | 0.01 |
| `--v2_atlas_n_r` / `--v2_atlas_n_phi` | atlas 分辨率 | 512 / 1024 |
| `--v2_shear_strength` | 傅里叶剪切强度（默认 0，关闭） | 0 |
| `--v2_disable_visual_atlas` | 关闭 visual atlas，回退 F_shear 路径 | False |

**V2 使用注意**：

- **视觉验收**以用户参考图为准，须人工看图；`white_ratio` 仅作诊断。
- 验收构图使用 `--ar1 2 --ar2 15`；大半径 demo 推荐 `--ar2 50`。
- V2 强制 `--ar1 ≥ 3 r_s`（Schwarzschild ISCO），小于该值会自动钳制并 warning。
- V2 当前仅支持 `--device gpu`。CPU 路径在小图下也可能耗时数分钟，CLI 会直接拒绝。
- 推荐默认加 `--v2_auto_exposure`；`--v2_print_stats` 可查看 `white_ratio` / `hdr_p99` 等诊断。
- `--v2_emission_scale` 保留为高级手动曝光；常规用户优先用 auto exposure。
- V2 当前不支持 `--video` / `--interactive`，仅单帧 PNG 输出。

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
