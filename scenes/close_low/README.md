# 场景：贴着云顶环绕（close_low）

一段 60 s 的匀速环绕镜头：相机在 r = 22、z = 0.45 处贴着吸积盘的云顶，逆着盘的旋转方向绕黑洞飞行。
黑洞在画面中线偏下，近处是被内盘照亮的云层，远处是盘面"地平线"与经透镜翻到上方的远端盘。
吸积盘为 V2 统一气体模型，开启小尺度温度湍流使近处云层出现细节。

| 文件 | 内容 |
|------|------|
| `close_low.json` | 运镜路径：3 个关键帧、节奏、测光与淡入淡出参数（字段说明见 [`docs/plans/v2_camera_path_plan.md`](../../docs/plans/v2_camera_path_plan.md) §5.1） |
| `README.md` | 本说明：镜头、渲染命令、耗时与注意事项 |

## 镜头

| 项目 | 取值 |
|------|------|
| 相机位置 | r = 22、z = 0.45（r_s），方位角从 −129.1° 匀速转到 −46.3° |
| 相机速度 | 1.38°/s（0.53 r_s/s），全程不变，无加减速；相对气体 3.65°/s |
| 视野与姿态 | 竖直视野 28°，滚转 10°，黑洞固定在画面 (0.50, 0.62)（横向居中、纵向偏下） |
| 淡入淡出 | 开头 2 s 淡入，结尾 3 s 淡出 |
| 曝光 | 渲染前沿路径测光、平滑，叠加曝光补偿 +1.5 档 |

镜头起点取自 v2_arts 场景约 40 s 处的相机位置与方向（半径与高度略作调整），视野减半（焦距加倍），改为保持高度的匀速圆周运动。

## 渲染命令

以下命令在仓库根目录执行。天空纹理 `-t` 为本机路径，可换成任意等距柱状星空图；不传则程序生成星空。

```bash
# 正片：1080p / 60 fps，3600 帧，优化级别 2（视频默认）
python render.py --disk_model v2 --video \
    --v2_camera_path scenes/close_low/close_low.json \
    --ar1 3 --ar2 30 --v2_reverse_rotation --v2_orbit_seconds 8 --v2_temp_turb 0.1 \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg \
    -r fhd --fps 60 --device gpu -o output/close_low/close_low_1080p60.mp4

# 中断后续传：同一条命令末尾加 --resume（每 240 帧一个分段，最多重渲 1 段）

# 小样：360p / 15 fps
python render.py --disk_model v2 --video \
    --v2_camera_path scenes/close_low/close_low.json \
    --ar1 3 --ar2 30 --v2_reverse_rotation --v2_orbit_seconds 8 --v2_temp_turb 0.1 \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg \
    -r sd --fps 15 --device gpu -o output/close_low/close_low_360p15.mp4

# 单帧：检查某一时刻的构图（例如 30 s）
python render.py --disk_model v2 \
    --v2_camera_path scenes/close_low/close_low.json --v2_camera_path_time 30 \
    --ar1 3 --ar2 30 --v2_reverse_rotation --v2_orbit_seconds 8 --v2_temp_turb 0.1 \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg \
    -r fhd --device gpu -o output/close_low/close_low_t30.png
```

- `--ar1 3 --ar2 30` 必须显式给出（`--ar1` / `--ar2` 的默认值属于 V1）。
- `--v2_reverse_rotation`：盘顺时针旋转（从 +z 看），相机逆着盘的旋转方向飞行，云层始终相对相机流动。
- `--v2_orbit_seconds 8`：内缘转一圈 8 视频秒（默认 16）；r = 22 处云层约 2.3°/s。
- `--v2_temp_turb 0.1`：小尺度温度湍流，强度 σ_T = 0.1（全部尺度可见时 ln T 起伏的标准差，即温度起伏约 10%）。关闭时近处云层是一片均匀的雾；原理、实测与限制见
  [`docs/plans/v2_temperature_turbulence_plan.md`](../../docs/plans/v2_temperature_turbulence_plan.md)。
- 运镜模式下帧数 = round(60 × `--fps`)，不需要传 `--n_frames`；`--pov`、`--fov`、`--v2_camera_roll` 不生效。

## 耗时

RTX 4090 实测（AutoDL；只计光线积分，后处理与编码按 8 线程与积分并行）：

| 输出 | 帧数 | 耗时 |
|------|------|------|
| 1080p 视频档（优化级别 2、超采样倍率 1） | 1 | 2.58–2.77 s/帧（关闭温度湍流时 1.00–1.16 s/帧） |
| 1080p 单帧（优化级别 1、超采样倍率 2） | 1 | 16.6 s（关闭时 6.5 s） |

1080p / 60 fps 正片 3600 帧按 2.7 s/帧估算约 2.7 h（未实测整片）。M 系列 GPU 约慢 6 倍，正片放到远端渲染。

## 注意事项

- **静止观者近似**：每一帧按位于该处的静止观者成像，不计相机速度带来的光行差与频移；全片最大局部速度约 0.09 c。
- **长时间渲染放到远端**：本机只做单帧与几秒的短片段验证；远程渲染的步骤见仓库根目录 `AGENTS.md` 的"远程渲染"一节。
