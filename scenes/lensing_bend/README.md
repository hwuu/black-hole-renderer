# 场景：透镜转弯处特写（lensing_bend）

一段 30 s 的长焦特写：相机在 15 r_s 处几乎贴着盘面平视（仰角约 1.5°），镜头偏向黑洞右侧，拍逼近侧的透镜转弯处——
经引力透镜翻到黑洞上方的远端盘像瀑布一样从画面顶部落下，在转弯处汇入近侧盘面，再向右流出。
黑洞本身在画面左侧之外，只露出光子环的一条细线。相机以 0.1°/s 极慢地逆着盘的旋转方向环绕，画面里的
运动几乎全部来自盘自身的转动。吸积盘为 V2 统一气体模型：开启分形温度湍流（瀑布中也保留，不按透镜淡出），
内边界保留一点力矩，使光子环前不再出现冷而不透明的暗弧；色调为明亮的橙金色。

| 文件 | 内容 |
|------|------|
| `lensing_bend.json` | 运镜路径：2 个关键帧（只差方位角）、节奏与淡入淡出参数（字段说明见 [`docs/plans/v2_camera_path_plan.md`](../../docs/plans/v2_camera_path_plan.md) §5.1） |
| `README.md` | 本说明：镜头、渲染命令、参数取值依据与注意事项 |

## 镜头

| 项目 | 取值 |
|------|------|
| 相机位置 | r = 15、z = 0.4（r_s），仰角约 1.5°；方位角从 −91.5° 匀速转到 −88.5° |
| 相机速度 | 0.1°/s（0.03 r_s/s），全程不变；逆着盘的旋转方向（相对气体约 4°/s） |
| 视野与姿态 | 竖直视野 9°，滚转 −2° |
| 黑洞位置 | 画面 (−0.571, 0.650)：横向在画面左边缘之外 0.57 个画面宽，纵向略低于中线（镜头略上仰，多拍瀑布） |
| 淡入淡出 | 开头 2 s 淡入，结尾 3 s 淡出 |
| 曝光 | 渲染前沿路径测光、平滑，叠加曝光补偿 +2 档（测光后全程 +1.7 … +2.3 档） |

黑洞所在距离上画面约宽 4.2 r_s、高 2.4 r_s，1080p 下每像素约 0.002 r_s。

## 渲染命令

以下命令在仓库根目录执行。天空纹理 `-t` 为本机路径，可换成任意等距柱状星空图；不传则程序生成星空。

```bash
# 正片：1080p / 60 fps，1800 帧，优化级别 2（视频默认）
python render.py --disk_model v2 --video \
    --v2_camera_path scenes/lensing_bend/lensing_bend.json \
    --ar1 3 --ar2 30 --v2_reverse_rotation --v2_orbit_seconds 8 \
    --v2_core_contrast 2.0 --v2_core_floor 0 --v2_isco_stress 0.005 --v2_plunge_width 0.1 \
    --v2_temp_density_coupling 0.25 --v2_lum_temp_scale 1.0 \
    --v2_temp_turb 0.134 --v2_temp_turb_coarse 3 --v2_temp_turb_gain 1 \
    --v2_temp_turb_clamp_px 1.5 --v2_temp_turb_lens_deg 360 \
    --v2_lens_glare 0.25 --v2_exposure_ev 2 \
    --v2_white_balance 25000 --v2_film_response 0.3 --v2_saturation 1.6 \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg \
    -r fhd --fps 60 --device gpu -o output/scenes/lensing_bend/lensing_bend_1080p60.mp4

# 中断后续传：同一条命令末尾加 --resume（每 240 帧一个分段，最多重渲 1 段）

# 单帧：检查某一时刻的画面（例如 15 s）；看细节须用 1080p
python render.py --disk_model v2 \
    --v2_camera_path scenes/lensing_bend/lensing_bend.json --v2_camera_path_time 15 \
    --ar1 3 --ar2 30 --v2_reverse_rotation --v2_orbit_seconds 8 \
    --v2_core_contrast 2.0 --v2_core_floor 0 --v2_isco_stress 0.005 --v2_plunge_width 0.1 \
    --v2_temp_density_coupling 0.25 --v2_lum_temp_scale 1.0 \
    --v2_temp_turb 0.134 --v2_temp_turb_coarse 3 --v2_temp_turb_gain 1 \
    --v2_temp_turb_clamp_px 1.5 --v2_temp_turb_lens_deg 360 \
    --v2_lens_glare 0.25 --v2_exposure_ev 2 \
    --v2_white_balance 25000 --v2_film_response 0.3 --v2_saturation 1.6 \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg \
    -r fhd --device gpu -o output/scenes/lensing_bend/lensing_bend_t15.png
```

- `--ar1 3 --ar2 30` 必须显式给出（`--ar1` / `--ar2` 的默认值属于 V1）；`--v2_isco_stress > 0` 要求 `--ar1 3`（内缘 = ISCO）。
- `--v2_reverse_rotation`：盘顺时针旋转（从 +z 看），画面右侧为逼近侧（与《星际穿越》一致），相机逆着盘的旋转方向环绕。
- `--v2_orbit_seconds 8`：内缘转一圈 8 视频秒（默认 16），转弯处（r ≈ 3–8）的气流约 10–45°/s。
- 运镜模式下帧数 = round(30 × `--fps`)，不需要传 `--n_frames`；`--pov`、`--fov`、`--v2_camera_roll` 不生效。
- 不提供 360p 小样命令：温度湍流按像素足迹淡出，360p 下几乎看不到，小样只能看构图与流动方向。

## 参数的取值依据

特写只拍到最内盘与经透镜的远端盘，画面几乎被亮盘铺满；默认参数为远景调定，在这里偏白、偏平。

| 层 | 参数 | 取值（默认） | 原因 |
|----|------|------|------|
| 场景·结构 | `--v2_core_contrast` | 2.0（0.8） | 主云明暗起伏加强，瀑布与近处盘面的团块更分明 |
| 场景·结构 | `--v2_core_floor` | 0（0.15） | 稀处可完全透空，斜看的瀑布里团块之间透出缝 |
| 场景·结构 | `--v2_isco_stress` / `--v2_plunge_width` | 0.005 / 0.1（0 / 0.1） | 零力矩时内缘又冷又不透明，掠射时在光子环前形成一道暗弧；β = 0.005 使内缘温度回到峰值的 0.8 倍，坠落气体在阴影前几乎看不到。原理见 [`docs/design_ad_v2.md`](../../docs/design_ad_v2.md) §3.2「内边界与坠落区」 |
| 场景·辐射 | `--v2_temp_density_coupling` | 0.25（0.05） | 取物理值：盘光学厚时亮度只取决于温度，团块明暗主要靠浓处更热 |
| 场景·辐射 | `--v2_lum_temp_scale` | 1.0（1.25） | 取物理值：特写里外盘不在画面中，不需要提亮外盘 |
| 场景·辐射 | `--v2_temp_turb` 等 | 0.134 / 粗尺度 3 / 等幅（关闭） | 分形温度湍流，与 interstellar_skim 相同的配置，近处盘面出现大小层次的冷暖云纹 |
| 求解精度 | `--v2_temp_turb_clamp_px` | 1.5（3） | 细节更多（interstellar_skim 实测 1 与 1.5 的闪烁相同） |
| 求解精度 | `--v2_temp_turb_lens_deg` | 360（10） | 瀑布是强透镜像，默认会淡出全部温度湍流、只剩大尺度结构；实测强透镜处足迹误差只有 13–35%，故不淡出。代价见下文 |
| 镜头 | `--v2_lens_glare` | 0.25（0.5） | 满屏亮盘时眩光均匀铺满画面、抹平局部明暗；0.5 → 0.2 时实测盘面细节对比度约提高 1.6 倍 |
| 传感器 | `--v2_exposure_ev` | +2（+1.5） | 配合饱和度提高后画面偏暗，补回亮度；最亮的团块烧白 |
| ISP | `--v2_white_balance` | 25000 K（4500 K） | 色度温度封顶到白平衡色温，默认下转弯处大片被截成白色；提高白平衡后盘面偏暖 |
| ISP | `--v2_film_response` | 0.3（0） | 少量逐通道响应，最亮处趋白、有发光感 |
| ISP | `--v2_saturation` | 1.6（1） | 白平衡 12000 K 以上颜色几乎不再变暖，ACES 压高光时色度又变淡，画面停在浅金；放大色度后为明亮的橙金色（火焰色）。原理见 [`docs/imaging_model.md`](../../docs/imaging_model.md) §6 |

## 注意事项

- **瀑布中的细纹理在视频里会闪烁**：瀑布在画面上流动约 20 px/帧（60 fps），温度湍流的细纹理在时间方向欠采样，
  按画面位移补偿后的相邻帧残差为 8.5%（透镜淡出角取默认 10° 时为 4.1%）。这不是空间锯齿（超采样倍率 2 与 1 的单帧差
  只有 0.8%），快门方向的频率钳制也只能降到 7.7%。接受不了时把 `--v2_temp_turb_lens_deg` 改回 10，瀑布会变平滑。
  实测见 [`docs/plans/v2_temperature_turbulence_plan.md`](../../docs/plans/v2_temperature_turbulence_plan.md) §10。
- **温度湍流只在高分辨率下可见**：它的尺度比主云更细，格宽小于 K 个像素足迹时按频率钳制淡出；4K 下细节更多。
- **黑洞在画面外**：`subject_uv` 横向为 −0.571，越出 [0, 1]（路径文件允许 [−1, 2]）；联系表脚本的黑洞标记会画在画面之外。
- **静止观者近似**：每一帧按位于该处的静止观者成像，不计相机速度带来的光行差与频移；相机速度不到 0.001 c。
- **耗时**：本机（M5）1080p 视频档实测约 82 s/帧（温度湍流不按透镜淡出，满屏都要算温度湍流），1800 帧约 41 h，
  只适合渲几秒的验证片段；远端 4090 未实测，按其余场景约 1/7 的耗时比估计约 12 s/帧、全片约 6 h。
- **长时间渲染放到远端**：本机只做单帧与几秒的短片段验证；远程渲染的步骤见仓库根目录 `AGENTS.md` 的"远程渲染"一节。
