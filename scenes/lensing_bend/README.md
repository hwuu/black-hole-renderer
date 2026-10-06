# 场景：透镜转弯处特写（lensing_bend）

一段 30 s 的长焦特写：相机在约 40 r_s 外几乎平视吸积盘，镜头偏向黑洞右侧，拍逼近侧的透镜转弯处——
经引力透镜翻到黑洞上方的远端盘像瀑布一样从画面顶部落下，在转弯处汇入近侧盘面平带，再向右流出。
黑洞本身在画面左侧之外，只露出光子环的一条细线。相机以 0.2°/s 极慢地逆着盘的旋转方向环绕，画面里的
运动几乎全部来自盘自身的转动。吸积盘为 V2 统一气体模型，开启小尺度温度湍流使近处盘面出现云团细节。

| 文件 | 内容 |
|------|------|
| `lensing_bend.json` | 运镜路径：2 个关键帧（只差方位角）、节奏与淡入淡出参数（字段说明见 [`docs/plans/v2_camera_path_plan.md`](../../docs/plans/v2_camera_path_plan.md) §5.1） |
| `README.md` | 本说明：镜头、渲染命令、参数取值依据与注意事项 |

## 镜头

| 项目 | 取值 |
|------|------|
| 相机位置 | r = 39.9、z = 2.8（r_s），仰角约 4°；方位角从 −93° 匀速转到 −87° |
| 相机速度 | 0.2°/s（0.14 r_s/s），全程不变；逆着盘的旋转方向 |
| 视野与姿态 | 竖直视野 6°（预设 M 的 38° 约 6 倍焦距），滚转 0° |
| 黑洞位置 | 画面 (−0.30, 0.60)：横向在画面左边缘之外 0.3 个画面宽，纵向略低于中线（镜头略上仰，多拍瀑布） |
| 淡入淡出 | 开头 2 s 淡入，结尾 3 s 淡出 |
| 曝光 | 渲染前沿路径测光、平滑，叠加曝光补偿 +0.5 档 |

画面约宽 7.5 r_s、高 4.2 r_s，1080p 下每像素约 0.004 r_s。

## 渲染命令

以下命令在仓库根目录执行。天空纹理 `-t` 为本机路径，可换成任意等距柱状星空图；不传则程序生成星空。

```bash
# 正片：1080p / 60 fps，1800 帧，优化级别 2（视频默认）
python render.py --disk_model v2 --video \
    --v2_camera_path scenes/lensing_bend/lensing_bend.json \
    --ar1 3 --ar2 30 --v2_reverse_rotation --v2_orbit_seconds 8 --v2_temp_turb 0.05 \
    --v2_white_balance 12000 --v2_exposure_ev 0.5 --v2_lens_glare 0.2 \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg \
    -r fhd --fps 60 --device gpu -o output/lensing_bend/lensing_bend_1080p60.mp4

# 中断后续传：同一条命令末尾加 --resume（每 240 帧一个分段，最多重渲 1 段）

# 小样：360p / 15 fps（只看构图与流动方向；温度湍流在 360p 下完全淡出，见下文）
python render.py --disk_model v2 --video \
    --v2_camera_path scenes/lensing_bend/lensing_bend.json \
    --ar1 3 --ar2 30 --v2_reverse_rotation --v2_orbit_seconds 8 --v2_temp_turb 0.05 \
    --v2_white_balance 12000 --v2_exposure_ev 0.5 --v2_lens_glare 0.2 \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg \
    -r sd --fps 15 --device gpu -o output/lensing_bend/lensing_bend_360p15.mp4

# 单帧：检查某一时刻的画面（例如 15 s）；看细节须用 1080p
python render.py --disk_model v2 \
    --v2_camera_path scenes/lensing_bend/lensing_bend.json --v2_camera_path_time 15 \
    --ar1 3 --ar2 30 --v2_reverse_rotation --v2_orbit_seconds 8 --v2_temp_turb 0.05 \
    --v2_white_balance 12000 --v2_exposure_ev 0.5 --v2_lens_glare 0.2 \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg \
    -r fhd --device gpu -o output/lensing_bend/lensing_bend_t15.png
```

- `--ar1 3 --ar2 30` 必须显式给出（`--ar1` / `--ar2` 的默认值属于 V1）。
- `--v2_reverse_rotation`：盘顺时针旋转（从 +z 看），画面右侧为逼近侧（与《星际穿越》一致），相机逆着盘的旋转方向环绕。
- `--v2_orbit_seconds 8`：内缘转一圈 8 视频秒（默认 16），转弯处（r ≈ 3–8）的气流约 10–45°/s。
- 运镜模式下帧数 = round(30 × `--fps`)，不需要传 `--n_frames`；`--pov`、`--fov`、`--v2_camera_roll` 不生效。

## 成像参数的取值依据

特写只拍到最内盘与经透镜的远端盘，画面几乎被亮盘铺满，默认成像参数（为远景调定）在这里会偏白、偏平：

| 参数 | 取值（默认） | 原因 |
|------|------|------|
| `--v2_white_balance` | 12000 K（4500 K） | 颜色温度封顶到白平衡色温。转弯处的气体温度大多高于 4500 K，默认下被统一截成白色；提高白平衡后内盘落在封顶以下，呈橘金色 |
| `--v2_exposure_ev` | +0.5（+1.5） | 默认补偿使逼近侧大片烧白、丢失层次 |
| `--v2_lens_glare` | 0.2（0.5） | 眩光把每个点约 ε 的能量散成长尾；满屏亮盘时散射光均匀铺满画面、抹平局部明暗。0.5 → 0.2 时盘面细节对比度约提高 1.6 倍 |
| `--v2_temp_turb` | 0.05（关闭） | 近处盘面的颗粒云团；0.1 过密，0.05 较平衡 |
| 多普勒 | 默认（亮度 0.25、颜色 0.75） | 逼近侧略亮略白，整体仍为橘金色 |

## 注意事项

- **温度湍流只在高分辨率下可见**：它的尺度比主云更细，格宽小于 3 个像素足迹时按频率钳制完全淡出。本镜头相机在约 40 r_s 外，
  360p 下全部淡出（开与不开逐位相同），1080p 下只在近处盘面（画面下半）可见，4K 下更多。
  瀑布（经强透镜、累计偏折角 > 10° 的光线）按透镜权重淡出，不受温度湍流影响，其流线状纹理来自主云结构与透镜拉伸。
  原理见 [`docs/plans/v2_temperature_turbulence_plan.md`](../../docs/plans/v2_temperature_turbulence_plan.md)。
- **黑洞在画面外**：`subject_uv` 横向为 −0.3，越出 [0, 1]（路径文件允许 [−1, 2]）；联系表脚本的黑洞标记会画在画面之外。
- **静止观者近似**：每一帧按位于该处的静止观者成像，不计相机速度带来的光行差与频移；全片最大局部速度约 0.02 c。
- **耗时未实测**：本机（M5）1080p 视频档单帧命令总耗时约 25 s（关闭温度湍流，含 kernel 编译与天空加载）、约 38 s（温度湍流 0.1）；
  远端 4090 的逐帧耗时尚未测量，可参考 `close_low`（1.8–1.9 s/帧，开温度湍流）估算，1800 帧约 1 h。
- **长时间渲染放到远端**：本机只做单帧与几秒的短片段验证；远程渲染的步骤见仓库根目录 `AGENTS.md` 的"远程渲染"一节。
