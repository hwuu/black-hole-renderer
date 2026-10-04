# 场景：掠过吸积盘（v2_arts）

一段 87.5 s 的一镜到底长镜头：从远处接近史瓦西黑洞，掠过盘面云顶，穿过盘面，在盘的下方横扫，
绕过盘外缘回到上方，最后拉远。吸积盘为 V2 统一气体模型。

| 文件 | 内容 |
|------|------|
| `interstellar_skim.json` | 运镜路径：14 个关键帧、节奏、测光与淡入淡出参数（字段说明见 [`docs/plans/v2_camera_path_plan.md`](../../docs/plans/v2_camera_path_plan.md) §5.1） |
| `README.md` | 本说明：分镜、渲染命令、耗时与注意事项 |

## 分镜

| 镜头 | 时间 | 画面 |
|------|------|------|
| S1 建立 | 0–13.2 s | 淡入；远景中的黑洞阴影、光子环与经透镜翻到上方的远端盘，黑洞在右三分线 |
| S2 接近 | 13.2–31.5 s | 弧线下降，盘面"地平线"上升，光子环逐渐变大 |
| S3 掠云 | 31.5–38.6 s | 低空掠过被内盘照亮的云顶，抵达云缘 |
| S4 入云 | 38.6–48.5 s | 沿云最浓的方位下降，光子环透雾发光，黑洞移向画面中央 |
| S5 穿越 | 48.5–53.1 s | 穿过盘面，约 2–3 s 满屏暗金色雾 |
| S6 盘底 | 53.1–64.6 s | 盘成为头顶的"天花板"，黑洞露出下半环；向盘外缘横扫 |
| S7 绕边回升 | 64.6–81.1 s | 从盘外缘外侧回到上方，约 72.9 s 平视经过一线盘面与完整光环 |
| S8 拉远 | 81.1–87.5 s | 盘面全貌，黑洞在左三分线，与开场镜像；淡出 |

- 节奏：全程不停顿，中段画面变化速度恒定（离黑洞或盘面越近，相机移动越慢），开头 6 s 缓慢加速、结尾 8 s 缓慢减速。
- 曝光：渲染前沿路径测光，平滑后按 0.6 的比例补偿亮度变化，叠加在曝光补偿 +1.5 档之上；进云变亮、盘底变暗的
  明暗变化得以保留。
- 盘的旋转方向需配合 `--v2_reverse_rotation`：相机全程逆着盘的旋转方向飞行，盘面纹理与云层始终相对相机流动。

## 渲染命令

以下命令在仓库根目录执行。天空纹理 `-t` 为本机路径，可换成任意等距柱状星空图；不传则程序生成星空。

```bash
# 正片：1080p / 60 fps，5250 帧，优化级别 2（视频默认）
python render.py --disk_model v2 --video \
    --v2_camera_path scenes/v2_arts/interstellar_skim.json \
    --ar1 3 --ar2 30 --v2_reverse_rotation \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg \
    -r fhd --fps 60 --device gpu -o output/v2_arts/interstellar_skim_1080p60.mp4

# 中断后续传：同一条命令末尾加 --resume（每 240 帧一个分段，最多重渲 1 段）

# 小样：360p / 15 fps
python render.py --disk_model v2 --video \
    --v2_camera_path scenes/v2_arts/interstellar_skim.json \
    --ar1 3 --ar2 30 --v2_reverse_rotation \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg \
    -r sd --fps 15 --device gpu -o output/v2_arts/interstellar_skim_360p15.mp4

# 单帧：检查某一时刻的构图（例如 58.7 s 的盘底）
python render.py --disk_model v2 \
    --v2_camera_path scenes/v2_arts/interstellar_skim.json --v2_camera_path_time 58.7 \
    --ar1 3 --ar2 30 --v2_reverse_rotation \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg \
    -r sd --device gpu -o output/v2_arts/interstellar_skim_t58.png

# 联系表：每个关键帧的经过时刻各一格，画三分线与黑洞目标位置
python scripts/contact_sheet.py scenes/v2_arts/interstellar_skim.json --v2_reverse_rotation \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg -o output/v2_arts/interstellar_skim_contact_sheet.png
```

- `--ar1 3 --ar2 30` 必须显式给出（`--ar1` / `--ar2` 的默认值属于 V1）。
- 运镜模式下帧数 = round(87.5 × `--fps`)，不需要传 `--n_frames`；`--pov`、`--fov`、`--v2_camera_roll` 不生效。
- 其余参数（盘结构、辐射、镜头、白平衡等）取默认值；如需改动，在命令中追加对应的 `--v2_*` 参数即可，路径不受影响。

## 耗时（M 系列 GPU）

| 输出 | 帧数 | 耗时 |
|------|------|------|
| 1080p / 60 fps | 5250 | 约 9.3 h（约 6.4 s/帧，估算） |
| 360p / 15 fps | 1312 | 约 21 min（实测 0.97 s/帧） |
| 测光（每次视频渲染前） | 176 张 128×72 | 约 1.5–2 min |

## 注意事项

- **静止观者近似**：每一帧按位于该处的静止观者成像，不计相机速度带来的光行差与频移。开场推近段（约 4.8 s）
  的局部速度约 1.01 c，真实飞船无法这样飞行；画面不受影响。渲染前的路径报告会打印这一数值。
- **路径报告**：渲染前打印每个关键帧的经过时刻、位置、速度、顺行 / 逆行、相对气体角速度，以及两次穿越 z = 0 的
  时刻（48.6 s 穿过盘内；72.4 s 在盘外，无气体）。
- **联系表的曝光**：联系表与单帧按单帧自动曝光成像，穿越盘面那一格偏亮；视频按测光曲线曝光。
