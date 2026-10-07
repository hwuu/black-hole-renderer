# 场景：掠过吸积盘（v2_arts）

一段 120 s 的一镜到底长镜头：从远处接近史瓦西黑洞，长时间低空掠过盘面云顶，斜穿盘面，在盘的下方横扫，
沿一条大弧绕过盘外缘回到上方，最后拉远。掠云期间镜头摇向前进方向，黑洞退到画面左缘。
吸积盘为 V2 统一气体模型，开启分形温度湍流（大片冷暖斑块由更细的起伏逐级细分），色调为橙金色的"火焰"观感。

| 文件 | 内容 |
|------|------|
| `interstellar_skim.json` | 运镜路径：18 个关键帧、1 段摇镜、节奏、测光与淡入淡出参数（字段说明见 [`docs/plans/v2_camera_path_plan.md`](../../docs/plans/v2_camera_path_plan.md) §5.1） |
| `README.md` | 本说明：分镜、渲染命令、耗时与注意事项 |

## 分镜

| 镜头 | 时间 | 时长 | 画面 |
|------|------|------|------|
| S1 建立 | 0–12.4 s | 12.4 s | 淡入；远景中的黑洞阴影、光子环与经透镜翻到上方的远端盘，黑洞在右三分线 |
| S2 接近 | 12.4–29.4 s | 17.0 s | 弧线下降，盘面"地平线"上升，光子环逐渐变大 |
| S3 掠云 | 29.4–43.2 s | 13.8 s | 在 r ≈ 22–23 处沿云顶低空掠过，高度从 0.67 缓降到 0.37 r_s；30–38 s 镜头摇向前进方向，黑洞退到画面左缘 |
| S4 入云 | 43.2–57.8 s | 14.6 s | 继续缓降到 0.11 r_s，前方云层迎面而来，光子环在左缘透雾发光 |
| S5 斜穿 | 57.8–70.2 s | 12.4 s | 以平缓坡度斜穿盘面（61.6 s 过中面），约 5–6 s 满屏金色雾；66–74 s 镜头摇回黑洞 |
| S6 盘底 | 70.2–83.3 s | 13.1 s | 盘成为头顶的"天花板"，黑洞露出下半环；向盘外缘横扫 |
| S7 绕边回升 | 83.3–109.7 s | 26.4 s | 沿盘外缘外侧画一条大弧缓缓回升，约 100.7 s 平视经过一线盘面与完整光环 |
| S8 拉远 | 109.7–120 s | 10.3 s | 盘面全貌，黑洞在左三分线，与开场镜像；淡出 |

- 节奏：全程不停顿，画面变化速度恒定（离黑洞或盘面越近，相机移动越慢），只在开头 6 s 缓慢加速、结尾 8 s 缓慢减速。
  贴盘、斜穿与绕回各段的时长来自更长的路线，不靠局部减速。
- 视野：全程固定 50°，不变焦（焦距不变）。
- 摇镜：路径文件的 `pans` 块。30 s 起用 8 s 把黑洞的画面横坐标从路径值（约 0.56）平滑移到 0.12，
  光轴随之转向前进方向（约 32°）；停留到 66 s，再用 8 s 摇回。摇镜角速度两端为 0，光子环全程留在画面内
  （左缘余量约 0.7°）。摇镜不改变相机位置与节奏。
- 贴近盘面核心（|z| < 0.2 r_s）的时间约 12.9 s，|z| < 0.1 r_s 约 6.1 s。
- 曝光：渲染前沿路径测光，平滑后按 0.6 的比例补偿亮度变化，叠加在曝光补偿 +3 档之上；进云变亮、盘底变暗的
  明暗变化得以保留。
- 色调：白平衡 10000 K、胶片响应 1（逐通道 ACES）、曝光补偿 +3 档。盘面主体约 3000–6000 K，低于白平衡色温，
  呈橙金色；最亮处沿"通往白色的路径"变淡。原型对比见 `docs/plans/v2_temperature_turbulence_plan.md` §9.4。
- 盘的旋转方向需配合 `--v2_reverse_rotation`：相机全程逆着盘的旋转方向飞行，盘面纹理与云层始终相对相机流动。
- 盘的转速：`--v2_orbit_seconds 8`，内缘转一圈 8 视频秒（默认 16），全片内缘转 15 圈；掠云处（r = 22）云层约
  2.3°/s，相对相机约 3.5°/s。
- 温度湍流：`--v2_temp_turb 0.134 --v2_temp_turb_coarse 3 --v2_temp_turb_gain 1 --v2_temp_turb_clamp_px 1`。
  温度起伏从主云第 2 级尺度一直延伸到最细一级（共 5 个八度，各八度等幅，每个约 6%），见
  [`docs/plans/v2_temperature_turbulence_plan.md`](../../docs/plans/v2_temperature_turbulence_plan.md) §9。

## 渲染命令

以下命令在仓库根目录执行。天空纹理 `-t` 为本机路径，可换成任意等距柱状星空图；不传则程序生成星空。

```bash
# 正片：1080p / 60 fps，7200 帧，优化级别 2（视频默认）
python render.py --disk_model v2 --video \
    --v2_camera_path scenes/v2_arts/interstellar_skim.json \
    --ar1 3 --ar2 30 --v2_reverse_rotation --v2_orbit_seconds 8 \
    --v2_temp_turb 0.134 --v2_temp_turb_coarse 3 --v2_temp_turb_gain 1 --v2_temp_turb_clamp_px 1 \
    --v2_white_balance 10000 --v2_film_response 1 --v2_exposure_ev 3 \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg \
    -r fhd --fps 60 --device gpu -o output/v2_arts/interstellar_skim_1080p60.mp4

# 中断后续传：同一条命令末尾加 --resume（每 240 帧一个分段，最多重渲 1 段）

# 小样：360p / 15 fps
python render.py --disk_model v2 --video \
    --v2_camera_path scenes/v2_arts/interstellar_skim.json \
    --ar1 3 --ar2 30 --v2_reverse_rotation --v2_orbit_seconds 8 \
    --v2_temp_turb 0.134 --v2_temp_turb_coarse 3 --v2_temp_turb_gain 1 --v2_temp_turb_clamp_px 1 \
    --v2_white_balance 10000 --v2_film_response 1 --v2_exposure_ev 3 \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg \
    -r sd --fps 15 --device gpu -o output/v2_arts/interstellar_skim_360p15.mp4

# 单帧：检查某一时刻的构图（例如 50 s 摇镜停留段）
python render.py --disk_model v2 \
    --v2_camera_path scenes/v2_arts/interstellar_skim.json --v2_camera_path_time 50 \
    --ar1 3 --ar2 30 --v2_reverse_rotation --v2_orbit_seconds 8 \
    --v2_temp_turb 0.134 --v2_temp_turb_coarse 3 --v2_temp_turb_gain 1 --v2_temp_turb_clamp_px 1 \
    --v2_white_balance 10000 --v2_film_response 1 --v2_exposure_ev 3 \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg \
    -r sd --device gpu -o output/v2_arts/interstellar_skim_t50.png

# 联系表：每个关键帧的经过时刻各一格，画三分线与黑洞目标位置
python scripts/contact_sheet.py scenes/v2_arts/interstellar_skim.json --v2_reverse_rotation --v2_orbit_seconds 8 \
    -t /Users/hwuu/TychoSkymapII.t5_8192x4096.jpg -o output/v2_arts/interstellar_skim_contact_sheet.png
```

- `--ar1 3 --ar2 30` 必须显式给出（`--ar1` / `--ar2` 的默认值属于 V1）。
- 运镜模式下帧数 = round(120 × `--fps`)，不需要传 `--n_frames`；`--pov`、`--fov`、`--v2_camera_roll` 不生效。
- 联系表脚本不读取温度湍流与色调参数，联系表只用于检查构图。
- 其余参数（盘结构、辐射、镜头等）取默认值；如需改动，在命令中追加对应的 `--v2_*` 参数即可，路径不受影响。

## 耗时

RTX 4090 实测（AutoDL）：

| 输出 | 帧数 | 耗时 |
|------|------|------|
| 1080p / 60 fps 正片 | 7200 | 预计约 7.5 h（分形温度湍流 5 个八度，约为旧版 2 个八度的 1.85 倍；旧版实测 4.1 h，平均 2.06 s/帧） |
| 360p / 15 fps 小样 | 1800 | 约 35 min（含测光；旧版 2 个八度约 19 min） |
| 测光（每次视频渲染前） | 241 张 128×72 | 约 50 s |

M 系列 GPU 约慢 6 倍，正片放到远端渲染。

## 注意事项

- **静止观者近似**：每一帧按位于该处的静止观者成像，不计相机速度带来的光行差与频移。全片最大局部速度
  约 0.54 c（开场推近段，约 4.8 s）。渲染前的路径报告会打印这一数值。
- **路径报告**：渲染前打印每个关键帧的经过时刻、位置、速度、顺行 / 逆行、相对气体角速度，以及两次穿越 z = 0 的
  时刻（61.6 s 穿过盘内；99.8 s 在盘外，无气体）。
- **开场与结尾的构图**：视野固定 50° 后，开场与结尾的黑洞比旧版（32–34°）小一圈，远景更空旷。
- **联系表的曝光**：联系表与单帧按单帧自动曝光成像，穿越盘面那一格偏亮；视频按测光曲线曝光。
- **长时间渲染放到远端**：本机只做单帧与几秒的短片段验证；远程渲染步骤见仓库根目录 `AGENTS.md` 的"远程渲染"一节。
