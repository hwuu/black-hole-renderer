# 吸积盘旋转方案实验

本目录包含三个实验性程序，用于对比不同的吸积盘旋转方案。

## 实验方案

### 1. Baseline: 固定纹理 + 刚体旋转
- **文件**: `experiment_1_baseline.py`
- **方案**: 生成一次纹理，然后按开普勒速度旋转采样点
- **优点**: 快速，只需生成一次纹理
- **缺点**: 差分旋转导致失真

### 2. 参数化旋转纹理
- **文件**: `experiment_2_parametric.py`
- **方案**: 每帧重新生成纹理，在生成时考虑旋转
- **优点**: 物理正确，无累积失真
- **缺点**: 速度慢，每帧需要生成纹理

### 3. 关键帧插值
- **文件**: `experiment_3_keyframes.py`
- **方案**: 预计算 N 个关键帧，运行时插值
- **优点**: 平衡速度和质量
- **缺点**: 需要预计算，内存占用大

## 使用方法

```bash
# 运行实验 1（Baseline）
python disk_rotation_experiments/experiment_1_baseline.py

# 运行实验 2（参数化旋转）
python disk_rotation_experiments/experiment_2_parametric.py

# 运行实验 3（关键帧插值）
python disk_rotation_experiments/experiment_3_keyframes.py
```

## 输出

每个实验会生成：
- 视频文件：`output/disk_rotation_exp{N}.mp4`
- 每帧右下角显示：
  - 方案名称
  - 参数配置
  - 当前帧生成时间
  - 总生成时间

## 参数配置

所有实验使用相同的参数：
- 分辨率：512x512（俯视图）
- 帧数：60 帧
- 帧率：30 fps
- 吸积盘：r_inner=2.0, r_outer=15.0
- 旋转速度：0.1（与当前 render.py 一致）
