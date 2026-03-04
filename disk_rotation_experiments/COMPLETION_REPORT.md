# 吸积盘旋转实验 - 完成报告

**日期**: 2026-03-04
**状态**: ✅ 实验代码已完成

---

## 已创建的文件

### 核心代码

1. **`disk_rotation_experiments/common.py`** (共享模块)
   - `generate_disk_texture_topview()` - 生成吸积盘纹理（支持 t_offset 参数）
   - `polar_to_cartesian()` - 极坐标转笛卡尔坐标
   - `sample_disk_with_rotation()` - Baseline 方案的采样函数
   - `add_text_overlay()` - 添加文字覆盖层

2. **`disk_rotation_experiments/experiment_1_baseline.py`** (实验 1)
   - 固定纹理 + 刚体旋转
   - 生成一次纹理，每帧旋转采样
   - 输出: `output/disk_rotation_exp1_baseline.mp4`

3. **`disk_rotation_experiments/experiment_2_parametric.py`** (实验 2)
   - 参数化旋转纹理
   - 每帧重新生成纹理（考虑 t_offset）
   - 输出: `output/disk_rotation_exp2_parametric.mp4`

4. **`disk_rotation_experiments/experiment_3_keyframes.py`** (实验 3)
   - 关键帧插值
   - 预计算 10 个关键帧，运行时插值
   - 输出: `output/disk_rotation_exp3_keyframes.mp4`

### 辅助脚本

5. **`disk_rotation_experiments/quick_test.py`**
   - 快速测试脚本（只生成几帧）
   - 验证代码是否正常工作
   - 输出测试图像到 `output/test_*.png`

6. **`disk_rotation_experiments/run_all.py`**
   - 运行所有三个实验
   - 生成性能对比报告

### 文档

7. **`disk_rotation_experiments/README.md`**
   - 实验说明文档
   - 使用方法

8. **`disk_rotation_experiments/REPORT_TEMPLATE.md`**
   - 实验报告模板
   - 用于填写实验结果

---

## 使用方法

### 快速测试（推荐先运行）

```bash
# 快速测试（只生成几帧，验证代码）
python disk_rotation_experiments/quick_test.py
```

这会生成三张测试图像：
- `output/test_baseline.png`
- `output/test_parametric.png`
- `output/test_with_text.png`

### 运行单个实验

```bash
# 实验 1: Baseline（最快，约 1-2 分钟）
python disk_rotation_experiments/experiment_1_baseline.py

# 实验 2: 参数化旋转（最慢，约 30-40 分钟）
python disk_rotation_experiments/experiment_2_parametric.py

# 实验 3: 关键帧插值（中等，约 5-10 分钟）
python disk_rotation_experiments/experiment_3_keyframes.py
```

### 运行所有实验

```bash
# 运行所有三个实验（总计约 40-50 分钟）
python disk_rotation_experiments/run_all.py
```

---

## 实验参数

所有实验使用相同的参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| 分辨率 | 512x512 | 俯视图 |
| 帧数 | 60 | 2 秒视频 @ 30fps |
| 帧率 | 30 fps | |
| r_inner | 2.0 | 吸积盘内半径 |
| r_outer | 15.0 | 吸积盘外半径 |
| 旋转速度 | 0.1 | 与 render.py 一致 |
| 随机种子 | 42 | 保证可重复性 |

---

## 预期结果

### 实验 1: Baseline

**速度**: ⚡⚡⚡ 最快
- 纹理生成: ~20 秒（只生成一次）
- 渲染: ~0.01 秒/帧
- 总时间: ~20-30 秒

**质量**: ⚠️ 有失真
- 60 帧内，内圈旋转约 0.3 圈，外圈旋转约 0.1 圈
- 差分旋转约 0.2 圈 = 72°
- 应该能看到明显的纹理错位

---

### 实验 2: 参数化旋转

**速度**: 🐌 最慢
- 纹理生成: ~20 秒/帧 × 60 = ~1200 秒（20 分钟）
- 渲染: ~0.01 秒/帧
- 总时间: ~20-30 分钟

**质量**: ✅ 完美
- 每帧都是物理正确的
- 无累积失真
- 帧间连续

---

### 实验 3: 关键帧插值

**速度**: ⚡⚡ 较快
- 预计算: ~20 秒/帧 × 10 = ~200 秒（3-4 分钟）
- 插值: ~0.001 秒/帧
- 总时间: ~3-5 分钟

**质量**: ✅ 较好
- 插值平滑
- 可能在关键帧切换时有轻微跳跃
- 整体质量接近参数化方案

---

## 视频输出

每个视频的右下角会显示：

```
Experiment N: [方案名称]
Method: [方法描述]
Frame: [当前帧]/60
[方案特定信息]
Frame time: [帧生成时间]
Total time: [总时间]
```

---

## 观察要点

观看视频时，重点观察：

### 1. 螺旋臂

- **Baseline**: 螺旋臂是否扭曲变形？内外圈是否错位？
- **Parametric**: 螺旋臂是否平滑旋转？
- **Keyframes**: 螺旋臂是否有跳跃？

### 2. Filaments（细丝）

- **Baseline**: 细丝是否被拉伸或断裂？
- **Parametric**: 细丝是否保持连续？
- **Keyframes**: 细丝是否平滑？

### 3. 整体连续性

- 暂停视频，逐帧播放
- 观察相邻帧之间的变化
- Baseline 应该有明显的"跳跃"感

---

## 下一步

1. **运行快速测试**:
   ```bash
   python disk_rotation_experiments/quick_test.py
   ```

2. **检查测试图像**:
   - 打开 `output/test_*.png`
   - 确认图像正常

3. **运行实验 1**（最快）:
   ```bash
   python disk_rotation_experiments/experiment_1_baseline.py
   ```

4. **观看视频**:
   ```bash
   vlc output/disk_rotation_exp1_baseline.mp4
   # 或
   ffplay output/disk_rotation_exp1_baseline.mp4
   ```

5. **如果满意，运行其他实验**:
   ```bash
   python disk_rotation_experiments/experiment_3_keyframes.py  # 先运行较快的
   python disk_rotation_experiments/experiment_2_parametric.py  # 最后运行最慢的
   ```

6. **填写实验报告**:
   - 复制 `REPORT_TEMPLATE.md` 为 `REPORT.md`
   - 填写实际结果和观察

---

## 故障排除

### 如果遇到导入错误

```bash
# 确保在项目根目录运行
cd C:\Users\hwuu\dev\hwuu\black-hole-renderer
python disk_rotation_experiments/quick_test.py
```

### 如果纹理生成太慢

可以修改参数（在各个实验脚本中）：
```python
# 降低分辨率
SIZE = 256  # 从 512 改为 256

# 减少帧数
N_FRAMES = 30  # 从 60 改为 30

# 降低纹理分辨率
n_phi=256,  # 从 512 改为 256
n_r=64,     # 从 128 改为 64
```

### 如果视频无法播放

```bash
# 检查视频文件是否生成
ls -lh output/disk_rotation_exp*.mp4

# 使用 ffprobe 检查视频信息
ffprobe output/disk_rotation_exp1_baseline.mp4
```

---

## 技术说明

### Baseline 方案的失真原理

```python
# 固定纹理在极坐标 (r, phi) 下生成
texture[r, phi] = generate(...)

# 采样时应用旋转
phi_rotated = phi + t_offset * omega(r)

# 问题：omega(r) = sqrt(0.5 / r)，内圈快、外圈慢
# 原本对齐的特征会被剪切错位
```

### 参数化方案的正确性

```python
# 生成时就考虑旋转
phi_grid_rotated = phi_grid + t_offset * omega(r_grid)

# 所有结构在旋转后的坐标系中生成
spiral = generate_spiral(phi_grid_rotated, ...)

# 结果：物理正确，无失真
```

### 关键帧插值的权衡

```python
# 预计算 N 个关键帧
keyframes = [generate(t_offset=i*dt) for i in range(N)]

# 运行时插值
image = lerp(keyframes[i], keyframes[i+1], alpha)

# 权衡：速度快，但插值可能不够平滑
```

---

## 总结

✅ **实验代码已完成**，包括：
- 3 个实验脚本
- 共享代码模块
- 快速测试脚本
- 批量运行脚本
- 完整文档

🎯 **下一步**：运行实验，观察结果，填写报告

⏱️ **预计时间**：
- 快速测试: 1-2 分钟
- 实验 1: 1-2 分钟
- 实验 3: 5-10 分钟
- 实验 2: 30-40 分钟
- 总计: 约 40-50 分钟

📊 **预期发现**：
- Baseline 方案在 60 帧内会有明显失真
- 参数化方案完美但太慢
- 关键帧方案是最佳平衡点
