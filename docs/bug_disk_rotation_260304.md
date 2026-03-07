# Bug Report: 吸积盘纹理旋转过快

**发现日期**: 2026-03-04
**报告人**: 用户
**严重程度**: 高（影响视频质量）

---

## 问题描述

在生成环绕视频时，第 0 帧和第 3599 帧的吸积盘纹理差异非常大，纹理严重错位。

**复现命令**:
```bash
python render.py --texture ... --pov 20 0 2 --fov 60 --ar1 2 --ar2 15 \
  --disk_tilt 20 --resolution fhd --video --orbit --n_frames 3600 \
  -o output/demo7-fhd.mp4 --resume
```

**观察结果**:
- `frame_0000.png`: 吸积盘纹理正常
- `frame_3599.png`: 吸积盘纹理完全不同，严重错位

---

## 根本原因

### 当前实现

**位置**: `render.py:1652`

```python
def render(self, cam_pos, fov, frame=0):
    ...
    t_offset = float(frame) * 0.1  # ❌ 问题在这里
    ...
```

**位置**: `render.py:1179`

```python
def _sample_disk(hit_x, hit_y, r_inner, r_outer, t_offset):
    r = ti.sqrt(hit_x ** 2 + hit_y ** 2)
    phi = ti.atan2(hit_y, hit_x)
    omega = ti.sqrt(0.5 / (r + 0.01))  # 开普勒角速度
    phi = phi + t_offset * omega  # ❌ 旋转角度过大
```

### 问题分析

1. **t_offset 值过大**:
   - frame=0: t_offset=0
   - frame=3599: t_offset=359.9

2. **旋转圈数过多**:
   - 内圈 (r=2): 旋转 **28.6 圈**
   - 外圈 (r=15): 旋转 **10.5 圈**

3. **差分旋转导致错位**:
   - 不同半径的旋转速度不同（开普勒旋转）
   - 内圈旋转快，外圈旋转慢
   - 导致纹理严重扭曲和错位

### 测试数据

```
第 0 帧，内圈 (r=2):
  t_offset = 0.0
  旋转圈数 = 0.0 圈

第 3599 帧，内圈 (r=2):
  t_offset = 359.9
  旋转圈数 = 28.6 圈  ❌ 太多了！

第 3599 帧，外圈 (r=15):
  t_offset = 359.9
  旋转圈数 = 10.5 圈  ❌ 太多了！
```

---

## 修复方案

### 方案 1: 降低旋转速度（推荐）

**修改**: `render.py:1652`

```python
# 修改前
t_offset = float(frame) * 0.1

# 修改后
t_offset = float(frame) * 0.01  # 降低 10 倍
```

**效果**:
- 3600 帧视频中，内圈旋转约 2.9 圈，外圈旋转约 1.0 圈
- 旋转速度更合理，视觉效果更自然

**优点**:
- 修改简单，只改一行
- 保持开普勒旋转特性
- 旋转速度可调（通过修改系数）

**缺点**:
- 旋转速度仍然是固定的，不随视频长度自适应

---

### 方案 2: 归一化旋转（更优雅）

**修改**: `render.py:1652` 和 `render_video` 函数

```python
# 在 render 方法中
def render(self, cam_pos, fov, frame=0, n_frames=1):
    ...
    # 归一化到 [0, 2π]，一个完整视频周期旋转一圈
    t_offset = float(frame) / float(n_frames) * 2.0 * np.pi
    ...

# 在 render_video 中传递 n_frames
img = renderer.render(cam_pos, fov, frame=frame, n_frames=n_frames)
```

**效果**:
- 一个完整的 360° 环绕视频，吸积盘旋转恰好一圈
- 视频长度自适应

**优点**:
- 旋转速度与视频长度匹配
- 物理意义更清晰
- 更容易理解和调整

**缺点**:
- 需要修改函数签名
- 需要传递 n_frames 参数

---

### 方案 3: 可配置旋转速度（最灵活）

**修改**: 添加 CLI 参数

```python
parser.add_argument("--disk_rotation_speed", type=float, default=0.01,
                    help="吸积盘旋转速度系数 (default: 0.01)")

# 在 render 方法中
t_offset = float(frame) * self.disk_rotation_speed
```

**优点**:
- 用户可以自定义旋转速度
- 可以设置为 0 禁用旋转
- 最灵活

**缺点**:
- 增加了 API 复杂度
- 需要用户理解参数含义

---

## 推荐修复

**立即修复**: 使用方案 1（降低旋转速度）

```python
# render.py:1652
t_offset = float(frame) * 0.01  # 从 0.1 改为 0.01
```

**理由**:
1. 修改最小，风险最低
2. 立即解决问题
3. 不破坏现有 API

**后续优化**: 考虑方案 2 或方案 3

---

## 测试计划

### 1. 单帧测试

生成两张图片，验证旋转效果：

```bash
# 第 0 帧
python render.py --pov 20 0 2 --fov 61 --ar1 2 --ar2 15 --disk_tilt 20 \
  --resolution sd -o output/test_frame0.png

# 模拟第 360 帧（旋转 1/10 周期）
# 需要修改代码临时设置 frame=360
```

### 2. 短视频测试

生成 10 帧短视频，检查旋转连续性：

```bash
python render.py --pov 20 0 2 --fov 61 --ar1 2 --ar2 15 --disk_tilt 20 \
  --resolution sd --video --orbit --n_frames 10 -o output/test_short.mp4
```

### 3. 视觉检查

- 检查吸积盘纹理是否连续
- 检查旋转速度是否合理
- 检查内外圈是否协调

---

## 相关代码位置

- `render.py:1652` - t_offset 计算
- `render.py:1179` - 吸积盘采样（_sample_disk）
- `render.py:1210` - 吸积盘采样（_sample_disk_mip）
- `render.py:1889` - render_video 调用 render

---

## 附加说明

### 为什么使用 t_offset * omega？

这是为了模拟吸积盘的**开普勒旋转**：
- 内圈角速度快（omega 大）
- 外圈角速度慢（omega 小）
- 符合物理规律：ω ∝ r^(-3/2)

### 为什么会出现这个 bug？

可能的原因：
1. 最初设计时假设视频帧数较少（如 360 帧）
2. 0.1 的系数在短视频中效果可能还可以
3. 但在 3600 帧的长视频中，旋转过快导致问题

### 其他项目的做法

参考 JaeHyunLee94/BlackHoleRendering：
- 可能使用更小的旋转系数
- 或者将旋转速度与视频长度关联

---

**状态**: 待修复
**优先级**: 高
**预计修复时间**: 5 分钟（方案 1）
