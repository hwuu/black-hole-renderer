# 吸积盘旋转的物理矛盾分析

## 问题本质

### 当前实现的矛盾

**静态纹理 + 开普勒旋转 = 必然失真**

```
初始状态（t=0）:
  内圈 (r=2):  纹理特征 A 在 φ=0°
  外圈 (r=15): 纹理特征 B 在 φ=0°

经过 N 帧后:
  内圈 (r=2):  特征 A 旋转到 φ=α°  (α 大)
  外圈 (r=15): 特征 B 旋转到 φ=β°  (β 小)

问题：α ≠ β，原本径向对齐的特征现在错位了！
```

### 物理真实情况

真实的吸积盘：
- **物质在旋转**，不是纹理在旋转
- 内圈物质快，外圈物质慢（开普勒旋转）
- 物质会被**剪切拉伸**，形成螺旋结构
- 湍流、磁场等会不断产生新的结构

### 当前方案的局限

```python
phi = phi + t_offset * omega  # 简单的刚体旋转
```

这相当于：
1. 生成一张静态纹理
2. 把纹理当作刚体，按开普勒速度旋转
3. **问题**：纹理不是刚体！物质会被剪切

---

## 解决方案对比

### 方案 A: 降低旋转速度（当前方案）

```python
t_offset = float(frame) * 0.01
```

**优点**:
- 简单，立即可用
- 短时间内失真不明显

**缺点**:
- 只是延缓问题，不能根治
- 长视频仍会失真
- 物理上不正确

**适用场景**:
- 短视频（< 360 帧）
- 快速预览
- 不追求物理准确性

---

### 方案 B: 每帧重新生成纹理

#### B1: 完全随机生成

```python
for frame in range(n_frames):
    seed = 42 + frame  # 不同的随机种子
    disk_tex = generate_disk_texture(seed=seed, ...)
```

**优点**:
- 每帧都是"新鲜"的吸积盘
- 没有累积失真

**缺点**:
- ❌ **帧间不连续**：噪声、filament 位置完全随机
- ❌ **视觉跳跃**：看起来像闪烁
- ❌ **速度慢**：每帧生成纹理需要 20-30 秒

**结论**: ❌ 不可行

---

#### B2: 参数化旋转生成

**核心思想**：在极坐标下生成时，直接考虑旋转

```python
def generate_disk_texture_with_rotation(n_phi, n_r, seed, t_offset):
    """
    在生成时就考虑开普勒旋转
    """
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r_norm = np.linspace(0, 1, n_r)
    phi_grid, r_norm_grid = np.meshgrid(phi, r_norm)

    # 关键：在生成时应用开普勒剪切
    omega = np.sqrt(0.5 / (r_inner + (r_outer - r_inner) * r_norm_grid))
    phi_rotated = phi_grid + t_offset * omega

    # 使用 phi_rotated 生成所有结构
    spiral = generate_spiral_arms(phi_rotated, r_norm_grid, ...)
    filaments = generate_filaments(phi_rotated, r_norm_grid, ...)
    ...
```

**优点**:
- ✅ 物理正确：考虑了开普勒剪切
- ✅ 帧间连续：相同的随机种子 + 不同的 t_offset
- ✅ 可以模拟螺旋臂的缠绕

**缺点**:
- ❌ **速度慢**：每帧仍需生成纹理（20-30 秒/帧）
- ❌ **内存大**：需要缓存或实时生成
- ⚠️ **复杂度高**：需要重写生成逻辑

**可行性**: ⚠️ 技术上可行，但性能是瓶颈

---

#### B3: 预计算 + 插值

**核心思想**：预先生成多个关键帧，运行时插值

```python
# 预计算阶段：生成 10 个关键帧
keyframes = []
for i in range(10):
    t = i * (2 * np.pi / 10)
    tex = generate_disk_texture_with_rotation(t_offset=t)
    keyframes.append(tex)

# 渲染阶段：插值
def get_texture_at_frame(frame, n_frames):
    t = (frame / n_frames) * 2 * np.pi
    # 找到相邻的两个关键帧
    idx0 = int(t / (2*np.pi/10)) % 10
    idx1 = (idx0 + 1) % 10
    # 线性插值
    alpha = (t % (2*np.pi/10)) / (2*np.pi/10)
    return keyframes[idx0] * (1-alpha) + keyframes[idx1] * alpha
```

**优点**:
- ✅ 帧间连续
- ✅ 运行时快速
- ✅ 内存可控（只存 10 个关键帧）

**缺点**:
- ⚠️ 预计算时间长（10 × 30 秒 = 5 分钟）
- ⚠️ 插值可能不够平滑
- ⚠️ 内存占用大（10 × 纹理大小）

**可行性**: ✅ 可行，但需要权衡

---

### 方案 C: 禁用旋转

```python
t_offset = 0  # 吸积盘不旋转
```

**优点**:
- ✅ 没有失真
- ✅ 简单

**缺点**:
- ❌ 失去了动态感
- ❌ 物理上不真实（吸积盘应该旋转）

**适用场景**:
- 静态展示
- 相机快速移动时（旋转不明显）

---

### 方案 D: 混合方案（推荐）

**核心思想**：慢速旋转 + 短视频

```python
# 1. 降低旋转速度
t_offset = float(frame) * 0.005  # 更慢

# 2. 限制视频长度
# 建议：360 帧以内，失真不明显

# 3. 添加 CLI 参数
parser.add_argument("--disk_rotation_speed", type=float, default=0.005)
parser.add_argument("--disable_disk_rotation", action="store_true")
```

**优点**:
- ✅ 简单实用
- ✅ 用户可控
- ✅ 短视频效果好

**缺点**:
- ⚠️ 长视频仍有失真
- ⚠️ 不是物理正确的解决方案

**适用场景**:
- 大多数实际使用场景
- 快速预览和演示

---

## 速度分析

### 当前纹理生成速度

测试配置：
- 分辨率：n_phi=784, n_r=128
- ar1=2, ar2=15

```
Spiral arms:     0.7s  (4 条)
Filaments:      17.0s  (2054 条)
Hotspots:        4.0s  (554 个)
其他:            2.0s
------------------------
总计:          ~24s/帧
```

### 3600 帧视频的时间成本

| 方案 | 纹理生成 | 光线追踪 | 总时间 |
|------|---------|---------|--------|
| 静态纹理 | 24s × 1 | 6s × 3600 | ~6 小时 |
| 每帧生成 | 24s × 3600 | 6s × 3600 | ~30 小时 |
| 10 关键帧 | 24s × 10 | 6s × 3600 | ~6.5 小时 |

**结论**: 每帧生成不现实（30 小时太长）

---

## 推荐方案

### 短期（立即可用）

**方案 D: 混合方案**

```python
# render.py:1652
t_offset = float(frame) * 0.005  # 降低到 0.005

# 添加 CLI 参数
parser.add_argument("--disk_rotation_speed", type=float, default=0.005,
                    help="吸积盘旋转速度系数，0 表示禁用旋转")
```

**使用建议**:
- 360 帧以内：使用默认 0.005
- 3600 帧：降低到 0.001 或禁用旋转
- 快速移动相机：可以提高到 0.01

---

### 中期（性能优化）

**方案 B3: 预计算关键帧**

1. 实现 `generate_disk_texture_with_rotation(t_offset)`
2. 预计算 10-20 个关键帧
3. 运行时插值

**预计开发时间**: 2-3 天

---

### 长期（物理正确）

**实时生成 + GPU 加速**

1. 将纹理生成移植到 Taichi GPU
2. 实时生成每帧的吸积盘
3. 目标：< 1 秒/帧

**预计开发时间**: 1-2 周

---

## 失真可接受性分析

### 测试不同旋转速度

| 系数 | 3600 帧内圈旋转 | 3600 帧外圈旋转 | 失真程度 |
|------|----------------|----------------|---------|
| 0.1  | 28.6 圈 | 10.5 圈 | ❌ 严重 |
| 0.01 | 2.9 圈  | 1.0 圈  | ⚠️ 明显 |
| 0.005| 1.4 圈  | 0.5 圈  | ✅ 可接受 |
| 0.001| 0.3 圈  | 0.1 圈  | ✅ 几乎无 |
| 0    | 0 圈    | 0 圈    | ✅ 无（但无动态） |

### 360 帧视频（10 秒）

| 系数 | 360 帧内圈旋转 | 360 帧外圈旋转 | 失真程度 |
|------|---------------|---------------|---------|
| 0.01 | 0.29 圈 | 0.10 圈 | ✅ 几乎无 |
| 0.005| 0.14 圈 | 0.05 圈 | ✅ 无 |

**结论**:
- 短视频（360 帧）：0.01 可接受
- 长视频（3600 帧）：建议 0.001-0.005

---

## 其他项目的做法

### JaeHyunLee94/BlackHoleRendering

查看其代码（如果有视频功能）：
- 可能使用非常小的旋转系数
- 或者禁用旋转
- 或者使用预计算方案

### 电影《星际穿越》

DNEG 的做法：
- 使用物理模拟生成吸积盘
- 每帧都计算物质的运动
- 使用超级计算机渲染
- 我们的项目无法达到这个级别

---

## 总结

### 问题本质

**静态纹理 + 差分旋转 = 必然失真**

这是当前方案的固有限制，无法完全避免。

### 实用建议

1. **已实现**: 添加 `--disk_rotation_algorithm` 参数，支持三种算法
2. **已实现**: 添加 `--disk_rotation_speed` 参数，让用户可以调整旋转速度
3. **已实现**: 添加 `--keyframes_count` 参数，用于关键帧算法
4. **未来优化**: 可以考虑 GPU 加速实时生成

### 使用方法

```bash
# 算法 1: baseline（固定纹理 + 刚体旋转，最快，长视频有失真）
python render.py --video --orbit --n_frames 360 \
  --disk_rotation_algorithm baseline \
  --disk_rotation_speed 0.05 \
  -o output/baseline.mp4

# 算法 2: parametric（每帧重新生成，物理正确，最慢）
python render.py --video --orbit --n_frames 360 \
  --disk_rotation_algorithm parametric \
  --disk_rotation_speed 0.05 \
  -o output/parametric.mp4

# 算法 3: keyframes（关键帧插值，平衡速度和质量）
python render.py --video --orbit --n_frames 360 \
  --disk_rotation_algorithm keyframes \
  --disk_rotation_speed 0.05 \
  --keyframes_count 10 \
  -o output/keyframes.mp4
```

### 算法对比

| 算法 | 速度 | 物理正确性 | 长视频失真 | 适用场景 |
|------|------|-----------|-----------|---------|
| baseline | 最快 | ❌ 刚体旋转 | ⚠️ 有 | 快速预览、短视频 |
| parametric | 最慢 | ✅ 开普勒剪切 | ✅ 无 | 高质量、物理准确 |
| keyframes | 中等 | ⚠️ 近似 | ⚠️ 轻微 | 平衡速度和质量 |

### 可接受的权衡

- ✅ 短视频（< 360 帧）：三种算法效果都很好
- ⚠️ 长视频（3600 帧）：baseline 有失真，建议用 keyframes 或 parametric
- ✅ 超长视频：keyframes 算法是最佳选择

这是**性能、质量、复杂度**之间的权衡。
