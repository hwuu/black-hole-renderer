# 问题记录

## Issue #1: 多普勒效应颜色/亮度方向问题

**状态**: 已解决视觉效果，物理原理待深入分析

**日期**: 2026-03-01

---

### 1. 初始状态

原始代码中多普勒效应的实现：

```python
# 原始速度方向（假设盘在 xy 平面）
v_hat = ti.Vector([-hit_pos[1], hit_pos[0], 0.0])

# 颜色调整
red_boost = 1.0 + 0.35 * ti.pow(neg_shift, 0.75)
r_scale = (1.0 + 0.6 * neg_shift - 0.25 * pos_shift) * red_boost
b_scale = 1.05 + 0.3 * pos_shift - 0.15 * neg_shift

# 亮度调整
brightness *= 1.05 + 0.18 * pos_shift - 0.08 * neg_shift + neg_lift
```

**问题表现**：用户反馈视频生成后，黑洞右侧应该更亮（蓝移），但实际效果相反。

---

### 2. 调试过程

#### 2.1 第一次尝试：交换颜色公式

初步分析认为颜色公式写反了，将 `pos_shift` 和 `neg_shift` 在颜色公式中交换：

```python
red_boost = 1.0 + 0.35 * ti.pow(pos_shift, 0.75)  # 改为 pos_shift
r_scale = (1.0 + 0.6 * pos_shift - 0.25 * neg_shift) * red_boost
b_scale = 1.05 + 0.3 * neg_shift - 0.15 * pos_shift  # 改为 neg_shift
```

**结果**：生成两张图对比，发现完全一样。

**原因**：Taichi 有编译缓存，代码修改后内核没有重新编译。

**解决**：禁用离线缓存 `ti.init(arch=..., offline_cache=False)`

#### 2.2 第二次尝试：再次颠倒颜色

重新生成两张图（正常版 vs 颠倒版），用户确认**颠倒版是正确的**。

同时发现亮度调整也是反的，用户要求先移除亮度调整，只看颜色。

#### 2.3 第三次尝试：修正亮度公式

直接反转亮度公式的正负号，但用户反馈"逻辑没变"。

分析后发现：亮度来自 g 因子的物理计算，不是手动加的：

```python
g = ti.min(g_doppler * g_grav, g_cap)
intensity = ti.pow(g, lum_power)
brightness = gain * intensity / (1.0 + intensity / g_cap)
```

物理上：g > 1 → 更亮，g < 1 → 更暗。

#### 2.4 关键发现：盘旋转方向问题

**用户确认**：图像右边应该是蓝移（物质向相机运动，蓝亮）。

分析相机设置：
- cam_pos = (20, 0, 2)
- cam_right = (0, 1, 0) → 图像右边对应 +y 方向

当前物理计算显示：
- 图像右边（+y）→ 红移（g < 1）
- 图像左边（-y）→ 蓝移（g > 1）

**与预期相反！**

**结论**：盘的旋转方向错了。

原始代码使用 `v_hat = (-y, x, 0)`，假设盘在 xy 平面逆时针旋转。
修改为正确考虑盘倾斜后的速度方向：

```python
# 盘法向量（考虑倾斜）
disk_normal = ti.Vector([0.0, -sin_tilt, cos_tilt])
r_hat = hit_pos.normalized()

# 关键修改：交换叉积顺序
v_hat = r_hat.cross(disk_normal)  # 原来是 disk_normal.cross(r_hat)
```

#### 2.5 第五次尝试：颜色再次调整

旋转方向修正后，亮度分布正确（右边更亮）。

但颜色需要再次调整，最终正确的颜色公式：

```python
# 蓝移(neg_shift>0): 偏红; 红移(pos_shift>0): 偏蓝
red_boost = 1.0 + 0.35 * ti.pow(neg_shift, 0.75)
r_scale = (1.0 + 0.6 * neg_shift - 0.25 * pos_shift) * red_boost
b_scale = 1.05 + 0.3 * pos_shift - 0.15 * neg_shift
```

**注意**：这里 neg_shift 对应蓝移，但颜色偏红。下面的问题部分会详细说明。

#### 2.6 第六次尝试：悬臂方向

修改盘旋转方向后，悬臂的拖拽方向也反了。

修改悬臂角度公式：

```python
# 原来
arm_angle = phi_grid - base_angle + r_norm_grid * rotations * 2 * np.pi

# 修改为
arm_angle = phi_grid - base_angle - r_norm_grid * rotations * 2 * np.pi
```

---

### 3. 最终正确实现

```python
# 盘旋转方向（顺时针，从 +z 看）
disk_normal = ti.Vector([0.0, -sin_tilt, cos_tilt])
r_hat = hit_pos.normalized()
v_hat = r_hat.cross(disk_normal)

# 颜色：蓝移偏红，红移偏蓝
red_boost = 1.0 + 0.35 * ti.pow(neg_shift, 0.75)
r_scale = (1.0 + 0.6 * neg_shift - 0.25 * pos_shift) * red_boost
b_scale = 1.05 + 0.3 * pos_shift - 0.15 * neg_shift

# 亮度：由 g 因子自动决定（不做额外手动调整）
# g > 1 → 蓝, 更亮
# g < 1 → 红, 更暗
```

---

### 4. 遗留问题：物理正确性待分析

#### 4.1 现象

最终实现中：
- **蓝移区域**（neg_shift > 0，g > 1）：偏红、更亮
- **红移区域**（pos_shift > 0，g < 1）：偏蓝、更暗

这与直觉相反！直觉上蓝移应该偏蓝，红移应该偏红。

#### 4.2 当前理解

**g 因子的物理定义**：

```
g = g_doppler × g_gravitational

g_doppler = 1 / (γ × (1 - β × cos(θ)))
```

其中：
- β = v/c（物质速度）
- θ = 速度方向与观察方向的夹角
- cos(θ) > 0 → 物质远离观察者
- cos(θ) < 0 → 物质接近观察者

**关键公式分析**：

当物质接近观察者（cos(θ) < 0）：
- 分母 `1 - β × cos(θ)` > 1
- g_doppler < 1
- **但这意味着蓝移！**

等等，这里可能有问题... 需要重新推导。

#### 4.3 待验证的假设

1. **shift 变量定义是否有误？**

   ```python
   shift = g - 1.0
   pos_shift = max(shift, 0)   # g > 1 时为正
   neg_shift = max(-shift, 0)  # g < 1 时为正
   ```

   当前假设：g > 1 对应蓝移。但根据上面的分析，可能相反。

2. **多普勒增益公式是否正确？**

   标准相对论多普勒因子：
   - 接近：频率增加（蓝移），g > 1？
   - 远离：频率减少（红移），g < 1？

   需要确认代码中的公式是否符合这个定义。

3. **视觉直觉 vs 物理现象**

   可能的解释：
   - 蓝移区域虽然波长变短，但因为 beaming 效应，辐射强度增强
   - 高能辐射（偏蓝）可能在视觉上呈现暖色调？（需要查证黑体辐射）

#### 4.4 建议的分析方向

1. 添加调试输出，打印 `g`, `g_doppler`, `g_grav`, `cos_theta` 的值
2. 在已知位置（如 +y 侧的点）验证这些值的物理意义
3. 参考 Interstellar 电影的科学论文或其他黑洞渲染器的实现
4. 咨询物理专业人士

---

### 5. 相关文件

- `render.py:679-749` - `apply_g_factor()` 函数
- `render.py:690-710` - g 因子计算
- `render.py:720-741` - shift 和颜色计算
- `render.py:504` - 悬臂角度

---

### 6. 参考资料

- [Doppelgängers: The science behind Interstellar's black hole](https://arxiv.org/abs/1502.03808)
- [Relativistic Doppler effect - Wikipedia](https://en.wikipedia.org/wiki/Relativistic_Doppler_effect)
