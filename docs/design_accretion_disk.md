# 吸积盘模拟设计文档

## 1. 物理背景

### 1.1 吸积盘基本结构

黑洞吸积盘是环绕黑洞旋转的高温气体盘，主要特征：

- **ISCO（最内稳定圆轨道）**：大约 3-6 倍史瓦西半径，是吸积盘内边界
- **差动旋转**：内圈快外圈慢（开普勒速度 `ω ∝ r^(-3/2)`）
- **温度梯度**：内圈更热、更亮，颜色偏蓝；外圈更冷、更暗，颜色偏红
- **引力红移**：内圈光子在逃逸前被强烈红移

### 1.2 吸积盘会有旋臂吗？

**答：会，但与星系旋臂不同——它们是瞬态的。**

| 特征 | 星系旋臂 | 黑洞吸积盘 |
|------|---------|-----------|
| 寿命 | 数十亿年（准稳态） | 数天-数年（瞬态） |
| 维持机制 | 密度波理论 | 潮汐力/不稳定性/粘滞 |
| 臂数 | 2-4 条（稳定） | 1-3 条（易破碎） |

**触发机制**：
1. **潮汐作用**：若存在伴星（如 X 射线双星），伴星引力激发潮汐螺旋臂
2. **Papaloizou-Pringle 不稳定性**：薄盘对扰动不稳定，形成旋涡模式
3. **磁旋转不稳定性（MRI）**：磁场与差动旋转耦合，产生准周期螺旋结构
4. **Rayleigh-Taylor 不稳定性**：在内边界（ISCO）附近产生尖峰和团块

**结论**：
- 吸积盘的旋臂不应是长期稳定的结构
- 应呈现不规则、破碎的特征
- 2-4 条宽臂比 8-15 条细臂更符合物理

---

## 2. 经典模拟方法

### 2.1 物理精确模拟（科研级）

#### 2.1.1 Shakura-Sunyaev 薄盘模型（1973）

**理论基础**：假设盘很薄（h << r），径向速度远小于转动速度，通过粘滞力角动量传递。

**关键方程**：
- 连续性方程：$\frac{\partial\Sigma}{\partial t} + \frac{1}{r}\frac{\partial}{\partial r}(r\Sigma v_r) = 0$
- 角动量方程：$\Sigma v_r \frac{d(h)}{dr} = \frac{1}{2\pi r}\frac{dG}{dr}$

**你的代码中的简化版本**：
```python
# 温度剖面（Novikov-Thorne）
T = T_max * (1 - sqrt(r_inner / r)) ^ 0.25
```

#### 2.1.2 网格流体动力学（MHD）

使用 **Athena++**、**PLUTO** 或 **Flash** 代码求解理想磁流体方程：

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$

$$\frac{\partial (\rho \mathbf{v})}{\partial t} + \nabla \cdot (\rho \mathbf{v}\mathbf{v} + P\mathbf{I} - \mathbf{B}\mathbf{B}) = -\rho\nabla\Phi + \mathbf{J}\times\mathbf{B}$$

**经典结果**：产生 MRI，形成湍流和准周期性的密度波（QPOs）。

#### 2.1.3 光滑粒子流体动力学（SPH）

将流体离散为粒子，适合大形变（如潮汐撕裂事件）。可追踪百万级粒子模拟盘的风和喷流。

---

### 2.2 实时视觉模拟（图形学/游戏）

#### 2.2.1 噪声合成（你已部分实现）

**FBM（分形布朗运动）+ 螺旋密度波**是行业标准。

**Warping FBM 算法**：
```python
def disk_density(r, phi, t):
    # 基础螺旋臂
    arm_angle = phi - log(r) * pitch_angle - omega * t
    
    # 多层噪声扰动
    noise = fbm(r * freq, phi * freq, octaves=4)
    
    # 将噪声映射到螺旋坐标（产生云状结构）
    warped = fbm(r * freq + noise * distortion, phi * freq + noise * distortion)
    
    return arm_profile(arm_angle) * warped
```

**参考**：《星际穿越》吸积盘使用基于物理的湍流合成（Ramanathan et al., SIGGRAPH 2015）。

#### 2.2.2 粒子系统（N-body 简化）

模拟 $10^4-10^6$ 个测试粒子在引力场中的运动：

```python
# 每个粒子：位置 (r, phi)，比角动量 h
omega = sqrt(G * M / r**3)  # 开普勒角速度
phi_dot = omega

# 径向漂移（模拟粘滞）
r_dot = -alpha * c_s * (dlnP / dlnr) / omega
```

适合产生动态团块（clumps）和旋涡效果。

#### 2.2.3 流体纹理（Fluid Texture）

使用 **2D Navier-Stokes 方程**在纹理空间模拟：
- 在 (r, φ) 坐标系中求解速度场
- 实时模拟湍流和对流单元
- 结果烘焙为密度纹理序列

**优势**：计算成本低，可产生真实的涡旋结构。

---

### 2.3 计算开销对比（基于 CPU 渲染）

假设当前 1080p 渲染需要 **5-30 秒**：

| 方案 | 每帧额外开销 | 总帧时间 | CPU 可行性 | 视觉效果 |
|------|-------------|----------|-----------|---------|
| **静态纹理（当前）** | 0 ms | 5-30s | ✅ | 无动态 |
| **预计算动画序列** | 2-5 ms | +0.1% | ✅✅ | 流畅旋转 |
| **实时 FBM 扰动** | 50-200 ms | +1-5% | ✅ | 缓慢漂移 |
| **2D 流体纹理 (256²)** | 500-2000 ms | +10-50% | ⚠️ | 真实湍流 |
| **SPH (10k 粒子)** | 3000-8000 ms | +100% | ❌ | 团块状 |

---

## 3. 本项目选择

### 3.1 约束条件

- **纯 CPU 渲染**：需保持帧率可用
- **实时性**：视频生成需在合理时间内完成
- **内存限制**：不能加载过大的纹理序列

### 3.2 最终方案

采用**静态纹理 + 渲染时扰动**方案：

1. **程序化生成**：极坐标下直接生成纹理，避免 phi 方向接缝
2. **多尺度噪声**：FBM 产生云雾效果
3. **旋臂**：2-4 条主臂（改进当前 8-15 条），加入噪声使其断断续续
4. **热点**：弧形亮斑，模拟局部高密度区
5. **内边界特殊处理**：RT 不稳定性产生尖峰结构
6. **动态效果**：通过渲染时的相位偏移模拟旋转

---

## 4. 当前实现详解

### 4.1 纹理坐标系

```python
# 极坐标参数
phi: 0 → 2π（角度方向，对应纹理 U）
r: r_inner → r_outer（径向方向，对应纹理 V）
```

### 4.2 纹理组成结构

```
吸积盘密度 = 
  ├── 基础温度剖面（Novikov-Thorne）
  ├── 螺旋臂（螺旋密度波，8-15条→改为2-4条）
  ├── 云雾（FBM 噪声，多尺度叠加）
  ├── 热点（弧形亮斑，15-35个）
  └── 内边界不稳定性（RT 尖峰，待改进）
```

### 4.3 各组件实现

#### 4.3.1 螺旋臂

```python
# 当前实现
arm_angle = phi - base_angle - r_norm * rotations * 2 * np.pi
arm_val = exp(kappa * (cos(arm_angle) - 1) / width_mod)

# 问题：太规则、太稳定
# 改进方向：
# - 减少到 2-4 条
# - 使用噪声调制宽度和强度，使其断断续续
# - 加入时间演化的相位漂移
```

#### 4.3.2 云雾（FBM 噪声）

```python
# 多层 Perlin/Simplex 噪声叠加
# 极坐标下生成以避免接缝
# 按面积比例分布
for octave in range(4):
    scale = base_scale * 2 ** octave
    amplitude = base_amplitude * 0.5 ** octave
    noise += fbm(phi * scale, r * scale) * amplitude
```

#### 4.3.3 热点

```python
# 弧形亮斑，位于不同半径
# 随机强度和位置
# 使用高斯衰减
hotspot = exp(-((phi - phi_0) / sigma_phi)^2 - ((r - r_0) / sigma_r)^2)
```

#### 4.3.4 内边界（待改进）

**当前**：无特殊处理

**应添加**：
- RT 不稳定性尖峰
- ISCO 附近的热点
- 更强的亮度梯度

---

## 5. 改进计划

### 5.1 短期（高优先级）

- [ ] **减少旋臂数量**：8-15 → 2-4 条
- [ ] **增加旋臂破碎效果**：用噪声调制使边缘不规则
- [ ] **增强内边界**：添加 RT 不稳定性尖峰
- [ ] **热点改进**：内边界附近增加高亮区域

### 5.2 中期（可选）

- [ ] **预计算动画序列**：30-60 帧循环，每帧不同湍流相位
- [ ] **动态噪声**：时间演化的 FBM 扰动
- [ ] **调整颜色**：内圈更蓝，外圈更红

### 5.3 长期（探索）

- [ ] GPU 加速后可用 SPH 粒子
- [ ] 实时流体模拟（MHD）
- [ ] 与黑洞阴影统一渲染

---

## 6. 视频编码质量

### 6.1 问题

FFmpeg 默认 CRF=23 压缩较多，可能在视频中产生摩尔纹。

### 6.2 解决方案

渲染后手动用 FFmpeg 重新编码：

```bash
ffmpeg -framerate 36 -i output/.frames_xxx/frame_%04d.png -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p output_high_quality.mp4
```

**参数说明**：
- `-crf 18`：更高质量（默认 23 压缩较多）
- `-preset slow`：更好的压缩算法
- `-pix_fmt yuv420p`：兼容播放

---

## 7. 参考资料

### 7.1 论文

- Shakura, N.I., Sunyaev, R.A. (1973). "Black holes in binary systems. Observational appearance." Astronomy & Astrophysics.
- Interstellar rendering paper (2015). DNEG's scientific approach.
- Kim, J., et al. (2019). "Warped FBM for realistic cloud rendering." SIGGRAPH.

### 7.2 代码

- **Athena++**：网格 MHD 模拟
- **PLUTO**：NASA 天体物理流体
- **Gadget-2**：SPH 粒子模拟
- **Pencil Code**：开源 MHD

### 7.3 软件

- **Three.js**：WebGL 渲染参考
- **Blender**：程序化纹理生成