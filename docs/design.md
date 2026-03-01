# Black Hole Renderer 设计文档

## 目录

- [1. 背景与目标](#1-背景与目标)
  - [1.1 项目简介](#11-项目简介)
  - [1.2 目标](#12-目标)
  - [1.3 非目标](#13-非目标)
- [2. 物理模型](#2-物理模型)
  - [2.1 史瓦西度规](#21-史瓦西度规)
  - [2.2 笛卡尔等效形式](#22-笛卡尔等效形式)
  - [2.3 守恒量](#23-守恒量)
- [3. 渲染管线](#3-渲染管线)
  - [3.1 整体流程](#31-整体流程)
  - [3.2 相机模型](#32-相机模型)
  - [3.3 光线初始化](#33-光线初始化)
  - [3.4 光线积分](#34-光线积分)
  - [3.5 天空盒采样](#35-天空盒采样)
- [4. 实现细节](#4-实现细节)
  - [4.1 向量化策略](#41-向量化策略)
  - [4.2 为什么选择笛卡尔方案](#42-为什么选择笛卡尔方案)
  - [4.3 文件结构](#43-文件结构)
- [5. 与 starless 的对比分析](#5-与-starless-的对比分析)
  - [5.1 技术路线对比](#51-技术路线对比)
  - [5.2 starless 的优势](#52-starless-的优势)
  - [5.3 本项目的优势](#53-本项目的优势)
- [6. 提升方案](#6-提升方案)
  - [6.1 吸积盘渲染](#61-吸积盘渲染)
  - [6.2 自适应步长](#62-自适应步长)
  - [6.3 实施优先级](#63-实施优先级)
- [7. 参考资料](#7-参考资料)
- [变更记录](#变更记录)

---

## 1. 背景与目标

### 1.1 项目简介

基于广义相对论的史瓦西黑洞光线追踪渲染器。通过数值积分零测地线方程，模拟光线在弯曲时空中的传播路径，生成黑洞引力透镜效果的图像。

### 1.2 目标

| 目标 | 说明 |
|------|------|
| **物理正确** | 基于史瓦西度规的零测地线方程，正确模拟引力透镜 |
| **可视化** | 生成包含黑洞阴影、光子环、引力透镜扭曲星空的图像 |
| **性能可用** | 支持多种分辨率，CPU 渲染可用 |

### 1.3 非目标

- **不做 Kerr 度规**：仅支持无自旋黑洞（史瓦西解）
- **单一框架**：只维护 Taichi 并行实现，不再保留 NumPy 备选
- **不做实时渲染**：非 GPU 加速（Taichi 可选 GPU），不追求交互帧率

---

## 2. 物理模型

### 2.1 史瓦西度规

采用几何单位制（G = c = 1），史瓦西半径 rs = 1（对应质量 M = 1/2）。

球坐标 (t, r, θ, φ) 下的线元：

```
ds² = -(1 - rs/r) dt² + (1 - rs/r)⁻¹ dr² + r² dθ² + r² sin²θ dφ²
```

### 2.2 笛卡尔等效形式

对于零测地线（光子），史瓦西度规下的运动方程可以等效地写成笛卡尔坐标中的二阶 ODE：

```
d²x/dλ² = -1.5 · L² · x / r⁵
```

其中：
- `x = (x, y, z)` 为笛卡尔位置向量
- `r = |x|` 为径向距离
- `L² = |v × x|²` 为角动量平方（守恒量）
- `λ` 为仿射参数

这个等效形式来自 Schwarzschild 有效势 `V_eff = L²/(2r²) - L²/(2r³)` 对 r 求导后的广义相对论修正项。牛顿引力的 `-1/r²` 项被角动量守恒吸收，剩下的 `-1.5L²/r⁵` 是纯 GR 效应，正是它导致了光线弯曲和黑洞阴影。

将上述笛卡尔方程沿径向投影，并记 `L² = |x × v|² = r⁴ φ̇²`，可得到熟悉的 Schwarzschild 零测地线径向方程：

```
r̈ = (L² / r³) - (1.5 L² / r⁴)
```

因此“缺失”的牛顿 `L²/r³` 项会在投影后自动回到径向方程里，等价于传统的 `-1/r² + L²/r³ - 1.5L²/r⁴` 分解；我们在数值测试中也验证了本公式能重现弱场偏折角与 photon ring 位置。

### 2.3 守恒量

角动量平方 `L² = |v × x|²` 在整个积分过程中守恒，只需在初始化时计算一次。这是因为 Schwarzschild 时空具有球对称性，角动量是运动积分。

---

## 3. 渲染管线

### 3.1 整体流程

渲染流程（Taichi 框架）：

```
+----------+  +----------+  +-------------------------+
| Camera   |->| Skybox   |->| Taichi Kernel           |
| Generate |  | Load/Gen |  | while_loop per ray      |
| Rays     |  |          |  | (parallel on CPU/GPU)   |
+----------+  +----------+  +-------------------------+
                         |->| Shade                   |
                           +-------------------------+
```

### 3.2 相机模型

相机采用针孔模型，对齐 JaeHyunLee94：

- **位置**：默认 `(6, 0, 0.5)`
- **朝向**：指向原点
- **FOV**：默认 90°
- **上方向**：+z 轴投影到垂直于视线的平面
- **右方向**：`forward × world_up`

像素映射（图像平面投影）：

```
image_plane_height = 2 * tan(fov/2)
image_plane_width  = image_plane_height * aspect
pixel_width  = image_plane_width / width
pixel_height = image_plane_height / height

pixel_pos = top_left + (px+0.5) * pixel_width * right - (py+0.5) * pixel_height * up
ray_dir = normalize(pixel_pos - camera_pos)
```

### 3.3 光线初始化

所有光线在笛卡尔空间中初始化，无需坐标变换：

```
position  = camera_pos                    # (x, y, z)
velocity  = normalize(ray_dir)            # unit direction vector
L²        = |velocity × position|²        # conserved angular momentum squared
```

### 3.4 光线积分

**积分器**：4 阶 Runge-Kutta（RK4）

**两种算法**：

所有光线在 Taichi kernel 的 while_loop 中独立积分，可运行于 CPU/GPU。

**加速度计算**：

```python
r = |pos|
r⁵ = r² · r² · r
acceleration = -1.5 * L² * pos / r⁵
```

**终止条件**：

| 条件 | 阈值 | 结果 |
|------|------|------|
| r < rs | 事件视界 | captured（黑色） |
| r > max(r_max, 2×distance) | 逃逸半径 | escaped（采样天空） |
| λ_accum > r_escape × 40 | 超时 | 视为 escaped（λ 为累计仿射参数） |

默认 `r_max = 10`，`dt = 0.1`（可通过参数调整）

### 3.5 天空盒采样

**纹理格式**：等距柱状投影（equirectangular），2048×1024，float32

**纹理生成**：
- 底色：深蓝 (0.005, 0.005, 0.005)
- 星云：低频随机噪声上采样，叠加微弱彩色变化
- 恒星：6000 颗随机分布在球面上，渲染为高斯 blob（σ ∈ [0.6, 1.5] 像素）
- 颜色分布：55% 白色、25% 蓝色、20% 暖色

**采样方式**：逃逸光线的速度方向 → (θ, φ) → 纹理坐标 (u, v) → 双线性插值

```
θ = arccos(dz)           -> v = θ/π · H
φ = atan2(dy, dx)        -> u = φ/(2π) · W    (φ 归一化到 [0, 2π])
```

水平方向环绕（wrap），垂直方向钳制（clamp）。

---

## 4. 实现细节

### 4.1 渲染框架

| 框架 | 算法 | 适用场景 | 性能 |
|------|------|----------|------|
| Taichi | while_loop | CPU/GPU 渲染 | 1080p 单帧 ~2s（CPU），GPU 更快 |

Taichi kernel 中每条光线使用 while_loop 独立积分，支持 CPU/GPU 并行；不再保留 NumPy 版本，调试模式可通过 `--device cpu` 和低分辨率实现。

### 4.2 为什么选择笛卡尔方案

对比球坐标 Christoffel 符号方案和笛卡尔等效势方案：

| 方面 | 球坐标 (Christoffel) | 笛卡尔 (等效势) |
|------|---------------------|-----------------|
| 状态维度 | 8 (t, r, θ, φ, ṫ, ṙ, θ̇, φ̇) | 6 (x, y, z, vx, vy, vz) |
| RHS 计算 | sin/cos/cot + 1/f 除法 | 仅 r⁵ 和乘法 |
| 极点奇异性 | θ→0,π 时 cot(θ)→∞ | 无 |
| 事件视界 | f→0 时 1/f→∞，数值爆炸 | r→0 时 1/r⁵ 增长，但光线已被捕获 |
| 收敛步数 | ~1500 步 | ~150 步 |
| 640×360 耗时 | ~60s | ~18s |

笛卡尔方案在所有方面都更优，且物理等价。

### 4.3 文件结构

```
black-hole-renderer/
+-- render.py           # main renderer (单帧 + 视频)
+-- output/             # output images (auto-created)
+-- docs/
|   +-- design.md       # this document
+-- CLAUDE.md           # dev conventions
```

### 4.4 命令行参数

**单帧模式**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pov` | 相机位置 (x, y, z) | 6 0 0.5 |
| `--fov` | 视野角度 (0-180°) | 90 |
| `--resolution`, `-r` | 分辨率: 4k/fhd/hd/sd | fhd |
| `--texture`, `-t` | 天空盒纹理路径 | 程序生成 |
| `--step_size`, `-s` | 积分步长 dt | 0.1 |
| `--r_max` | 逃逸半径 | 10 |
| `--n_stars` | 程序天空盒恒星数量 | 6000 |
| `--disk_texture` | 吸积盘纹理路径 | 程序生成 |
| `--ar1` | 吸积盘内半径 | 2.0 rs |
| `--ar2` | 吸积盘外半径 | 3.5 rs |
| `--output`, `-o` | 输出文件路径 | output/blackhole.png |
| `--device`, `-d` | Taichi 设备: cpu/gpu | cpu |

**视频模式**（添加 `--video` 开启）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video` | 开启视频模式 | - |
| `--orbit` | 相机围绕原点旋转 | - |
| `--orbit_radius` | 轨道半径 | 8.0 |
| `--orbit_z` | 轨道高度 | 0.5 |
| `--n_frames` | 视频帧数 | 3600 |
| `--fps` | 视频帧率 | 36 |
| `--resume` | 尝试从断点恢复（默认从头开始） | - |

---

## 5. 与 starless 的对比分析

[rantonels/starless](https://github.com/rantonels/starless)（[原理文档](https://rantonels.github.io/starless/)）是一个成熟的 Schwarzschild 黑洞渲染器，采用了不同的技术路线。

### 5.1 技术路线对比

| 维度 | starless | 本项目 |
|------|----------|--------|
| 坐标系 | 球坐标 2D 降维（利用球对称性投影到光线平面） | 笛卡尔 3D 直接积分 |
| 核心公式 | `d²r/dλ² = -1/r² + L²/r³ - 1.5L²/r⁴` | `d²x/dλ² = -1.5 L² x / r⁵` |
| 积分器 | Velocity Verlet（2 阶，辛） | RK4（4 阶，非辛） |
| 每步力计算 | 1 次 | 4 次 |
| 状态维度 | 2 (r, φ) | 6 (x, y, z, vx, vy, vz) |
| 坐标变换 | 需要 3D↔2D 旋转矩阵 | 不需要 |
| 吸积盘 | 支持（温度模型 + 黑体辐射） | 不支持 |
| 并行框架 | 纯 NumPy | Taichi (CPU/GPU) |

### 5.2 starless 的优势

- **2D 降维**：每条光线只积分 2 个自由度，理论计算量更少
- **Verlet 积分器**：辛积分器保证长期能量守恒，单步仅 1 次力计算
- **吸积盘**：支持 Shakura-Sunyaev 温度模型 + 黑体辐射着色，视觉效果丰富

### 5.3 本项目的优势

- **公式极简**：一行加速度公式，无坐标变换、无奇点
- **RK4 高精度**：4 阶精度允许更大步长，总步数远少于 Verlet
- **compact 算法**：动态移除终止光线，避免无效计算
- **Taichi 并行**：支持 CPU/GPU，640×360 渲染 < 1s

---

## 6. 已实现的提升

### 6.1 自适应步长 ✅

基于径向距离 + 曲率抑制的启发式步长调整：

```
dt = dt_base · clamp( √(r/rs) / (1 + 2 (rs / r)³), 0.2, 8 )
```

- 远场 (r >> rs)：`√(r/rs)` 主导，步长最多放大到 8×，大幅减少逃逸射线的步数
- 近场 (r ≳ rs)：`(rs/r)³` 抑制因子让步长自动缩到 ~0.3×，保护 photon ring / ghost image 的积分精度
- 终止条件改为累计仿射参数 `λ_accum > r_escape × 40`，不再依赖瞬时步长，避免远场大步长误判超时

### 6.2 吸积盘渲染 ✅

**对齐 JaeHyunLee94 实现**：

- 赤道面穿越检测：`z_old * z_new < 0`，线性插值到 z=0 平面
- 纹理贴图：柱面 UV 映射（`u = φ/2π`, `v = (r-r1)/(r2-r1)`），支持外部纹理或程序生成
- 光线穿透：命中吸积盘后光线继续传播，多次命中累加颜色（ghost images）
- 加法混合：`final = clamp(background + Σ disk_hits, 0, 1)`
- 透明度：体积积分和直接命中时的 alpha 都会乘以 `disk_alpha_gain = 1.5`，整体更不透明

```
每步积分后:
  if z_old * z_new < 0:                        # 穿越赤道面
    t = -z_old / (z_new - z_old)               # 线性插值
    hit_xy = pos_old + t * (pos_new - pos_old)  # 交点
    r_hit = |hit_xy|
    if r_inner <= r_hit <= r_outer:
      color = sample_disk_texture(hit_x, hit_y)   # 纹理采样
      color = apply_g_factor(color, hit_xy, ray_dir, camera_pos)  # 多普勒+引力红移
      accumulated_color += color                   # 累加，光线继续
```

**关键参数**（对齐 JaeHyunLee94）：

| 参数 | 默认值 | CLI | 说明 |
|------|--------|-----|------|
| r_inner | 2.0 rs | `--ar1` | 吸积盘内半径 |
| r_outer | 3.5 rs | `--ar2` | 吸积盘外半径 |
| disk_texture | 程序生成 | `--disk_texture` | 吸积盘纹理路径 |

**程序纹理细节**（极坐标直接生成，无缝周期）：

纹理直接在极坐标 `(r, φ)` 下生成，避免笛卡尔→极坐标映射的接缝问题。

1. **螺旋臂**（8-15 条）：
   - 2-3 条从圆心开始，其余从随机半径开始
   - 每条旋转 2-5 圈，宽 0.1-0.2（von Mises 分布，kappa≈1.5/width²）
   - 噪声调制宽度和强度，产生断断续续、时宽时窄的效果

2. **云雾**（30-60 条弧线叠加）：
   - 每条弧线：角度宽度 0.15-0.5，径向宽度 0.03-0.08
   - 半径按面积比例分布（`r = sqrt(uniform(0,1))`），外围更多
   - 强度 0.03-0.12，kappa=0.6，模糊而淡

3. **热点**（15-35 个弧形亮斑）：
   - von Mises 角度分布 + 高斯径向分布
   - 角度宽度 0.08-0.2，径向宽度 0.01-0.03
   - 随机位置、强度 0.4-1.0

4. **底色**：均匀低亮度背景（权重 0.1）

密度组合：`0.1*底色 + 0.3*螺旋臂 + 0.3*云雾 + 0.2*热点 + 0.1*弧形`

### 6.3 多普勒 & 引力红移（g 因子） ✅

参考 [rantonels/starless](https://github.com/rantonels/starless)（[Redshift 文档](https://rantonels.github.io/starless/)）的推导，我们用统一的 g 因子来同时处理 relativistic beaming 与 gravitational redshift：

```
g = (k · u_obs) / (k · u_em) = g_grav · g_doppler
```

实现细节：

1. **盘元四速度**：吸积盘在赤道面做 Kepler 轨道，角速度 `Ω = sqrt(M / r³)`（rs=1 ⇒ M=1/2）。
   ```
   β = r · Ω / sqrt(1 - rs/r)
   γ = 1 / sqrt(1 - β²)
   v_disk = β · (-sin φ, cos φ, 0)
   ```
2. **Doppler 因子**：令 `cosθ = dot(normalize(v_disk), -normalize(ray_dir))`，并将 `1 - β·cosθ` 钳制到 `[1e-3, +∞)`，则
   ```
   g_doppler = 1 / (γ · (1 - β · cosθ))
   ```
3. **引力红移**：静止观察者位于 `r_obs`（通常是相机到原点的距离），发射点 `r_em = |hit_xy|`，有
   ```
   g_grav = sqrt(1 - rs / r_obs) / sqrt(1 - rs / r_em)
   ```
   若观察者在无穷远，可令分子≈1。
4. **着色**：
   ```
  color = apply_wavelength_shift(texture_color, g)
  brightness *= gain · g^p / (1 + g^p / g_max)  # p≈2, gain≈0.35, g_max≈3，Reinhard 风格
   ```
   纹理仍在累加管线中（ghost images），Bloom 只作用于吸积盘层。

代码注释与本文均说明：g 因子公式引用自 starless，用于保证多普勒/红移处理与主流实现一致。

效果：朝观察者运动的一侧更亮且偏蓝，远离的一侧更暗且偏红。

### 6.4 边缘软化 ✅

吸积盘边缘（内外圈）渐变透明，避免硬边界：

```python
def compute_edge_alpha(height, inner_soft=0.1, outer_soft=0.2):
    v = np.linspace(0, 1, height)
    alpha = np.ones_like(v)
    # 内圈：vv < inner_soft -> 逐渐透明
    alpha[inner_mask] = (v[inner_mask] / inner_soft) ** 3.0
    # 外圈：vv > (1 - outer_soft) -> 逐渐透明
    alpha[outer_mask] = ((1 - v[outer_mask]) / outer_soft) ** 2.0
    return alpha
```

- 程序生成纹理和外部纹理统一使用此逻辑
- 中间区域完全不透明

### 6.5 分离式 Bloom（色散效果） ✅

模拟相机镜头遇到强光的泛光效果，只对吸积盘层做 bloom，并加入色散：

```
+------------------+     +------------------+
| Render Kernel    |     |                  |
| background ------+---->| image_field      |
| disk_layer ------+--+  +------------------+
+------------------+  |
                      v  +------------------+
                      +->| disk_layer_field |
                      |  +------------------+
                      |         |
                      |         v
                      |  +------------------+
                      +->| Bloom Kernel     |
                          | (RGB 不同半径)   |
                          +------------------+
                                   |
                                   v
                          +------------------+
                          | final = bg +     |
                          | disk + disk_bloom|
                          +------------------+
```

**色散效果**：RGB 三通道使用不同的高斯半径，模拟真实镜头的色散现象：
- 红色：σ²=25（更锐利）
- 绿色：σ²=80（中等）
- 蓝色：σ²=1600（更模糊，边缘蓝紫色光晕）

**分辨率缩放**：卷积范围和 sigma 按输出分辨率动态缩放，确保不同分辨率下视觉效果一致：
- `kernel_radius = width × 0.02`
- `sigma_scale = (width / 640)²`

**优点**：
- 只对吸积盘做 bloom，不影响背景星空
- 色散效果使光晕边缘呈现自然的蓝紫色调
- 保持星空清晰，吸积盘有真实的镜头光晕感

**参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| threshold | 0 | 高亮提取阈值（0=全部） |
| intensity | 0.4 | bloom 强度 |
| kernel_radius | width×0.02 | 卷积范围（动态缩放） |

### 6.6 吸积盘倾角 ✅

吸积盘可绕 x 轴倾斜，模拟非对称视角：

```
平面方程: z = y * tan(tilt)
```

**参数**：
| 参数 | 默认值 | CLI | 说明 |
|------|--------|-----|------|
| disk_tilt | 0° | `--disk_tilt` | 吸积盘倾角（度） |

正值表示前端翘起，负值表示后端翘起。

### 6.7 摩尔纹（Aliasing）问题 ✅

#### 问题描述

摩尔纹（Moire pattern）是黑洞吸积盘渲染中的**经典问题**，本质是**高频周期性纹理在极端引力透镜压缩下的欠采样混叠**。

吸积盘远端被黑洞引力场压缩到接近事件视界的薄环中，纹理频率远超像素奈奎斯特极限，导致高频信号失真。

#### 技术方案对比

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **Ray Differentials** | 追踪光线微分，计算纹理梯度，动态选择 LOD | 物理正确，效果最好 | 计算量大（约 3x） |
| **自适应超采样** | 高频区域增加采样密度 | 质量高 | 需要 TAA 累积多帧 |
| **预过滤纹理** | 纹理生成时预模糊高频区域 | 简单快速 | 物理上不准确 |
| **改变纹理设计** | 避免纯周期性纹理 | 根治 | 需要重新设计纹理 |

本项目采用 **Ray Differentials** 方案，参考《星际穿越》渲染团队（DNEG）的实现。

#### 实现细节

```
1. 光线初始化时，同时初始化微分状态：
   - d_pos_dx: 位置对屏幕 x 的导数（初始为 0）
   - d_dir_dx: 相邻像素方向差 = ray_dir(x+1) - ray_dir(x)（精确计算）

2. RK4 积分时同步追踪微分光线：
   - 主光线: pos, dir_
- 微分光线: d_pos_dx, d_dir_dx, d_pos_dy, d_dir_dy（X 和 Y 两个方向）
   
   微分光线的加速度需要计算雅可比矩阵：
   d(acc)/d(pos) = -1.5*L2 * (I/r^5 - 5*pos*pos^T/r^7)

3. 击中吸积盘时计算纹理梯度（X 和 Y 两个方向）：
   # X 方向
   dr/dpixel_x = (hit_x * d_pos_dx[0] + hit_y * d_pos_dx[1]) / r
   dphi/dpixel_x = (-hit_y * d_pos_dx[0] + hit_x * d_pos_dx[1]) / r²
   
   # Y 方向
   dr/dpixel_y = (hit_x * d_pos_dy[0] + hit_y * d_pos_dy[1]) / r
   dphi/dpixel_y = (-hit_y * d_pos_dy[0] + hit_x * d_pos_dy[1]) / r²
   
   dudx = dphi_x * dtex_w / (2π)
   dvdx = dr_x * dtex_h / (r_outer - r_inner)
   dudy = dphi_y * dtex_w / (2π)
   dvdy = dr_y * dtex_h / (r_outer - r_inner)

4. 根据梯度幅值选择 LOD（取两个方向的最大值）：
   grad_sq = max(dudx² + dvdx², dudy² + dvdy²)
   LOD = log₂(grad_sq) * strength
   LOD = clamp(LOD, 0, 3)
   
   说明：当 grad_sq > 1 时，纹理变化超过 1 像素，需要更高 LOD

5. 从 mipmap 金字塔采样对应层级
```

**Mipmap 生成**：
```
base_tex → 2x2 avg → level 1 → 2x2 avg → level 2 → ...
```

**参数**：
| 参数 | 默认值 | CLI | 说明 |
|------|--------|-----|------|
| anti_alias | disabled | `--anti_alias` | 算法开关：disabled/lod_radius |
| aa_strength | 1.0 | `--aa_strength` | 抗锯齿强度，>1 更模糊 |

#### 效果

- 有效减少远端吸积盘的摩尔纹
- 同时追踪 X 和 Y 方向的压缩，更全面
- 强度可调，适应不同分辨率
- 性能开销约 2x

---

## 7. 参考资料

- [JaeHyunLee94/BlackHoleRendering](https://github.com/JaeHyunLee94/BlackHoleRendering) — Python + Taichi GPU 实现，笛卡尔等效势方案 (`-1.5L²x/r⁵`)，支持 5 种积分器，含吸积盘
- [rantonels/starless](https://github.com/rantonels/starless)（[原理文档](https://rantonels.github.io/starless/)）— Python + NumPy 实现，球坐标 2D 降维 + Velocity Verlet，含吸积盘温度模型
- [ArrayFire Blog: Raytracing a Black Hole](https://arrayfire.com/blog/raytracing-a-black-hole/) — C++ + ArrayFire GPU 实现，球坐标 Christoffel 方案（数值微分），支持 Schwarzschild/Kerr，自适应 RK3(4)
- Schwarzschild, K. (1916). "Über das Gravitationsfeld eines Massenpunktes nach der Einsteinschen Theorie"

---

- v5.16 (2026-03-01): 抗锯齿增加 Y 方向微分追踪，全面检测摩尔纹；添加视频高质量编码提示

- v5.15 (2026-03-01): 新增抗锯齿功能（Ray Differentials + Mipmap LOD），添加 --anti_alias 和 --aa_strength 参数
- v5.13 (2026-03-01): 修正多普勒效应和悬臂方向；盘旋转改为顺时针（v_hat = r_hat × n）；颜色调整为蓝移偏红、红移偏蓝
- v5.12 (2026-02-28): 新增 lens flare 效果（ghost 光斑、多层环形、六边形光环、星芒，默认关闭，--lens_flare 开启）；Bloom 优化为分离式 1D 卷积，性能提升 18 倍
- v5.11 (2026-02-28): Bloom 加入色散效果（RGB 三通道不同高斯半径），模拟真实镜头的蓝紫边缘光晕
- v5.10 (2026-02-28): 吸积盘纹理改为极坐标直接生成（phi 方向无缝周期）；悬臂数量增至 8-15 条，加入噪声调制使其断断续续；云雾改为多弧线叠加（按面积比例分布）；热点改为弧形亮斑
- v5.9 (2026-02-28): 吸积盘纹理加入方位热点（azimuthal hotspot）与更强螺旋剪切，对应 g 因子自转时亮斑更加明显
- v5.8 (2026-02-28): 吸积盘程序纹理重写（多尺度湍流+螺旋剪切+温度/密度分离），盘面结构更破碎、热点更丰富
- v5.7 (2026-02-28): 重写吸积盘半径亮度加权（亮度 ∝ (1 - (r-r_inner)/(r_outer-r_inner))^p），新增 `DISK_RADIAL_BRIGHTNESS_[MIN,MAX]` 钳制，方便拉开内亮外暗对比
- v5.6 (2026-02-28): 调低吸积盘基础色温（DISK_BASE_TINT 采用暖色 1.1/0.92/0.75），默认视觉更接近日落/炽热尘埃的色调
- v5.5 (2026-02-28): 移除 NumPy 渲染路径，改为纯 Taichi；新增 g 因子亮度 tone mapping、吸积盘透明度指数增益，CLI/文档同步更新
- v5.4 (2026-02-28): 实现分离式 Bloom（只对吸积盘层泛光）、添加吸积盘倾角参数 --disk_tilt、简化 orbit 参数
- v5.3 (2026-02-27): 多普勒颜色偏移（蓝移偏蓝、红移偏红）、边缘软化公共函数（compute_edge_alpha）、降低亮度避免过曝
- v5.2 (2026-02-27): 吸积盘纹理大幅改进：FBM噪声絮状云雾、螺旋臂结构、径向丝状条纹、间隙效果、alpha合成（多次穿透衰减）、双线性插值、体积渲染（高斯密度剖面）
- v5.1 (2026-02-27): 视频模式新增 --resume 断点续传，参数变化时自动从头开始；吸积盘改进：Doppler boost 上限提升、温度剖面纹理、真实自转（开普勒角速度）、边缘软化、角度方向条纹细节
- v5.0 (2026-02-27): 合并 render.py 和 demo.py，提取 TaichiRenderer 类（kernel 一次编译），新增视频模式（--video, --orbit），默认框架改为 taichi
- v4.0 (2026-02-27): 吸积盘渲染（纹理贴图 + 光线穿透累加 + 多普勒 beaming）、自适应步长、环绕 demo 视频
- v3.1 (2026-02-27): bug 修复（Taichi max_steps 保护、逃逸半径对齐、双线性插值统一、ti.acos 边界保护），新增 starless 对比分析和提升方案
- v3.0 (2026-02-26): 添加 Taichi 框架支持，默认 fov=90°, focal=1.8，重构代码提取公共模块
- v2.1 (2026-02-26): 相机参数对齐 JaeHyunLee94 默认值 — cam_pos=(6,0,0.5), fov=60°，改为直接指定笛卡尔坐标位置
- v2.0 (2026-02-26): 重写为笛卡尔等效势方案，对齐 JaeHyunLee94 实现 — 加速度 `-1.5L²x/r⁵`，角动量守恒，无坐标变换，性能提升 3.5 倍
- v1.0 (2026-02-26): 初始版本 — 球坐标 Christoffel 符号 + RK4 积分
