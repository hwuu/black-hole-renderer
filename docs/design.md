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
- **双重框架**：支持 NumPy（compact 算法）和 Taichi（while_loop 算法）
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

### 2.3 守恒量

角动量平方 `L² = |v × x|²` 在整个积分过程中守恒，只需在初始化时计算一次。这是因为 Schwarzschild 时空具有球对称性，角动量是运动积分。

---

## 3. 渲染管线

### 3.1 整体流程

两种渲染框架：

**NumPy 框架（compact 算法）**：
```
+----------+  +----------+  +----------+  +----------+
| Camera   |->| Skybox   |->| RK4      |->| Classify |->| Shade   |
| Generate |  | Load/Gen |  | Compact  |  | Rays     |  | Escape  |
| Rays     |  |          |  | Batch    |  |          |  | Capture |
+----------+  +----------+  +----------+  +----------+
```

**Taichi 框架（while_loop 算法）**：
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

| 算法 | 框架 | 描述 |
|------|------|------|
| **compact** | NumPy | 向量化批量处理，每步移除终止光线 |
| **while_loop** | Taichi | 每条光线独立 while 循环，并行执行 |

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
| step > r_escape×20/dt | 超时 | 视为 escaped |

默认 `r_max = 10`，`dt = 0.5`（可通过参数调整）

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
| NumPy | compact | CPU 开发调试 | 慢 (~10s for 640×360) |
| Taichi | while_loop | 正式渲染 | 快 (~0.1s for 640×360) |

**NumPy compact 策略**：向量化批量处理，每步检测并移除终止光线。

**Taichi while_loop**：每条光线独立 while 循环，Taichi 自动并行化（CPU/GPU）。

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
| `--framework`, `-f` | 渲染框架: numpy/taichi | **taichi** |
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
| 并行框架 | 纯 NumPy | NumPy + Taichi (CPU/GPU) |

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

基于径向距离的启发式步长调整：

```
dt = dt_base · min(r / rs, 10)
```

- 远场 (r >> rs)：步长放大至 10×，加速逃逸
- 近场 (r ~ rs)：步长缩小，提高精度
- 实测：固定步长 272 步 → 自适应 101 步，渲染时间减少 ~60%

### 6.2 吸积盘渲染 ✅

**对齐 JaeHyunLee94 实现**：

- 赤道面穿越检测：`z_old * z_new < 0`，线性插值到 z=0 平面
- 纹理贴图：柱面 UV 映射（`u = φ/2π`, `v = (r-r1)/(r2-r1)`），支持外部纹理或程序生成
- 光线穿透：命中吸积盘后光线继续传播，多次命中累加颜色（ghost images）
- 加法混合：`final = clamp(background + Σ disk_hits, 0, 1)`

```
每步积分后:
  if z_old * z_new < 0:                        # 穿越赤道面
    t = -z_old / (z_new - z_old)               # 线性插值
    hit_xy = pos_old + t * (pos_new - pos_old)  # 交点
    r_hit = |hit_xy|
    if r_inner <= r_hit <= r_outer:
      color = sample_disk_texture(hit_x, hit_y)  # 纹理采样
      color *= doppler_boost(hit_xy, ray_dir)     # 多普勒 beaming
      accumulated_color += color                   # 累加，光线继续
```

**关键参数**（对齐 JaeHyunLee94）：

| 参数 | 默认值 | CLI | 说明 |
|------|--------|-----|------|
| r_inner | 2.0 rs | `--ar1` | 吸积盘内半径 |
| r_outer | 3.5 rs | `--ar2` | 吸积盘外半径 |
| disk_texture | 程序生成 | `--disk_texture` | 吸积盘纹理路径 |

### 6.3 多普勒 Beaming ✅

吸积盘物质沿开普勒轨道旋转，产生相对论多普勒效应：

```
v_orbital = sqrt(M/r) = sqrt(1/(2r))        # 开普勒速度
v_disk = v_orbital · (-sin φ, cos φ, 0)     # 逆时针轨道方向
doppler = 1 / (1 - v_disk · ray_dir)        # 多普勒因子
brightness *= clamp(doppler^1.0, 0, 1.5) * 0.6  # 亮度调制
```

**颜色偏移**（蓝移偏蓝、红移偏红）：

```
shift = doppler - 1.0
r = color.r * (1 - shift * 0.6)      # 蓝移时红减少
g = color.g * (1 - |shift| * 0.3)    # 中间色影响较小
b = color.b * (1 + shift * 0.4)      # 蓝移时蓝增加
```

效果：朝观察者运动的一侧更亮且偏蓝，远离的一侧更暗且偏红。

### 6.4 边缘软化 ✅

吸积盘边缘（内外圈）渐变透明，避免硬边界：

```python
def compute_edge_alpha(height, inner_soft=0.1, outer_soft=0.1):
    # 内圈：vv < inner_soft -> 逐渐透明
    alpha[inner_mask] = (v[inner_mask] / inner_soft) ** 3.0
    # 外圈：vv > (1 - outer_soft) -> 逐渐透明
    alpha[outer_mask] = ((1 - v[outer_mask]) / outer_soft) ** 5.0
```

- 程序生成纹理和外部纹理统一使用此逻辑
- 中间区域完全不透明

### 6.5 分离式 Bloom ✅

模拟相机镜头遇到强光的泛光效果，只对吸积盘层做 bloom：

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
                         | (disk only)      |
                         +------------------+
                                  |
                                  v
                         +------------------+
                         | final = bg +     |
                         | disk + disk_bloom|
                         +------------------+
```

**优点**：
- 只对吸积盘做 bloom，不影响背景星空
- 保持星空清晰，吸积盘有自然的光晕

**参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| threshold | 0 | 高亮提取阈值（0=全部） |
| intensity | 0.1 | bloom 强度 |
| blur_sigma | 64 | 高斯模糊半径 |

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

---

## 7. 参考资料

- [JaeHyunLee94/BlackHoleRendering](https://github.com/JaeHyunLee94/BlackHoleRendering) — Python + Taichi GPU 实现，笛卡尔等效势方案 (`-1.5L²x/r⁵`)，支持 5 种积分器，含吸积盘
- [rantonels/starless](https://github.com/rantonels/starless)（[原理文档](https://rantonels.github.io/starless/)）— Python + NumPy 实现，球坐标 2D 降维 + Velocity Verlet，含吸积盘温度模型
- [ArrayFire Blog: Raytracing a Black Hole](https://arrayfire.com/blog/raytracing-a-black-hole/) — C++ + ArrayFire GPU 实现，球坐标 Christoffel 方案（数值微分），支持 Schwarzschild/Kerr，自适应 RK3(4)
- Schwarzschild, K. (1916). "Über das Gravitationsfeld eines Massenpunktes nach der Einsteinschen Theorie"

---

## 变更记录

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
