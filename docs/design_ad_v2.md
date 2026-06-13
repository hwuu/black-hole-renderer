# 吸积盘 V2 设计文档

> **状态**：v2.1 修订（2026-06-13）。基于对 v1.0 已实现部分（`disk_v2/` 四个基础模块）的实物诊断，确认 v1.0 的物理场建模与显示映射存在系统性问题，会导致渲染结果"黄乎乎一片、缺乏细节"。本版保留 v1.0 的四层架构与命名规范，**重构基础物理场、结构调制、调色与实现路径**。v2.1 相对 v2.0 修正温度跨度论述、调整 `r_out` 默认值、补 parity 测试与 `F_clump` 算法决策点。
>
> **真源关系**：本文件是吸积盘 V2 的设计真源。实施排期与跟踪由 [`docs/plans/realism_uplift_plan.md`](plans/realism_uplift_plan.md) 负责。

## 目录

- [1. 背景与目标](#1-背景与目标)
  - [1.1 v1.0 留下的问题](#11-v10-留下的问题)
  - [1.2 v2.0 目标](#12-v20-目标)
  - [1.3 非目标](#13-非目标)
- [2. 设计总览](#2-设计总览)
  - [2.1 v2.0 的核心改动](#21-v20-的核心改动)
  - [2.2 五层模型](#22-五层模型)
  - [2.3 视觉验收标准](#23-视觉验收标准)
- [3. 盘体模型](#3-盘体模型)
  - [3.1 几何模型](#31-几何模型)
  - [3.2 厚度与密度场](#32-厚度与密度场)
  - [3.3 温度与发射率](#33-温度与发射率)
  - [3.4 结构调制（三维）](#34-结构调制三维)
  - [3.5 差动旋转与统一平流](#35-差动旋转与统一平流)
  - [3.6 调色与色调映射](#36-调色与色调映射)
- [4. 成像模型](#4-成像模型)
  - [4.1 整体流程](#41-整体流程)
  - [4.2 局部坐标与盘体相交](#42-局部坐标与盘体相交)
  - [4.3 有限厚度积分](#43-有限厚度积分)
- [5. 实现方案](#5-实现方案)
  - [5.1 直接 Taichi 实现](#51-直接-taichi-实现)
  - [5.2 规划文件结构](#52-规划文件结构)
  - [5.3 参数草案](#53-参数草案)
  - [5.4 性能预估](#54-性能预估)
- [6. 测试方案](#6-测试方案)
- [7. 实施顺序](#7-实施顺序)
- [8. 风险与取舍](#8-风险与取舍)
- [9. 参考资料](#9-参考资料)
- [附录 A：v1.0 实物诊断报告](#附录-av10-实物诊断报告)
- [变更记录](#变更记录)

---

## 1. 背景与目标

### 1.1 v1.0 留下的问题

v1.0 实施到 `disk_v2/{geometry, physical_fields, structure_modulations, params}.py` 四个基础模块后，用一个临时脚本调用这些模块出预览图。结果是"黄乎乎一片，没有任何细节"。

实物诊断（详见 [附录 A](#附录-av10-实物诊断报告)）确认了五条根因：

- **R1**：温度场峰值环存在，但用 v1.0 默认 `r_in=2, r_out=10` 时，未乘径向边界权重的 raw profile 动态范围仅约 1.89 倍（实算 `T_peak / T_raw(r_out) ≈ 1.89`），撑不起细节层次。完整温度场还会在精确外边界被 `W_r` 收口到 0，因此不能把边界点当作盘内有效色阶。要拉开跨度，需要同时调 `r_out` 与显示映射，靠温度量纲化本身解决不了问题。
- **R2**：密度场没有内边界压制，与温度场的内边界形态不一致，造成内缘亮度结构混乱。
- **R3**：结构调制总振幅 ±50% 以内，且全是连续平滑的傅里叶叠加，**无团块边界**。
- **R4**：垂向密度是高斯钟形，无 `(r, φ, z)` 三维结构，光线穿盘时没有"反复进入亮区"的体感。
- **R5**：没有 `palette.py`，没有"温度→黑体色"映射、没有色调映射，输出只剩单一色相。

R1 + R3 + R5 三条任何一条都能让输出变成"黄乎乎一片"，三条同时存在等于三层串联压扁。v2.0 必须同时解决这五条。

### 1.2 v2.0 目标

v2.0 在保留 v1.0 四层架构与命名规范的前提下，引入以下能力：

| 目标 | 说明 |
|------|------|
| **三维结构调制** | `F_struct` 从 `(r, φ)` 升级到 `(r, φ, z)`，新增体积噪声分量 |
| **稀疏团块项** | 新增 `clump_modulation`，作为主结构来源，与剪切和热斑互补 |
| **拉开动态范围** | 温度场采用带量纲开氏度与更大的默认外半径，配合密度场、黑体色映射和 HDR 色调映射形成层次 |
| **HDR 调色与色调映射** | 物理温度（开氏度）→ 黑体色，渲染中间结果保持 HDR，最后做色调映射 |
| **直接 Taichi 实现** | 不再走 NumPy reference → Taichi 二次迁移；最小骨架直接进 Taichi |
| **ISCO 默认约束** | `r_in ≥ 3 · r_s`（Schwarzschild 内最稳定圆轨道），低于该值给出警告 |

### 1.3 非目标

- 不引入 Kerr 黑洞、磁流体力学、辐射转移、偏振、康普顿散射。
- 不重写 Schwarzschild 测地线积分器。
- 不做实时交互。
- 不在 v2.0 内引入 V3 概念。盘体模型改动落进 `disk_v2/`；主渲染接线按 Phase 4 修改 `render.py`，不另起 V3 分支。

---

## 2. 设计总览

### 2.1 v2.0 的核心改动

以下五项是 v2.0 与 v1.0 的实质区别，称为"五项大改"。各项与诊断根因的对应见 [附录 A](#附录-av10-实物诊断报告)。

| 大改 | 改什么 | 解决根因 |
|------|--------|----------|
| **大改 1** | `F_struct` 升级到 `(r, φ, z)` 三维，引入 3D 体积噪声分量 | R4 |
| **大改 2** | 新增 `clump_modulation`：稀疏、高对比、边界清晰的团块项 | R3 |
| **大改 3** | 直接在 Taichi 主路径里实现 V2，砍掉 NumPy reference | 实施可行性 |
| **大改 4** | 补齐 `palette.py`：温度→黑体色映射 + HDR 色调映射 | R1、R5 |
| **大改 5** | 调整温度场量纲与密度场内边界，让动态范围拉开 | R1、R2 |

### 2.2 五层模型

v2.0 仍采用四层分离，但显式加入"显示映射层"作为第五层：

1. **Geometry Layer**：盘的局部坐标、内外半径、半厚度函数 `H(r)`。
2. **Physical Fields Layer**：密度 `ρ(r, z)`、温度 `T(r, z)`（带量纲）、角速度 `Ω(r)`。
3. **Structure Modulations Layer**：在 `(r, φ_adv, z)` 空间中定义体积噪声、剪切纹理、团块、热斑。
4. **Imaging Layer**：沿光线做发射-吸收积分，得到 HDR 强度。
5. **Display Mapping Layer**：温度→黑体色、HDR→LDR 色调映射。

```
+----------------+    +----------------+    +-------------------+
| Geometry Layer | -> | Physical Fields| -> | Phase Advection   |
+----------------+    +----------------+    +-------------------+
                                                     |
                                                     v
+----------------+    +----------------+    +-------------------+
| Display Mapping| <- | Path Integrate | <- | Structure Mods 3D |
+----------------+    +----------------+    +-------------------+
```

### 2.3 视觉验收标准

v2.0 满足以下硬指标视为通过，否则视为未达成：

- **三维体感**：高倾角观察下，光线穿过盘体的路径上至少经过 3 ~ 5 个亮 → 暗 → 亮振荡。
- **团块边界**：盘面正视图中存在大量边界清晰（非渐变）的局部亮块，团块尺度在 `0.1 · r_in ~ 0.5 · r_in` 之间。
- **温度物理跨度可达**：默认 `r_in=3, r_out=50` 下，未乘径向边界权重的 raw profile 温度跨度约 4 倍（`T_peak / T_raw(r_out) ≈ 4.3`）。这一跨度在黑体色映射中表现为内圈紫蓝、外圈紫白的**连续过渡**；完整温度场仍由 `W_r` 在精确边界收口。
- **色相层次显式**：内圈、中段、外圈在最终图像中有可辨色相差异。该差异主要由 cinematic palette 的色温响应曲线和 [§7 Phase 5 之后的 g-factor 颜色偏移](plans/realism_uplift_plan.md) 共同提供；温度场只负责自然的连续过渡。
- **HDR 不饱和**：默认参数下，关闭色调映射时图像会出现明显饱和；开启色调映射后细节恢复。
- **差动剪切**：动态视频中，同一团块在 `Δt` 内被内圈快速拖拽、被外圈拉长。

---

## 3. 盘体模型

### 3.1 几何模型

与 v1.0 保持一致。半厚度函数：

```
H(r) = h0 · r · (r / r_in)^beta_h
```

边界约定（v1.0 已有，继续保留）：

- `*_mask`：硬判定，闭区间 membership。
- `*_weight`：软判定，边界处收口到 0。

**v2.0 新增约束**：

- `r_in ≥ 3 · r_s` 强制校验。若用户传入更小的值，参数对象在 `__post_init__` 中发出 warning 并把值钳制为 `3 · r_s`，避免在 ISCO 内部继续应用经典薄盘公式。

### 3.2 厚度与密度场

中面密度采用径向幂律 + **内外边界双重压制**（v2.0 新增内边界项）：

```
ρ_mid(r) = (r / r_in)^(-rho_power)
         · [1 - sqrt(r_in / r)]^(1/2)    # v2.0 新增：内边界压制
         · W_r(r)
```

二维密度场保留高斯垂向轮廓作为**几何包络**：

```
ρ_envelope(r, z) = ρ_mid(r) · exp(-0.5 · (z / H(r))²) · W_z(r, z)
```

注意：v2.0 中 `ρ_envelope` 只表达盘体的**平均密度包络**，真正的"密度涨落"由 [§3.4 三维结构调制](#34-结构调制三维) 注入：

```
ρ(r, φ, z, t) = ρ_envelope(r, z) · F_struct_density(r, φ_adv, z, t)
```

`F_struct_density` 围绕 1 波动，但允许的波动幅度比 v1.0 大得多（详见 [§3.4](#34-结构调制三维)）。

**v2.0 参数默认值变化**：

- `rho_power`: `1.0` → `1.5`
- `edge_softness`: `0.1` 保持不变

### 3.3 温度与发射率

中面温度公式形式与 v1.0 一致，但**输出单位改为开氏度（K）**：

```
T_mid(r) = T_peak_K · norm_factor
         · (r / r_in)^(-3/4)
         · [1 - sqrt(r_in / r)]^(1/4)
         · W_r(r)
```

其中：

- `T_peak_K`：用户参数（默认 `1.0e7` K，典型 X 射线吸积盘内缘温度量级）。
- `norm_factor`：解析常数，使得 `max_r T_mid(r) ≈ T_peak_K`。具体值由公式在 `r ≈ 1.36 · r_in` 取峰值推得，固定为 `norm_factor = (1.36)^(3/4) / (1 - 1/sqrt(1.36))^(1/4) ≈ 4.1`。

这里的 `norm_factor` 针对**未乘径向边界权重 `W_r` 的 raw thin-disk profile** 归一化。取 `r_in=3, r_out=50` 默认值时，raw profile 从 `r_in` 稍外的峰值约 `T_peak_K` 衰减到 `r_out` 处的 `T_peak_K · 0.23`（实算 `T_raw(r_out) / T_peak ≈ 0.232`），物理跨度约 4.3 倍。完整 `T_mid(r)` 还会乘上 `W_r(r)`；按照几何边界约定，`W_r(r_out)=0`，因此精确外边界上的完整温度为 0，不能用来衡量盘内温度跨度。这个跨度不是 v1.0 留下错误说法里的"3 个数量级"，标准薄盘 `T(r) ∝ r^(-3/4)` 的衰减天然就是慢的。色相层次主要靠以下三个机制叠加提供：

- 温度 raw profile 提供的连续平滑过渡（约 4 倍跨度），完整温度场再由 `W_r` 负责边界收口。
- 黑体色查表的非线性色温响应（`1e7 K` 紫蓝、`5e6 K` 紫白、`2e6 K` 浅蓝白）。
- Cinematic palette 在物理色基础上的饱和度增强。

如果用户想看到"内缘蓝白、外圈暗红"那种夸张色阶，需要把 `r_out` 进一步加大（如 `200 r_s`，对应外缘温度 `< 1e6 K`）或显式启用 cinematic palette 的高饱和度模式。这超出 v2.0 默认配置，不在第一版承诺范围内。

垂向温度衰减保持 v1.0 形式：

```
T(r, z) = T_mid(r) · clip(1 - 0.25 · |z| / H(r), 0, 1) · W_z(r, z)
```

发射率引入"基础发射率 + 结构调制"：

```
j_base(r, z) = Cj · ρ_envelope(r, z)^α · T(r, z)^β
j(r, φ, z, t) = j_base(r, z) · F_struct_emission(r, φ_adv, z, t)
```

其中 `α`、`β` 用于在密度主导和温度主导之间平衡视觉。建议默认 `α = 1.0`、`β = 1.0`。

**v2.0 参数默认值**：

- `temp_scale` → 重命名为 `T_peak_K`，默认 `1.0e7`。
- 新增 `alpha_density`: `1.0`、`beta_temperature`: `1.0`。

### 3.4 结构调制（三维）

v2.0 的结构调制是三维场，由四个分量构成（**v1.0 的三个 → v2.0 的四个**）：

```
F_struct = F_mode · F_shear · F_clump · F_hotspot
```

也可以分别用于密度与发射率：

- `F_struct_density`：作用于密度，主要由 `F_shear` + `F_clump` 提供。
- `F_struct_emission`：作用于发射率，主要由 `F_clump` + `F_hotspot` 提供。

**分量职责（v2.0 调整）**：

| 分量 | 作用空间 | 角色 | v2.0 振幅范围 |
|------|---------|------|-------------|
| `F_mode` | `(r, φ)` 二维 | 大尺度弱不对称 | ±5% |
| `F_shear` | `(r, φ_adv)` 二维 | 差动剪切丝状结构（保留） | ±30% |
| **`F_clump`** | **`(r, φ_adv, z)` 三维** | **主结构来源：稀疏团块** | **±60%** |
| `F_hotspot` | `(r, φ_adv)` 二维 | 极亮极少数点缀 | ±20% |

**`F_clump` 的实现要点**（v2.0 新增）：

- **首版实现采用显式点云团**：在盘内随机撒 200 ~ 800 个团块中心，每个团块用锐利衰减核（如 `(1 - r_local / r_clump)^2`，r_local 是到团块中心的局部距离）。理由：参数直观可控、调试容易、容易写 parity 测试。
- Worley/Voronoi noise 作为**首版不达标时的回退方案**，详见 [§8 风险与取舍](#8-风险与取舍)。
- 每个团块有清晰的**核**（亮度 ≈ 1 + clump_strength）和**锐利衰减**（边界处亮度回到 ≈ 1）。
- 团块在 `(r, φ_adv, z)` 三维空间中分布，垂向（`z` 方向）的团块数量足以让光线穿盘时经过 3 ~ 5 次亮 → 暗振荡。
- 团块尺度：径向 `0.1 · r_in ~ 0.5 · r_in`、角向对应弧长 `0.1 · r ~ 0.3 · r`、垂向 `0.3 · H(r) ~ 0.8 · H(r)`。

**`F_shear` 的调整**（v2.0）：

- 保留多频傅里叶叠加作为剪切结构。
- 但频谱衰减改为 `amplitude ∝ 1 / k^(1/2)`（v1.0 是 `0.5^k`，衰减过快），让高频细节保留。
- `shear_components` 默认从 8 提升到 16。
- 不再在 `_normalize_signed` 里做全局归一化（这会压扁高频细节），改为按高斯分布的 3σ 截断保证振幅可控。

**约定（与 v1.0 保持一致）**：

- 各分量都是乘性调制，盘内围绕 `1` 波动，盘外返回中性值 `1`。
- 振幅参数对应分量内部的**最大振幅**，不是 RMS。

### 3.5 差动旋转与统一平流

与 v1.0 完全一致：

```
Ω(r) = omega_scale · (r / r_in)^(-3/2)
φ_adv = wrap(φ - Ω(r) · t)
```

所有动态分量（`F_shear`、`F_clump`、`F_hotspot`）统一通过 `φ_adv` 采样。`F_clump` 的体积噪声本身也通过 `(r, φ_adv, z)` 采样，保证团块也随差动旋转推进。

### 3.6 调色与色调映射

v2.0 新增的"显示映射层"。其作用是把物理量（带量纲温度、HDR 强度）转为最终的 LDR RGB。

**温度 → 颜色**：

- 复用 `render.py:136` 的 `_blackbody_rgb(T_K)` 实现（基于 Tanner Helland 色温查表）。
- 输入是真实开氏度温度，不是 0 ~ 1 归一化。
- 提供 `palette_mode`：
  - `physical`：直接黑体色，物理展示用。
  - `cinematic`：在 `physical` 基础上增强饱和度、稍提升暖色调，演示输出用。

**HDR 强度 → LDR**：

- 渲染管线**所有中间结果保持 HDR 浮点**，不在中间步骤做 `clamp(0, 10)` 一类硬截断。
- 在最终输出前做色调映射。v2.0 第一版使用 **Reinhard**：

```
rgb_ldr = rgb_hdr / (1 + rgb_hdr)
```

- 后续可切换 ACES Filmic；调色映射模块要预留切换接口。
- 色调映射之后做 sRGB 伽马校正（`x^(1/2.2)`）。

**Bloom 位置**：

- Bloom 必须在色调映射**之前**做（HDR 域），否则会丢失高动态范围的真实辉光感。

---

## 4. 成像模型

### 4.1 整体流程

```
+------------+    +-------------+    +----------------+
| World Ray  | -> | Local Frame | -> | Disk Bounds    |
+------------+    +-------------+    +----------------+
                                            |
                                            v
+------------+    +-------------+    +----------------+
| HDR Color  | <- | Integrate   | <- | Sample Fields  |
+------------+    +-------------+    +----------------+
        |
        v
+------------+    +-------------+
| Tonemap    | -> | LDR Output  |
+------------+    +-------------+
```

### 4.2 局部坐标与盘体相交

与 v1.0 设计一致：

- 用 `r_out` 与 `H_max = max_r H(r)` 构造保守包围体。
- 求出光线进入/离开包围体的参数区间 `[s_enter, s_exit]`。
- 仅在该区间内取有限采样点。

### 4.3 有限厚度积分

发射-吸收积分（与 v1.0 一致的离散形式）：

```
L_hdr = 0
tau = 0
for k in samples:
    if disk_volume_mask(sample_k):
        j_k = j_base · F_struct_emission(r_k, phi_adv_k, z_k, t)
        T_k = T(r_k, z_k)
        color_k = blackbody_rgb(T_k)             # HDR 颜色
        alpha_k = opacity_scale · density(r_k, phi_adv_k, z_k, t)
        transmittance = exp(-tau)
        L_hdr += transmittance * j_k * color_k * ds
        tau += alpha_k * ds
```

**v2.0 关键变化**：

- 颜色由**沿光线每个采样点**的局部温度决定，而不是一次性把整段路径用同一颜色着色。这样才能呈现"内缘蓝白、外侧红橙"沿视线的自然过渡。
- 输出 `L_hdr` 保留为 HDR，留待 [§3.6](#36-调色与色调映射) 的色调映射处理。

**掠射角增益**（保留 v1.0 设计）：

```
alpha_eff = alpha_k · [1 + kg · (1 - |dot(d, disk_normal)|)]
```

---

## 5. 实现方案

### 5.1 直接 Taichi 实现

v2.0 **不再走** v1.0 的"NumPy reference → 单独预览 → 再迁移 Taichi"路线。原因：

- NumPy 预览 + 黑洞光追的联合渲染估算耗时 5 ~ 10 秒/帧，调参不可行。
- 视觉验收必须包含黑洞透镜、photon ring、次像，NumPy 单独出图无法承担。
- 二次迁移会让参数和接口被调两次，浪费工作量。

新的实施路线：

1. 保留 `disk_v2/{geometry, physical_fields, structure_modulations, params}.py` 的**接口设计与单测基准**，作为参考实现。
2. 在 `render.py` 内（或新建 `disk_v2/taichi_impl.py`）用 Taichi 实现：
   - 三维体积噪声采样（`F_clump`）。
   - HDR 颜色累积与色调映射。
   - 盘体包围体求交与有限步长采样。
3. 通过 `--disk_model {v1, v2}` CLI 开关切换。
4. 仍提供 `disk_v2/preview.py`，但它只用于**单测和静态参数校验**，不承担视觉验收。

### 5.2 规划文件结构

```
black-hole-renderer/
+-- render.py                        # 主路径：相机、Schwarzschild 光追、v1/v2 切换
+-- docs/
|   +-- design.md                    # 项目主设计
|   +-- design_ad_v2.md              # 本文件
+-- disk_v2/
|   +-- __init__.py
|   +-- _array_utils.py              # 已实现
|   +-- params.py                    # 已实现，需扩展参数
|   +-- geometry.py                  # 已实现
|   +-- physical_fields.py           # 已实现，需调整温度量纲、密度内边界
|   +-- structure_modulations.py     # 已实现，需重构为 (r, φ, z) 三维 + clump
|   +-- palette.py                   # 新增：温度→颜色、HDR→LDR
|   +-- taichi_impl.py               # 新增：上述场的 Taichi 实现
|   +-- preview.py                   # 新增：静态参数校验预览
+-- tests/
    +-- unit/
        +-- test_disk_v2_array_utils.py
        +-- test_disk_v2_physical_fields.py
        +-- test_disk_v2_structure_modulations.py
        +-- test_disk_v2_clump.py            # 新增
        +-- test_disk_v2_palette.py          # 新增
        +-- test_disk_v2_advection.py        # 规划中
        +-- test_disk_v2_integrator.py       # 规划中
        +-- test_disk_v2_snapshot.py         # 规划中
```

### 5.3 参数草案

**基础盘体参数（`DiskV2Params`）**：

| 参数 | v1.0 默认 | v2.0 默认 | 说明 |
|------|---------|---------|------|
| `r_in` | 2.0 | **3.0** | 强制 ≥ 3.0（ISCO） |
| `r_out` | 10.0 | **50.0** | 加大外缘，让温度物理跨度从 1.5 倍提到 4.3 倍 |
| `h0` | 0.05 | 0.05 | 不变 |
| `beta_h` | 0.05 | 0.05 | 不变 |
| `rho_power` | 1.0 | **1.5** | 加大径向衰减 |
| **`T_peak_K`** | （不存在） | **1.0e7** | 替换原 `temp_scale`，stellar-mass BH preset |
| `omega_scale` | 1.0 | 1.0 | 不变 |
| `edge_softness` | 0.1 | 0.1 | 不变 |
| **`alpha_density`** | （不存在） | **1.0** | 发射率密度指数 |
| **`beta_temperature`** | （不存在） | **1.0** | 发射率温度指数 |

**结构调制参数（`DiskV2StructureParams`）**：

| 参数 | v1.0 默认 | v2.0 默认 | 说明 |
|------|---------|---------|------|
| `mode1_strength` | 0.03 | 0.03 | 不变 |
| `mode2_strength` | 0.05 | 0.05 | 不变 |
| `shear_strength` | 0.22 | **0.30** | 略增 |
| `shear_components` | 8 | **16** | 加密高频 |
| **`clump_strength`** | — | **0.60** | 新增：主结构振幅 |
| **`clump_count`** | — | **400** | 新增：团块数量 |
| **`clump_radial_sigma`** | — | **0.2 · r_in** | 新增：径向尺度 |
| **`clump_vertical_sigma`** | — | **0.5 · H** | 新增：垂向尺度 |
| `hotspot_strength` | 0.16 | 0.20 | 略增 |
| `hotspot_count` | 8 | 8 | 不变 |

**调色参数（`DiskV2PaletteParams`，新增）**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `palette_mode` | `"physical"` | `physical` 或 `cinematic` |
| `tonemap_mode` | `"reinhard"` | 当前只实现 Reinhard，预留 `aces` |
| `gamma` | 2.2 | sRGB 伽马 |
| `opacity_scale` | 0.5 | 有限厚度积分的不透明度缩放 |

**约束（保留并加强 v1.0）**：

- 各乘性调制必须保证盘内取值 > 0。当总振幅 > 1 时（如 `clump_strength = 0.6`），单独乘起来仍可能 > 0，但要保证 `1 + signed_value · strength > 0`。`F_clump` 的实现应用 clamp 保证。
- `r_in` 强制 ≥ 3.0。

### 5.4 性能预估

v2.0 全部在 Taichi 上做。预估基于 [`docs/design.md` 中 1080p ~2s CPU 基线](design.md) 与 Taichi GPU 加速比经验值（5 ~ 20 倍），目标机器仍是 T480s + i7 + 集显或 CPU。

| 阶段 | T480s CPU 估算 | T480s GPU 估算 |
|------|-------------|--------------|
| Phase 1：基础场 Taichi 化 | 720p 单帧 1 ~ 2s | 0.2 ~ 0.5s |
| Phase 2：三维 `F_clump` + HDR + tonemap | 720p 单帧 2 ~ 4s | 0.4 ~ 1.0s |
| Phase 3：差动旋转动画 | 每帧 + 0.1 ~ 0.3s | + 0.05s |
| Phase 4：接入主光追 | 720p 单帧 4 ~ 8s | 0.6 ~ 1.5s |

数字均为**估算**，不是实测，待 Phase 0 基线后修订。

---

## 6. 测试方案

**单元测试（保留 v1.0 + 新增）**：

- 已实现：`test_disk_v2_array_utils.py`、`test_disk_v2_physical_fields.py`、`test_disk_v2_structure_modulations.py`。
- 新增：
  - `test_disk_v2_clump.py`：团块数量分布、径向 / 角向 / 垂向尺度、振幅范围、`r_in` 内为 1。
  - `test_disk_v2_palette.py`：温度→颜色（高温偏紫蓝、低温偏浅蓝白）、tonemap 在 `[0, ∞)` 输入下输出 `[0, 1)`。
- 规划中（与 v1.0 一致）：`test_disk_v2_advection.py`、`test_disk_v2_integrator.py`、`test_disk_v2_snapshot.py`。

**v2.0 新增的硬指标测试**（覆盖 [§2.3 视觉验收](#23-视觉验收标准)）：

- `test_v2_temperature_range_default`：在默认 `r_in=3, r_out=50` 下，未乘径向边界权重的 raw profile 满足 `T_raw(r ≈ 1.36 r_in) / T_raw(r_out)` 落在 `[4.0, 4.6]` 区间内（实算值 ≈ 4.32）。这个值不是夸张目标，而是物理标准薄盘在当前半径范围内的真实跨度；测试不能直接使用完整 `T_mid(r_out)`，因为 `W_r(r_out)=0`。
- `test_v2_volumetric_oscillation`：高倾角光线穿盘路径上，`F_struct_density` 应至少出现 3 次极值。
- `test_v2_clump_boundary_sharpness`：团块从核到外缘 0.1 倍尺度内幅度跌至 50% 以下（验证"边界清晰"）。

**v2.1 新增：NumPy 与 Taichi 参考实现 parity 测试**

Phase 4 引入 `taichi_impl.py` 后，原有的 NumPy 实现（`physical_fields.py`、`structure_modulations.py`、`palette.py`）作为**参考实现**保留。视觉验收走 Taichi 路径，但单测继续测 NumPy 路径。这会产生双实现漂移风险：NumPy 单测全绿不代表 Taichi 路径正确。

为了兜住这条，必须新增 parity 测试：

- `test_disk_v2_numpy_taichi_parity.py`：
  - 固定小网格输入（例如 `r ∈ [3, 50]` 上 16 个点、`φ ∈ [0, 2π]` 上 16 个点、`z ∈ [-H, H]` 上 8 个点）。
  - 对 `density_field`、`temperature_field`、`structure_modulation`、`blackbody_color` 四个核心函数，比较 NumPy 与 Taichi 实现的输出。
  - 容差：相对误差 `< 1e-5`（除非 Taichi 端用 fp16 / fp32 而 NumPy 用 fp64，则放宽到 `< 1e-3`）。
  - 该测试在 Phase 4 落地时与 `taichi_impl.py` 一起入仓。

没有 parity 测试时，Phase 4 之后任何"出图不对"的视觉问题都会陷入"是 V2 物理不对、是 Taichi 实现写错、还是主光追集成出错"的多重不确定，调试成本爆炸。

---

## 7. 实施顺序

### 7.1 Phase 1：调整基础物理场量纲与密度内边界

实现内容：

- 温度场参数从 `temp_scale` 改名为 `T_peak_K`，默认 `1.0e7`。
- 密度场加入内边界压制项 `[1 - sqrt(r_in/r)]^(1/2)`。
- `r_in` 强制 ≥ 3.0 校验，越界发出 warning 并钳制。
- `r_out` 默认改为 50.0，让温度物理跨度从 1.5 倍提到 4.3 倍。
- `rho_power` 默认改为 1.5。

验收：

- 现有单测调整后仍通过。
- `test_v2_temperature_range_default` 通过（温度峰值/外缘比值落在 `[4.0, 4.6]`）。
- 在静态打印中，温度场的峰值位置接近 `r ≈ 1.36 · r_in = 4.08`。

### 7.2 Phase 2：三维结构调制与团块项

实现内容：

- `structure_modulations.py` 重构为 `(r, φ, z)` 三维输入。
- 新增 `F_clump`：首版采用显式点云团，参数详见 [§5.3](#53-参数草案)；Worley/Voronoi 只作为首版不达标时的回退方案。
- `F_shear` 频谱衰减改为 `1 / k^(1/2)`，分量数加到 16。
- 区分 `F_struct_density` 与 `F_struct_emission` 两种合成方式。

验收：

- `test_v2_volumetric_oscillation`、`test_v2_clump_boundary_sharpness` 通过。
- 单测覆盖团块尺度、数量与振幅范围。

### 7.3 Phase 3：调色与色调映射

实现内容：

- 新增 `disk_v2/palette.py`：温度→颜色（复用 `_blackbody_rgb`）、Reinhard tonemap、sRGB 伽马。
- 渲染管线中间结果改为 HDR 浮点。
- 移除现有 `clamp(0, 10)` 硬截断，改由色调映射兜底。
- Bloom 移到色调映射之前。

验收：

- `test_disk_v2_palette.py` 通过。
- 关闭色调映射时输出明显饱和；开启后细节恢复。

### 7.4 Phase 4：Taichi 实现与接入主光追

实现内容：

- 新增 `disk_v2/taichi_impl.py`：上述所有场的 Taichi 版本。
- 同步新增 `tests/unit/test_disk_v2_numpy_taichi_parity.py`，覆盖 `density_field`、`temperature_field`、`structure_modulation`、`blackbody_color` 四个核心函数。
- 在 `render.py` 加入 `--disk_model {v1, v2}` 开关。
- 命中盘体后调用 V2 采样器与有限厚度积分。
- 端到端回归测试更新。

验收：

- V1 默认输出哈希不变。
- `test_disk_v2_numpy_taichi_parity` 通过（容差见 [§6](#6-测试方案)）。
- V2 在三组参考相机下（正视、倾视、侧视）输出符合 [§2.3 视觉验收](#23-视觉验收标准)。

---

## 8. 风险与取舍

| 风险 | 说明 | 应对 |
|------|------|------|
| **团块算法决策** | Worley/Voronoi 与显式点云团两种实现各有优势：Worley 在 Taichi 中天然支持周期边界，但参数不直观；显式点云团参数直观可控、容易调，但团块边界需要手工设计衰减核 | **Phase 2 开始前必须决策只选一种作为首版**，避免同时实现两套违反"每步只专注一件事"流程。当前 reference 建议从**显式点云团**起步（参数可控性强、调试容易），Worley 留作首版不达标后的回退方案 |
| **HDR 调参成本** | Reinhard 简单但容易整体偏灰；可能需要换 ACES | 第一版用 Reinhard；预留切换接口；Phase 4 后评估 |
| **Taichi 兼容性** | Worley/3D noise 在 Taichi 中实现可能受限 | 必要时回退为查表预生成的体素数据 |
| **`T_peak_K = 1e7` 偏热** | 真实 stellar-mass BH 吸积盘内缘 `~1e7 K`，supermassive BH 内缘 `~1e5 K`；用户也许希望 supermassive 视觉 | `T_peak_K` 是用户参数，提供 preset |
| **接入主光追后耗时上升** | 三维体积采样比二维贵 | 用 Taichi GPU 跑；提供低质量预览档（4 步采样） |

v2.0 的核心取舍：

- 放弃 v1.0 的 NumPy reference 路线，换更快的视觉验收闭环。
- 放弃 v1.0 "F_struct 是平滑场"的简化，换"团块边界清晰"的真实感。
- 引入物理温度量纲，换稳定的 HDR 调色基础。

---

## 9. 参考资料

- `docs/design.md`：项目主设计文档。
- `docs/design_accretion_disk.md`：早期吸积盘设计与背景整理。
- `docs/plans/realism_uplift_plan.md`：v2.0 的实施排期与跟踪。
- Shakura, N. I., & Sunyaev, R. A. (1973). Black holes in binary systems. Observational appearance.
- Novikov, I. D., & Thorne, K. S. (1973). Astrophysics of black holes.
- Worley, S. (1996). A cellular texture basis function. SIGGRAPH 1996.
- Reinhard, E. et al. (2002). Photographic tone reproduction for digital images.

---

## 附录 A：v1.0 实物诊断报告

v1.0 已实现的四个基础模块在用临时脚本预览时输出"黄乎乎一片、无细节"。以下是代码级诊断。

### A.1 诊断方法

- 不依赖临时脚本（已遗失），直接从 `disk_v2/` 现有代码推断各场的实际取值范围与形态。
- 数值推演基于 v1.0 默认参数（[`disk_v2/params.py`](../disk_v2/params.py)）。

### A.2 五条根因

#### R1：温度场动态范围只有 1.9 倍

未乘径向边界权重的 raw temperature profile `T_raw(r) = (r / r_in)^(-3/4) · [1 - sqrt(r_in/r)]^(1/4)` 在 v1.0 默认 `r_in=2, r_out=10` 上的实算值（完整实现位于 [`disk_v2/physical_fields.py:95`](../disk_v2/physical_fields.py)，随后还会乘 `W_r`）：

| r | T_raw / T_peak |
|---|---|
| 2.0 | 0.000 |
| 2.72 (= 1.36 r_in，峰值) | 1.000 |
| 4.0 | 0.897 |
| 6.0 | 0.725 |
| 10.0 | 0.529 |

除去内边界 `r = r_in` 的 0 起点，raw profile 在当前半径范围内只有 **1.89 倍**（`1.0 / 0.529`）。完整温度场再乘 `W_r` 后，会在精确内外边界收口到 0；这个边界 0 值不能被拿来当作盘内有效动态范围。结论是：v1.0 的盘内有效温度跨度太窄，**无法支撑细节层次**。

要注意：温度场量纲化（从 0~1 改为开氏度）**本身解决不了动态范围问题**——公式的形状决定了在窄 `r_out / r_in` 比例下温度天然就是窄分布。v2.0 的解决方案是同时：

- 把 `r_out` 默认从 10 提到 50（跨度提升到约 4.3 倍）。
- 引入 `palette.py` 的非线性黑体色查表与 cinematic palette。
- 用 `g-factor` 颜色偏移在 Phase 5 之后进一步增加方向性色阶。

#### R2：密度场没有内边界压制

`midplane_density_field` 在 `r = r_in` 处取 1，不经过 0 起步。温度场和密度场的内边界形态不一致，发射率 `ρ^α · T^β` 在内缘附近出现"有密度无温度"的诡异暗带。代码位置：[`disk_v2/physical_fields.py:78`](../disk_v2/physical_fields.py)。

#### R3：结构调制总振幅 ±50% 以内，且全是连续光滑场

- `shear_strength = 0.22`、`hotspot_strength = 0.16`、`mode1+mode2 = 0.08`（[`disk_v2/params.py:97`](../disk_v2/params.py)）。
- `shear_modulation` 用 `amplitude = 0.5^k` 衰减，第 4 个分量只剩 1/16 振幅（[`disk_v2/structure_modulations.py:194`](../disk_v2/structure_modulations.py)）。频谱衰减太快。
- 全部分量是傅里叶余弦/高斯叠加，**连续可微，无团块边界**。
- 三个分量乘起来盘内最大对比度约 1.6 倍。

#### R4：垂向只有高斯钟形，无 3D 体感

`density_field` 沿 z 方向单调钟形（[`disk_v2/physical_fields.py:157`](../disk_v2/physical_fields.py)）。光线穿盘时密度是"低 → 高 → 低"一次，**不可能出现"亮 → 暗 → 亮 → 暗"的多次振荡**。这是云雾粒子感的本质缺失。

#### R5：没有调色层，没有色调映射

- `disk_v2/` 缺少 `palette.py`。
- 临时预览脚本（已遗失）大概率用了 matplotlib 的 cmap 直接上色，把 2~4 倍动态范围的灰阶直接映射到 cmap 的橘黄段。
- 没有 HDR 浮点保留、没有 Reinhard / ACES 一类色调映射。
- 没有"温度→黑体色"的物理映射。

### A.3 根因 → 大改映射

| 根因 | 解决方案 | 在本文档位置 |
|------|---------|------------|
| R1 | 温度量纲化 + `r_out` 默认 50（跨度从 1.89 倍提到 4.32 倍） + 非线性 palette | [§3.3](#33-温度与发射率)、[§3.6](#36-调色与色调映射)、[§5.3](#53-参数草案) |
| R2 | 密度场加内边界压制项 | [§3.2](#32-厚度与密度场) |
| R3 | `F_shear` 频谱调整 + 新增 `F_clump`（高对比、边界清晰） | [§3.4](#34-结构调制三维) |
| R4 | `F_struct` 升级为 `(r, φ, z)` 三维 | [§3.4](#34-结构调制三维) |
| R5 | 新增 `palette.py`：温度→黑体色 + HDR 色调映射 | [§3.6](#36-调色与色调映射) |

---

## 变更记录

- **v2.1 (2026-06-13)**：修正 v2.0 的温度跨度论述硬伤——v2.0 错误地声称"跨 3 个数量级"，实算 SS 公式 `r_in=3, r_out=10` 跨度仅 1.47 倍。本版采用实算结果：把 `r_out` 默认从 10 提到 50，物理跨度提升到 4.32 倍；同步修正 §1.1 R1 描述、§2.3 视觉验收（色相层次主要由 palette + g-factor 贡献，不再夸大温度跨度作用）、§3.3 温度场说明、§5.3 参数草案、§6 测试方案（替换不可能的 `T_max/T_min ≥ 100` 指标为基于实算的 `[4.0, 4.6]` 区间检查）、附录 A R1 实算数据。新增 §6 NumPy/Taichi parity 测试要求与对应单测；§8 把 `F_clump` 算法决策从"同时实现两种"改为"Phase 2 开始前必选一种、显式点云团为首选、Worley 作为回退"；§3.4 同步显式点云团为首版实现。Phase 1 验收指标更新。
- **v2.0 (2026-06-13)**：基于 v1.0 实物诊断的重大修订。新增五项大改（三维 F_struct、clump 项、Taichi 直接实现、palette + HDR、温度密度动态范围）。引入第五层"显示映射层"。新增 §3.6 调色与色调映射、§5.1 直接 Taichi 实现、§5.3 参数草案 v2.0 默认值、附录 A 诊断报告。`r_in` 强制 ≥ 3 · r_s。删除 v1.0 NumPy reference 路线相关描述。
- v1.0 (2026-03-07)：初版 V2 设计文档；明确平行实现策略、2.5D 有限厚度模型、统一平流约束、发射-吸收积分与分阶段实施计划。
