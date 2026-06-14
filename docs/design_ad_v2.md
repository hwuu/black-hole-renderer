# 吸积盘 V2 设计文档

> **状态**：v2.2 修订（2026-06-14）。基于 V2 visual recovery 的实物反馈，确认“visual atlas 主导亮度/色温/alpha”的分支会绕过物理主链，导致外缘硬边、灰盘、细节生硬和长视频不可持续。本版冻结新的成像契约：V2 默认走 `support -> rho/T/tau -> F_phys -> g-factor -> HDR/tonemap` 主链，cinematic 只改变显示映射和受限扰动；visual atlas 降级为 bounded turbulence 输入，不再接管主亮度、主色温或 alpha。
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
  - [2.4 v2.2 成像契约](#24-v22-成像契约)
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

> **2026-06-14 视觉恢复修订**：最终验收以用户提供的 Interstellar 风格参考图为标准（完整横贯盘面、上下透镜弧、内缘细亮环、沿轨道方向的细丝烟雾、暖白/金/褐色带、暗蓝黑背景与柔和 Bloom）。固定验收相机：`pov=24 0 8, fov=90, ar1=2, ar2=15, disk_tilt=20`，沿原 `6 0 2` 方向后撤到远景以保留吸积盘边缘。运行 `scripts/v2_visual_acceptance.sh` 生成对照图。

v2.1 满足以下指标视为通过（**删除 v2.0「团块边界清晰」硬指标**）：

- **三维体感**：高倾角观察下，光线穿过盘体的路径上至少经过 3 ~ 5 个亮 → 暗 → 亮振荡。
- **主结构为絮状/细丝云雾**：盘面沿轨道方向拉伸的细丝/烟雾纹理，**无大块孤立圆斑、鬣狗斑或条带/斑马纹**；v2.2 起主亮度与透明度来自物理主链，visual atlas 仅可作为 bounded turbulence 扰动输入，不能直接接管主亮度、主色温或 alpha。
- **温度物理跨度可达**：默认 `r_in=3, r_out=50` 下，未乘径向边界权重的 raw profile 温度跨度约 4 倍（`T_peak / T_raw(r_out) ≈ 4.3`）。验收构图使用 `ar1=2, ar2=15` 时跨度较小，色相层次主要由 cinematic palette 与 g-factor 贡献。
- **色相层次显式**：内圈、中段、外圈在最终图像中有可辨暖白 / 金黄 / 褐黑过渡；不能灰脏，也不能红黄白靶盘。
- **HDR 不饱和**：默认参数 + auto exposure 下，LDR 保留中间调纹理；`white_ratio` 仅作诊断，**必须人工看图**。
- **差动剪切**：动态视频中，结构随差动旋转推进（atlas 静态首版；动态平流后续迭代）。

### 2.4 v2.2 成像契约

v2.2 的核心目标是把 cinematic 输出重新约束到物理主链上。渲染、测试和文档统一采用以下语义：

```
support(r) -> rho_raw_mid(r), T_raw_mid(r), tau_effective(r)
           -> F_phys(r)
           -> g-factor
           -> exposure / tonemap / bloom
```

**物理场域 reference**：

```
tau_effective(r) = opacity_scale · rho_raw_mid(r) · H(r)
F_phys(r) = W_r(r) · tau_effective(r) · [T_raw_mid(r) / T_peak_K]^4
```

注意：`rho_raw_mid(r)` 与 `T_raw_mid(r)` 不包含边界 support，`F_phys` 最终只乘一次 `W_r(r)`，避免边界处变成 `W_r^5`。当 `tau_effective << 1` 时，该模型解释为 optically-thin effective opacity；只有当 `tau_effective ≈ 1` 时，才可近似称为 thin-disk surface/photosphere 发射。

**cinematic 曝光 reference**：

曝光基准定义在物理场域，而不是屏幕空间。reference 通量与体积渲染量纲一致（D3 物理一致性已建立）：

```
F_ref = p99_r(F_phys_volume(r))   # 沿 z 方向解析积分，含密度高斯、温度衰减与 W_z
F_phys_volume(r) = W_r(r) · ∫ opacity_scale · rho_raw(r) · exp[-0.5(z/H)^2] · W_z(z)
                            · [T_raw(r) · V_T(z/H) · W_z(z) / T_peak]^4 dz
```

注意 W_z 在密度部分也出现一次——与 NumPy `density_field(r,z) = ρ_mid·exp·W_z` 一致。
Taichi 端 `sample_emission(r, phi, z)` 是真 z 函数：

```
j(r, z) = support · opacity · ρ_envelope(r, z) · (T(r,z) / T_peak)^4 · F_struct
ρ_envelope(r, z) = ρ_raw · exp(-0.5 (z/H)^2) · W_z(z)
T(r, z)         = T_raw · V_T(|z|/H) · W_z(z)
```

`∫ j(r, z) dz`（face-on）严格等于 `F_phys_volume(r)`——量纲一致性由
`test_parity_volume_emission_integral` 保护，相对误差 < 5%。

选择 `exposure` 使 `reinhard(exposure · F_ref) ≈ 0.7`；renderer 内
`white_point = 1 / exposure`。face-on 参考相机只用于校验该量级在图像上的
显示效果，不参与定义 `F_ref`，避免曝光随相机/FOV/分辨率漂移。

**reference 与 actual HDR 的 fallback 窗口（D3）**：

`reference_exposure` 是相机无关的物理量，与实际渲染 HDR 之间存在自然偏差（g-factor、disk_tilt、cinematic palette、transmittance 累积、盘体在屏幕上占比导致的 percentile 取值差异）。renderer 通过 `_compute_white_point` 在 reference 与 HDR p{white_point_percentile} 之间选择：

| ratio = actual_hdr_p{n} / reference_wp | 行为 |
|---|---|
| `[0.1, 10]` | 使用 reference（trusted 窗口，曝光基线稳定） |
| `[0.01, 100]` 但越出 trusted | 仍使用 reference，打印 warning 提示检查物理参数 |
| 超出 `[0.01, 100]` | 完全 fallback 到 HDR p{n}，避免单帧黑掉或全白 |

注：`n = white_point_percentile`，CLI 默认 99，`interstellar` preset 改为 96。
术语统一为 `actual_hdr_p{n}`，避免 p99/p96/p95 混用。

D3 是**策略选择**而非"量纲严格一致"：reference 量纲与 ray-march 路径一致
（由 `test_parity_volume_emission_integral` 保护），但相机、FOV、构图等
"画面层面"差异让 ratio 偏离 1 是预期的。"宁可用 reference 也不切走"是为
跨场景曝光基线稳定，而非反映物理精确性。

**温度显示链**：

物理温度 `T_peak_K≈1e7K` 不直接送入 LDR 黑体色。cinematic 模式先做 log 映射：

```
T_visible = log_map(T_phys, T_outer, T_peak, Tvis_min, Tvis_max)
```

g-factor 的颜色偏移作用在可见色温上：

```
T_visible_obs = clamp(g · T_visible_em, Tvis_min, Tvis_max)
```

这是 band-limited cinematic 显示近似；亮度仍由 `g^4 · F_phys` 的相对强度链负责，二者不能混成两套独立美术补丁。

**模式语义**：

- `physical`：诊断/验证用，弱扰动、保守曝光。硬性剖面指标在该模式或无暖色偏移的 log-T display 链上验证。
- `cinematic`：同一物理主链 + 可见色温映射、曝光、bloom、受限扰动。人工视觉验收看该模式。
- `interstellar` preset：v2.2 起必须是“统一物理路径 + cinematic 参数组合”，不能依赖已删除或停用的 visual 美术分支。

**薄层快速路径**：

`use_visual_atlas=True` 时，当前 Taichi renderer 仍保留倾斜中面单次命中的 thin-layer
快速路径，用于单帧视觉验收和 atlas 过渡阶段。该路径不是完整有限厚度体积积分；它必须调用
同一套 `j/rho/T/g/support` helper，并进入同一 HDR/tonemap 链。`use_visual_atlas=False`
时走有限厚度体积积分。

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

v2.2 起，默认主发射率改为 effective thin-disk emission。`ρ^α·T^β`
保留为 optically-thin 体积诊断路径，不再作为 cinematic 默认主亮度：

```
tau_effective(r) = opacity_scale · rho_raw_mid(r) · H(r)
F_phys(r) = W_r(r) · tau_effective(r) · [T_raw_mid(r) / T_peak_K]^4
j_surface(r, φ, t) = F_phys(r) · F_turb_bounded(r, φ, t)
```

其中 `F_turb_bounded` 只能在有限范围内围绕 1 扰动，不能改变径向平均亮度主趋势。
旧的体积诊断路径仍可写为：

```
j_base(r, z) = Cj · ρ_envelope(r, z)^α · T(r, z)^β
j(r, φ, z, t) = j_base(r, z) · F_struct_emission(r, φ_adv, z, t)
```

其中 `α`、`β` 用于诊断 optically-thin 体积发射，不作为 v2.2 cinematic 默认调参入口。

**v2.0 参数默认值**：

- `temp_scale` → 重命名为 `T_peak_K`，默认 `1.0e7`。
- 新增 `alpha_density`: `1.0`、`beta_temperature`: `1.0`。

### 3.4 结构调制（三维）

> **2026-06-14 视觉恢复历史基线**：v2.1 visual recovery 曾把主发射/密度结构改由预烘焙 **visual atlas** 提供；`F_clump` 仅弱体积自遮挡；`F_shear` 默认关闭（`shear_strength=0`）。v2.2 已判定该路径会绕过物理主链，因此下述 v2.1 公式只作为历史基线和回归对照，不再作为新默认成像契约。

v2.1 的结构调制曾由 atlas、弱模态、弱 clump 与热斑构成：

```
# 视觉 atlas 开启时（默认）
j ∝ ρ_envelope^α · T^β · emission_atlas(r, φ) · F_mode · F_hotspot
ρ ∝ ρ_envelope · density_atlas(r, φ) · F_clump_weak

# atlas 关闭时（回退）
F_struct = F_mode · F_shear · F_clump · F_hotspot
```

**v2.1 分量职责（历史基线）**：

| 分量 | 作用空间 | 角色 | 默认振幅 |
|------|---------|------|-------------|
| **`F_visual_atlas`** | **`(r, φ)` 二维预烘焙** | **主结构：V1 云雾 + spiral warp + alpha clip** | turbulence ~±35% |
| `F_mode` | `(r, φ)` 二维 | 大尺度弱不对称 | ±5% |
| `F_shear` | `(r, φ_adv)` 二维 | 傅里叶剪切（**默认关闭**） | 0 |
| `F_clump` | `(r, φ_adv, z)` 三维 | **弱体积自遮挡**，不进发射 | ~±12% |
| `F_hotspot` | `(r, φ_adv)` 二维 | 极亮极少数点缀 | ±20% |

**`F_visual_atlas` 的实现要点**（`disk_v2/visual_atlas.py`）：

- 复用 V1 多层 tileable noise + Kepler shear 烘焙到 `(r, φ)` atlas。
- Blender 字幕思路：径向 gradient 驱动 φ 方向 spiral warp；Alpha Clip 去掉弱发光灰雾。
- `emission_weight` 驱动暖色云雾主视觉；`density_weight` 强度较弱，负责体积遮挡。
- Taichi 侧双线性查表，不改变测地线积分路径。

**`F_clump` 的调整**（视觉恢复后）：

- 保留显式点云团实现，但 **默认 `clump_emission_weight=0`**，仅经密度路径提供弱自遮挡。
- 不再作为主发射纹理来源；避免出现大块孤立亮斑（鬣狗斑）。

**`F_shear` 的调整**（视觉恢复后）：

- 默认 `shear_strength=0`，避免与 atlas 叠加产生斑马纹。
- 频谱范围回滚至稳定区间（`phi: 2~14`, `log_r: 1~8`）。

**v2.2 分量职责（新默认）**：

| 分量 | 作用空间 | v2.2 角色 | 默认约束 |
|------|---------|-----------|---------|
| `F_turb_bounded` | `(r, φ[, t])` | 有界湍流扰动，只调制 `j/tau/T` 的小幅偏差 | 不改变径向平均亮度主趋势 |
| `F_tau` | `(r, φ, z[, t])` | 光学厚度 / 密度扰动，产生暗丝、烟雾和遮挡层次 | 与 `ρ` 同号或受约束相关 |
| `F_T` | `(r, φ[, t])` | 小幅温度扰动，影响可见色温与 `T^4` 发射 | 典型限制在 `[0.85, 1.25]` |
| `F_clump` | `(r, φ, z)` | 历史弱自遮挡项；后续可被有限寿命 eddy 取代 | 不作为主发射结构 |
| `F_hotspot` | `(r, φ)` | cinematic 稀疏点缀，需受物理发射链约束 | 不覆盖 `F_phys` 主趋势 |

长期视频不得使用固定 visual atlas 做 `φ - Ω(r)t` 的持续剪切；有限寿命 eddy 属于后续独立里程碑。

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

- **v2.2.4 (2026-06-14)**：修 C：bloom 默认参数重校。D2+D3+路径 B 之后 HDR max ≈ 0.025，旧 `interstellar` preset 默认 `bloom_threshold=0.15` 直接过滤掉所有 bloom 信号，`v2_acceptance_bloom.png` 与 `v2_acceptance_no_bloom.png` 字节相同。修 C 把 preset 默认调到 `bloom_threshold=5e-4`（约 HDR p99.5 量级）、`bloom_intensity=1.5`、`bloom_radius=8.0`——bloom 实际起作用：mean +5%、p99 +3%、>0.7 高亮像素 +36%。同时修复 preset 偷改 bug：`v2_bloom_*` 三个 CLI 默认值改为 `None`，preset 用 `is None` 判定"用户未指定"，让用户显式 `--v2_bloom_intensity 0` 仍能覆盖 preset 默认。新增 4 个单测覆盖 bloom 配方与覆盖逻辑。已知遗留（v2.2.3）：主验收 ratio ≈ 0.02 仍在 warn 区下端；这是 reference 与"屏幕 p{n}"两种量的本质差异，下一轮可考虑 face-on smoke 校准。
- **v2.2.3 (2026-06-14)**：D3 reference 一致性与测试漏洞修复（gpt 5.5 review 后）。**路径 B 物理修正**：Taichi `sample_emission(r, phi, z)` 之前接受 z 但完全不用，违反 thin disk emissivity 真实物理；本版改为真 z 函数，沿 z 高斯衰减 + 垂向温度衰减 + W_z 收口，与 NumPy `density_field/temperature_field` 一致。NumPy `physical_baseline_volume_flux` 之前在 vertical_density 项漏乘 W_z（少乘 1 次），同步补上。新增 `test_parity_volume_emission_integral`：在结构调制全部关闭的"纯物理基线"配置下，face-on Taichi 沿 z 数值积分与 NumPy reference 量纲一致（相对误差 < 5%）。**注意**：这条只证明"reference 与 sample_emission 的物理公式已对齐"；实际渲染叠加 g-factor、cinematic palette、transmittance 累积、构图相关 percentile 取值后，actual HDR p{n} 与 reference 之间仍会偏离一个数量级，由 `_compute_white_point` 的策略性 trusted/warn 窗口吸收。**`RenderStats` 暴露 `actual_hdr_white_point` 与 `white_point_percentile`**：之前测试只测 `used_wp == ref_wp`（fallback 后总成立、自证），现在测 `actual_hdr_wp / ref_wp` 落入 trusted/warn 窗口的真实物理 ratio。`_compute_white_point` 返回元组 `(used_wp, actual_hdr_wp)`，stats summary 显示 `actual_hdr_wp (p{n})` 让 percentile 口径明确。D3 fallback 测试改为通过 `resolve_v2_render_options(interstellar preset)` 触发主验收，不再单独构造 renderer 绕过 preset。design §2.4 说明 D3 是**策略选择**而非"actual 与 reference 已严格匹配"，相机/FOV/构图层面差异让 ratio 偏离 1 是预期的。已知遗留（不阻塞当前分支）：主验收实测 ratio ≈ 0.02 落在 warn 区下端，说明 reference 曝光基线比实际画面强一个量级；这不是"配方错误"，而是 reference 的物理量纲与"屏幕 p{n}"本质上是两个量。下一轮做 Bloom 阈值/强度重校（修 C）时一并评估是否需要 face-on smoke 校准。
- **v2.2.2 (2026-06-14)**：修 A + 修 B。修 A：acceptance 主验收相机距离从 `pov="30 0 10"` (≈ 31.6 r_s) 拉到 `pov="95 0 32"` (≈ 100.2 r_s)，让 `r_out=50` 的盘外缘完整可见、画面占比 30%~46%；保持 ~18.6° 仰角与 `disk_tilt=20°` 接近。修 B：撤销 `interstellar` preset 把 `lum_power` 从 4.0 降到 2.5 的偷改，让 plan Step 4 line 176 严格 g^4 物理在 cinematic 主验收中保持成立。D3 后曝光 reference 物理可控、HDR 由 Reinhard 自然压缩，不再需要靠降 `lum_power` 来防止单帧饱和（之前 2.5 让多普勒视觉显著性减少约 14 倍：`6^4 / 6^2.5 ≈ 14.7`）。配套 5 个新单测（`test_v2_render_options`），ACCEPT_POV 期望值更新。已知遗留：volume 路径下 HDR max ≈ 0.03，远低于 `interstellar` preset 的 `bloom_threshold=0.15`，bloom 在主验收下不再触发；这是 preset 与新量纲对不齐的副作用，留待下一轮 bloom 默认值重校（plan Step 8 范围）。
- **v2.2.1 (2026-06-14)**：D2 + D3 落地。D2：acceptance 主验收参数从 v1 兼容范围（`ar1=2, ar2=15, pov="24 0 8"`）切换到 v2.2 推荐值（`ar1=3, ar2=50, pov="30 0 10"`），让标准薄盘温度跨度从约 1.5 倍提到 4.32 倍，色相层次承诺可达成；小盘参数保留为独立的 `v2_compat_small_disk.png` 对照。D3：实测 D2 后 ratio = actual_hdr_p99 / reference_white_point ≈ 4.25，落在 reasonable 物理偏差范围内，原 `[0.3, 3.0]` fallback 窗口判定过严反而切走 reference；现把 trusted 窗口放宽到 `[0.1, 10]`、warn 窗口扩展到 `[0.01, 100]`，warn 区仍优先使用 reference 并打印诊断信息，只在比值越出 warn 窗口才完全 fallback 到 HDR p99。HD 主验收实测 reference 现在生效（white_point=5.30e-3），LDR p99 从 0.73 提到 0.91，达成 plan Step 1 line 77 "参考点经 tonemap 后约落在 0.7" 的承诺。配套 13 个新单测保护 D2 脚本配置与 D3 fallback 行为。
- **v2.2 (2026-06-14)**：基于 V2 visual recovery 的实物反馈，冻结新的物理可信 cinematic 成像契约。默认主链改为 `support/rho/T/tau -> F_phys -> g-factor -> HDR/tonemap`；`F_phys = W_r · tau_raw · (T_raw/T_peak)^4` 使用相对通量归一化，避免 `T_peak_K≈1e7` 绝对值直接进入 LDR，并修复边界处 `W_r` 被重复乘成 `W_r^5` 的黑环问题。cinematic 曝光 reference 改为相机无关的物理场域垂向积分 p99，并接入 auto exposure 的 reference white point；颜色链改为 `T_phys -> log -> T_visible`，g-factor 色偏作用在 `T_visible` 上，physical 模式不再把 g 重复计入颜色。visual atlas 从主亮度/主色温/主 alpha 来源降级为 bounded turbulence 扰动输入；`interstellar` preset 后续必须重新定义为统一物理路径上的参数组合。V1 路径不纳入本轮改造。
- **v2.1 (2026-06-13)**：修正 v2.0 的温度跨度论述硬伤——v2.0 错误地声称"跨 3 个数量级"，实算 SS 公式 `r_in=3, r_out=10` 跨度仅 1.47 倍。本版采用实算结果：把 `r_out` 默认从 10 提到 50，物理跨度提升到 4.32 倍；同步修正 §1.1 R1 描述、§2.3 视觉验收（色相层次主要由 palette + g-factor 贡献，不再夸大温度跨度作用）、§3.3 温度场说明、§5.3 参数草案、§6 测试方案（替换不可能的 `T_max/T_min ≥ 100` 指标为基于实算的 `[4.0, 4.6]` 区间检查）、附录 A R1 实算数据。新增 §6 NumPy/Taichi parity 测试要求与对应单测；§8 把 `F_clump` 算法决策从"同时实现两种"改为"Phase 2 开始前必选一种、显式点云团为首选、Worley 作为回退"；§3.4 同步显式点云团为首版实现。Phase 1 验收指标更新。
- **v2.0 (2026-06-13)**：基于 v1.0 实物诊断的重大修订。新增五项大改（三维 F_struct、clump 项、Taichi 直接实现、palette + HDR、温度密度动态范围）。引入第五层"显示映射层"。新增 §3.6 调色与色调映射、§5.1 直接 Taichi 实现、§5.3 参数草案 v2.0 默认值、附录 A 诊断报告。`r_in` 强制 ≥ 3 · r_s。删除 v1.0 NumPy reference 路线相关描述。
- v1.0 (2026-03-07)：初版 V2 设计文档；明确平行实现策略、2.5D 有限厚度模型、统一平流约束、发射-吸收积分与分阶段实施计划。
