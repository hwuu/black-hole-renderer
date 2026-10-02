# Disk V2 设计：体积吸积盘

> **版本**：v2.3（2026-10-02）。v2.0–v2.2 的 atlas / 团块模型已归档至
> [`docs/archived/design_ad_v2_atlas_model.md`](archived/design_ad_v2_atlas_model.md)。
>
> **真源关系**：本文给出 V2 的模型、分层与边界；参数定稿值与调参经验见
> [`docs/plans/v2_volumetric_video_plan.md`](plans/v2_volumetric_video_plan.md) §2；
> 数值行为以参考实现 `scripts/proto_disk_reference.py`（预设 M、mode 3）为基准。

---

## 1. 问题陈述

V1 吸积盘是零厚度倾斜平面 + 程序纹理，存在三个问题：

- **P1 无体积感**：纹理贴在平面上，看不到烟雾、半透明层与盘面上方的结构。
- **P2 卷绕**：任何 `f(φ − Ω(r)·t)` 形式的纹理在 t 增大时被开普勒差速卷成同心圆，
  螺旋倾角 `tan i = 1/(1.5·Ω·t)` 单调趋零。
- **P3 颜色与亮度不物理**：颜色映射与多普勒亮度是经验公式，冷暖与亮暗关系难以解释。

v2.0–v2.2 用"预烘焙 atlas + 团块 + 薄层"缓解 P1，但 atlas 只有 `(r, φ)`、没有 z 结构，
且烘焙阶段就带同心弧，P1、P2 都没有根本解决。

## 2. 分析与选型

- **体积密度场 + 辐射转移**取代表面纹理：盘是 3D 半透明介质，体积感来自沿光线积分
  而非贴图，直接解决 P1。
- **刚体环平流**取代逐半径平流：按 ln r 分带，带内以带中心角速度刚体旋转、带间平滑混合、
  种子有限寿命交叉淡化，任意时长都不卷绕（P2）。
- **物理亮度与颜色**：观测谱为温度 `g·T` 的黑体（`I_ν/ν³` 不变），亮度取可见光亮度
  `Y(g·T)`、颜色取 CIE 黑体色度（P3）；偏离物理的取值全部作为显式第 3 层旋钮。
- 选型依据是参考实现的原型阶段：用户看图确认预设 M，V2 的目标是**等价移植**，
  而不是重新调参；等价性由 `scripts/compare_v2_proto.py` 逐像素验证。

## 3. 方案设计

### 3.1 模块分层与数据流

```
+--------------------+     +----------------------+     +----------------------+
| advection.py       |---->| taichi_impl.py       |---->| taichi_render.py     |
| rigid-ring bands   |     | DiskV2Taichi         |     | geodesic RK4         |
| (CPU f64 phases)   |     | density_I (3D field) |     | volume integration   |
+--------------------+     +----------------------+     +----------------------+
          ^                           ^                            |
          |                           |                            v
+--------------------+     +----------------------+     +----------------------+
| render.py          |     | noise_ti.py          |     | postfx.py            |
| frame time t       |     | cascade / fbm / hash |     | WB + bloom + CA      |
| exposure lock      |     | physical_fields.py   |     | + chroma ACES + sRGB |
+--------------------+     +----------------------+     +----------------------+
```

1. `render.py` 给出帧物理时间 `t` 与相机。
2. `advection.py` 在 CPU 上用 float64 计算每条带的转角、种子进度与周期索引，上传为小 field
   （Metal 无 f64，长视频下 `Ω_b·t` 的 f32 误差会到 1e-2 rad）。
3. `taichi_render.py` 沿测地线步进，在每段中点调用 `DiskV2Taichi.density_I` 求 3D 密度，
   累积线性 HDR。
4. `postfx.py` 曝光、后处理并输出 sRGB。

| 模块 | 职责 |
|------|------|
| `params.py` | `DiskV2Params`（盘内外半径）、`DiskV2VolumeParams`（预设 M 全部参数） |
| `physical_fields.py` | Page–Thorne 通量 / 温度、`derive_t_peak`、SS 外区 `H(r)`、`Σ(r)`（NumPy 参考） |
| `relativity.py` | 频移 `g` 的 NumPy 参考与严格 GR 对照 |
| `palette.py` | CIE 黑体色度 / 亮度查找表、von Kries 白平衡 |
| `noise_ti.py` | 梯度噪声、value noise、乘性级联、fBm |
| `advection.py` | 刚体环带表与每帧相位表 |
| `taichi_impl.py` | 体积密度场、Taichi 端频移、光行时间相位换算、噪声与 κ 标定 |
| `taichi_render.py` | `DiskV2Renderer`：主光追 kernel、超采样、曝光 |
| `postfx.py` | 后处理链 |

### 3.2 结构场：盘"长什么样"

| 分量 | 公式 | 说明 |
|------|------|------|
| 标高 | `H(r) = HR_REF·r·(r/10)^{1/8}·(f/f_10)^{3/20}`，`f = 1 − sqrt(r_in/r)` | SS 外区（气压主导、Kramers） |
| 柱密度 | `Σ(r) = (r/10)^{−3/4}·(f/f_10)^{7/10}·外缘截断·LN(σ_L·n_L)` | 叠加大尺度低频 lognormal 明暗 |
| 表面 | `H_s = H·(1 − SURF_NOISE + SURF_NOISE·softsat(tn))` | 噪声调制的云顶轮廓 |
| 核心密度 | `ρ = Σ/(√(2π)·H_s)·exp(−z²/2H_s²)·cfac`，`cfac = FLOOR + (1 − FLOOR)·c/⟨c⟩` | 乘性级联 `c`，温和起伏、无空洞 |
| 烟雾 | 7 层薄片 `Σ·SMOKE_I·A_k·exp(−dz_k²/2)·LN(n_c)·sigmoid(n_c)` | 盘风团块，温度 `SMOKE_TR·T` |
| 尘埃 | `DUST_EM·(1 − (z/H_d)²)⁺·c_dust` | 内区稀薄尘埃 |

所有噪声 `n_L`、`n_c` 归一到单位方差（`_calibrate_volume` 实测 std）；`σ_L`、`sigma_c`、
`cloud_c0` 等参数都按单位方差定义。噪声坐标使用刚体环流坐标 `φ0 = φ − Ω_b·(t − Δt) − φ_b`，
`Δt` 为光行时间延迟。

### 3.3 辐射转移：盘"发出什么光"

- 温度：Page–Thorne 相对论薄盘 `T(r)`，峰值 `T_peak` 由黑洞质量与吸积率推出；
  核心叠加灰大气竖直温度 `T⁴ = ¾T_eff⁴(τ_z + ⅔)`（上限 `GREY_CAP`）与湍流温度起伏。
- 局部热平衡：吸收 `α = κ·(ab_o + CORE_OPAC·ab_c)`，发射 `j = CORE_OPAC·em_c·S_c + em_o·S_o + em_s·S_s`，
  源函数 `S(T, g) = Y(g^{s_L}·T)/Y(T_peak)·χ(T·g^{s_C})`。`κ` 按 `TAU_I` 在 r ≈ 6 处标定。
- 段内精确均匀解：`ΔI = T·(j/α)·(1 − e^{−αΔs})`，`T ← T·e^{−αΔs}`；光学薄极限退化为 `T·j·Δs`。
- 像素值：盘发射 `Σ T·ΔI` 与透过的天空 `T_end·I_sky` 分两路累积。先积分本段再判视界，
  落入视界前的发射保留（光子环下半部来自这部分光线）。
- 频移：`g = g_grav / (γ(1 − β·cosθ_loc))`，`β = sqrt(M/(r − 2M))`，`θ_loc` 为光子在本地静止观者
  标架下与轨道速度的夹角；相机同样取静止观者本地标架。

### 3.4 后处理与视频

- 曝光：盘区亮度（`L > 1e-4`）p99.9 映射到 0.9，只按盘发射计算；视频首帧计算后锁定。
- 天空：天空盒（sRGB 编码）解码为线性光、双线性采样；合成 `x = exposure·disk + sky_gain·sky`
  后整体进入后处理（星点参与 bloom），天空亮度不随盘曝光变化。
- 倾角：盘面绕世界 x 轴旋转 θ；相机在 y-z 平面时与"盘不倾、相机仰角 + θ"严格等价（单测保护）。
- 后处理：von Kries 白平衡 → 高光 bloom（HDR 域、只散射超过阈值的部分）→ 高光镶边 →
  横向色散 → 保色度 ACES → sRGB。参数与参考实现一致。
- 视频：`t = 2000 + frame·dt`，`dt = P(r_in)/(v2_orbit_seconds·fps)`；相机支持环绕。
- 超采样：`ss×ss` 内部分辨率积分后盒式下采样，配合首步抖动构成蒙特卡洛体积积分。

### 3.5 概念边界与命名

| 层 | 含义 | 所在位置 | 命名 |
|----|------|----------|------|
| 结构场 | 盘的几何与物质分布：标高、柱密度、噪声结构、平流 | `physical_fields.py`、`noise_ti.py`、`advection.py`、`DiskV2Taichi` 的 `_ss_*` / `_flow_*` / `_turb_*` / `density_I` | `*_half_thickness`、`*_surface_density`、`*_flow`、`density_*` |
| 辐射转移 | 温度、源函数、频移、发射-吸收积分 | `physical_fields.py`（温度）、`palette.py`、`relativity.py`、`taichi_render.py` | `*_temperature`、`blackbody_*`、`*_g_factor` |
| 后处理 | 曝光、相机效应、显示编码 | `postfx.py`、`DiskV2Renderer.render` | `apply_*`、`tonemap_*`、`srgb_encode` |

约定：同一概念在测试名、docstring、本文中叫法一致；NumPy 参考函数与 Taichi 实现以 `_ti`
后缀区分并由单测做 parity。

## 4. 验收

- `python scripts/compare_v2_proto.py`：1080p、ss = 2 同参数对比，指标为盘覆盖率、R/B、
  左右通量比、光子环下半部通量占比、逐像素 `|ln(V2/Proto)|`；2026-10-01 修复后中位数 0.062。
- `tests/unit/test_disk_v2_*.py`：物理参考函数、Taichi parity、噪声归一化、光行时间相位、
  视界发射保留、超采样形状。
- V1 e2e 基线保持不变（`python tests/e2e_render.py --verify`）。

## 5. 非目标

Kerr 度规；MHD / 真实流体模拟；`--interactive` 接入 V2；V1 路径的任何行为改变。

## 参考资料

- D. N. Page, K. S. Thorne, "Disk-Accretion onto a Black Hole", ApJ 191, 499 (1974)。
- N. I. Shakura, R. A. Sunyaev, "Black holes in binary systems", A&A 24, 337 (1973)。
- C. Wyman, P.-P. Sloan, P. Shirley, "Simple Analytic Approximations to the CIE XYZ Color Matching Functions", JCGT 2(2), 2013。
- E. Bruneton, "Real-time High-Quality Rendering of Non-Rotating Black Holes", 2020（刚体环平流思路）。
