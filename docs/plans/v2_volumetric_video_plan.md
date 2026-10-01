# V2 体积云雾 + 动态旋转 + 视频接入方案

> **状态**：v0.3 已冻结（2026-10-01），按 §5 实施中。
>
> **触发原因**：
>
> 1. V2 单帧"烟雾/粒子感看不出来"，结构像贴图。
> 2. 动态模拟时开普勒差速把纹理卷成同心圈（winding problem）。
> 3. V2 不支持视频。
>
> **验收基准**：参考实现 `scripts/proto_disk_reference.py` 默认输出（预设 H，用户已看图确认，见 §2）。本方案的目标是把原型**等价地**移植进 V2，而不是重新调参。
>
> **真源关系**：盘体几何/物理场/调制的数学定义仍以 [`docs/design_ad_v2.md`](../design_ad_v2.md) 为准；本方案落地后需同步更新其对应章节（§9）。本方案替代 [`v2_visual_recovery_plan.md`](v2_visual_recovery_plan.md) 中"预烘焙 visual atlas 作为主结构"的决定。
>
> **非目标**：Kerr 度规；MHD/真实流体模拟；交互模式（`--interactive`）接入 V2；V1 路径的任何行为改变（V1 e2e 基线 `df4ccd8f…` 必须保持）。

---

## 1. 根因回顾（原型阶段已证实）

| 编号 | 现象 | 根因 | 证据 |
|------|------|------|------|
| R1 | 烟雾/体积感看不出来 | V2 默认 `use_visual_atlas=True`，主光追退化为"中面单次命中 + 2D atlas 查表"（`disk_v2/taichi_render.py:471`），没有真正的体积积分；atlas 只有 `(r, φ)`，无 z 结构 | 代码阅读 |
| R2 | 纹理卷成圈 | 任何 `f(φ − Ω(r)·t)` 形式在 t 无界时螺旋倾角 `tan i = 1/(1.5·Ω·t)` 单调趋零 | 原型 `winding`：naive 30 圈后倾角 6°→1.5° |
| R3 | 多普勒强度偏高 | `schwarzschild_orbital_beta_ti` 用 `1 − 3M/r`，应为 `1 − 2M/r`（ISCO 处 0.577 vs 0.5） | `disk_v2/taichi_impl.py:67`、`disk_v2/relativity.py:131` |
| R4 | 多普勒角度偏差 | `cosθ` 用坐标方向，未变换到本地静止观者方向；与严格 GR 最大差 5.6% | 原型 `physcheck` [1] |
| R5 | 颜色发灰、蓝移被吃掉 | Tanner Helland 输出是 sRGB 编码值，被当线性光使用，末端再做一次 gamma（双重 gamma） | 原型对比：修正后饱和度 0.33→0.49 |
| R6 | bloom 盖住云雾与蓝移 | bloom 在 LDR 域对整盘加性叠加（`_bloom_kernel`），并用逐通道衰减硬凑色偏 | 原型：整盘 bloom 使局部对比损失 40% |
| R7 | 静态 atlas 本身带同心弧 | `visual_atlas.py:96-98` 烘焙阶段就做了 Kepler roll + V1 弧线噪声 | 代码阅读 |

---

## 2. 原型定稿参数（验收基准）

参考实现：`scripts/proto_disk_reference.py`（S0 已入库）。**默认命令即定稿（预设 H）**，V2 移植后需与其输出等价：

```bash
conda run -n black-hole python scripts/proto_disk_reference.py frame --ss 2   # 1080p 单帧
conda run -n black-hole python scripts/proto_disk_reference.py physcheck      # 物理自检
```

| 类别 | 参数 | 值 | 含义 |
|------|------|----|------|
| 几何 | `R_IN / R_OUT` | 3 / 30 | 内缘 ISCO；外缘（用户确认） |
| 几何 | `HR` | 0.012 | 核心层标高 H/r |
| 相机 | `dist / fov / elev` | 60 / 38° / 7° | 相机距离、竖直视场、仰角 |
| 核心结构 | `FR / NPHI / FZ / WARP` | 16 / 7 / 0.9 / 0.9 | 噪声频率（方位拉伸约 14 倍）与域扭曲 |
| 核心结构 | `SIGMA_LN` | 1.3 | lognormal 密度涨落 |
| 核心结构 | `RIDGE_POW / FIL_GAIN / FIL_MIX` | 6 / 5 / 0.6 | ridged multifractal 丝状项 |
| 核心结构 | `COV_LO / COV_HI / COV_RIDGE` | −0.1 / 0.7 / 3 | 空洞覆盖率（丝所在处不挖空） |
| 大尺度 | `LOWF_SIGMA / FR_L / NPHI_L` | 0.9 / 3 / 3 | 低频 lognormal 明暗调制（尺度 Δln r ≈ 0.33） |
| 大尺度 | `DLN_L / K_RIGID_L` | ln 1.7 / 6 | 低频层宽带刚体环，避免被细带切成同心环 |
| 烟雾 | `N_CL_HALF` | 3 | 7 层，k = −3..3 |
| 烟雾 | `CL_SPACING / CL_WIDTH` | 0.014 / 0.011 | 层中心 z_k = k·0.014·r，层厚 σ = 0.011·r（重叠 → 近似连续烟雾冕） |
| 烟雾 | `CL_DECAY` | 0.35 | 层振幅 ∝ exp(−0.35·|k|) |
| 烟雾 | `CLOUD_COL` | 0.8 | 烟雾总柱密度 / 核心柱密度 |
| 烟雾 | `CLOUD_EMIT` | 0.33 | 烟雾单位密度发射率 / 核心（偏冷、以吸收为主 → 背光暗色丝缕） |
| 烟雾结构 | `FR_C / NPHI_C / FZ_C / SIGMA_C` | 8 / 10 / 0.9 / 1.2 | 烟雾噪声（低各向异性，蓬松） |
| 烟雾结构 | `CLOUD_C0 / CLOUD_SOFT` | 0.4 / 0.35 | sigmoid 软覆盖率（软边缘） |
| 温度 | `T_PEAK_VIS` | 4500 K | 可视色温峰值（Novikov-Thorne 形状） |
| 温度 | `T_CLUMP` | 0.06 | T ← T·(1 + 0.06·tanh(n_a)) |
| 发射 | `EMIT_POW` | 2.5 | j ∝ ρ·(T/T_peak)^2.5（非物理旋钮，4 = bolometric；控制内亮外暗的衰减速度） |
| 光学深度 | `TAU0` | 2.0 | r = 6 处 face-on 竖直光学深度 |
| 动态 | `DLN_R` | ln 1.22 | 刚体环带宽（ln r），带边界 1D 噪声扰动 0.35 |
| 动态 | `K_RIGID` | 4.0 | 结构种子寿命（本地轨道周期） |
| 相对论 | `doppler_lum` | 0.5 | 亮度频移指数（非物理旋钮，1 = 物理） |
| 相对论 | `doppler_color` | 2.2 | 颜色频移指数（非物理旋钮，1 = 物理；用户选"稍稍夸张"） |
| 颜色 | 黑体色度 | CIE 普朗克表 | 普朗克谱 × CIE 1931 色匹配（Wyman 2013 近似）→ 线性 sRGB |
| 相机 | `white_balance_K` | 5000 K | 相机白平衡（von Kries） |
| 后处理 | `bloom_src / bloom_gain` | 0.3 / 4.0 | 只散射高光 |
| 后处理 | `CA_AXIAL` | (1.0, 1.0, 1.15) | 轴向色散：只让蓝光稍外扩（R、B 同时外扩会形成品红光晕） |
| 后处理 | `CA_FRINGE / FRINGE_SRC / FRINGE_COLOR` | 0.4 / 0.7 / (0.3, 0.45, 1.0) | 饱和高光外缘偏蓝镶边 |
| 后处理 | `CA_LATERAL` | 0.0025 | 横向色散 |
| 曝光 | `auto_exposure` | p99.9 → 0.7 | 盘像素亮度分位映射值 |
| 曝光 | `WHITE_BLEND` | 0.12 | 超色域高光向白混合斜率 |

原型实测（1080p，2×2 超采样，M5 GPU）：单帧 8.7 s，首次编译约 2–3 min；`physcheck` 全部通过（频移与严格 GR 误差 < 0.01%）。

调参过程中的经验（防止 V2 移植时重犯）：

- 任何在"最亮区域"生效的后处理（轴向色散、镶边、高光向白混合）都会直接改写逼近侧颜色，必须单独验证逼近侧色相。
- 结构只在一个频段时观感单调；需同时有低频大尺度调制。低频层必须用比细结构更宽的刚体带，否则会被切成同心环。
- 烟雾感来自盘面上方偏冷、以吸收为主的半透明层（背光暗丝缕），而不是更多的发光层。

## 3. 总体架构

```
+------------------+     +-------------------+     +--------------------+
| advection.py     |---->| volume_field_ti   |---->| taichi_render.py   |
| rigid-ring bands |     | core + 7 smoke    |     | volume ray-march   |
| (CPU f64 phase)  |     | (Taichi noise)    |     | + g-factor (fixed) |
+------------------+     +-------------------+     +--------------------+
         ^                        ^                          |
         |                        |                          v
+------------------+     +-------------------+     +--------------------+
| video driver     |     | noise_ti.py       |     | postfx.py          |
| time t per frame |     | gnoise/fbm/ridged |     | WB + bloom + CA    |
| lock exposure    |     |                   |     | + chroma ACES      |
+------------------+     +-------------------+     +--------------------+
```

数据流：视频驱动给出物理时间 `t` → `advection.py` 在 CPU 上用 float64 算出每条带的相位/种子/权重表并上传 → 体积核在每个采样点求 3D 密度 → 光追累积 HDR → 后处理输出 LDR。

### 3.1 概念边界与命名（遵循 AGENTS.md Disk V2 规则）

| 新概念 | 归属层 | 命名 | 返回值 |
|--------|--------|------|--------|
| 核心层竖直包络 `exp(−ζ²/2)` | `geometry` | 复用 `density_field` 的竖直因子 | `[0, 1]` |
| 烟雾层 k 的竖直包络 `exp(−dz_k²/2)` | `geometry` | `cloud_layer_weight(r, z, k)` | `[0, 1]` |
| 核心/云层基础柱密度分配 | `physical_fields` | `core_density_field` / `cloud_density_field` | 物理密度 |
| lognormal × 丝 × 覆盖率 | `structure_modulations` | `core_structure_modulation` | 围绕 1 波动，≥ 0 |
| 云层 lognormal × sigmoid 覆盖率 | `structure_modulations` | `cloud_structure_modulation` | 围绕 1 波动，≥ 0 |
| 温度弱团块响应 | `structure_modulations` | `temperature_clump_modulation` | 围绕 1，`[0.94, 1.06]` |
| 大尺度低频明暗 | `structure_modulations` | `large_scale_modulation` | 围绕 1 波动（lognormal 均值 1） |
| 烟雾发射比例 | `physical_fields` | `cloud_emission_fraction` 参数 | 标量 `[0, 1]` |

注意：覆盖率会产生接近 0 的空洞，`*_modulation` 的"围绕 1 波动"指均值约为 1（lognormal 已做 `−σ²/2` 均值校正），不是逐点。docstring 中需写明。

---

## 4. 关键实现思路

### 4.1 刚体环平流（mode 3）

- 按 ln r 分带，带中心 `ln r_b = LNR0 + b·DLN_R`；带坐标 `fb = (ln r − LNR0)/DLN_R + 0.35·n1d(ln r)`（1D 噪声使带间距不规则，扰动斜率 < 1 保证单调）。
- 相邻两带以 `cos²/sin²` 权重混合（权重和恒为 1）。
- 带内结构以带中心角速度 `Ω_b = sqrt(M / r_b³)` **刚体**旋转：`φ0 = φ − Ω_b·t − φ_b`，带内无剪切，因此任意时长都不卷绕。
- 结构演化：每带两套噪声种子，相位差半周期，周期 `T_b = K_RIGID·2π/Ω_b`，三角权重 `sin²(π·frac)`；种子切换瞬间权重为 0，无跳变。
- 混合方式：主涨落 `n_a` 用方差保持（`Σw·n / sqrt(Σw²)`）；丝状项 `n_b`（非零均值）用线性加权。
- **float32 精度处理**：Metal 无 f64。`Ω_b·t` 在长视频中数值很大，f32 相位精度会损失。做法是每帧在 CPU 上用 float64 为每条带预计算 `(phase_b mod 2π, cycle_index_p, frac_p)`，上传为小 field（带数约 10），kernel 只查表。
- 简化假设：带内角速度统一取带中心值，局部偏差约 ±10%（`DLN_R = ln 1.22` 时 `Ω` 在带内变化约 ±15%，经两带混合后可见偏差更小）；物理上对应"结构有有限寿命，被湍流不断重建"。

### 4.2 3D 密度场

```
ρ(r, φ, z, t) = ρ_core + Σ_k ρ_cloud,k

ρ_env'(r) = ρ_env(r) · LN_L(n_L)                # 大尺度低频调制，n_L 由宽带刚体环求值
ρ_core    = ρ_env'(r) · exp(−ζ²/2) · LN(n_a) · F(n_b) · C(n_a, n_b)
ρ_cloud,k = CLOUD_COL · (HR/CL_WIDTH) · A_k · ρ_env'(r) · exp(−dz_k²/2) · LN(n_c,k) · S(n_c,k)

发射源：S_em = (ρ_core + CLOUD_EMIT·Σρ_cloud,k)/ρ · (T/T_peak)^EMIT_POW · B_550(g_lum) · χ(T·g_col)
吸收：  dτ = κ · ρ · ds

ζ = z / (HR·r),  dz_k = (z − k·CL_SPACING·r) / (CL_WIDTH·r)
LN(n) = exp(σ·n − σ²/2)                       # 均值为 1 的 lognormal
F(n_b) = (1 − FIL_MIX) + FIL_MIX·FIL_GAIN·n_b  # ridged 丝状项
C = smoothstep(COV_LO, COV_HI, n_a + COV_RIDGE·(n_b − 0.1))
S(n) = 1 / (1 + exp(−(n − CLOUD_C0)/CLOUD_SOFT))
A_k = exp(−CL_DECAY·|k|) / Σ_j exp(−CL_DECAY·|j|)
```

- 噪声坐标：拉格朗日坐标 `(ln r·FR, φ0/(2π)·NPHI, ζ·FZ)`，φ 方向用整数周期格点实现无缝。
- 烟雾 7 层在 kernel 里用**运行时循环**（非 `ti.static` 展开），只算 `|dz_k| < 3` 的层；原型实测 static 展开会让编译时间超过 20 分钟。
- `ρ_env(r) = (r/r_in)^(−1.5)·sqrt(1 − sqrt(r_in/r))·外缘收口`，沿用现有 `physical_fields` 语义。

### 4.3 相对论修正

- `β = sqrt(M/(r − 2M))`（本地静止观者测得的圆轨道速度）。
- 光子方向变换到本地静止观者：`k_loc ∝ k_rad + sqrt(1 − r_s/r)·k_tan`。
- `g = g_grav · 1/(γ(1 − β·cosθ_loc))`，`g_grav = sqrt(1 − r_s/r_em)/sqrt(1 − r_s/r_obs)`。
- 严格验证公式（赤道面圆轨道，`L_z/E` 取光子真实传播方向、沿盘旋转方向为正）：`g_exact = sqrt(1 − 3M/r) / (1 − Ω·L_z/E) / sqrt(1 − r_s/r_obs)`；原型逐像素误差 < 0.01%，NumPy reference 对 200 组随机方向误差 < 1e-9。
- 亮度：观测谱为温度 `g·T` 的黑体，550 nm 处 `B_ν(g_lum·T)/B_ν(T)`；颜色：`blackbody(T·g_col)`。`g_lum = g^doppler_lum`、`g_col = g^doppler_color`，两个指数为显式非物理旋钮，默认值见 §2，CLI 帮助中注明"1 = 物理"。
- 现有 `lum_power`（g⁴ 近似）由波段 Planck 比值替代，参数删除（§8-3）。

### 4.4 颜色与后处理

1. 黑体色：普朗克谱 × CIE 1931 色匹配函数 → XYZ → 线性 sRGB(D65)，负值裁 0、亮度归一，预计算为 log T 查找表（512 项，1000–40000 K）。替代 Tanner Helland（其输出为 sRGB 编码值，且白平衡下易偏青/品红）。
2. 相机白平衡：von Kries 增益 `1/rgb_lin(T_wb)`，保持亮度。
3. 高光 bloom（HDR 域）：`x += G·Σ_c PSF_c ∗ max(x − S, 0)`，三尺度盒式模糊，通道半径乘 `CA_AXIAL`。
4. 镶边：`ring = blur_wide(hs) − blur_narrow(hs)`，`hs = max(L − FRINGE_SRC, 0)`，染 `FRINGE_COLOR`。
5. 横向色散：R 放大 `(1+k)`、B 缩小 `(1−k)`，双线性重采样。
6. 保色度 ACES：只对亮度做 ACES，RGB 等比缩放；超出色域通道按比例收回并向白少量混合（`w = clip((max−1)·0.12, 0, 0.25)`）。
7. sRGB 编码。天空盒在相同链路之前以线性光合成（天空盒 PNG 先解码）。
- 实现位置：新增 `disk_v2/postfx.py`（NumPy，1080p 约 0.3 s）；若视频性能不够再移到 Taichi（见 §6）。
- 替换 `taichi_render.py` 现有 `_disk_tonemap_kernel` + `_bloom_kernel` + `_compose_kernel`。

### 4.5 视频

- 新增 `render_video_v2()`（`render.py`），复用 V1 `render_video` 的帧目录、`--resume`、异步 PNG 保存与编码逻辑。
- 时间：`t = t0 + frame · dt`，`dt = P(r_in) / (orbit_s · fps)`；新增 `--v2_orbit_seconds`（内缘一圈对应视频秒数，默认 16）。
- 曝光：首帧计算后锁定，避免逐帧自动曝光导致闪烁。
- 相机：支持 `--orbit` / `--orbit_degrees`，与 V1 一致。

---

## 5. 实施步骤（按依赖排序，每步单独 Review）

```
S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9
proto   rel    color  noise  advec  field  march  postfx video  docs
```

| 步骤 | 内容 | 修改/新增文件 | 测试要点 |
|------|------|---------------|----------|
| S0 ✅ | 原型入库作为参考实现与验收脚本 | `scripts/proto_disk_reference.py` | 默认输出与用户确认的预设 H 图逐像素一致；`physcheck` 通过 |
| S1 ✅ | 相对论修正：β、本地方向 cosθ、Planck 波段增强、`doppler_lum/color`（helper；渲染核接线在 S6） | `disk_v2/relativity.py`、`disk_v2/taichi_impl.py`、新增 `tests/unit/test_disk_v2_relativity_s1.py` | β(ISCO)=0.5；g 与严格 GR 误差 < 0.1%；逼近侧 g>1、远离侧 g<1；Planck 比值随 g 单调；指数为 0 时 g=1 |
| S2 | 颜色链路：CIE 黑体查找表、白平衡、删除 cinematic palette | `disk_v2/palette.py`、`disk_v2/taichi_impl.py`、`disk_v2/params.py` | 6500K 近中性（色度偏差 < 3%）；低温偏红、高温偏蓝单调；WB=T 时该温度黑体输出中性；亮度守恒；cinematic 相关测试删除 |
| S3 | Taichi 噪声库：周期梯度噪声、fBm、ridged、1D 噪声 | 新增 `disk_v2/noise_ti.py` | φ 周期无缝（首尾差 < 1e-5）；零均值；归一后单位方差；同种子可复现；ridged 值域 `[0, 1]` |
| S4 | 刚体环平流 + CPU f64 相位表 | 新增 `disk_v2/advection.py` | 带权重和 = 1；种子切换时权重为 0；长时间（30 圈）倾角不衰减；纹理 Δφ 与 Ω·Δt 一致（误差 < 1°）；t = 1e6 时相位精度（与 f64 参考对比） |
| S5 | 3D 密度场（核心 + 7 层烟雾 + 大尺度低频）替换 atlas | `disk_v2/geometry.py`、`disk_v2/physical_fields.py`、`disk_v2/structure_modulations.py`、`disk_v2/taichi_impl.py`、`disk_v2/params.py` | 盘外 = 0；ρ ≥ 0；调制均值约 1；空洞率 15–25%；竖直剖面在核心外有烟雾包络；低频层不产生径向环（径向自相关无周期峰）；径向自相关次峰 < 0.1；各层随 Ω 同步转动 |
| S6 | 体积光追为默认路径、步长控制、g 接入、烟雾发射比例；删除 atlas / thin-layer | `disk_v2/taichi_render.py`、删除 `disk_v2/visual_atlas.py` | 渲染可复现（同参数 hash 一致）；1080p ss=2 GPU 单帧 ≤ 10 s；与参考实现同参数时逼近/远离侧色相与盘带亮度剖面一致 |
| S7 | 后处理替换 | 新增 `disk_v2/postfx.py`，修改 `disk_v2/taichi_render.py` | bloom 对盘面局部对比损失 < 35%；光晕点亮暗区 5–15%；高光外缘 B/G 升高；零强度时各效果为恒等 |
| S8 | V2 视频 | `render.py` | 相邻帧差无尖峰；亮度波动 < 1%；`--resume` 结果与连续渲染逐帧一致；V1 e2e 基线不变 |
| S9 | CLI、README、设计文档、AGENTS 踩坑 | `render.py`、`README.md`、`docs/design_ad_v2.md`、`docs/design.md`、`AGENTS.md` | 文档与参数名一致 |

每步都跑：V2 单测全集 + `python tests/e2e_render.py --verify`（V1 不变）。已知与本方案无关的 6 条 `test_gpu_texture_compose` 失败不在范围内。

---

## 6. 性能预算

| 项 | 原型实测 | 目标 |
|----|----------|------|
| 1080p 单帧（ss=1） | 约 2.3 s | ≤ 3 s |
| 1080p 单帧（ss=2） | 8.7 s | ≤ 10 s |
| 首次编译 | 约 2 min | ≤ 3 min（有离线缓存后秒级） |
| 后处理（NumPy） | 约 0.3 s | 视频时若成为瓶颈移到 Taichi |

---

## 7. 风险

- **编译时间**：7 层烟雾 × 两带 × 两相位的噪声内联量大。缓解：云层运行时循环；噪声函数不 static 展开八度以外的维度。
- **与原型的数值偏差**：V2 坐标系是 `disk_tilt` 绕 x 轴倾斜，原型是盘在 z=0、相机仰角。移植时在盘局部坐标内计算一切，用 `physcheck` 等价测试保证结果一致。
- **外半径**：定稿 30；外圈亮度由 `EMIT_POW = 2.5` 控制衰减速度（用户在 1.7 / 2.0 / 2.5 中选定 2.5）。
- **现有 V2 测试大面积失效**：atlas / cinematic palette 相关测试需按新语义重写，每步明确列出被修改的测试及原因。

---

## 8. 已决事项（2026-10-01 用户确认）

1. **原型入库**：`scripts/proto_disk_reference.py`，作为参考实现与 `physcheck` 验收工具（S0 已完成）。
2. **atlas 路径删除**：删除 `disk_v2/visual_atlas.py`、thin-layer 分支、`--v2_turbulence_strength` / `--v2_spiral_warp_strength` / `--v2_alpha_clip_threshold` / `--v2_atlas_*` / `--v2_disable_visual_atlas`，以及 `tests/unit/test_disk_v2_visual_atlas.py`（S5/S6）。
3. **`--v2_lum_power` 删除**：由 Planck 波段增强 + `--v2_doppler_lum` 取代（helper 在 S1 提供，CLI 删除与渲染核接线在 S6）。
4. **V2 默认外半径 30**：`ar1 = 3, ar2 = 30`；推荐相机 `dist = 60`、竖直 fov 38°、仰角 7°。观感已确认（预设 H）。
5. **删除 cinematic palette 与 `--v2_visual_preset interstellar`**：删除 `palette_mode = cinematic`、warm_shift / saturation / visual_temp 映射及对应测试；颜色只走"线性化黑体 + 白平衡 + 频移"（S2）。

## 9. 文档同步

- `docs/design_ad_v2.md`：结构层（atlas → 3D 程序化密度 + 7 层烟雾 + 大尺度低频）、动态（刚体环）、相对论修正、颜色链路。
- `docs/design.md`：V2 后处理与视频管线。
- `README.md`：新增/废弃的 `--v2_*` 参数与视频用法。
- `AGENTS.md` 踩坑记录：Taichi kernel 内不能用 `math.exp`；`ti.static` 展开多层噪声导致编译爆炸；`from __future__ import annotations` 与 `ti.template()` 冲突；Tanner Helland 为 sRGB 编码值。
- `docs/plans/realism_uplift_plan.md` 变更记录追加本方案引用。

---

## 变更记录

- **v0.3 (2026-10-01)**：定稿预设 H（稀疏核心纹理 + 大尺度低频调制 + 7 层半吸收烟雾 + EMIT_POW 2.5 + CIE 黑体 + `doppler_color` 2.2 + 曝光 0.7 + 去品红色散）；S0 完成；方案冻结。
- **v0.2 (2026-10-01)**：§8 待决事项全部定稿（原型入库、删 atlas、删 `--v2_lum_power`、`ar2 = 30`、删 cinematic palette/preset）。
- **v0.1 (2026-10-01)**：首版。基于已确认的原型（mode 3 刚体环、核心 + 5 云层、修正 g、线性化颜色 + WB 5000K、`doppler_color = 1.6`、高光 bloom + 色散）整理移植方案。
