# V2 体积云雾 + 动态旋转 + 视频接入方案

> **状态**：v0.5 已冻结（2026-10-01），按 §5 实施中。定稿为预设 M（§2，参数分三层）；L / J2 / H 保留为可退回预设（§2.4）。
>
> **后续（2026-10-04）**：烟雾层、尘埃、`core_opac` 与旧主云级联（`core_az_stretch` / `outer_detail_fade` 等）已被统一
> 气体模型取代并删除；`scripts/proto_disk_reference.py`、`scripts/compare_v2_proto.py` 与 `test_disk_v2_proto_parity.py`
> 随之删除。本文中的相关内容为历史记录，现行模型见 [`v2_unified_gas_plan.md`](v2_unified_gas_plan.md)。
>
> **触发原因**：
>
> 1. V2 单帧"烟雾/粒子感看不出来"，结构像贴图。
> 2. 动态模拟时开普勒差速把纹理卷成同心圈（winding problem）。
> 3. V2 不支持视频。
>
> **验收基准**：参考实现 `scripts/proto_disk_reference.py` 的默认输出（预设 M + 刚体环平流，用户已看图确认，见 §2）。本方案的目标是把原型**等价地**移植进 V2，避免二次调参引入偏差。
>
> **术语表**（本文为已冻结的实施记录，保留当时的用语）：
> - **参考实现 / Proto**：`scripts/proto_disk_reference.py`，独立单文件渲染器，参数组合的定稿名为预设 M。
> - **mode 3 / 刚体环平流**：按 ln r 分带、带内刚体旋转、带间平滑混合、种子有限寿命交叉淡化的平流方案（§4.1）。
> - **V2**：`src/v2/` 包的工程实现，与参考实现逐像素对齐（§10、§12）。
> - **S0–S9**：本方案 §5 的实施步骤编号。
>
> **真源关系**：模型、分层与概念边界以 [`docs/design_ad_v2.md`](../design_ad_v2.md)（v2.3）为准。本方案替代已归档的 [`v2_visual_recovery_plan.md`](../archived/v2_visual_recovery_plan.md) 中"预烘焙 visual atlas 作为主结构"的决定；旧 atlas 模型已于 2026-10-02 删除（§11）。
>
> **非目标**：Kerr 度规；MHD/真实流体模拟；交互模式（`--interactive`）接入 V2；V1 路径的任何行为改变（V1 e2e 基线 `df4ccd8f…` 必须保持）。

---

## 1. 根因回顾（原型阶段已证实）

| 编号 | 现象 | 根因 | 证据 |
|------|------|------|------|
| R1 | 烟雾/体积感看不出来 | V2 默认 `use_visual_atlas=True`，主光追退化为"中面单次命中 + 2D atlas 查表"（`src/v2/taichi_render.py:471`），没有真正的体积积分；atlas 只有 `(r, φ)`，无 z 结构 | 代码阅读 |
| R2 | 纹理卷成圈 | 任何 `f(φ − Ω(r)·t)` 形式在 t 无界时螺旋倾角 `tan i = 1/(1.5·Ω·t)` 单调趋零 | 原型 `winding`：naive 30 圈后倾角 6°→1.5° |
| R3 | 多普勒强度偏高 | `schwarzschild_orbital_beta_ti` 用 `1 − 3M/r`，应为 `1 − 2M/r`（ISCO 处 0.577 vs 0.5） | `src/v2/taichi_impl.py:67`、`src/v2/relativity.py:131` |
| R4 | 多普勒角度偏差 | `cosθ` 用坐标方向，未变换到本地静止观者方向；与严格 GR 最大差 5.6% | 原型 `physcheck` [1] |
| R5 | 颜色发灰、蓝移被吃掉 | Tanner Helland 输出是 sRGB 编码值，被当线性光使用，末端再做一次 gamma（双重 gamma） | 原型对比：修正后饱和度 0.33→0.49 |
| R6 | bloom 盖住云雾与蓝移 | bloom 在 LDR 域对整盘加性叠加（`_bloom_kernel`），并用逐通道衰减硬凑色偏 | 原型：整盘 bloom 使局部对比损失 40% |
| R7 | 静态 atlas 本身带同心弧 | `visual_atlas.py:96-98` 烘焙阶段就做了 Kepler roll + V1 弧线噪声 | 代码阅读 |

---

## 2. 原型定稿参数（验收基准）

参考实现：`scripts/proto_disk_reference.py`。**默认命令即定稿（预设 M、mode 3）**，V2 移植后需与其输出等价：

```bash
conda run -n black-hole python scripts/proto_disk_reference.py frame --ss 2   # 1080p 单帧
conda run -n black-hole python scripts/proto_disk_reference.py physcheck      # 物理自检
conda run -n black-hole python scripts/proto_disk_reference.py params         # 三层参数表（当前值 + 物理值）
```

演进：H → J2（借鉴 NPGS / Baopinsui 最新 shader 的模型 I + H 烟雾）→ L（物理化亮度 / 温度 / 密度）→ M（全面 review 后的物理修正 + 三层参数）。原则：**所有主要物理效应必须具备，可近似**；偏离物理的取值全部放在第 3 层并标注物理值。

### 2.1 第 1 层：基本参数（场景设定）

| 参数 | CLI / 全局 | 定稿值 | 说明 |
|------|-----------|--------|------|
| 黑洞质量 | `--bh_mass` / `BH_MASS_MSUN` | 1e8 M☉ | 图像与尺度无关，只经 Ṁ 决定温度 |
| 吸积率 | `--mdot_edd` / `MDOT_EDD` | 1.7e-6 Ṁ_Edd | 与 M 一起经 Page–Thorne 绝对通量推出 T_peak = 4509 K（η = 1 − sqrt(8/9)） |
| 自旋 | — | 0 | 史瓦西黑洞 |
| 盘内 / 外半径 | `R_IN` / `--r_out` | 3（ISCO）/ 30 r_s | |
| 相机 | `--dist` / `--elev` / `--fov` | 40 r_s / 7° / 38° | |
| 分辨率 / 超采样 | `--w --h` / `--ss` | 1920×1080 / 2 | |
| 视频 | `--orbit_s` / `--fps` | 16 s / 24 | 内缘一圈对应的视频秒数 |

### 2.2 第 2 层：物理模型（定稿值即物理默认值）

| 模型 | CLI / 全局 | 定稿值 | 公式 / 说明 |
|------|-----------|--------|-------------|
| 光线 | — | 史瓦西零测地线 | `d²x/dλ² = −1.5 L² x / r⁵`，RK4 |
| 相机标架 | `STATIC_CAM` | 1 | 静止观者本地标架：`tanψ_coord = tanψ_local / sqrt(1 − r_s/r)` |
| 物质运动 | — | 开普勒圆轨道 | `Ω = sqrt(M/r³)`；mode 3 刚体环（带内取带中心 Ω，±15%） |
| 频移 | — | 严格 g | 本地静止观者方向 × 引力红移（§4.3） |
| 温度剖面 | `TEMP_PT` | 1 | Page–Thorne 相对论薄盘，峰值 4.8 r_s |
| 峰值温度 | `--t_peak` | 0（由 M、Ṁ 推出） | 非 0 时直接覆盖 |
| 盘结构 | `PHYS_STRUCT` | 1 | SS 外区（气压主导、Kramers）：`H/r ∝ r^{1/8} f^{3/20}`，`Σ ∝ r^{-3/4} f^{7/10}`，`f = 1 − sqrt(r_in/r)`；竖直等温高斯 |
| 亮度 | `PHYS_LUM` | 1 | 观测谱为温度 g·T 的黑体（I_ν/ν³ 不变）→ 亮度 `Y(g·T) = ∫B_λ(gT)·ȳ dλ` |
| 颜色 | — | CIE 黑体色度 | 普朗克谱 × CIE 1931 → 线性 sRGB(D65) |
| 竖直温度 | `GREY_ATM` | 1 | 灰大气 `T⁴ = ¾T_eff⁴(τ_z + ⅔)`，`τ_z = κΣ/2·erfc(|z|/(√2 H_s))`；上限 `GREY_CAP` = 1.19（τ≈2，见 §2.5） |
| 辐射转移 | — | 局部热平衡 | 发射 = 吸收 × 源函数，源函数由温度决定 |
| 湍流密度 | `CORE_FLOOR` | 0.35 | 核心密度 = FLOOR + (1 − FLOOR)·c/⟨c⟩，无空洞 |
| 湍流温度 | `DT_I` | 0.05 | 发热率起伏 δT/T |
| 结构寿命 | `--k_rigid` | 4 | 本地轨道周期 |
| 盘风团块（烟雾） | `SMOKE_I` / `SMOKE_TR` | 0.8 / 0.85 | 柱密度 = 0.8·Σ，温度 = 0.85·T(r)，7 层半透明 |
| 内区尘埃 | `DUST_EM` / `DUST_KEPLER` | 0.02 / 1 | 稀薄尘埃，按开普勒刚体环旋转 |
| 光行时间 | `LIGHT_DELAY` | 1 | 采样时间 = t − 光程 |

### 2.3 第 3 层：视觉调节（右列为物理值）

| 参数 | CLI / 全局 | 定稿值 | 物理值 / 说明 |
|------|-----------|--------|---------------|
| 多普勒亮度强度 | `--doppler_lum` | 0.55 | 1；亮度用 `Y(g^s·T)`。4500 K 时可见光对频移的等效指数约 hν/kT ≈ 6，物理值下左右亮度比 > 5，用户选 0.55 |
| 多普勒颜色强度 | `--doppler_color` | 1.5 | 1；色度用 `χ(T·g^s)` |
| 灰大气强度 | `GREY_MIX` | 0.5 | 1 |
| 核心光学深度 | `CORE_OPAC`（τ = 1.5 × 倍率） | 2（τ≈3） | 真实薄盘 ≫ 100；τ≈3 为通透感 |
| 盘厚 | `HR_REF`（r = 10 处 H/r） | 0.027 | SS 理论对本场景 ≪ 0.01 |
| 大尺度明暗幅度 | `LOWF_SIGMA` | 0.9 | 约 0.3 |
| 程序化结构 | `CON_I` 50、`KR_I` 0.2、`NPHI_I` 2、`L0_I` 3、`SURF_NOISE` 0.6、`DLN_R` ln 1.22、烟雾 `N_CL_HALF` 3 / `CL_SPACING` 0.014 / `CL_WIDTH` 0.011 / `CL_DECAY` 0.35 / `FR_C` 8 / `NPHI_C` 10 / `SIGMA_C` 1.2 / `CLOUD_C0` 0.4 / `CLOUD_SOFT` 0.35、尘埃八度 0–6 | 见左 | 噪声频率、八度、对比度、表面起伏、烟雾分层 |
| 曝光 | `--exp` / `EXP_TARGET` | 0.9 | 盘区亮度 p99.9 映射值 |
| 白平衡 | `--wb` | 5000 K | |
| bloom | `--bloom_src` / `--bloom_gain` | 0.3 / 4.0 | 只散射高光 |
| 色散 | `CA_AXIAL` / `--ca_fringe` / `--ca_lateral` | (1, 1, 1.15) / 0.4 / 0.0025 | 轴向色散、高光镶边、横向色散 |
| 色调映射 | `WHITE_BLEND` | 0.12 | 保色度 ACES |

对照预设 `Mphys`（另加 `--doppler_lum 1 --doppler_color 1`）：第 3 层中多普勒、灰大气强度、核心光学深度（τ≈100）、大尺度明暗（0.3）取物理值。

原型实测（1080p，2×2 超采样，M5 GPU）：单帧约 51 s（主要为高斯竖直包络与尘埃流场，移植时优化）；`physcheck` 全部通过；慢速视频无闪烁。

### 2.4 退回路径（与历史图逐像素一致，最大差 1 个灰度级为浮点舍入）

```bash
... frame --ss 2 --preset L  --doppler_lum 0.75
... frame --ss 2 --preset J2 --dist 60 --doppler_lum 0.5 --doppler_color 2.2
... frame --ss 2 --preset H  --dist 60 --doppler_lum 0.5 --doppler_color 2.2
```

### 2.5 调参与物理经验（防止 V2 移植时重犯）

- 任何在"最亮区域"生效的后处理（轴向色散、镶边、高光向白混合）都会直接改写逼近侧颜色，必须单独验证逼近侧色相。
- 结构只在一个频段时观感单调；需同时有低频大尺度调制，且低频层用比细结构更宽的刚体带，否则会被切成同心环。
- 烟雾感来自盘面上方偏冷、以吸收为主的半透明层；体积感另一来源是噪声调制的表面高度（云顶轮廓）。
- NPGS 原式"发射 ∝ ρ、吸收 ∝ ρ²"等价于 S ∝ 1/ρ（越密越暗），必须用局部热平衡；截面形状只决定几何厚度，柱密度用物理剖面。
- 亮度必须用可见光亮度 Y(g·T)，不能用 (T/T_peak)^p：4500 K 盘在 r = 30 处物理亮度只有峰值的 1.4e-5，旧的 T^2.5 给出 6%。
- 4500 K 与吸积率绑定：1e8 M☉ 需 Ṁ ≈ 1.7e-6 Ṁ_Edd（真实流体此时应为 ADAF，这里仍按薄盘近似，Interstellar 同样如此）。温度越高，可见光处于 Rayleigh–Jeans 段，亮度衰减越慢、外圈越不黑。
- 灰大气用竖直 τ_z 的平行平面近似时，光线从起伏侧壁斜入会被误判为深层高温，出现少数极亮热点并拉低整体曝光；可见层温度需加上限（τ≈2）。
- 体积积分已经计入掠射时的长光程，不能再乘掠射增亮因子。
- 步长必须是位置的连续函数（取多个连续函数的 min），体积内配合起点抖动 + 超采样。

## 3. 总体架构

```
+------------------+     +-------------------+     +--------------------+
| advection.py     |---->| volume_field_ti   |---->| taichi_render.py   |
| rigid-ring bands |     | lens core + smoke |     | volume ray-march   |
| (CPU f64 phase)  |     | (Taichi noise)    |     | + g-factor (fixed) |
+------------------+     +-------------------+     +--------------------+
         ^                        ^                          |
         |                        |                          v
+------------------+     +-------------------+     +--------------------+
| video driver     |     | noise_ti.py       |     | postfx.py          |
| time t per frame |     | cascade/vnoise/fbm|     | WB + bloom + CA    |
| lock exposure    |     |                   |     | + chroma ACES      |
+------------------+     +-------------------+     +--------------------+
```

数据流：视频驱动给出物理时间 `t` → `advection.py` 在 CPU 上用 float64 算出每条带的相位/种子/权重表并上传 → 体积核在每个采样点求 3D 密度 → 光追累积 HDR → 后处理输出 LDR。

### 3.1 概念边界与命名

实施初期规划的 `geometry` / `structure_modulations` 三层划分已随旧模块删除而废止（2026-10-02）。
现行分层（结构场 / 辐射转移 / 后处理）与命名见 [`docs/design_ad_v2.md`](../design_ad_v2.md) §3.5。

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

### 4.2 3D 密度场与源函数（M）

```
f(r)      = 1 − sqrt(r_in/r)
H(r)      = HR_REF · r · (r/10)^{1/8} · (f/f_10)^{3/20}            # SS 外区标高
Σ'(r)     = (r/10)^{-3/4} · (f/f_10)^{7/10} · 外缘截断 · LN_L(n_L)   # SS 柱密度 × 大尺度低频
H_s       = H · (1 − SURF_NOISE + SURF_NOISE·softsat(tn))           # 噪声调制表面（云顶轮廓）
ρ_core    = Σ'/(sqrt(2π)·H_s) · exp(−z²/2H_s²)                     # ∫ρ dz = Σ'
c         = softplus(CON · Σ_l w_l ln(1 + 0.1·n(3^l p)))            # 乘性级联（value noise）
cfac      = FLOOR + (1 − FLOOR)·c/⟨c⟩                               # 温和密度起伏，无空洞
α_core    = κ · CORE_OPAC · cfac · ρ_core
T_core    = T_PT(r) · [1 + GREY_MIX·(min((¾(τ_z + ⅔))^{1/4}, GREY_CAP) − 1)] · (1 + δT·(c/⟨c⟩ − 1))
τ_z       = κ · CORE_OPAC · cfac · Σ'/2 · erfc(|z|/(√2 H_s))
α_smoke   = κ · SMOKE_I · Σ' · Σ_k A_k exp(−dz_k²/2)/(sqrt(2π)·CL_WIDTH·r) · LN(n_c,k) · sigmoid(n_c,k),  T_smoke = SMOKE_TR·T_PT
α_dust    = κ · DUST_EM · (1 − (z/H_d)²)⁺ · c_dust（开普勒刚体环流场），T_dust = T_PT

dI = 透过率 · α · S(T, g) · ds，透过率 ← 透过率 · exp(−α ds)
S(T, g)   = Y(g_lum·T)/Y(T_peak) · χ(T·g_col)，g_lum = g^{doppler_lum}，g_col = g^{doppler_color}
```

- 噪声坐标：核心 `(KR_I·r, φ0/(2π)·NPHI_I, KR_I·z)`，φ0 为 mode 3 刚体环流坐标；φ 方向整数周期，无接缝。
- 烟雾 7 层与级联噪声八度都用**运行时循环**（非 `ti.static` 展开），控制编译时间。
- κ 由 `TAU_I` = 1.5 在 r ≈ 6 处标定（不含 CORE_OPAC）。

### 4.3 相对论修正

- `β = sqrt(M/(r − 2M))`（本地静止观者测得的圆轨道速度）。
- 光子方向变换到本地静止观者：`k_loc ∝ k_rad + sqrt(1 − r_s/r)·k_tan`。
- `g = g_grav · 1/(γ(1 − β·cosθ_loc))`，`g_grav = sqrt(1 − r_s/r_em)/sqrt(1 − r_s/r_obs)`。
- 严格验证公式（赤道面圆轨道，`L_z/E` 取光子真实传播方向、沿盘旋转方向为正）：`g_exact = sqrt(1 − 3M/r) / (1 − Ω·L_z/E) / sqrt(1 − r_s/r_obs)`；原型逐像素误差 < 0.01%，NumPy reference 对 200 组随机方向误差 < 1e-9。
- 亮度：观测谱为温度 `g·T` 的黑体，亮度 `Y(g_lum·T)`（普朗克谱 × CIE ȳ 积分，ln Y 查找表，300–60000 K）；颜色：`χ(T·g_col)`。`g_lum = g^doppler_lum`、`g_col = g^doppler_color`，两个指数为显式非物理旋钮，默认值见 §2，CLI 帮助中注明"1 = 物理"。
- 现有 `lum_power`（g⁴ 近似）由波段 Planck 比值替代，参数删除（§8-3）。

### 4.4 颜色与后处理

1. 黑体色：普朗克谱 × CIE 1931 色匹配函数 → XYZ → 线性 sRGB(D65)，负值裁 0、亮度归一，预计算为 log T 查找表（512 项，1000–40000 K）。替代 Tanner Helland（其输出为 sRGB 编码值，且白平衡下易偏青/品红）。
2. 相机白平衡：von Kries 增益 `1/rgb_lin(T_wb)`，保持亮度。
3. 高光 bloom（HDR 域）：`x += G·Σ_c PSF_c ∗ max(x − S, 0)`，三尺度盒式模糊，通道半径乘 `CA_AXIAL`。
4. 镶边：`ring = blur_wide(hs) − blur_narrow(hs)`，`hs = max(L − FRINGE_SRC, 0)`，染 `FRINGE_COLOR`。
5. 横向色散：R 放大 `(1+k)`、B 缩小 `(1−k)`，双线性重采样。
6. 保色度 ACES：只对亮度做 ACES，RGB 等比缩放；超出色域通道按比例收回并向白少量混合（`w = clip((max−1)·0.12, 0, 0.25)`）。
7. sRGB 编码。天空盒在相同链路之前以线性光合成（天空盒 PNG 先解码）。
- 实现位置：新增 `src/v2/postfx.py`（NumPy，1080p 约 0.3 s）；若视频性能不够再移到 Taichi（见 §6）。
- 替换 `taichi_render.py` 现有 `_disk_tonemap_kernel` + `_bloom_kernel` + `_compose_kernel`。

### 4.5 视频

- 新增 `render_video_v2()`（`render.py`），复用 V1 `render_video` 的帧目录、`--resume`、异步 PNG 保存与编码逻辑。
- 时间：`t = t0 + frame · dt`，`dt = P(r_in) / (orbit_s · fps)`；新增 `--v2_orbit_seconds`（内缘一圈对应视频秒数，默认 16）。
- 曝光：首帧计算后锁定，避免逐帧自动曝光导致闪烁。
- 光行时间：采样时间 `t_s = t − 光程`（从相机沿光线累计的路径长度），使近侧/远侧纹理相位符合物理。
- 步长：位置的连续函数 `h = min(h_far(r), H_IN_I + 0.3·d_core, max(0.4·CL_WIDTH·R, H_IN_I) + 0.3·d_smoke)`，首步乘随机因子（每帧种子不同）。
- 相机：支持 `--orbit` / `--orbit_degrees`，与 V1 一致；相机为静止观者本地标架（§2.2）。

---

## 5. 实施步骤（按依赖排序，每步单独 Review）

```
S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9
proto   rel    color  noise  advec  field  march  postfx video  docs
```

| 步骤 | 内容 | 修改/新增文件 | 测试要点 |
|------|------|---------------|----------|
| S0 ✅ | 原型入库作为参考实现与验收脚本（v0.5 更新为 M，L / J2 / H 保留为预设） | `scripts/proto_disk_reference.py` | 默认输出与用户确认的 M 图逐像素一致；L / J2 / H 复现命令与历史图一致；`physcheck` 通过 |
| S1 ✅ | 相对论修正：β、本地方向 cosθ、Planck 波段增强、`doppler_lum/color`（helper；渲染核接线在 S6） | `src/v2/relativity.py`、`src/v2/taichi_impl.py`、新增 `tests/unit/test_disk_v2_relativity_s1.py` | β(ISCO)=0.5；g 与严格 GR 误差 < 0.1%；逼近侧 g>1、远离侧 g<1；Planck 比值随 g 单调；指数为 0 时 g=1 |
| S2 | 颜色与亮度链路：CIE 黑体色度表、可见光亮度 ln Y(T) 表、白平衡、删除 cinematic palette | `src/v2/palette.py`、`src/v2/taichi_impl.py`、`src/v2/params.py` | 6500K 近中性（色度偏差 < 3%）；低温偏红、高温偏蓝单调；Y(T) 单调且与数值积分误差 < 1%；WB=T 时该温度黑体中性；cinematic 相关测试删除 |
| S3 | Taichi 噪声库：周期 value noise、乘性级联（小数八度 + softplus）、周期梯度噪声 fBm（烟雾 / 低频层）、1D 噪声 | 新增 `src/v2/noise_ti.py` | φ 周期无缝（首尾差 < 1e-5）；同种子可复现；级联输出 ≥ 0、小数八度在整数处连续；fBm 归一后单位方差 |
| S4 | 刚体环平流（主云、厚度扰动、烟雾、低频层共用）+ CPU f64 相位表 + 光行时间 | 新增 `src/v2/advection.py` | 带权重和 = 1；种子切换时权重为 0；长时间（30 圈）倾角不衰减；纹理 Δφ 与 Ω·Δt 一致（误差 < 1°）；t = 1e6 时相位精度（与 f64 参考对比） |
| S5 | 3D 场（M）：第 1 层 M、Ṁ → T_peak；Page–Thorne T(r)；SS H(r)/Σ(r) + 竖直高斯；噪声表面；灰大气 + 温度起伏；温和密度起伏；烟雾（温度比）；开普勒尘埃；大尺度低频；替换 atlas | `src/v2/geometry.py`、`src/v2/physical_fields.py`、`src/v2/structure_modulations.py`、`src/v2/taichi_impl.py`、`src/v2/params.py` | 盘外 = 0；ρ ≥ 0；T_peak(1e8 M☉, 1.7e-6) = 4509 K；PT 峰值 4.8 r_s；核心柱密度 = Σ'（误差 < 2%）；τ_z 与数值积分一致；灰大气倍率 ∈ [1, GREY_CAP]；各层随 Ω 同步转动；与参考实现同参数 parity |
| S6 | 体积光追为默认路径、连续步长 + 起点抖动、静止观者相机、g 接入、Y(g·T) 源函数（核心 / 烟雾 / 尘埃分温度）；删除 atlas / thin-layer | `src/v2/taichi_render.py`、删除 `src/v2/visual_atlas.py` | 渲染可复现（同参数、同抖动种子 hash 一致）；1080p ss=2 GPU 单帧 ≤ 30 s；与参考实现同参数时逼近/远离侧色相与盘带亮度剖面一致 |
| S7 | 后处理替换 | 新增 `src/v2/postfx.py`，修改 `src/v2/taichi_render.py` | bloom 对盘面局部对比损失 < 35%；光晕点亮暗区 5–15%；高光外缘 B/G 升高；零强度时各效果为恒等 |
| S8 | V2 视频 | `render.py` | 相邻帧差无尖峰；亮度波动 < 1%；`--resume` 结果与连续渲染逐帧一致；V1 e2e 基线不变 |
| S9 | CLI 按三层分组（与参考实现一致）、README、设计文档、AGENTS 踩坑 | `render.py`、`README.md`、`docs/design_ad_v2.md`、`docs/design.md`、`AGENTS.md` | 文档与参数名一致；`--help` 分组与 §2 三层一致 |

每步都跑：V2 单测全集 + `python tests/e2e_render.py --verify`（V1 不变）。已知与本方案无关的 6 条 `test_gpu_texture_compose` 失败不在范围内。

---

## 6. 性能预算

| 项 | 原型实测 | 目标 |
|----|----------|------|
| 1080p 单帧（ss=1） | 约 13 s | ≤ 8 s |
| 1080p 单帧（ss=2） | 51 s | ≤ 30 s（尘埃八度、高斯包络剔除、烟雾噪声优化后） |
| 首次编译 | 约 2 min | ≤ 3 min（有离线缓存后秒级） |
| 后处理（NumPy） | 约 0.3 s | 视频时若成为瓶颈移到 Taichi |

---

## 7. 风险

- **编译时间与性能**：7 层烟雾 × 两带 × 两相位的噪声内联量大，且烟雾噪声占单帧耗时约 95%。缓解：运行时循环；移植时烟雾改用乘性级联 value noise（单次求值更便宜），需与参考实现做观感对比后再替换。
- **长视频精度**：刚体环相位在 CPU 用 float64 计算（§4.1）；光行时间只做相对偏移，不引入新的大数。
- **与原型的数值偏差**：V2 坐标系是 `disk_tilt` 绕 x 轴倾斜，原型是盘在 z=0、相机仰角。移植时在盘局部坐标内计算一切，用 `physcheck` 等价测试保证结果一致。
- **外半径与外圈亮度**：定稿 30；外圈亮度由物理 Y(T) 决定（4509 K 盘在 r ≳ 20 已接近黑），不再有 `EMIT_POW` 旋钮。
- **现有 V2 测试大面积失效**：atlas / cinematic palette 相关测试需按新语义重写，每步明确列出被修改的测试及原因。

---

## 8. 已决事项（2026-10-01 用户确认）

1. **原型入库**：`scripts/proto_disk_reference.py`，作为参考实现与 `physcheck` 验收工具（S0 已完成）。
2. **atlas 路径删除**：删除 `src/v2/visual_atlas.py`、thin-layer 分支、`--v2_turbulence_strength` / `--v2_spiral_warp_strength` / `--v2_alpha_clip_threshold` / `--v2_atlas_*` / `--v2_disable_visual_atlas`，以及 `tests/unit/test_disk_v2_visual_atlas.py`（S5/S6）。
3. **`--v2_lum_power` 删除**：由 Planck 波段增强 + `--v2_doppler_lum` 取代（helper 在 S1 提供，CLI 删除与渲染核接线在 S6）。
4. **V2 默认外半径 30**：`ar1 = 3, ar2 = 30`；推荐相机 `dist = 60`、竖直 fov 38°、仰角 7°。
6. **盘体形态定稿 J2**（v0.4，已被 7 取代）：模型 I + H 烟雾 + 大尺度明暗，对比度 50；旋转保留 mode 3（mode 4 螺线内流对比后未采用）；H 保留为可退回预设。
7. **物理化与三层参数，定稿 M**（v0.5）：亮度 Y(g·T)、Page–Thorne、灰大气（强度 0.5）、核心 τ≈3 温和起伏、δT 5%、SS 外区结构 + 竖直高斯、开普勒尘埃、烟雾温度比、静止观者相机、T_peak 由 M、Ṁ 推出；多普勒 0.55 / 1.5、曝光 0.9、相机 dist 40；L / J2 / H 保留为退回预设。
5. **删除 cinematic palette 与 `--v2_visual_preset interstellar`**：删除 `palette_mode = cinematic`、warm_shift / saturation / visual_temp 映射及对应测试；颜色只走"线性化黑体 + 白平衡 + 频移"（S2）。

## 9. 文档同步

- `docs/design_ad_v2.md`：参数三层；结构层（atlas → 3D 程序化密度：SS 外区 + 噪声表面 + 7 层烟雾 + 大尺度低频）；温度（Page–Thorne、灰大气）；亮度 Y(g·T)、动态（刚体环）、相对论修正、颜色链路。
- `docs/design.md`：V2 后处理与视频管线。
- `README.md`：新增/废弃的 `--v2_*` 参数与视频用法。
- `AGENTS.md` 踩坑记录：亮度不能用 (T/T_peak)^p；灰大气侧壁热点；掠射增亮重复计算；NPGS 式 S ∝ 1/ρ 导致越密越暗；Taichi kernel 内不能用 `math.exp`；`ti.static` 展开多层噪声导致编译爆炸；`from __future__ import annotations` 与 `ti.template()` 冲突；Tanner Helland 为 sRGB 编码值。
- `docs/archived/realism_uplift_plan.md`：已归档（2026-10-02）。

---

## 10. 移植一致性修复（2026-10-01，V2 vs Proto 逐段对照）

验收工具：`python scripts/compare_v2_proto.py`（默认 1920×1080、ss = 2，同相机 / 同参数 / t = 2000；
输出终端指标 + `output/v2_dev/2026-10-02_外区与边侧视角/compare_v2_proto.png`，上 V2、下 Proto）。

| # | 偏差 | 修复 |
|---|---|---|
| 1 | 烟雾 / 低频噪声未归一（std ≈ 0.22，烟雾覆盖率 3% vs 35%） | `_calibrate_volume` 标定 `smoke_norm` / `low_norm`（与 Proto 同网格） |
| 2 | 低频层带中心约定不同（`r_b` 少 0.5 带、原点差 2 带） | `RigidRingBands(lnr0_bands=0, center_frac=0.5)` |
| 3 | 合成 `disk·α + sky·(1−α)` 重复乘透射率 | 体积路径输出 `I = Σ T·ΔI + T_end·I_sky`（`hdr_field`） |
| 4 | 落入视界的光线盘发射被清零（光子环下半部消失） | 先积分本段再判视界，发射保留 |
| 5 | 核心发射漏乘 `CORE_OPAC` | 发射与吸收同乘（源函数不变） |
| 6 | 体积路径混入 V1 `opacity_scale` | 去掉（κ 已按 `tau_i` 标定） |
| 7 | 光行时间无效（相位表全帧统一） | `_delayed_rot` / `_delayed_seed` 按每样本光程换算相位 |
| 8 | 静止观者相机未实现 | 主核内做方向变换 |
| 9 | 烟雾竖直剔除范围偏窄、步长细化缺烟雾层 | `max(3H, CL_EXTENT·r)`；步长函数与 Proto 一致 |
| 10 | bloom 半径 / 权重与 Proto 不同 | 改为 Proto 值（0.25/0.35/0.40，h/120、h/25、h/7） |
| 11 | 倾角下 g 因子坐标混用 | 光线方向转盘局部坐标 |
| 12 | 超采样白渲一遍 + 每帧重编译 | 构造时按 `ss` 分配内部分辨率，NumPy 盒式下采样 |
| 13 | CLI 单帧未传 `volume_params`（实际走旧 atlas 路径）；视频不传 t、曝光不锁 | 单帧走体积模型；`render(t=...)`；首帧曝光锁定；新增 `--v2_ss` |

修复后 1080p 指标（Proto / V2）：覆盖率 0.111 / 0.111，R/B 2.49 / 2.55，左右通量比 3.61 / 3.73，
光子环下半部通量占比 0.021 / 0.022，逐像素 |ln(V2/Proto)| 中位数 0.062（修复前 0.202）。
保护单测：`tests/unit/test_disk_v2_proto_parity.py`。

## 11. 旧模型清理（2026-10-02）

V2 只保留体积模型这一条路径：

- 删除模块：`visual_atlas.py`、`structure_modulations.py`、`preview.py`、`imaging.py`、`geometry.py`、`stats.py`
  （`hdr_luminance` 迁入 `postfx.py`）；`physical_fields.py` / `palette.py` / `relativity.py` 只保留体积模型用到的参考函数。
- 删除参数类 `DiskV2StructureParams`、`DiskV2PaletteParams`；`DiskV2Params` 只保留 `r_in`、`r_out`（默认 30）。
- `DiskV2Renderer` 只保留体积积分 kernel，构造参数精简为分辨率、`params`、`skybox`、`volume_params`、
  `r_max`、倾角、多普勒强度、`ss`。
- `render.py` 删除 24 个失效的 `--v2_*` 参数（含 `--v2_disable_g_factor`），单帧与视频共用 `_make_v2_renderer`；
  保留 `--disk_model`、`--v2_ss`、`--v2_orbit_seconds`。
- 删除 `scripts/v2_visual_acceptance.sh` 与 13 个只测旧代码的单测文件；有效用例迁入现存测试。
- 旧文档归档到 `docs/archived/`。
- 验证：清理前后 V2 HDR（960×540）逐位相同；V2 单测与 V1 e2e 全部通过。

## 12. 天空、倾角与视频验收（2026-10-02）

- **天空**：渲染核把盘发射与透过的天空分成两个缓冲；天空 sRGB 解码为线性光、双线性采样（V1 同约定），
  以 `--v2_sky_gain`（默认 0.5）在曝光之后叠加，星点参与 bloom。修复前程序星空使自动曝光按星点计算，
  盘被压成黑色剪影。黑天空下盘 HDR 与清理前逐位相同；亮天空下曝光不变（单测）。
- **倾角**：盘倾 θ、相机仰角 e ≡ 盘不倾、仰角 e + θ。960×540 实测 θ = 10° / 20°：逐像素
  `|ln|` 中位数 1e-4、总通量差 < 1e-5（f32 舍入）；亮侧 = 蓝移侧（左右通量比 4.1，B/R 0.49 vs 0.10）。无需修改，固化为单测。
- **多时刻对齐 Proto**（960×540，ss = 1；内缘周期 P = 2π/√(0.5/27) ≈ 46.2 r_s/c）：

  | t | 0 | P/4 | P | 5P |
  |---|---|---|---|---|
  | `|ln(V2/Proto)|` 中位数 | 0.064 | 0.052 | 0.066 | 0.068 |

- **卷绕**（结构场 log c 平均螺旋倾角，参考实现 `pitch_deg` 同式；内 / 中 / 外三段）：

  | 内缘圈数 | 0 | 1 | 3 | 10 | 20 |
  |---|---|---|---|---|---|
  | 刚体环（V2） | 19.4 / 11.4 / 6.6 | 19.6 / 11.6 / 6.5 | 19.1 / 12.0 / 6.4 | 19.5 / 11.9 / 7.0 | 19.4 / 11.9 / 6.8 |
  | naive `φ − Ω(r)t` | 19.3 / 11.8 / 6.5 | 10.6 / 10.9 / 6.5 | 6.5 / 8.0 / 6.3 | 5.3 / 5.2 / 5.1 | 5.1 / 4.3 / 3.9 |

- **延时视频**（`output/v2_dev/2026-10-02_外区与边侧视角/v2_longrun.mp4`，960×540，240 帧覆盖 20 圈）：近侧纹理 98% 的帧向右移动；
  逐像素相邻帧差 max/median = 1.35；盘总亮度逐帧相对变化中位数 0.9%、峰值 4.4%（孤立、不成簇，
  为大尺度明暗与种子更替在延时下的演化，不是曝光跳变）。
- **实时视频**（`output/v2_dev/2026-10-02_外区与边侧视角/v2_realtime.mp4`，1080p、ss = 2、240 帧 = 10 s、程序星空，经 CLI 渲染，耗时 3.2 h）：
  画面平均亮度逐帧相对变化中位数 0.13%、max 0.42%；逐像素相邻帧差中位数 0.06/255、max/median 1.64，无可见闪烁。
  10 s 内整体亮度缓慢下降 5.8%（曝光锁定 + 大尺度明暗演化）。
- **速度**（稳态每帧）：960×540 4.8 s；1080p 15.8 s；1080p ss = 2 52 s。

---

## 变更记录

- **v0.8 (2026-10-02)**：§12 天空（分缓冲 + 线性化 + 双线性 + `--v2_sky_gain`）、倾角验证、视频验收与测速。

- **v0.7 (2026-10-02)**：§11 旧模型清理；§3.1 概念边界改为指向 `design_ad_v2.md` §3.5。

- **v0.6 (2026-10-01)**：§10 移植一致性修复，V2 与 Proto 同参数逐像素对齐。
- **v0.5 (2026-10-01)**：全面物理 review 后定稿 M，参数分三层（基本参数 / 物理模型 / 视觉调节，§2）。新增：可见光亮度 Y(g·T)、Page–Thorne 温度（M、Ṁ 推 T_peak）、灰大气竖直温度、核心 τ≈3 温和起伏 + 温度起伏、SS 外区 H/Σ + 竖直高斯、开普勒尘埃（去重复掠射增亮）、烟雾温度比、静止观者相机；去掉 `EMIT_POW`、表面增亮、透镜状截面。S2/S5/S6/S9 与性能预算更新。
- **v0.4 (2026-10-01)**：学习 NPGS 最新 shader 后定稿改为 J2（模型 I：绝对厚度 + 透镜状截面 + 噪声表面、乘性级联 value noise、Σ(r) 解耦的局部热平衡、表面增亮、内区尘埃、连续步长 + 抖动、光行时间；叠加 H 的 7 层烟雾与大尺度明暗；对比度 50）；mode 3 保留；H 保留为退回预设。S3/S4/S5/S6 内容与性能预算更新。
- **v0.3 (2026-10-01)**：定稿预设 H（稀疏核心纹理 + 大尺度低频调制 + 7 层半吸收烟雾 + EMIT_POW 2.5 + CIE 黑体 + `doppler_color` 2.2 + 曝光 0.7 + 去品红色散）；S0 完成；方案冻结。
- **v0.2 (2026-10-01)**：§8 待决事项全部定稿（原型入库、删 atlas、删 `--v2_lum_power`、`ar2 = 30`、删 cinematic palette/preset）。
- **v0.1 (2026-10-01)**：首版。基于已确认的原型（mode 3 刚体环、核心 + 5 云层、修正 g、线性化颜色 + WB 5000K、`doppler_color = 1.6`、高光 bloom + 色散）整理移植方案。
