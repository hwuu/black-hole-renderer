# 黑洞渲染真实感提升计划

> **状态**：草稿（v0.3），尚未冻结，正在与既有设计真源对齐。
>
> **跟踪对象**：
>
> - 吸积盘体本身：以 [`docs/design_ad_v2.md`](../design_ad_v2.md) v2.1 为真源，本计划只引用、不重复定义。
> - 主渲染管线、相对论成像、后处理：以 [`docs/design.md`](../design.md) 为真源；本计划中涉及该范围的新增内容，目前在本文件内**临时落地**，待补独立设计文档后再迁出。
>
> **最近更新**：2026-06-13。
>
> **本文定位**：把"渲染效果不真实"的问题拆成可追踪的根因，给出按依赖排序的推进阶段。不重新定义吸积盘体模型本身，那由 `design_ad_v2.md` 负责。
>
> **非目标**：不解决 Kerr 黑洞、磁流体力学、全频谱辐射转移。不重写 Schwarzschild 测地线积分器。不引入 V3。
>
> **范围边界**：V2 盘体模型本身（几何、物理场、结构调制、调色）的改动落进 `disk_v2/`；主渲染接线、CLI 开关、Bloom 重排、相对论 g-factor 增强等按对应 Phase 修改 `render.py`、`docs/design.md`、`README.md`。

---

## 1. 背景

当前渲染器在 [`render.py`](../../render.py) 中已经具备：

- Schwarzschild 笛卡尔等效势光线追踪（[`render.py:2519`](../../render.py)）。
- 吸积盘程序纹理（[`render.py:920`](../../render.py) `build_disk_texture_rotating_state` 等）。
- 多次穿越倾斜平面 + 透明度累加（[`render.py:2941`](../../render.py) ~ [`render.py:3002`](../../render.py)）。
- 多普勒亮度 + 颜色偏移（[`render.py:2440`](../../render.py) `_apply_g_factor`）。
- Bloom + 色散 + 可选镜头光晕。

实际观感与"Interstellar 风格"或 EHT 风格存在明显差距。用户反馈集中在两点：

- 整体偏"程序贴图"或"高斯模糊画出来的发光盘"。
- 没有"很多发光粒子/气体团沿视线累积"的体积感、云雾感。

并且 v1.0 阶段的 `disk_v2/` 已经实现了几何 / 物理场 / 结构调制 / 参数四个基础模块，但用临时脚本预览出来"黄乎乎一片、无细节"。这迫使我们在推进新阶段前，先把 V2 v1.0 的根因看清——见 [`docs/design_ad_v2.md` 附录 A](../design_ad_v2.md)。

本计划的职责：

- 把视觉上的"不真实"拆成可追踪的根因 P1 ~ P5。
- 把各根因映射到 `design_ad_v2.md` v2.1 的五项大改，明确"哪些根因 V2 大改覆盖、哪些没覆盖"。
- 为 V2 没覆盖的部分（相对论成像增强、高阶像、天空盒、后处理校准）排出推进顺序。
- 在用户视觉体验最差的根因上优先投入。

---

## 2. 根因分析

下面五条是当前视觉问题的根因。后续所有阶段都会明确对应到这里的编号。

### 2.1 P1：吸积盘没有体积，光线只与一个零厚度平面相交

**事实**：[`render.py:2941`](../../render.py) 用 `f_old * f_new < 0` 检测光线穿过倾斜平面 `z = y · tan(tilt)`。命中后调用 `_sample_disk()`（[`render.py:2959`](../../render.py)）取出一张二维 RGBA 纹理，再用 `disk_alpha_total` 做前向不透明度合成（[`render.py:2994`](../../render.py) ~ [`render.py:3002`](../../render.py)）。

**推断**：

- 盘体感、自遮挡、边缘增亮全部来自"沿光线穿过等离子体云团"的累积。零厚度平面无法表达这一过程。
- 即使纹理本身已经包含 FBM、热斑、丝状结构，这些细节也只能以二维投影形式出现，不会随观察角度改变积分长度。

**工程判断**：在不引入有限厚度积分的前提下，仅靠继续加强 2D 纹理无法根治"贴图感"。

### 2.2 P2：温度、密度、调色都没有承担真实感

**事实**：

- V1：[`render.py:795`](../../render.py) `_generate_temperature_base` 用 FBM 噪声 + 径向衰减组合出"温度"。颜色由 `_blackbody_rgb()`（[`render.py:136`](../../render.py)）按这条温度查表得到。
- V2 v1.0：[`disk_v2/physical_fields.py:95`](../../disk_v2/physical_fields.py) 实现了标准薄盘温度峰值环，但实物诊断（见 [`design_ad_v2.md` 附录 A](../design_ad_v2.md)）显示未乘径向边界权重的 raw temperature profile 在默认半径范围内只有约 1.89 倍跨度，且没有调色层（`palette.py` 缺失）。

**与真实差距**：

- 标准薄盘启发式给出 `T_raw(r) ∝ r^(-3/4) · [1 - sqrt(r_in/r)]^(1/4)`，量纲应是开氏度。V2 v1.0 把它当作 0 ~ 1 的归一量来用；在默认 `r_in=2, r_out=10` 下，除去精确内边界的 0 起点，raw profile 的盘内有效温度跨度只有约 1.89 倍，完整温度场还会由 `W_r` 在精确边界收口，整体撑不起明显色阶。
- V1 的 `_blackbody_rgb` 设计是按真实温度查表，但接收的"温度"也是程序图案，与物理温度不挂钩。
- ISCO 约束：Schwarzschild 的 ISCO 等于 `3 · r_s`。当前默认 `--ar1 = 2.0`（[`README.md:69`](../../README.md)）落在 ISCO 内部的 plunging region，没有物理对应的稳定圆轨道。

### 2.3 P3：差动旋转与结构剪切表达不一致

**事实**：

- V1 视频模式提供三种旋转算法（[`AGENTS.md:75`](../../AGENTS.md) 视频旋转算法速记）。
- `parametric` 模式已经修过"组件方向不一致"问题（[`AGENTS.md:106`](../../AGENTS.md)、[`tests/unit/test_parametric_rotation_direction.py`](../../tests/unit/test_parametric_rotation_direction.py)）。
- 但结构本质上还是"一张二维纹理在旋转"，不是"流体被持续剪切"。
- V2 v1.0 的 `F_shear`（[`disk_v2/structure_modulations.py:194`](../../disk_v2/structure_modulations.py)）虽然在 `(r, φ_adv)` 空间叠加傅里叶分量，但频谱衰减 `0.5^k` 太快，高频细节几乎丢失。

**推断**：

- 缺少"内圈被快速拖拽、外圈缓慢"导致的拉伸、缠绕、撕裂特征。
- V1 已用 `omega` 表达内快外慢，但调制信号本身仍是局部图案，被剪切后只表现为整体位移。
- V2 v1.0 的 `F_shear` 频谱过低，看不到细丝感。

### 2.4 P4：相对论亮度与颜色的物理一致性不足

**事实**：

- `_apply_g_factor`（[`render.py:2440`](../../render.py)）实现了 Doppler g 因子 + 引力红移因子。
- 亮度按 `g^lum_power` 缩放（[`render.py:2478`](../../render.py)），由参数 `lum_power` 控制，不是严格 `g^4` beaming。
- 颜色按 Wien 近似平移，基准温度被固定写为 `~10000 K`（[`render.py:2495`](../../render.py) ~ [`render.py:2503`](../../render.py)）。
- 输出经 `clamp(..., 0.0, 10.0)`（[`render.py:2516`](../../render.py)）压制。
- 圆轨道角速度用 `omega = sqrt(0.5 / r³)`（[`render.py:2452`](../../render.py)），未带 Schwarzschild 度规校正。

**工程判断**：

- 物理上，方向性 beaming 强度按强度量变换应是 `g^4`。当前 `lum_power` 偏小时蓝移侧的高亮表达不足。
- Wien 平移基准温度与盘上实际温度脱钩，导致颜色偏移方向与真实情况不一定一致。
- `clamp(0, 10)` 是后处理性质的硬截断，但渲染管线没有色调映射兜底，提升 `lum_power` 到 4 时会直接饱和。

### 2.5 P5：高阶像、光子环、天空盒、后处理共同削弱真实感

**事实**：

- 命中循环不限制穿越次数（[`render.py:2854`](../../render.py) ~ [`render.py:3006`](../../render.py)），透明度按 `front_factor = 1 - disk_alpha_total` 收敛，理论上能产生主像 + 次像 + 高阶像。
- `r_escape = max(r_max, distance * 2)`（[`render.py:3830`](../../render.py)），`max_affine = r_escape * 40`（[`render.py:2818`](../../render.py)），不是固定 `r_max = 10`。
- 默认天空盒在没传纹理时由程序生成（[`README.md:65`](../../README.md) `--texture` 默认"程序生成"）。
- Bloom + 色散在最后阶段叠加（[`render.py:3022`](../../render.py) 附近 `_bloom_kernel`）。

**推断与未验证项**：

- 主像、次像在亮度和颜色层次上是否清晰可辨，目前没有独立验收口径或回归图。
- `r_escape` 在小相机距离（如 `d = 3`）下取 `max(10, 6) = 10`，光线绕回余地是否充足，未实测。
- 程序生成天空盒削弱"扭曲星空"层次，是合理推断；具体影响大小未测。
- Bloom 强度可能掩盖物理细节（光子环、内缘结构），合理推断；未量化。

后三条要避免写成事实，需在 Phase 0 实测后再下结论。

---

## 3. 目标视觉特征

按优先级从高到低：

| 优先级 | 特征 | 对应根因 |
|--------|------|----------|
| 1 | **云雾粒子感**：盘看起来像许多热气体团沿视线累积发光，光线穿盘时多次进入暗区与亮区 | P1、P3 |
| 2 | **有限厚度感**：高倾角时盘边缘增厚、近侧遮挡远侧、边缘路径更长更亮 | P1 |
| 3 | **团块边界**：盘面正视图中存在大量边界清晰（非渐变）的局部亮块 | P3 |
| 4 | **温度色阶**：内缘呈蓝白、中段呈橙黄、外圈呈暗红，至少三种可辨色相 | P2 |
| 5 | **相对论方向性**：朝观察者一侧明显增亮变蓝，远离一侧变暗变红 | P4 |
| 6 | **高阶像层次**：主像、次像、光子环在亮度、宽度、颜色上可分辨 | P5 |
| 7 | **干净的后处理**：Bloom 与色散增强氛围，不吞掉上述特征 | P5 |

---

## 4. 与现有设计真源的关系

### 4.1 `docs/design_ad_v2.md` v2.1 覆盖的根因

`design_ad_v2.md` v2.1 通过五项大改覆盖以下根因：

| 大改 | 内容 | 解决根因 |
|------|------|----------|
| 大改 1 | `F_struct` 升级到 `(r, φ, z)` 三维，引入 3D 体积噪声 | P1 部分、P3 |
| 大改 2 | 新增 `F_clump`：稀疏、高对比、边界清晰团块项 | P3 |
| 大改 3 | 直接 Taichi 实现，砍掉 NumPy reference | 实施可行性 |
| 大改 4 | `palette.py`：温度→黑体色 + HDR 色调映射 | P2、为 P4 兜底 |
| 大改 5 | 温度量纲化、密度内边界压制、`r_in ≥ 3 r_s` | P2 |

### 4.2 `docs/design_ad_v2.md` 没有覆盖的根因

- **P1 完整解决** 需要：V2 五项大改 + 有限厚度积分接入主光追。`design_ad_v2.md` Phase 4 覆盖了接入部分，但本计划要给出对应的端到端验收。
- **P4 完整解决** 需要：在主渲染器层面调整 `_apply_g_factor`，把 `lum_power` 推到 4、把 Wien 基准温度改为从 V2 真温度场采样。`design_ad_v2.md` v2.1 §1.3 仍将完整相对论光学列为非目标。
- **P5（高阶像、天空盒、后处理）**：`design_ad_v2.md` 不涉及，本计划独立推进。

### 4.3 本计划与 `design_ad_v2.md` 的分工

- **本计划**：跟踪推进节奏、视觉验收基线、与主渲染器的集成节奏、相对论增强与后处理的设计与排期。
- **`design_ad_v2.md`**：定义 V2 各场的数学形式、参数默认值、接口设计、单元测试硬指标。

---

## 5. 总体路线

```
  +-----------+   +---------------+   +-----------+   +-------------+
  | Baseline  |-->| Disk V2 Build |-->| GR Look   |-->| Composition |
  +-----------+   +---------------+   +-----------+   +-------------+
       |                |                 |                 |
       v                v                 v                 v
  +-----------+   +---------------+   +-----------+   +-------------+
  | Capture   |   | Cover P1+P2   |   | Cover P4  |   | Cover P5    |
  | Problems  |   | + P3 (V2)     |   |           |   |             |
  +-----------+   +---------------+   +-----------+   +-------------+
```

四个阶段块、内嵌九个 Phase：

- **Baseline 块**：Phase 0。
- **Disk V2 Build 块**：Phase 1 → Phase 2 → Phase 3 → Phase 4（对应 `design_ad_v2.md` Phase 1 ~ Phase 4）。
- **GR Look 块**：Phase 5（相对论亮度颜色）。
- **Composition 块**：Phase 6（高阶像 + 天空盒）→ Phase 7（后处理校准）。

关键依赖：

- **V2 的 HDR 与色调映射先于 GR 增强**：`design_ad_v2.md` 大改 4 在 Phase 3 落地。之后再调 `lum_power` 到 4 才不会饱和。
- **盘接入主渲染器后必须 Checkpoint**：Phase 4 结束后做一次评估，决定 Phase 5 ~ Phase 7 是否需要调整或合并。

---

## 6. 跟踪总览

| 阶段 | 标题 | 解决根因 | 状态 | 验收口径真源 | 证据 / 说明 |
|------|------|----------|------|-------------|-----------|
| Phase 0 | 建立问题基线 | P1 ~ P5 | 未开始 | 本计划 §7.1 | 暂无 |
| Phase 1 | V2 基础物理场调整（温度量纲、密度内边界、ISCO、r_out=50） | P2 | 未开始 | [`design_ad_v2.md` §7.1](../design_ad_v2.md) | 暂无 |
| Phase 2 | V2 三维结构调制与团块项 | P3、P1 部分 | 未开始 | [`design_ad_v2.md` §7.2](../design_ad_v2.md) | 暂无 |
| Phase 3 | V2 调色与色调映射 | P2、为 P4 兜底 | 未开始 | [`design_ad_v2.md` §7.3](../design_ad_v2.md) | 暂无 |
| Phase 4 | V2 Taichi 实现 + parity 测试 + 接入主光追 | P1 完整 + 集成 | 未开始 | [`design_ad_v2.md` §7.4](../design_ad_v2.md) | 暂无 |
| **Checkpoint** | Phase 4 后重评 | — | 未开始 | 本计划 §7.6 | 暂无 |
| Phase 5 | 相对论亮度与颜色 | P4 | 未开始 | 本计划 §7.7 | 暂无 |
| Phase 6 | 高阶像、光子环、天空盒 | P5 | 未开始 | 本计划 §7.8 | 暂无 |
| Phase 7 | 后处理校准 | P5 | 未开始 | 本计划 §7.9 | 暂无 |

每个 Phase 完成后，请在 **状态** 列改为"进行中 / 完成"，并在 **证据 / 说明** 列追加 PR 链接、关键命令、或关键文件路径，使新读者无需翻历史也能复核进度。

---

## 7. 分阶段计划

### 7.1 Phase 0：建立问题基线

**目标**：固定参考画面，避免后续只凭主观感受判断改动是否有效。

**实现内容**：

- 选定三组参考相机参数：
  - 正视（盘面正对相机）：`python render.py --pov 6 0 0.5 --fov 90 --ar1 3 --ar2 50 --disk_tilt 0 -r hd`
  - 倾视（典型 Interstellar 角度）：`python render.py --pov 30 0 10 --fov 90 --ar1 3 --ar2 50 --disk_tilt 20 -r hd`
  - 侧视（高倾角，强化厚度问题）：`python render.py --pov 20 0 1 --fov 90 --ar1 3 --ar2 50 --disk_tilt 80 -r hd`
- 注意：`--ar1` 已经从 v1.0 的默认 2.0 提升到 3.0，对齐 Schwarzschild ISCO；`--ar2` 从 10 提到 50，让温度物理跨度从 1.89 倍提到 4.32 倍。相机距离 `--pov` 已经按盘体变大同步拉远，避免盘体撑出画面。如果想保留 v1.0 默认值的对照，加做一组 `--ar1 2.0 --ar2 10` 的版本（注意 V1 模式下不强制 ISCO）。
- 每组渲染两版：默认开启 Bloom 的版本、关闭或降低 Bloom 强度的版本。
- 把所有基线图存到 `output/baseline/`，并在本节追加一个"基线观察"小节，逐张写明能看到的 P1 ~ P5 现象。
- 不修改源码，不修改默认参数。

**涉及文件**：

- [`docs/plans/realism_uplift_plan.md`](./realism_uplift_plan.md)（本文件）。
- `output/baseline/`（临时目录，不进版本控制）。

**验收**：

- 所有基线图可由记录的命令复现。
- 文档新增"基线观察"小节，每张图至少对应一条 P? 现象。
- 后续 Phase 的视觉对比有可锚定的起点。

### 7.2 Phase 1：V2 基础物理场调整

**目标**：覆盖 P2 中"温度场、密度场、ISCO"三条物理基础。

**实现内容（详见 [`design_ad_v2.md` §7.1`](../design_ad_v2.md)）**：

- `params.py`：`temp_scale` → `T_peak_K`（默认 `1.0e7`），`rho_power` 默认 `1.5`，`r_in` 默认 `3.0` 并强制校验 ≥ `3.0`，`r_out` 默认 `50.0`。
- `physical_fields.py`：温度公式用 `T_peak_K · norm_factor` 让未乘径向边界权重的 raw profile 峰值落在 `T_peak_K`；密度公式加入 `[1 - sqrt(r_in/r)]^(1/2)` 内边界压制。
- 新增 `alpha_density`、`beta_temperature` 参数控制发射率指数。

**测试要点**：

- 单元测试：
  - `test_v2_temperature_range_default`：未乘 `W_r` 的 raw thin-disk profile 在默认 `r_in=3, r_out=50` 下，峰值 / `r_out` 处比值落在 `[4.0, 4.6]`（实算 ≈ 4.32）。注意不要直接测完整 `T_mid(r_out)`，因为径向边界权重在精确外边界收口为 0。
  - 密度场在 `r = r_in` 处取 0、在 `r ≈ 1.5 · r_in` 处取峰值。
  - `r_in < 3.0` 时发出 warning 并钳制。
- 静态参数校验（不出图，用打印数值）：
  - 温度场峰值位置接近 `r ≈ 1.36 · r_in`。
  - 密度场峰值位置接近 `r ≈ 1.5 · r_in`。

**涉及文件**：

- [`disk_v2/params.py`](../../disk_v2/params.py)、[`disk_v2/physical_fields.py`](../../disk_v2/physical_fields.py)。
- 现有单测调整。

### 7.3 Phase 2：V2 三维结构调制与团块项

**目标**：覆盖 P3 与 P1 的部分（云雾粒子感的"团块边界 + 三维振荡"）。

**Phase 2 启动前的决策点**：根据 [`design_ad_v2.md` §8`](../design_ad_v2.md)，`F_clump` 首版**只实现一种算法**。当前推荐显式点云团（参数直观、调试容易），Worley/Voronoi noise 作为首版不达标时的回退方案。Phase 2 不允许并行实现两种。

**实现内容（详见 [`design_ad_v2.md` §7.2`](../design_ad_v2.md)）**：

- 重构 `structure_modulations.py` 接受 `(r, φ, z)` 三维输入。
- 新增 `F_clump`：**显式点云团**首版实现，`clump_count = 400` 起步。
  - 径向尺度 `0.2 · r_in`、角向弧长 `0.1 · r ~ 0.3 · r`、垂向 `0.5 · H`。
  - 振幅 `clump_strength = 0.6`，远大于现有 `shear_strength = 0.22`。
  - 边界锐利：从核到外缘 0.1 倍尺度内幅度跌至 50% 以下。
- 调整 `F_shear`：频谱从 `0.5^k` 改为 `1/k^(1/2)`；分量数 8 → 16；去除全局 `_normalize_signed`，改高斯 3σ 截断。
- 区分 `F_struct_density`（主要 = `F_shear · F_clump`）与 `F_struct_emission`（主要 = `F_clump · F_hotspot`）。

**测试要点**：

- 单元测试：
  - `F_clump` 团块数量在容差内匹配 `clump_count`。
  - `F_clump` 边界锐度满足"0.1 倍尺度内跌至 50%"。
  - 高倾角光线穿盘路径上 `F_struct_density` 至少出现 3 次极值。
  - 所有分量保持 `F > 0` 且围绕 1 波动，盘外返回 1。
- 静态参数校验：
  - 单帧二维投影上观察团块分布。

**涉及文件**：

- [`disk_v2/params.py`](../../disk_v2/params.py)、[`disk_v2/structure_modulations.py`](../../disk_v2/structure_modulations.py)。
- 新增 `tests/unit/test_disk_v2_clump.py`。

### 7.4 Phase 3：V2 调色与色调映射

**目标**：覆盖 P2 中"颜色"部分，并为 P4 提供 HDR 兜底。

**实现内容（详见 [`design_ad_v2.md` §7.3`](../design_ad_v2.md)）**：

- 新增 [`disk_v2/palette.py`](../../disk_v2/palette.py)：
  - `blackbody_color(T_K)` 复用 `render.py:136` 的查表。
  - `tonemap_reinhard(rgb_hdr)`、`gamma_correct(rgb)`。
  - `palette_mode = "physical" | "cinematic"`。
- 新增 `DiskV2PaletteParams`。
- 渲染管线中间保持 HDR 浮点；Bloom 移到色调映射之前；移除 `clamp(0, 10)` 硬截断。

**测试要点**：

- 单元测试：
  - 高温（≥ `1e7 K`）输出偏蓝白，低温（~`3000 K`）输出偏红。
  - Reinhard 在 `[0, ∞)` 输入下输出 `[0, 1)`。
  - 伽马校正满足 `gamma_correct(x)^2.2 ≈ x`（容差内）。
- 静态参数校验：
  - 关闭色调映射时已知 HDR 输入饱和；开启后细节恢复。

**涉及文件**：

- [`disk_v2/palette.py`](../../disk_v2/palette.py) 新建。
- 新增 `tests/unit/test_disk_v2_palette.py`。

### 7.5 Phase 4：V2 Taichi 实现与接入主光追

**目标**：把 V2 接到 Schwarzschild 光追管线里，覆盖 P1 的完整体积感。

**实现内容（详见 [`design_ad_v2.md` §7.4`](../design_ad_v2.md)）**：

- 新增 [`disk_v2/taichi_impl.py`](../../disk_v2/taichi_impl.py)：上述场的 Taichi 版本。
- 新增 [`disk_v2/preview.py`](../../disk_v2/preview.py)：单独的静态参数校验入口（不承担视觉验收，只用于排错）。
- **新增 `tests/unit/test_disk_v2_numpy_taichi_parity.py`**：固定小网格下，比较 NumPy 参考实现（`physical_fields.py`、`structure_modulations.py`、`palette.py`）与 Taichi 实现的输出，相对误差 `< 1e-5`（fp32 路径可放宽到 `< 1e-3`）。**这条测试必须与 `taichi_impl.py` 同步入仓**——没有 parity 测试时，后续任何"出图不对"的视觉问题都会陷入"是 V2 物理不对、是 Taichi 实现写错、还是主光追集成出错"的多重不确定。
- `render.py` 加 `--disk_model {v1, v2}` CLI 开关，默认 `v1`。
- 命中盘体的部分改为：先判定光线是否进入盘体包围体，进入则做有限步长发射-吸收积分。
- V1 保留为完整回退路径。
- 接入完成后用 Phase 0 的三组参考相机重渲染一遍，与基线对照存档。

**测试要点**：

- 端到端回归（基于 [`tests/e2e_render.py`](../../tests/e2e_render.py) 风格）：
  - V1 默认渲染哈希不变。
  - V2 固定参数下输出可复现。
- NumPy/Taichi parity 测试通过。
- 人工视觉验收：覆盖 [`design_ad_v2.md` §2.3`](../design_ad_v2.md) 的硬指标。
- 性能验收：720p 单帧 GPU 估算 ≤ `1.5 s`，CPU 估算 ≤ `8 s`（详见 [`design_ad_v2.md` §5.4`](../design_ad_v2.md)）。

**涉及文件**：

- [`disk_v2/taichi_impl.py`](../../disk_v2/taichi_impl.py)、[`disk_v2/preview.py`](../../disk_v2/preview.py) 新建。
- `tests/unit/test_disk_v2_numpy_taichi_parity.py` 新建。
- [`render.py`](../../render.py)。
- [`tests/e2e_render.py`](../../tests/e2e_render.py) 增加 V2 回归。

### 7.6 Checkpoint：Phase 4 完成后重评

**目标**：避免线性推进到 Phase 7，确认剩余阶段是否仍是用户最关心的项。

**做法**：

- 重新评估 P4、P5 在当前 V2 输出下的实际严重程度。
- 决定 Phase 5 ~ Phase 7 的顺序、合并或裁剪。
- 在本节末尾追加一段"重评结论"，作为后续阶段的依据。

### 7.7 Phase 5：相对论亮度与颜色

**目标**：覆盖 P4。

**实现内容**：

- 调整 `_apply_g_factor`（[`render.py:2440`](../../render.py)）：
  - 默认 `lum_power = 4`，对应 `I_obs / I_em = g^4`，在 docstring 中写清公式与符号约定。
  - 圆轨道角速度替换为 Schwarzschild 度规对应的物理表达，明确推导。
- 颜色偏移改成由实际温度场驱动：
  - 把 Wien 平移的基准温度改成从 V2 温度场实际采样得到，而不是固定 `1e4 K`。
  - V1 模式仍在使用时，保留旧行为作为兼容路径。

**前置条件**：

- Phase 3 已落地，HDR + Reinhard 提供动态范围兜底。否则 `g^4` 会直接饱和。

**涉及文件**：

- [`render.py`](../../render.py)。
- [`disk_v2/physical_fields.py`](../../disk_v2/physical_fields.py)、[`disk_v2/palette.py`](../../disk_v2/palette.py)：温度采样接口。
- [`docs/design.md`](../design.md)。

**测试要点**：

- 单元测试：
  - 朝向观察者的速度方向 → `g > 1`。
  - 远离观察者的速度方向 → `g < 1`。
  - `g = 1` 时颜色不偏移。
- 视觉验收：
  - 固定相机下，盘的一侧明显增亮变蓝，另一侧变暗变红。
  - 提升 `lum_power` 到 4 后整体亮度保持在色调映射可控范围内。

### 7.8 Phase 6：高阶像、光子环、天空盒

**目标**：覆盖 P5 中"高阶像可辨识"与"扭曲星空"两条。

**实现内容**：

- 高阶像与光子环：
  - 用 Phase 0 中"侧视 + 关闭 Bloom"的基线观察实际是否能看到次像。如能看到但偏弱，调 `--r_max`、`--step_size` 评估稳定性；如看不到，先在固定相机下加日志统计每像素的命中次数分布，再决定是否调整积分参数。
  - 是否需要为高阶像引入光程衰减，依据观察决定，不预设结论。
- 天空盒：
  - 评估替换为静态银河系背景图的成本。
  - 若决定支持，扩展现有 `--texture` 入口（[`README.md:65`](../../README.md)）。

**涉及文件**：

- [`render.py`](../../render.py)。
- [`tests/e2e_render.py`](../../tests/e2e_render.py)：新增高倾角参考回归。
- [`docs/design.md`](../design.md)。

**测试要点**：

- 端到端回归：
  - 高倾角参考相机下，固定参数渲染哈希在多次运行间稳定。
- 视觉验收：
  - 主像、次像、光子环在亮度、宽度上可分辨。
  - 替换天空盒后透镜扭曲层次清晰。

### 7.9 Phase 7：后处理校准

**目标**：覆盖 P5 中"后处理掩盖物理细节"。

**实现内容**：

- 重新校准 Bloom 阈值、强度、半径，使其默认值不再吞掉光子环和内缘细节。
- 提供两档：弱 Bloom 作为默认、强 Bloom 作为电影感预设。

**涉及文件**：

- [`render.py`](../../render.py)。
- [`docs/design.md`](../design.md)。

**测试要点**：

- 视觉验收：
  - 关闭 Bloom 时图像本身仍然成立。
  - 开启 Bloom 后光子环、内缘结构、云絮细节仍可见。

---

## 8. 假设、风险与未覆盖项

### 8.1 假设

- 视觉真实感的瓶颈集中在 P1 ~ P5 这五条。若实际瓶颈在别处（例如 BRDF、散射、偏振），本计划顺序需要重排。
- 用户对"Interstellar 风格"和"EHT 风格"的接受度大致一致。若用户实际只想要其中一种，参数取值范围会不同。
- 项目仍以 CPU + Taichi 为主要运行环境。`design_ad_v2.md` Phase 4 性能估算同时给出 CPU 和 GPU 估算，决策由实施时机决定。

### 8.2 已知风险

- **`F_clump` 算法决策**：Phase 2 启动前必须选定一种首版算法。`design_ad_v2.md` v2.1 已经确定首版采用**显式点云团**，Worley/Voronoi 作为首版不达标的回退方案。Phase 2 不并行实现两套。
- **HDR 调参成本**：Reinhard 简单但可能整体偏灰。需要在 Phase 3 后单独评估是否切换 ACES。
- **NumPy 参考实现与 Taichi 实现漂移**：单测继续测 NumPy 路径，视觉验收走 Taichi 路径。Phase 4 必须加 parity 测试（见 §7.5），否则后续视觉问题难以定位是物理错还是 Taichi 写错。
- **Taichi 兼容性**：3D 体积噪声在 Taichi 中可能受限。`design_ad_v2.md` v2.1 §8 提到回退方案为预生成体素数据。
- **回归测试稳定性**：色调映射会改变所有像素，e2e 哈希回归可能频繁失效。需要为每个阶段单独冻结回归基线。

### 8.3 未覆盖项

- 偏振、康普顿散射、辐射转移等纯科研项。
- Kerr 黑洞、磁流体力学、热致风。
- 实时交互（项目目标是离线渲染）。

---

## 9. 文档同步要求

每完成一个 Phase：

- 同步更新 `docs/plans/realism_uplift_plan.md` 第 6 节跟踪表的状态列与证据列。
- 涉及吸积盘体本身的改动 → 更新 [`docs/design_ad_v2.md`](../design_ad_v2.md)。
- 涉及主渲染路径、相对论模型、HDR、后处理 → 更新 [`docs/design.md`](../design.md)。
- 涉及用户可见参数 → 更新 [`README.md`](../../README.md)。
- 涉及踩坑超过两次的环境或方向问题 → 追加到 [`AGENTS.md`](../../AGENTS.md) 踩坑记录。

---

## 10. 评审检查问题

留给评审人在下一轮 Review 前追问：

- 第 4 节里把哪些根因分给了 V2 大改，是否准确？V2 团队是否同意 P3 完整解决在 V2 Phase 2 内？
- Phase 1 ~ Phase 4 是否需要并行某些子任务以缩短总耗时？
- Phase 0 基线参数已经把 `--ar1` 提升到 3.0、`--ar2` 提升到 50.0，是否也要保留 v1.0 默认值 `--ar1 2.0 --ar2 10` 的对照？
- `--ar2 = 50` 把盘体半径从 10 提到 50（5 倍）。相机距离 `--pov` 已经在 Phase 0 命令里同步拉远（从 6/4 → 30/20），是否仍会出现盘体撑出画面或视觉构图失衡？
- `T_peak_K = 1e7` 适合 stellar-mass BH，supermassive BH（如 M87 量级）对应 `~1e5 K`，是否在 v2.1 内直接提供 `--bh_type` preset？
- Phase 6 是否值得引入静态银河系天空盒？如果引入，由谁提供版权清晰的纹理？
- Phase 4 的视觉验收是否要新增"对照 V1 + 对照真实 EHT 图"的两组参考？

---

## 11. V1.0 V2 实施回顾

v1.0 阶段的 V2 已实施 `disk_v2/{geometry, physical_fields, structure_modulations, params}.py` 四个基础模块和对应单测，但用临时脚本预览结果是"黄乎乎一片、无细节"。

对应的代码级诊断写在 [`docs/design_ad_v2.md` 附录 A`](../design_ad_v2.md)，简要回顾如下：

- **R1**：温度场动态范围在 v1.0 默认 `r_in=2, r_out=10` 下只有 1.89 倍（公式量纲化为 0~1，没用真实 K）。要拉开跨度需要 `r_out` 加大 + palette 非线性查表，光靠量纲化解决不了。
- **R2**：密度场无内边界压制，与温度场形态不一致。
- **R3**：结构调制振幅 ±50% 以内，频谱衰减 `0.5^k` 太快，全是连续光滑场，无团块边界。
- **R4**：垂向只有高斯钟形，无 `(r, φ, z)` 三维结构。
- **R5**：没有 `palette.py`，没有调色映射，没有色调映射。

这些诊断驱动了 `design_ad_v2.md` v2.1 的五项大改，并直接决定了本计划 Phase 1 ~ Phase 4 的内容。

**经验沉淀**：

- 类似项目下次启动时，**先把 palette + tonemap 这条"显示链路"和"测一个单峰静态场看出图"的预览闭环建好**，再开始堆积物理场。不然到了后面所有东西都做完了，出图还是看不出对错。
- **物理量纲不能省**。即使最终输出归一化到 RGB，物理量纲是判断每一层是否合理的基础。`T_peak_K = 1e7` 这种约束一开始就该立。
- **写"跨 X 个数量级"这种数字之前先实算**。v0.2 的本计划和 v2.0 的 `design_ad_v2.md` 都写过"温度跨 3 个数量级"，实算后跨度只有 1.89 倍（`r_out=10` 时）或 4.32 倍（`r_out=50` 时）。错误的物理结论会直接误导后续 Phase 验收指标。

---

## 变更记录

- **v0.5 (2026-06-13)**：吸收 gpt 5.5 修订 + 版本号同步。P2 诊断和 Phase 1 验收明确测未乘 `W_r` 的 raw temperature profile（避免完整 `T_mid(r_out)=0` 与 `T_peak/T_at_r_out ∈ [4.0, 4.6]` 字面冲突）。正文 6 处真源版本号统一引用 `design_ad_v2.md` v2.1。
- **v0.4 (2026-06-13)**：按 gpt 反馈修正硬伤。修正温度跨度伪具体（v0.3 写"3 个数量级"，实算 1.89 倍，要求 `r_out=50` 才能到 4.32 倍）。范围边界句拆开：V2 盘体改动落 `disk_v2/`，主光追接线和后处理按阶段改 `render.py`。第 6 节跟踪表加"证据 / 说明"列。Phase 0 命令同步改 `--ar2 50`，相机距离 `--pov` 同步拉远。Phase 1 验收指标从不可能的 `≥100` 改为基于实算的 `[4.0, 4.6]`。Phase 4 加 NumPy/Taichi parity 测试。风险表 F_clump 改为决策点（显式点云团首选、Worley 回退）。第 11 节经验沉淀加"写'跨 X 个数量级'之前先实算"。
- **v0.3 (2026-06-13)**：基于"V2 v1.0 实物诊断"重大修订。删除"V3 备选"等所有 V3 描述。Phase 1 ~ Phase 4 重排为对接 `design_ad_v2.md` v2.0 的五项大改。新增第 11 节回顾 V1.0 V2 的实施经验。删除原 "Phase 6 HDR" 独立阶段（合并进 V2 Phase 3）。`--ar1` 默认值在 Phase 0 基线命令里改为 3.0。
- v0.2 (2026-06-13)：按 Review 意见全面重写。补元信息块；P1 ~ P5 编号；关键事实加 `file:line` 引用；Phase 顺序调整为"先体积积分、再调结构"；新增 HDR + 色调映射阶段；新增 ISCO 默认值约束；去中英夹杂、去辩论句式；新增第 6 节跟踪总览表与第 10 节评审检查问题。
- v0.1 (2026-06-13)：首版草稿，提出根因分析与九阶段计划。
