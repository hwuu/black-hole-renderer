# V2 视觉恢复方案

> **状态**：已实施（2026-06-14）。Step 0~5 完成；最终人工验收以用户参考图为准。
>
> **触发原因**：Phase 1~7 代码链路已通，但 V2 单帧视觉三轮迭代均失败——全白（save_image bug）、鬣狗斑（团块进发射）、斑马纹（纯傅里叶 F_shear）。用户已明确：最终验收标准是外部参考图，而不是仓库内 V1 输出。V1 经典参数（`output/v1_classic_darksky.png`）只作为技术对照。
>
> **真源关系**：盘体架构仍以 [`docs/design_ad_v2.md`](../design_ad_v2.md) 为准；本方案只修正 **结构纹理层职责划分** 与 **视觉验收口径**，不推翻几何 / 物理场 / palette / 体积积分 / g-factor 已有实现。
>
> **非目标**：本方案不引入 Kerr、不做视频模式、不重写测地线积分器、不在 Taichi 内实时算 FBM。

---

## 1. 问题复盘

### 1.1 已确认失败模式

| 迭代 | 做法 | 视觉结果 | 根因 |
|------|------|----------|------|
| A | `save_image` 把 uint8 当 float clip | 99% 死白 | 保存 bug（已修） |
| B | `F_clump` 全强度进发射 / 调制密度 | 大块亮斑（鬣狗斑） | 显式团块尺度远大于像素，不适合做主发射纹理 |
| C | 只用高频 `F_shear` + Bloom | 条带状、发灰发脏 | 傅里叶叠加产生周期纹，无 V1 多尺度 FBM 的絮状连续感 |

### 1.2 对照基准

**最终验收参考图**：

`/Users/hwuu/.cursor/projects/Users-hwuu-dev-github-hwuu-black-hole-renderer/assets/__2026-06-14_13.24.27-e87096b9-f8b3-41e9-8d81-9ceb9b51d0fb.png`

**仓库内技术对照**：`output/v1_classic_darksky.png`

说明：V1 classic 只用于确认构图、透镜环与基础色带没有跑偏；最终人工验收以用户参考图为准。

**验收相机（固定，不再随迭代漂移；沿原方向后撤到远景以看到吸积盘边缘）**：

```bash
--texture output/black_sky.png \
--pov 24 0 8 --fov 90 --ar1 2 --ar2 15 --disk_tilt 20 \
-r hd --device gpu
```

**用户外部参考的目标特征**：Interstellar 式盘——完整透镜环 + 横贯盘面 + 内缘细亮环 + 沿轨道方向拉伸的细丝/烟雾 + 暖白/金/褐色带 + 暗蓝黑背景 + 柔和 Bloom。其纹理特征接近 V1 `_generate_turbulence` + `_generate_filaments`，**不是** V2 当前显式点云团。

### 1.3 设计文档冲突（需本方案修正）

[`design_ad_v2.md` §2.3 / §3.4](../design_ad_v2.md) 把 **`F_clump` 定为主结构来源**、并把「团块边界清晰」列为硬指标。实物验收证明：在体积积分 + cinematic palette 下，该路线产生 **鬣狗斑**，与 Interstellar 美学相反。

**本方案主张**：主结构来源改为 **`F_turbulence`（V1 多尺度云雾）+ 可选 `F_filament`**；`F_clump` 降级为 **弱体积自遮挡**（仅密度路径，强度 ≤ 0.15）。

---

## 2. 目标视觉特征（修订版）

| 编号 | 特征 | 可观测标准 |
|------|------|------------|
| V1 | **完整盘面** | 接近参考图：横向盘面贯穿画面，上下透镜弧都清晰，黑洞阴影不是孤立小黑点 |
| V2 | **絮状云雾** | 接近参考图：无 > `0.05·r_in` 尺度的孤立圆斑；纹理沿 `φ` 方向拉伸成细丝/烟雾 |
| V3 | **暖色 cinematic** | 接近参考图：暖白/cream → pale gold → smoky brown，背景暗蓝黑；无红黄白靶盘、无全局死白 |
| V4 | **体积感** | 高倾角侧视 preview 上，沿 z 积分至少 2 次可辨亮暗起伏（不要求 3~5 次硬达标，团块弱贡献即可） |
| V5 | **HDR 可控** | `--v2_auto_exposure` 下 `ldr_white_ratio < 5%`，且盘面中间调有可辨结构（LDR std > 0.08） |

**明确放弃**：「团块边界清晰、尺度 0.1~0.5·r_in」作为 V2 主美学指标（保留为可选弱效果，不作硬验收）。

---

## 3. 结构层职责（修订）

### 3.1 新合成公式

**发射率**（主视觉）：

```
j = ρ_envelope^α · T^β · F_turbulence(r, φ) · F_filament(r, φ) · F_mode · F_hotspot
```

**密度 / 吸收**（体积自遮挡）：

```
ρ = ρ_envelope · F_turbulence · F_filament · F_clump_weak(r, φ, z)
```

其中 `F_clump_weak = 1 + clump_strength_weak · signed`，`clump_strength_weak ≤ 0.15`。

**`F_shear`（傅里叶）**：从发射路径 **移除**；密度路径 **默认关闭**（`shear_strength = 0`），避免与 `F_turbulence` 打架产生斑马纹。代码保留 API，CLI 可手动开启做实验。

### 3.2 `F_turbulence` 实现思路

不移植整个 V1 纹理管线，只抽取 **已验证的视觉核心**：

1. 从 `render.py` 抽出 `_tileable_noise`、`_periodic_pixel_noise`、开普勒剪切滚动逻辑到独立模块（见 §5 文件表）。
2. 在 `(n_r, n_phi)` 极坐标栅格上预计算 **`turbulence` 场**（与 V1 `_generate_turbulence` 同权重：5 层 tileable + pixel noise + Kepler roll）。
3. 初始化 `DiskV2Taichi` 时 bake 一次，上传到 Taichi `ti.field` `(n_r, n_phi)`。
4. 光追采样时用 `(r, φ)` 双线性查表，返回 `[0, 1]` 调制因子，映射为 `F_turb = 1 + turb_strength · (2·sample - 1)`。

**为何预烘焙而非 Taichi 实时噪声**：V1 美学已被验收；预烘焙 parity 简单、性能可控；与现有 clump centers 上传模式一致。

### 3.3 `F_filament`（Phase 2，可选叠加）

V1 `_generate_filaments` 生成的弧状细丝对 Interstellar **轨道条纹**贡献大。Phase 1 仅 turbulence 达标后再叠加，避免一次改太多难以归因。

实现同 turbulence：预烘焙 `(n_r, n_phi)` atlas + Taichi 双线性采样。

### 3.4 垂向 z 的处理

V1 atlas 是 `(r, φ)` 二维。体积感通过：

- 已有 `ρ_envelope(r,z)` 垂向高斯包络；
- 弱 `F_clump` 在 z 方向有 σ_z；
- 沿光线多步体积积分；

不在 Phase 1 引入 `(r, φ, z)` 三维噪声（留作后续增强）。

---

## 4. 流程框图

```
+-------------+     +------------------+     +-------------------+
| V1 noise    | --> | Bake (r,phi)     | --> | Upload Taichi     |
| extract     |     | turbulence atlas |     | atlas fields      |
+-------------+     +------------------+     +-------------------+
                                                      |
                                                      v
+-------------+     +------------------+     +-------------------+
| Ray march   | --> | Sample j, rho    | --> | HDR + tonemap     |
| (existing)  |     | w/ F_turb (+F_cl)|     | + auto exposure   |
+-------------+     +------------------+     +-------------------+
```

---

## 5. 实施步骤（按依赖排序）

### Step 0：冻结基线 + 回滚有害调参

**目的**：避免在错误默认值上叠加新功能。

**改动**：

| 文件 | 动作 |
|------|------|
| `disk_v2/params.py` | 回滚 `shear` 高频试验默认值；`clump_strength` 默认 → `0.12`；`clump_emission_weight` 保持 `0` |
| `disk_v2/structure_modulations.py` | 回滚 `phi_frequency` 范围到 `2~14`（与 Taichi 上传一致） |
| `disk_v2/taichi_impl.py` | 保持 `save_image` 修复后的发射公式，待 Step 2 替换 |

**测试**：现有 V2 单测全绿（允许 snapshot 数值因默认参数微调而更新）。

**产出**：`output/v2_baseline_step0.png`（验收相机，确认无全白/无斑马纹恶化）。

---

### Step 1：抽取 V1 噪声为可复用模块

**新增** `disk_v2/v1_texture_bridge.py`：

- `build_turbulence_atlas(params, seed, n_r, n_phi) -> np.ndarray`
- （Phase 2）`build_filament_atlas(...)`
- 内部调用从 `render.py` **搬迁**（不是 import render 循环依赖）的 `_tileable_noise` 等纯函数

**新增** `tests/unit/test_disk_v2_turbulence_atlas.py`：

- 同 seed 确定性
- 值域 `[0, 1]`
- 与 `render._generate_turbulence` 在固定网格上 **MSE < 0.02**（允许 upscale 路径差异）

**不改** `taichi_render.py`。

---

### Step 2：Taichi 侧 atlas 采样

**改动**：

| 文件 | 动作 |
|------|------|
| `disk_v2/taichi_impl.py` | 上传 `turbulence_atlas` field；新增 `@ti.func sample_turbulence_ti(r, phi)` 双线性采样 |
| `disk_v2/taichi_impl.py` | `sample_emission` 改用 §3.1 公式 |
| `disk_v2/taichi_impl.py` | `sample_density` 用 `F_turb · F_clump_weak`，去掉 `F_shear` |
| `disk_v2/params.py` | 新增 `DiskV2StructureParams.turbulence_strength`（默认 `0.35`）、`atlas_n_r` / `atlas_n_phi` |

**新增** `tests/unit/test_disk_v2_numpy_taichi_parity.py` 用例：`sample_turbulence` 逐点 parity。

**产出**：`output/v2_step2_turbulence.png` —— 人工对比 `v1_classic_darksky.png`。

---

### Step 3：视觉验收脚本 + CLI

**新增** `scripts/v2_visual_acceptance.sh`（或 `tests/visual/v2_acceptance_render.py`）：

- 固定验收相机命令
- 输出到 `output/v2_acceptance_<date>.png`
- 打印 `compute_render_stats` + 盘面区域 LDR std

**改动** `render.py`：

- `--v2_turbulence_strength`
- `--v2_atlas_n_r` / `--v2_atlas_n_phi`（默认 `512 × 1024`）
- `--v2_shear_strength`（默认 `0`，高级参数）

**不改** Bloom 默认值（验收阶段手动 `--v2_bloom_intensity 0.3` 做 A/B，不写入默认）。

---

### Step 4：叠加 Filament（可选，Step 2 达标后）

**改动**：`v1_texture_bridge.build_filament_atlas` + Taichi 第二 atlas + 发射/密度合成。

**验收**：轨道方向细条纹可辨，仍无大块圆斑。

---

### Step 5：文档同步

| 文件 | 内容 |
|------|------|
| `docs/design_ad_v2.md` §2.3 / §3.4 | 主结构来源改为 `F_turbulence`；团块降级 |
| `docs/plans/realism_uplift_plan.md` | 增加「视觉恢复」交叉引用与 Phase 状态 |
| `README.md` | V2 推荐命令改回验收相机；说明 V2 视觉仍 experimental |
| `AGENTS.md` | 踩坑 #27：V2 团块/傅里叶不宜做主发射纹理；验收图路径 |

---

## 6. 测试要点汇总

| 阶段 | 单测 | 视觉 |
|------|------|------|
| Step 0 | 全量 V2 unit | 无死白、无大面积圆斑 |
| Step 1 | atlas 确定性 + V1 MSE | `preview.py` face-on 可见絮状（非团块） |
| Step 2 | Taichi parity + g_factor smoke | **对比 v1_classic 同相机** |
| Step 3 | acceptance 脚本可重复 | 你 Review 通过/打回 |
| Step 4 | filament atlas 单测 | 细丝可辨、无斑马纹 |

---

## 7. 待你决策的不确定项

实施前需要你拍板：

### Q1：验收盘尺寸

- **A（推荐）**：验收只用 `ar1=2, ar2=15`（与 V1 classic 一致，对比最直接）
- **B**：两套都要（`ar2=15` + `ar2=50`），V2 默认物理半径保留 50

### Q2：Step 4 Filament 是否纳入首版

- **A（推荐）**：Step 1~3 只做 turbulence，Step 4 你 Review 过 Step 3 再加 filament
- **B**：Step 2 同时上 turbulence + filament（更快接近参考，但难归因）

### Q3：是否修订 `design_ad_v2.md` 硬指标

- **A（推荐）**：删除「团块边界清晰」硬指标，改为「絮状/细丝云雾，无大块孤立斑」
- **B**：保留团块指标但降为软指标

---

## 8. 风险

| 风险 | 缓解 |
|------|------|
| 从 `render.py` 抽噪声函数导致重复代码 | Step 1 搬迁后 V1 改 import 新模块（Step 5 或单独 PR） |
| 2D atlas 在极大 `r_out=50` 下径向采样稀疏 | atlas 默认 512×1024；`r` 用 log 或 lin 采样需与 V1 一致用 lin |
| 体积积分仍抹平细节 | 保持 `volume_samples=32`；弱 clump 只做遮挡 |
| 与 V1 仍差 Bloom / g-factor 微调 | Step 3 验收脚本提供 `--v2_bloom_*` A/B 组，不阻塞 Step 2 |

---

## 9. 变更记录

- **v0.1 (2026-06-14)**：初稿。基于用户三轮视觉反馈 + V1/V2 同构图对比；提出 `F_turbulence` 预烘焙 atlas 路线；团块/傅里叶降级；固定验收相机。
