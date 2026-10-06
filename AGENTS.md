## 开发规范

### 流程控制

1. 方案讨论阶段不要修改代码，方案确定后才可以动手
2. 方案讨论需要双方都没疑问才输出具体方案文档
3. 严格按步骤执行，每次只专注当前步骤。不允许跨步骤实现或"顺便"完成其他任务。每步完成后汇报，等待 Review 确认后进入下一步
4. 没有我的明确指令不许 commit / push

### 方案设计

5. 方案评估主动思考需求边界，合理质疑方案完善性。方案需包含：重要逻辑的实现思路、按依赖关系拆解排序、修改/新增文件路径、测试要点
6. 遇到争议或不确定性主动告知我，让我决策而不是默认采用一种方案
7. 文档中流程框图文字用英文，框线要对齐；其余内容保持中文

### 编码规范

8. 最小改动原则，除非我主动要求优化或重构
9. 优先参考和复用现有代码风格，避免重复造轮子
10. 不要在源码中插入 mock 的硬编码数据
11. 使用 TDD 开发模式，小步快跑，每一步都测试，保证不影响现有用例
12. 及时在 `tests/unit` 中添加单元测试
13. 测试完后清理测试文件
14. bug 修复超过 2 次失败，主动添加关键日志再尝试，修复后清除日志
15. 使用中文回答
16. 同步更新相关文档
17. 需要完整函数文档字符串，把公式、变量含义和简化假设写清楚；重要代码段也要有注释

### 文档字符串标准

- 对外函数、类、数据结构必须写完整 docstring；不能有的函数很详细、有的函数只写一句话
- docstring 结构默认统一为：一句话摘要、`Args:`、`Returns:`、`Formula:`、`Physical Meaning:`、`Simplifications:`；没有对应内容时才可以省略该段
- 涉及数学公式的函数，`Formula:` 必须写出明确公式；`Args:` 或正文中必须解释公式里的变量、坐标、量纲/无量纲约定
- `Returns:` 必须写清楚返回值的物理/几何意义、标量还是数组、形状如何跟随输入广播、值域范围（如布尔值、`[0, 1]`、围绕 `1` 波动等）
- `Args:` 不能只写参数名，要写清楚输入代表什么坐标/物理量、是否允许标量和数组、关键取值范围或边界语义
- `Physical Meaning:` 说明“这个函数在模型里表示什么”，不能只重复代码行为；`Simplifications:` 说明当前采用了哪些简化假设
- 私有辅助函数也要写 docstring；简单工具函数至少要有一句话摘要、`Args:`、`Returns:`，若是非平凡数学变换则补 `Formula:` 或 `Notes:`
- 同一层级的函数说明风格要一致：同一后缀（如 `*_half_thickness`、`*_temperature`、`*_g_factor`、`*_ti`）的措辞、值域描述、盘内/盘外语义要统一
- 测试名称、设计文档、源码 docstring 对同一概念的叫法必须一致，避免一处叫“结构场”、另一处叫“结构调制”

### Disk V2 概念边界

详见 `docs/design_ad_v2.md` §3.5 与 `docs/imaging_model.md` §7。

- 结构场：盘的几何与物质分布（标高、柱密度、剪切级联湍流、高斯核心 + 指数大气剖面、刚体环平流），表示盘"长什么样"；
  统一气体模型：盘面与大气是同一团气体、同一湍流场（见 `docs/plans/v2_unified_gas_plan.md`）
- 辐射转移：温度、源函数、散射反照率、频移与发射-吸收-散射积分，表示盘"发出什么光"
- 成像链（旧称"后处理"，原理见 `docs/imaging_model.md`）按真实相机分三层：
  - 镜头：点扩散函数（眩光长尾、轴向 / 横向色差），必须线性、能量守恒、作用于全部光，不设阈值
  - 传感器：曝光（盘区 p99.9 → 0.9，再乘曝光补偿 2^EV，默认 +1.5 档；视频锁定）
  - ISP：白平衡、色调映射、饱和度、sRGB 编码
- 参数按"场景·结构 / 场景·辐射 / 场景·背景 / 相机 / 镜头 / 传感器与 ISP / 求解精度 / 输出"分层，
  新参数先确定所属层，CLI 放进对应的参数组

### Disk V2 命名规则

- `*_half_thickness` / `*_surface_density`：结构场的标高（r_s）/ 柱密度（无量纲形状）
- `density_*`：体积密度场，返回非负发射 / 吸收系数
- `*_temperature`：温度（K）或温度倍率（标量，围绕 1）
- `*_g_factor`：频移 `ν_obs/ν_em`（> 0，> 1 为蓝移）
- `*_ti`：与同名 NumPy 参考函数逐值一致的 Taichi 实现（由单测做 parity）

### 提交规范

18. 提交前先梳理内容，等待 Review 确认后才能提交
19. commit message 使用中文
20. 每个 commit 必须添加 `Co-Authored-By` trailer

---

## 项目速记

### 项目简介

- 本项目是一个 **Schwarzschild 黑洞光线追踪渲染器**，核心目标是生成黑洞阴影、光子环、引力透镜扭曲星空和吸积盘效果。
- 当前主实现为 **Taichi**，主入口基本集中在 `render.py`；设计文档以 `docs/design.md` 为准。
- 当前不做 Kerr 度规，只关注无自旋黑洞；物理积分核心采用 **笛卡尔等效势形式**，而非球坐标 Christoffel 方案。

### 核心实现约定

- 核心光线方程：`d²x/dλ² = -1.5 * L² * x / r⁵`
- 单帧/视频主入口：`render.py`
- 设计与背景说明：`docs/design.md`；Disk V2：`docs/design_ad_v2.md`；旧文档：`docs/archived/`
- 端到端渲染测试：`tests/e2e_render.py`
- 方向相关单测：`tests/unit/test_parametric_rotation_direction.py`

### 代码结构速记

- 代码按层放在 `src/` 下，`render.py` 只是兼容门面与命令行入口（`python render.py ...` 用法不变）
- `src/core/`
  - `constants.py` 全部公共常量；`camera.py` 相机；`skybox.py` 天空盒生成/加载；`imaging.py` 存图
- `src/v1/`
  - `texture.py` V1 程序纹理生成与缓存（约 1450 行）；`lifecycle.py` 实体生命周期
  - `renderer.py` `TaichiRenderer`（V1 光追核）；`pipeline.py` 单帧/交互/视频管线
- `src/v2/`
  - 体积吸积盘（V2），见 `docs/design_ad_v2.md`
  - 运镜：`camera_compose.py` 插值 / 平滑 / 构图求解；`camera_path.py` 路径与节奏；`exposure_curve.py` 测光 + 平滑曝光、淡入淡出；
    `path_video.py` 路径文件加载与按路径渲染（见 `docs/plans/v2_camera_path_plan.md`）
  - 小尺度温度湍流：`temperature_turbulence.py` 几何、频率钳制 / 透镜权重与 NumPy 参考；Taichi 实现为
    `taichi_impl.DiskV2Taichi._temp_turb_*`（见 `docs/plans/v2_temperature_turbulence_plan.md`）
- `src/cli.py`
  - 参数解析与模式分发（V1 / V2、单帧 / 视频 / 运镜）
- `scenes/`
  - 成品场景，每个场景一个目录，内含说明（README.md）、渲染命令与运镜路径文件；
    `v2_arts/interstellar_skim.json` 为示例路径（120 s，开启温度湍流）；`close_low/close_low.json` 为贴云顶匀速环绕（60 s，开启温度湍流）；
    `lensing_bend/lensing_bend.json` 为逼近侧透镜转弯处的长焦特写（30 s，视野 6°、黑洞在画面外、0.2°/s 极慢环绕，
    白平衡 12000 K 等成像参数见其 README）
- `scripts/`
  - `contact_sheet.py` 运镜联系表（审构图）
- `tests/unit` 轻量定向单测；`tests/e2e_render.py` V1 固定参数渲染 + hash 校验

### 视频旋转算法速记

- 当前视频模式支持三种吸积盘旋转算法：
  - `baseline`：固定纹理，在采样阶段按 `frame` 做旋转
  - `parametric`：每帧重新生成带时间偏移的程序纹理
  - `keyframes`：预生成关键帧纹理并插值
- 当排查视频中“结构旋转方向不一致”问题时，**先确认用户实际使用的是哪种算法**。
- `parametric` 模式下，最容易出问题的是：
  - `phi_grid` 路线的相位旋转
  - `np.roll(...)` 路线的动态滚动
  - 两者若符号不一致，就会出现“有些组件方向对、有些组件方向反”的现象。

### 测试速记

- 定向方向单测：

```bash
python -m unittest tests/unit/test_parametric_rotation_direction.py
```

- 端到端渲染校验：

```bash
python tests/e2e_render.py --verify    # unittest 方式收集不到用例（脚本自带 argparse 入口）
```

### 踩坑记录

21. 重试过 2 次以上的环境配置问题或重复犯错的问题，记录在本文件

22. **多普勒效应颜色/亮度方向问题** (待深入分析)
    - 现象：直觉认为蓝移应偏蓝、红移应偏红，但实际效果相反
    - 最终修正：
      - 盘旋转方向：`v_hat = r_hat × disk_normal`（而非 `disk_normal × r_hat`）
      - 颜色：蓝移(neg_shift>0)时 r_scale 增大偏红，红移(pos_shift>0)时 b_scale 增大偏蓝
      - 亮度：由 g 因子自动决定，g>1 亮，g<1 暗
    - 待分析：为什么物理正确的多普勒效应在视觉上呈现"反直觉"的颜色？
      - 可能方向：shift 的定义是否有误？g>1 实际对应什么物理条件？
      - 文件位置：`render.py:716-741` 的 `apply_g_factor` 函数

23. **`parametric` 模式下动态旋转方向不一致** (已修复)
    - 现象：视频中吸积盘不同组件旋转方向不一致，部分结构看起来"反着转"
    - 根因：`phi_grid = phi_grid_base + t_offset * omega_grid` 与 `np.roll(..., +rotation_pixels)` 使用了相反的方向约定
    - 最终修正：
      - 统一以 `phi_grid` / `baseline` 采样方向为准
      - 动态时间旋转使用 `np.roll(..., -rotation_pixels)`
    - 重点文件：`render.py`
    - 保护测试：`tests/unit/test_parametric_rotation_direction.py`

24. **Taichi `@ti.func` 不接受 dataclass 实例作为常量** (V2 实施时踩到)
    - 现象：在 `@ti.func` 内访问 `self.params.r_in` 抛 `TaichiTypeError: Invalid constant scalar data type`
    - 根因：Taichi 1.7.4 只允许 Python float/int/bool 作为 runtime 常量，不接受任意 dataclass 字段
    - 修复：在 `__init__` 里把 dataclass 字段平铺为 `self._r_in = float(params.r_in)`，
      `@ti.func` 内只访问平铺字段
    - 文件位置：`src/v2/taichi_impl.py:DiskV2Taichi.__init__`

25. **Taichi `@ti.func` 不接受类型注解** (V2 实施时踩到)
    - 现象：`def foo(x: ti.f32) -> ti.f32:` 抛 `TaichiSyntaxError: Invalid type annotation`
    - 根因：Taichi 1.7.4 的 `@ti.func` 解析参数注解时拒绝直接的 `ti.f32` / `ti.types.vector(...)` 标注
    - 修复：去掉所有 `@ti.func` 的参数和返回类型注解；类型由 Taichi 编译期推断
    - 参考：`render.py` 现有 `@ti.func` 同样无注解

26. **V2 模式下 CLI 默认 `--ar1` / `--ar2` 不匹配 V2 推荐范围** (V2 接入时踩到)
    - 现象：`--disk_model v2` 未传 `--ar2` 时使用 V1 默认 `15`，导致温度跨度只有 ~2.5 倍
      （V2 推荐 `r_out=50` 可达 ~4.3 倍）
    - 根因：CLI `--ar1` / `--ar2` 的默认值是 V1 全局常量 `R_DISK_INNER_DEFAULT=2`、`R_DISK_OUTER_DEFAULT=15`
    - 当前修复：V2 路径检测 `--ar2 < 20` 时打印 warning，提示用户加上 `--ar2 50`
    - 文件位置：`render.py` 的 `if args.disk_model == "v2"` 分支
    - **现状（2026-10-02）**：warning 随旧参数清理删除；V2 必须显式传 `--ar1 3 --ar2 30`（预设 M）

27. **V2 团块 / 纯 Fourier shear 不适合做主发射纹理** (视觉恢复 2026-06-14；**已过时**：atlas / 团块 / shear 已于 2026-10-02 删除，仅作历史经验)
    - 现象：`F_clump` 全强度进发射 → 鬣狗斑；高频 `F_shear` 试验 → 斑马纹；二者叠加易全白/灰脏
    - 修复：主结构改预烘焙 `visual_atlas`（V1 云雾 + spiral warp + alpha clip）；`F_clump` 仅弱密度自遮挡（`clump_strength≈0.12`, `clump_emission_weight=0`）；`shear_strength` 默认 0
 - 验收：`bash scripts/v2_visual_acceptance.sh`；固定相机 `pov=24 0 8, ar1=2, ar2=15`
    - 文件：`src/v2/visual_atlas.py`、`src/v2/taichi_impl.py`、`src/v2/params.py`

---

### 常用命令

```bash
# V1 默认渲染
python render.py --pov 20 0 2 --fov 60 --ar1 2 --ar2 10 --disk_tilt 20 --resolution hd -o output/*.png

# V2 渲染（预设 M 构图；默认优化级别 1、超采样倍率 2；--v2_opt / --v2_supersample 可覆盖）
python render.py --disk_model v2 --pov 0 -39.7 4.87 --fov 38 \
                 --ar1 3 --ar2 30 -r fhd --device gpu -o output/v2.png

# V2 视频（内缘一圈 16 s；默认优化级别 2、超采样倍率 1）
python render.py --disk_model v2 --video --pov 0 -39.7 4.87 --fov 38 \
                 --ar1 3 --ar2 30 -r hd --device gpu --n_frames 240 --fps 24 -o output/v2.mp4

# V2 视频中断后续传：同一条命令加 --resume（每 240 帧一个分段，最多重渲 1 段）

# V2 运镜视频（帧数 = round(路径时长 × fps)；先测光约 2 分钟）与联系表
python render.py --disk_model v2 --video --v2_camera_path scenes/v2_arts/interstellar_skim.json \
                 --ar1 3 --ar2 30 --v2_reverse_rotation -r fhd --fps 60 --device gpu -o output/v2_path.mp4
python scripts/contact_sheet.py scenes/v2_arts/interstellar_skim.json --v2_reverse_rotation -o output/v2_arts/contact_sheet.png

# V2 单测全部
python -m unittest $(ls tests/unit/test_disk_v2_*.py | sed 's#/#.#g; s#\.py$##')

```

### 远程渲染（AutoDL RTX 4090）

长视频与大图放到 AutoDL 的 RTX 4090 实例上渲染，本机只做开发、小样审阅与短验证。

- **实例**：Ubuntu 22.04、RTX 4090（24 GB）、NVIDIA 驱动 580（CUDA 13.0）、Xeon Gold 6430（128 线程）。
  SSH 地址与端口以 AutoDL 控制台为准；本机用 `ssh -i ~/.ssh/id_ed25519_mac -p <端口> root@<主机>` 免密登录。
- **环境**：系统没有 `python` / `python3`，统一用 `/root/miniconda3/bin/python`（3.12）。
  首次开机安装 `pip install taichi==1.7.4 av imageio`（numpy、pillow 已有）。Taichi 只依赖驱动，不需要 CUDA Toolkit；
  `--device gpu` 在 Linux 上自动走 CUDA，命令与本机相同。
- **目录**：代码 `/root/autodl-tmp/bhr`，天空贴图 `/root/autodl-tmp/TychoSkymapII.t5_8192x4096.jpg`，产物 `/root/autodl-tmp/out`
  （`/root/autodl-tmp` 是数据盘，关机后保留 15 天）。

```bash
# 1. 同步代码（本机执行；包含未提交的改动时用 tar，否则可用 git archive HEAD）
tar czf /tmp/bhr.tar.gz src tests scenes scripts render.py
scp -i ~/.ssh/id_ed25519_mac -P <端口> /tmp/bhr.tar.gz root@<主机>:/tmp/
ssh -i ~/.ssh/id_ed25519_mac -p <端口> root@<主机> 'mkdir -p /root/autodl-tmp/bhr && tar -xzf /tmp/bhr.tar.gz -C /root/autodl-tmp/bhr'

# 2. 验证环境（远端）
nvidia-smi
/root/miniconda3/bin/python -c "import taichi as ti; ti.init(arch=ti.cuda); print('ok')"

# 3. 渲染（远端，nohup 后台运行，断线不影响；中断后同一命令加 --resume 续传）
cd /root/autodl-tmp/bhr && nohup /root/miniconda3/bin/python render.py --disk_model v2 --video \
    --v2_camera_path scenes/v2_arts/interstellar_skim.json --ar1 3 --ar2 30 --v2_reverse_rotation --v2_orbit_seconds 8 \
    -t /root/autodl-tmp/TychoSkymapII.t5_8192x4096.jpg -r fhd --fps 60 --device gpu \
    -o /root/autodl-tmp/out/interstellar_skim_1080p60.mp4 > /root/autodl-tmp/out/interstellar_skim_1080p60.log 2>&1 &

# 4. 取回产物（本机执行）
scp -i ~/.ssh/id_ed25519_mac -P <端口> root@<主机>:/root/autodl-tmp/out/<文件> output/<场景>/
```

- **速度（实测，1080p、优化级别 2）**：GPU 积分 0.37–0.81 s/帧（M5 为 2.5–7.1 s/帧）；后处理按 `src/cli.py` 的
  `POSTFX_THREADS = 8` 多线程执行，单线程 3.7 s/帧、8 线程约 0.55 s/帧；首次 kernel 编译约 40 s。
  360p / 15 fps 的 60 s 运镜小样（900 帧）约 6 分钟。
- **注意**：
  - 按秒计费，渲染完及时关机；关机前把产物 scp 回本机。
  - 本机不要跑完整长视频做验证：会让 GPU 长时间满载、机器卡顿；验证用几秒的短路径，或放到远端跑。


### 踩坑记录（v2.3 S0-S9 实施期间）

28. **`blackbody_luminance_ti` 多包一层 `exp()` → 亮度比值全错（S6 修复）**
    - 现象：V2 渲染全红，HDR 值 ~1e13（应为 ~1）
    - 根因：`blackbody_luminance_ti` 返回 `exp(lnY) = Y`，而 kernel 用
      `exp(Y - lnY_peak)` 计算 `Y/Y_peak` 比值——多了一层 exp 使所有温度的比值
      都变成 ~exp(32) ≈ 1e14，冷区红光不被压制
    - 修复：返回 `lnY`（与参考实现 `ln_luminance` 一致），kernel 用
      `exp(lnY(T) - lnY(T_peak))` 得到正确比值
    - 文件位置：`src/v2/taichi_impl.py:blackbody_luminance_ti`

29. **`from __future__ import annotations` 使 `ti.template()` 失效（S5 踩到）**
    - 现象：标定 kernel 编译报 `TaichiSyntaxError: Invalid type annotation`
    - 根因：Python 3.12 中 `from __future__ import annotations` 把所有注解变成
      字符串，Taichi 无法把字符串 `'ti.template()'` 解析回类型
    - 修复：`taichi_impl.py` 移除该 import（其余模块可用，但含 `@ti.kernel`
      定义的文件不可用）
    - 文件位置：`src/v2/taichi_impl.py`

30. **渲染核内 `self.xxx` 属性必须在 `_compile_kernels()` 之前赋值（S6 踩到）**
    - 现象：`AttributeError: 'DiskV2Renderer' object has no attribute 'volume_params'`
    - 根因：`_compile_kernels()` 在 `__init__` 中被调用时通过闭包读取
      `self.volume_params`，但该属性在 `_compile_kernels()` 之后才赋值
    - 修复：把 `self.volume_params = volume_params` 移到
      `self._compile_kernels()` 之前
    - 文件位置：`src/v2/taichi_render.py:__init__`

31. **移植 Proto 时凭印象改写 → V2 与 Proto 长期对不上（2026-10 修复）**
    - 现象：多轮"看图 → 猜原因 → 补一处"仍差距大：烟雾看不见、光子环下半部消失、锯齿
    - 根因：未逐行对照移植，累积 13 处偏差（噪声未归一化、视界清零发射、合成重复乘透射率、
      `CORE_OPAC` 漏乘发射、混入 `opacity_scale`、低频带中心约定不同、光行时间无效等）
    - 规则（已废止，2026-10-02）：当前 V2 视觉已优于 Proto，**不再**与 Proto 对比；新功能改为保证
      "参数取旧值时与改动前逐位一致"（用 `git archive HEAD src` 导出旧版对比渲染）；
      详见 `docs/plans/v2_volumetric_video_plan.md` §10
    - 保护测试：`tests/unit/test_disk_v2_proto_parity.py`（已随旧模型于 2026-10-04 删除）
    - 延续（2026-10-04，统一气体模型）：先在原型中定稿观感，再分步移植，每步用"同参数渲染 HDR 逐位一致"验收
      （360p、多视角、HDR 与天空两路、优化级别 0 与 3），见 `docs/plans/v2_unified_gas_plan.md` §11

32. **Taichi `@ti.func` 内不能把 dataclass 绑定到局部变量**
    - 现象：`f = self._adv_core_f` 抛 `Invalid constant scalar data type: RigidRingFields`
    - 修复：直接写全路径 `self._adv_core_f.rot[idx]`（与 #24 同源）

33. **V2 单测慢的主要原因是 Taichi 编译**（2026-10-02 实测）
    - 现象：`test_disk_v2_proto_parity.py`（已删除）单文件约 12.5 分钟，其余 V2 单测合计仅数秒
    - 根因：每构造一个 `DiskV2Renderer`，体积 kernel 首帧编译约 60–70 s；该文件内构造了 6 个
    - 做法：迭代期只跑改动相关的轻量单测（`test_disk_v2_volume_fields` / `noise_ti` / `advection` / `v2_cli` 等）；
      需要 `DiskV2Taichi` 的测试在 `setUpClass` 里只构造一次（如 `test_disk_v2_unified` / `shear_cascade`）；
      提交前再跑一次全量；小样渲染用 360p（`640×360`）

34. **画面"发红"先查后处理再查物理颜色**（2026-10-03）
    - 现象：调低多普勒指数、给颜色温度加下限后，金色区仍偏橙红
    - 根因：bloom 逐通道扣阈值 `max(hdr_c − th, 0)`，G/B 偏低的像素散射光只剩 R
    - 修复：按亮度扣阈值保持色度（`postfx.apply_bloom(luma_threshold=True)`）
    - 做法：颜色问题先逐级打印后处理各阶段 R:G:B 比例，定位是哪一步改了色相

35. **镜头效果先问"模拟的是什么器件"，参数调不好先怀疑模型**（2026-10-03）
    - 现象：阈值 bloom 把辉光后面的盘面点亮；改阈值、解耦镶边、能量守恒阈值版、取 max 都治标不治本
    - 根因：bloom 不是镜头模型——提取光 ×4 加回（造光）、6 px 窄核复制纹理、阈值落在盘面亮度分布中间、复制边缘补边造光
    - 修复：镜头 = 能量守恒 PSF `(1 − ε)·x + ε·(K ∗ x)`，作用于全部光、无阈值（`postfx.apply_lens_psf`）
    - 做法：在 HDR 域对每一步做"输出 − 输入"，区分"散射进来的光"与"自身散出的光"，并检查总能量比（应 ≤ 1）

36. **盘面"惨白 / 没有发光感 / 像大理石"先查曝光与白平衡，再查结构**（2026-10-04）
    - 现象：统一气体模型下盘面偏白、远看像大理石在转，光子环与内盘不"发光"
    - 根因：(1) 曝光 p99.9 → 0.9 使画面没有任何像素越过白点（旧版也如此，被烟雾的柔光掩盖）；
      (2) 白平衡 4000 K = 色温上限，去掉偏冷的烟雾层后内盘大片纯白，亮区饱和度减半；
      (3) 次要：剪切级联小尺度条纹满盘同样锐利
    - 修复：曝光补偿 `--v2_exposure_ev` 默认 +1.5 档、白平衡 4500 K、眩光 0.5、`core_oct_gain` 0.6
    - 做法：同相机对比量化——曝光后盘区越过白点的比例、亮区饱和度、黑洞阴影内的 LDR 均值（纯镜头散射光），
      并逐项关闭（散射 / 大气）排除结构层；先定位到层再改参数

37. **V2 在 CUDA（4090）上首帧 `CUDA_ERROR_ILLEGAL_ADDRESS`，Metal / CPU 正常**（2026-10-05）
    - 现象：同一份代码在 M5 上跑了几千帧都正常，换到 4090 首帧即崩
    - 根因：`taichi_impl` 的三处查表（黑体色、亮度、Page–Thorne 温度）只钳制了上端索引；表下限处运行时 f32 的 `log`
      与编译期常量相差 1 ulp，`u` 变成小负数，`floor` 后 `i0 = −1`。CUDA 读到池基址之前的未映射显存而崩溃，
      Metal / x64 越界落在已映射内存里、静默读到垃圾值
    - 修复：两端钳制 `max(min(i0, N − 2), 0)`，与 NumPy 参考 `palette._lut_lookup` 的 `np.clip` 一致
    - 做法：`compute-sanitizer --tool memcheck` 给出越界地址与所属内核；在远端临时副本里逐个把查表替换成常量二分定位；
      注意 `DiskV2Renderer` 构造时会自动 `ti.init`（已改为仅在没有 Taichi 程序时才初始化），否则测试脚本指定的 arch 会被覆盖
    - 保护测试：`tests/unit/test_disk_v2_palette_s2.py::TaichiLutFloorParityTest`（有 CUDA 时在 CUDA 上跑）

38. **后台渲染的等待循环不要用 `pgrep -f <输出文件名>`**（2026-10-05）
    - 现象：`while pgrep -f "x.mp4"; do sleep 30; done` 永不退出，渲染在后台长时间占满本机
    - 根因：等待循环所在 shell 的命令行本身包含该字符串，`pgrep -f` 匹配到自己
    - 做法：记录后台进程 PID（`$!`）后用 `kill -0 $PID` 判断；长渲染放远端，本机只跑几秒的短验证
