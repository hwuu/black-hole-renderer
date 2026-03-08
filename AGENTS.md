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
- 同一层级的函数说明风格要一致：`*_mask`、`*_weight`、`*_field`、`*_modulation` 的措辞、值域描述、盘内/盘外语义要统一
- 测试名称、设计文档、源码 docstring 对同一概念的叫法必须一致，避免一处叫“结构场”、另一处叫“结构调制”

### Disk V2 概念边界

- `geometry`：盘的空间几何脚手架，包含“多厚”“是否在盘内”“边界如何平滑过渡”
- `physical_fields`：基础物理量分布，表示盘“本来是什么”
- `structure_modulations`：对基础物理场的无量纲调制，表示盘“被扰动成什么样”

### Disk V2 命名规则

- `*_mask`：硬判定，返回布尔值
- `*_weight`：软判定，返回 `[0, 1]` 权重
- `*_field`：基础物理场，返回物理量分布
- `*_modulation`：结构调制，返回围绕 `1` 波动的乘性因子

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
- 设计与背景说明：`docs/design.md`
- 端到端渲染测试：`tests/e2e_render.py`
- 方向相关单测：`tests/unit/test_parametric_rotation_direction.py`

### 代码结构速记

- `render.py`
  - 相机、天空盒、吸积盘程序纹理、Taichi 渲染器、视频渲染 CLI 都在这个文件
  - `TaichiRenderer` 是核心渲染类
  - `render_video()` 支持视频模式
- `docs/design.md`
  - 项目目标、物理模型、渲染管线、实现取舍的主文档
- `tests/unit`
  - 放轻量、定向、快速运行的单元测试
- `tests/e2e_render.py`
  - 放固定参数渲染 + hash 校验的端到端测试

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
python -m unittest tests/e2e_render.py
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
    - 现象：视频中吸积盘不同组件旋转方向不一致，部分结构看起来“反着转”
    - 根因：`phi_grid = phi_grid_base + t_offset * omega_grid` 与 `np.roll(..., +rotation_pixels)` 使用了相反的方向约定
    - 最终修正：
      - 统一以 `phi_grid` / `baseline` 采样方向为准
      - 动态时间旋转使用 `np.roll(..., -rotation_pixels)`
    - 重点文件：`render.py`
    - 保护测试：`tests/unit/test_parametric_rotation_direction.py`

---

### 常用命令

```bash
# 渲染测试图像
python render.py --pov 20 0 2 --fov 60 --ar1 2 --ar2 10 --disk_tilt 20 --resolution hd -o output/*.png
```
