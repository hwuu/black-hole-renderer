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

### 提交规范

17. 提交前先梳理内容，等待 Review 确认后才能提交
18. commit message 使用中文
19. 每个 commit 必须添加 `Co-Authored-By` trailer：
    - OpenCode 实现：`Co-Authored-By: OpenCode (GLM-5) <noreply@opencode.ai>`
    - Claude Code 实现：Claude Code 默认

### Code Review

20. 完成一个编码步骤后，使用 OpenCode review 代码：

```
opencode run "<prompt>"
```

prompt 示例：

```
Claude Code 完成了编码工作，请你 review，看看是否符合设计、是否有潜在 bug、是否有不完善的地方、现有架构是否没有冲突、没有引入冗余实现。

设计：...

实现代码 (diff)：...
```

### 踩坑记录

21. 重试过 2 次以上的环境配置问题或重复犯错的问题，记录在本文件

---

## 环境配置备忘

### Go 代理

默认 Go 代理 (proxy.golang.org) 在当前网络环境下超时，需使用国内镜像：

```bash
export GOPROXY=https://goproxy.cn,direct
```

### Go SDK 路径

Go 安装在用户目录（无 sudo 权限），需确保 PATH 包含：

```bash
export PATH=$HOME/go-sdk/go/bin:$HOME/go/bin:$PATH
```

### 常用命令

```bash
# 构建
make build

# 运行测试
make test

# 清理构建产物
make clean
```
