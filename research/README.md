# 科研工作区

读者：接手项目的人与 AI。更新：2026-11-17。

这里管理“研究什么、为什么做、下一步做什么、凭什么判断”，不替代代码、配置和原始运行目录。不搬迁历史栈，不新增训练框架，不自动执行实验。

## 五分钟接手

1. 读 [CHARTER](CHARTER.md) 确认目标与结论边界。
2. 读 [STATUS](STATUS.md) 和 [TASKS](TASKS.md)，只领取一个符合本次授权的任务。
3. 从 [EVIDENCE](EVIDENCE.md) 找到相关原始证据；需要方法细节再读 [目标与路线报告](../docs/reports/项目目标与路线决策_20261117.md)。
4. 看该任务的 [协议](protocols/README.md)，选一个 [角色提示词](../prompts/README.md)。
5. 完成后记录验证与未完成项，更新任务和状态；不要用维护文件替代下一项实际科研工作。

## 目录及职责

```text
experiment/
  AGENTS.md                 AI 协作、保护与授权边界
  research/
    CHARTER.md              当前研究目标与范围
    STATUS.md               当前能力/证据边界快照
    TASKS.md                执行任务板与依赖
    tasks/                  需要展开的单项研发任务
    EVIDENCE.md             结论 → 原始证据索引
    protocols/              已实现环境规范与待锁定训练协议
    experiments/            实验记录索引，不存 checkpoint
    decisions/              路线/组织决策及理由
    templates/              任务、协议、结果、决策、交接模板
  prompts/                  通用约束与各角色可复制提示词
  scripts/check_research_structure.py  本地结构检查器
  code/                     现有实验代码与测试，路径不变
  configs_stage/ configs_turn/  现有可执行配置，路径不变
  configs_turn_train/       新短任务real/frontend开发配置（正式协议待锁）
  runs_stage_20261026/ runs_turn_20261117/  原始产物，路径不变
  docs/                     历史进度、详细报告、验收记录
```

不预建空 datasets/notebooks/paper 目录。确有数据处理、探索 notebook 或论文写作时再建，并说明来源与去向。已有缺失的历史 models/configs_next/runs 不恢复，也不假装仍在本机。

## 信息由谁负责

| 信息 | 唯一维护位置 | 其他文件怎么用 |
|---|---|---|
| 当前目标、非目标与结论边界 | [CHARTER](CHARTER.md) | 只作简要导航，不复制完整目标 |
| 当前能力、未完成与版本边界 | [STATUS](STATUS.md) | 每次实质变化后同步，不当实时进程监控 |
| 下一步、依赖、任务状态 | [TASKS](TASKS.md) | 提示词必须读取，不写死“下一步” |
| 某次实验的方法、预算与判定 | [protocols/](protocols/README.md) | 配置负责执行；两者冲突则停止核查 |
| 实际结果与来源 | 原始 run 内的 result/provenance/日志 | [EVIDENCE](EVIDENCE.md) 与 [experiments](experiments/README.md) 只索引 |
| 为什么选这条路线 | [decisions/](decisions/README.md) 与详细路线报告 | 历史决策保留，用新条目标记替代关系 |
| 逐次进度历史 | [PROGRESS](../docs/PROGRESS.md) | 追加更正，不抹除失败历史 |
| AI 通用工作边界 | [AGENTS](../AGENTS.md) | [prompts](../prompts/README.md) 引用而非各自重写 |

原始记录描述实际发生了什么；当前协议描述接下来应如何做，两者不能互相覆盖。来源冲突时回到同一 checkpoint/协议的原始记录和源码核实。新文件不得把过往事实“升级”为已通过。

## 最小科研闭环

问题 → 预锁协议 → 在授权内实现/验收 → 限额实验 → 冻结分析 → 决策与交接。

- 环境协议已实现，不等于训练协议已锁定；locked 不等于用户已批准执行。
- 在 dev 上做筛查，不用 confirm 调参。确认结果失败也必须保留，不能换名重开为“第一次”。
- 每次只用与任务有关的模板，不要求填完所有模板才能工作。
- 不新建一份每日“总计划”。当前文件持续维护，实验/决策以唯一 ID 留痕。

## 检查结构

在工作区用既有 Python 运行：

```powershell
& 'D:/fly/doomfly/.venv-neural/Scripts/python.exe' -I -B 'D:/fly/experiment/scripts/check_research_structure.py'
```

检查器只读文件，验证科研目录、相对链接、提示词结构、任务 ID 与关键状态；不运行模型、游戏、网络或脑图哈希。历史文档的全部旧链接不在此次扫描范围，不能把结构检查通过称为科学验收。加 `--strict-evidence` 可将缺失原始run链接视为错误；另一台仅同步文字的机器缺少run时，默认给警告而不伪造结果。检查器自身的10项轻量回归可运行 [test_check_research_structure.py](../scripts/test_check_research_structure.py)。
