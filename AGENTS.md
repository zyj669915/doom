# AGENTS.md — 科研协作规范与运行事实

读者：本地研究者、实现者与协作 AI。更新日期：2026-11-19。
本文是工作约束，**不是实验执行授权**；还须遵守 [工作区约束](../AGENTS.md)。
2026-11-17 的科研准则原样保留；2026-11-19 追加"运行事实"与"自转循环"两节（来自另一台机器的实测）。

## 接手顺序

1. [research/README.md](research/README.md)：文档职责与证据优先级。
2. [research/CHARTER.md](research/CHARTER.md)：研究目标、范围和结论边界。
3. [research/STATUS.md](research/STATUS.md)、[research/TASKS.md](research/TASKS.md)：当前状态与下一件任务。
4. 该任务关联的协议、原始结果及实际源码；不要一次扫全仓库、所有旧日志或大图。
5. 按需选 [prompts/](prompts/README.md) 中一个角色提示词。它们是可复制文本，不会自动启动多 agent 或调用 API。

## 科研规则

- 区分 G0 工程、G1 行为学习、G2 结构贡献、G3 脑内适应、G4 生理机制；检验一个问题，不替其他层级下结论。
- 保留完整 MaleCNS 及约定集合间全部发布边，不裁回路、不剪弱边/自环、不用游戏策略替代脑路径。纯 frontend 只能是明确标注的对照。
- RGB 是策略的外部任务观测；受体采样、脑传播、读出学习、奖励与未来可塑性必须分别可追溯。遥测、布局 ID、镜像标签、成功/剩余路程不能进入策略或脱困控制。
- 预先说明假设、变化因素、对照、预算、数据划分、主指标和停止/分流条件。不要同时改算法、前端、脑动力学与奖励再作单因素归因。
- 开环脚本成功、回报提高、放电增加、有梯度或预算跑满，都不等于学会。报告逐条件/逐训练 seed；续训谱系不是独立重复，case 数不是训练重复数。
- 保留阴性、中断、拒绝启动及来源不一致记录。全组地板不能判拓扑无用，不显著不等于等效。文献启发不等于本机已经支持。

## 文件与复现

- 不改 backup/、brain-v0-clean/ 或任何权威 graph.npz，不搬迁旧运行、checkpoint、失败材料。权威图预期 SHA256：f4d41f011e97e510761d011634891b1c082f61f91463eb586ee9cf8c6371b1d1；这是既有基线，不代表本次重新计算。
- 旧 flyrl/、flyrl_next/、flyrl_stage/ 及 doomfly/doom/ 含 checkpoint 绑定源码。新增适配另起文件，先检查 manifest，不能放宽来源验证或重写旧 manifest 来接受漂移。
- checkpoint 先核对来源与哈希，仅使用受支持的 weights_only=True 安全加载，不加载未知 pickle。
- run 保存配置快照、输入/源码/资产哈希、依赖版本、seed/谱系、交互与校准/评估成本、停止原因、完整性。原始结果不被后续修复覆盖。
- 从原始 run 事实推导状态；文档摘要冲突时核查而不是选择较乐观版本。新结构只索引证据，不复制大文件。
- 下载/解压资料放在各自新空目录，自写脚本另放；读取不可信资料的 Python 使用 -I，不从资料目录运行解释器/构建工具。
- 不恢复既有删除、不 git clean、不自动整理未知文件。Windows 优先使用既有解释器与 -I -B；不能将历史依赖表视为当前环境实测。

## 授权、资源与停止

- 按当前用户请求确定操作范围。讨论、提示词、任务板 pending、协议 locked 都不构成训练授权；历史授权不自动覆盖新任务。
- 2026-11-17 结构整理任务的授权仅限本地文档、提示词、导航与轻量静态检查，未包含训练、长评估、游戏、加载脑或调用外部模型。这是该次任务的范围记录，不是禁止未来经用户明确授权的研发或实验；后续仍按当次请求判断。
- 后续真实执行时先做当前资源门禁，默认单全脑槽位；遵守既有预算与系统余量，不降低阈值绕门禁、不无限重试、不关闭用户程序或改系统设置。
- 来源/输入不一致、非有限、资源未知、STOP 或墙钟到限必须保存并停止；不能算方法阴性。资源检查只在执行关口做，不替代研究目标。
- 安装升级、commit/push、同步脚本、对外上传与部署须有相应明确授权。不要运行 sync-docs.ps1。公开 launch 还须满足 [上游发布要求](../doomfly/AGENTS.md)，软件通过不等于科学门槛通过。

## 完成与交接

- 修改前读文件；只改与任务直接相关的内容，保留他人变更与历史事实。
- 先记录实际做过、测过、没测的内容，再更新任务板与状态；不因"计划已写"标记实现完成。
- 新结果先存原始 run，再按 [证据索引](research/EVIDENCE.md) 关联；路线改变写 [决策记录](research/decisions/README.md)，不重复造"最终最新版"。
- 任务状态仅用 pending / in_progress / blocked / done / deferred；结束会话时无实际继续执行的 in_progress。
- 给出可点击的本地文件引用。报告测试命令、数量、失败/跳过与版本边界；没有运行就明确说没有运行。

---

# 运行事实（2026-11-19 实测，两台机器路径不同）

## R1. 仓库与路径

| 项 | 值 |
|---|---|
| 远端 | `https://github.com/zyj669915/doom`（分支 `main`） |
| Python | `D:\fly\doomfly\.venv-neural\Scripts\python.exe`（3.11.9；torch/numba/vizdoom/pyarrow/pandas 齐备） |
| 连接组原始数据 | `D:\fly\brain-snapshot\v0-original\malecns_v1\`（1,369.7 MB，**v0 只读**） |
| 上游 vendor | `D:\fly\doomfly`（nftechie 克隆，**只读**，不要往里写本项目文件） |

**代码按固定结构解析路径**：`ROOT/experiment/code/...` 与 `ROOT/doomfly/doom`。因此仓库必须位于 `<ROOT>\experiment`，vendor 必须位于 `<ROOT>\doomfly`。

| 机器 | 仓库路径 | ROOT | 是否原生匹配 |
|---|---|---|---|
| ZYJ（本机） | `D:\fly\experiment` | `D:\fly` | ✓ 原生匹配 |
| 旧机 | `D:\doom` + 联接 `D:\experiment\code` | `D:\` | ✗ 需联接；且 `ROOT\doomfly` 不存在，provenance 里的 doomfly 源码会缺失 |

⚠️ 文档中的 `D:/fly/experiment` 在本机是**真实路径**；在没有 `D:\fly\experiment` 的机器上要按上表处理。

## R2. 入口

| 入口 | 用途 |
|---|---|
| `code/flyrl_stage_cli.py` | 三 arm 限额训练：`doctor / validate / task-probe / accept / train / evaluate / suite` |
| `code/flyrl_next_cli.py` | A1–A7 多路线栈，配置在 `configs_next/` |
| `code/flyrl_turn_cli.py`、`code/flyrl_turn_train_cli.py` | 新增 turn / turn-train 链路（2026-11-19 本机新增，未合入旧 manifest） |
| `code/flyrl_cli.py` | 旧栈（线性读出），已判死，仅作历史对照 |

调用带 `-I -B`：

```powershell
$py = 'D:\fly\doomfly\.venv-neural\Scripts\python.exe'
cd D:\fly\experiment
& $py -I -B code\flyrl_stage_cli.py doctor --config configs_stage\near_turn_three_arm_v1.json
```

## R3. 门禁（真实瓶颈是 commit，不是 RAM）

- 需要 `2 GiB 预算 + 3 GiB 固定余量 = 5.37 GB` commit。
- 2026-11-19 旧机实测：RAM 可用 11.4 GB（够），commit 可用 5.20 GB → **差 170 MB 被拒**，`stop: resource_gate_rejected`。
- 处置：**不降阈值、不关用户程序、不自动重试**；释放 commit 后重新过门禁。
- 已成功的 `doctor`：`allowed: true`、`errors: []`。

## R4. 成本量级（全脑在 CPU，约 2.46 tic/s）

| tics | 墙钟 |
|---|---|
| 10,000 | ≈ 1.1 h |
| 30,000 | ≈ 3.4 h |
| 50,000 | ≈ 5.6 h |

一轮 pilot 是**小时级**，不是分钟级。

## R5. 文件权限

| 可编辑 | 只读 |
|---|---|
| `configs_stage/**`、`configs_next/**` | 本文件、`program.md` |
| `code/flyrl_stage/**`（新版本目录） | 冻结源码清单内的文件 |
| `docs/experiments.md`、`docs/handoff.md` | `data-reference/**`、`models/**`、`code/tests/**` |
| `research/**`、`prompts/**`（按各自职责） | 上游 `doomfly/**` |

## R6. 自转循环

判定门、噪声带、回退、best 保存、崩溃分级、交接格式全部在 [program.md](program.md)。开工顺序：本文件 → `program.md` → `research/STATUS.md` → 具体任务源码。
