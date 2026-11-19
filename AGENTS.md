# AGENTS.md — 本仓库的硬约束与事实

> 所有 agent 开工前必读。这里写**事实**，不写愿望；任何一条被证伪就改这里，不要改结论。

## 1. 仓库与位置

| 项 | 值 |
|---|---|
| 远端 | `https://github.com/zyj669915/doom`（分支 `main`） |
| 本地仓库 | `D:\doom` |
| 运行产物根 | `D:\experiment\runs_stage_20261026\`（**在仓库外**，不受 git 回退影响） |
| Python | `D:\fly\doomfly\.venv-neural\Scripts\python.exe`（3.11.9，torch/numba/vizdoom/pyarrow/pandas 齐备） |
| 连接组原始数据 | `D:\fly\brain-snapshot\v0-original\malecns_v1\`（1,369.7 MB，v0 不可改） |
| 上游 vendor 仓 | `D:\fly\doomfly`（nftechie 克隆，**只读**，不要把本项目文件写进去） |

⚠️ 文档里出现的 `D:/fly/experiment/...` 是历史路径，**该目录已不存在**。入口改用 `D:\doom\code\...`。

### 1.1 运行布局（代码按固定结构解析路径，必须知道）

代码用 `ROOT/experiment/code/...` 定位源码，实测 `ROOT = D:\`。因此本机建了目录联接：

| 联接 | 指向 | 原因 |
|---|---|---|
| `D:\experiment\code` | `D:\doom\code` | 代码找 `D:\experiment\code\flyrl_stage_cli.py` |

- 仓库仍在 `D:\doom`，联接只是让旧路径能解析，**不改代码**。
- 产物落 `D:\experiment\runs_stage_20261026\<run>\`，与仓库分离。
- 若换机器，需重建同样布局，否则 `train` 会报 `FileNotFoundError: D:\experiment\code\flyrl_stage_cli.py`。

## 2. 入口

| 入口 | 用途 |
|---|---|
| `code/flyrl_stage_cli.py` | 三 arm 限额训练（当前主线）：`doctor / validate / task-probe / accept / train / evaluate / suite` |
| `code/flyrl_next_cli.py` | A1–A7 多路线栈，配置在 `configs_next/` |
| `code/flyrl_cli.py` | 旧栈（线性读出），已判死，仅作历史对照 |

调用必须带 `-I -B`（隔离模式，不依赖 PYTHONPATH）：

```powershell
$py = 'D:\fly\doomfly\.venv-neural\Scripts\python.exe'
cd D:\doom
& $py -I -B code\flyrl_stage_cli.py doctor --config configs_stage\near_turn_three_arm_v1.json
```

## 3. 硬约束（违反即作废，不得"为了结果好看"放宽）

1. **不改权威图与 v0 快照**：`D:\fly\brain-snapshot\v0-original\` 只读；`graph.npz`、原始 feather、clean 图一律不动。
2. **不污染冻结源码**：`check_stage_frozen.py` 校验的源码清单不得改动；要改就新建独立版本。
3. **不降资源余量**：门禁要求 `ram_gib 1.5 + 预留 3 GiB`。资源不足就等，**不关用户程序、不降阈值、不自动重试**。
4. **不自动启动长训**：`wall_hours 12` 的正式训练必须由人类明确授权，agent 不得自行开跑。
5. **不在 test 上调参**：`test` 只在所有选择结束后跑一次；反复用 test 选模型即作弊。
6. **不删负结果**：失败 run、crash、未过门的记录全部保留，它们是证据。
7. **不加载来源不明的 pickle**：`torch.load` 只允许 `weights_only=True`。
8. **不把全脑特征 / 坐标 / 生命值喂给策略**：只用于奖励与审计。

## 4. 已核验状态（2026-11-19）

- `doctor --config configs_stage/near_turn_three_arm_v1.json` → `"allowed": true`，`"errors": []`。
- 资源：RAM 可用 14.6 GiB，commit 可用 9.1 GiB，满足 1.5+3 GiB 需求。
- 报告：`D:\experiment\runs_stage_20261026\preflight-20261119-175733-eb8811a7.json`。
- **门禁的真实瓶颈是 commit，不是 RAM**：需要 `2 GiB 预算 + 3 GiB 固定余量 = 5.37 GB`。2026-11-19 实测 RAM 可用 11.4 GB（够），commit 可用 5.20 GB（**差 170 MB 被拒**，`stop: resource_gate_rejected`）。
- 处置口径：**不降阈值、不关用户程序、不自动重试**。释放 commit（关掉占内存的程序）后重新过门禁。

## 5. 当前真实进度（不许美化）

- 已完成：完整 MaleCNS 视觉—动作闭环；旧全脑 20k tic 训练；A4/A5/A6 纯视觉训练与四控/三 seed 评测。
- **未完成**：完整脑的可靠游戏学习、real/rewired/frontend 匹配闭环比较、生理验证、脑内学习。
- A6 的 val6/12 不含脑，**不能称为"果蝇脑学会游戏"**。
- 三 arm 正式训练与冻结评估仍待完成。

## 6. 文件权限

| 可编辑 | 只读 |
|---|---|
| `configs_stage/**`、`configs_next/**` | `AGENTS.md`、`program.md` |
| `code/flyrl_stage/**`（新版本目录） | 冻结源码清单内的文件 |
| `docs/experiments.md`、`docs/handoff.md`（只追加/覆盖写） | `data-reference/**`、`models/**` |
| | `code/tests/**`（改需人类批准） |
