# 研究组织手册（DOOM / 三 arm 主线）

> 人类只改本文件，agent 只改 §2 里"可编辑"的文件。
> 协议骨架取自 `karpathy/autoresearch` 的 ratchet 循环、`ml-agent-orchestrator` 的裁判规则、`XScientist` 的失败分支留证。

## 0. 成本模型（先算清再开跑）

全脑在 CPU，实测吞吐约 2.46 tic/s：

| tics | 墙钟量级 |
|---|---|
| 10,000 | ≈ 1.1 h |
| 30,000 | ≈ 3.4 h |
| 50,000 | ≈ 5.6 h |

所以**一轮 = 一次 10k tic pilot + 冻结评估**，不是几分钟的小实验。别照搬"5 分钟一次"的模板。

### 0.1 路径（两台机器不同，命令里的 `<REPO>` 按下表替换）

| 机器 | `<REPO>` | run 产物 |
|---|---|---|
| ZYJ（本机） | `D:\fly\experiment` | `D:\fly\experiment\runs_stage_20261026\` |
| 旧机 | `D:\doom` | `D:\experiment\runs_stage_20261026\` |

`python` 两机相同：`D:\fly\doomfly\.venv-neural\Scripts\python.exe`，调用带 `-I -B`。

## 1. 预算

- 单轮 ≤ **2 小时**（10k tic + 评估 + 记账）。
- 单次会话 ≤ **3 轮**（约 4–6 小时）；超出要人类明确授权。
- `wall_hours: 12` 的正式训练**必须人类点头**，agent 不得自行开跑。
- 资源门禁不通过 → 停，不降阈值、不关用户程序、不自动重试。

## 2. 文件权限

| 可编辑 | 只读 |
|---|---|
| `configs_stage/**`、`configs_next/**` | `AGENTS.md`、`program.md` |
| `code/flyrl_stage/**`（新版本目录） | 冻结源码清单内的文件、`code/tests/**` |
| `docs/experiments.md`、`docs/handoff.md` | `data-reference/**`、`models/**` |

## 3. 噪声带（这个项目最容易自欺的地方）

默认 val 只有 2 个布局——**2 局里多赢 1 局是 50% 相对波动，根本不是信号**。

判定规则：

1. **只有 2 布局时**：任何结果只能标 `pilot-only`，禁止写"提升""学会"。
2. **扩到 ≥4 个 val 布局后**（需人类授权改配置）才允许判定：
   - 成功率绝对提升 **≥1 局**，且
   - 四控（black / scramble / random / untrained）**不高于 untrained 基线**，且
   - 同配置重复 2 次，两次差异 <1 局
3. 三条缺一条 → `noise`，回退。
4. 变差 → `discard`，回退。

**"权重变了""活得久了""单局跑通"都不构成学习证据。**

## 4. 循环（每轮严格按序）

1. **看盘**：`git status` 干净；在 `exp/<tag>` 分支。工作区有不是本次实验的改动 → 停，要求人类先提交或 stash；**agent 不得替人类提交在研内容**。
2. **门禁**：`flyrl_stage_cli.py doctor --config <cfg>`。`allowed: false` → 停，报错误原因，**这不叫崩溃，不许重试绕过**。
3. **改**：只动 §2 可编辑文件；一次只改一个变量（读出头宽 / lr / gamma / 课程 / 观测表征）。
4. **先提交**：`git add -A; git commit -m "exp(<id>): <一句话>"`。
5. **跑**：`flyrl_stage_cli.py train --config <cfg> --max-game-tics 10000`，输出重定向到 `<REPO>\logs\<id>.log`，不要让日志灌进上下文。
6. **评**：`evaluate --checkpoint <本次 checkpoint 目录> --split val`；需要归因时再跑 `--policy random` / `--policy untrained`。
7. **判**：按 §3 判定 keep / pilot-only / noise / discard / crash。
8. **处置**：keep → §5 保存；其余 → `git reset --hard HEAD~1`。
9. **记账**：`results.tsv` + `docs/experiments.md`。
10. **继续**：回第 1 步。

## 5. 保存（什么值得留）

`keep` 必须四件齐全：

1. **代码**：commit 在 `exp/<tag>` 分支（该分支只进不退）。
2. **checkpoint**：把 `<REPO>\runs_stage_20261026\<run>\checkpoint-<steps>-<id>\` 整个目录复制到 `<REPO>\best\<id>\`（不进 git）。
3. **manifest**：同目录写 `manifest.json`：配置快照、seed、tic 数、val 各布局结果、四控结果、`sha256`（checkpoint 内每个文件）、**完整复现命令**。
4. **登记**：`docs/experiments.md` 追加一行，含 id / commit / val 成绩 / 四控 / sha256 前 16 位 / 复现命令。

只留最近 3 个 best，更早的删 checkpoint、留 manifest 与登记。

## 6. 崩溃处理（三级）

| 级别 | 判据 | 动作 |
|---|---|---|
| 1 琐碎 | 拼写、import、参数名、路径 | 修，`--amend`，重跑，沿用同一 id，最多 2 次 |
| 2 可诊断 | OOM、形状不符、NaN、数据缺列 | 读日志尾部 50 行，降 `ram_gib`/batch 重试；再失败转 3 级 |
| 3 放弃 | 同一想法连续 3 次崩 | `git reset --hard HEAD~1`，记 `crash`，写进"已试过"，换想法 |

**门禁拒绝 ≠ 崩溃**：那说明环境不够，等资源，不许改阈值。

## 7. 裁判规则（防自欺）

- 结论必须有 `result.json` / `evaluate` 的数字支撑；判断与数字矛盾时以数字为准并降级。
- `git status` no-op（说改了但没差异）→ 本次不计入轮次。
- 不得改评测脚本、四控定义、val/test 划分、源码哈希清单。
- 报"提升"必须给：val 布局数、基线局数、本次局数、四控结果、重复性。缺一不算。
- 别把"课程走到位"当"完整入口学会"；别把 A6 的 6/12 说成"果蝇脑成功"。

## 8. 止损

- 连续 3 轮 `noise`/`discard` → 停该方向，写清"为什么走不通"，换下一个。
- 同一方向累计 10 轮仍无实质推进 → 放弃该方向。
- 方向用尽 → 写交接并停，不要自己发明新假设。

## 9. 记账

`results.tsv`（制表符分隔，**不入 git**）：

```
id	commit	tics	val	ctrl	status	desc
0	abc1234	0	2/2	-	baseline	未改动基线
1	def5678	10000	3/4	0/4	keep	readout hidden 32
2	9876543	10000	2/4	1/4	discard	gamma 0.99→0.995
```

`docs/experiments.md`（入库，只追加）：每条 keep / crash 写"改了什么、为什么、结果、结论"。

## 10. 交接（每次会话结束前必写）

覆盖写 `docs/handoff.md`：分支与 commit、当前 best 的 id 与 val 成绩、这轮试过什么、被否掉什么、下一步跑什么（配置字段+取值）、卡在哪。

## 11. NEVER STOP

循环开始后不要停下来问人类是否继续。没想法就继续读代码、读 `docs/PROGRESS.md` 的证据口径纠正、组合之前的近似成功。

**只有这些可以停**：资源门禁不通过、需要人类授权（12h 长训 / 改 val 划分 / 改冻结源码）、连续 3 次同类崩溃、达到 §1 预算、方向用尽、触发合规红线。

## 12. 回退纪律

- 只回退一个实验：`git reset --hard HEAD~1`，且仅当该 commit 只含本次实验改动。
- **训练产物不进 git**（`<REPO>\runs_stage_20261026\`、`<REPO>\best\` 均在 `.gitignore` 内），回退不会删 run 与 checkpoint——这是故意的。
- 回退前确认 `results.tsv` 已写入该行。
- 永不 `git clean -fd`。
