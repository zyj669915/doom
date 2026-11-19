---
name: exp-loop
description: 执行恰好一轮 ratchet 循环后停下报告。用户说"跑一轮"时使用。
---

严格按 `program.md` §4 执行**一轮**：

1. **看盘**：`git status` 干净。无 `exp/<tag>` 分支则 `git checkout -b exp/<tag>`。有非本次实验的改动 → 停下，等人类提交或 stash。
2. **门禁**：`& D:\fly\doomfly\.venv-neural\Scripts\python.exe -I -B code\flyrl_stage_cli.py doctor --config <cfg>`。`allowed:false` → 停并报原因。
3. **改**：一处变量。
4. **先提交**：`git add -A; git commit -m "exp(<id>): <一句话>"`。
5. **跑**：`... train --config <cfg> --max-game-tics 10000`，日志重定向到 `D:\experiment\logs\<id>.log`。
6. **评**：`... evaluate --checkpoint D:\experiment\runs_stage_20261026\<run>\checkpoint-<steps>-<id> --split val`。
7. **判**：按 §3（val 布局数 / 绝对提升局数 / 四控 / 重复性）。
8. **处置**：keep → 存 `D:\experiment\best\<id>\` + manifest；其余 → `git reset --hard HEAD~1`。
9. **记账**：`results.tsv` + `docs/experiments.md`。

跑完一轮就停，不自动进下一轮。

报告格式：

| id | 改了什么 | val 布局数 | 基线 | 本次 | 四控 | 判定 | 回退 | 复现命令 |

val 只有 2 布局时，结论只能写 `pilot-only`。
