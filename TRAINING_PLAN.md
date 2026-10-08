# 训练提示词 / 策略（可直接喂给 agent 继续）

> 这份文件是给下一个接手者（人或 AI agent）的完整交接：**目标是什么、已经做到哪、哪些坑踩过、下一步每一步怎么做、用什么超参、怎么算成功**。
> 读完这份文件 + `docs/PROGRESS.md` 即可无缝继续，不需要重新摸索。

---

## 0. 一句话目标

让**真实果蝇连接组（MaleCNS v1.0，冻结拓扑）**通过训练，在 ViZDoom `combat_survival` 竞技场里把击杀从 baseline 的 1.0/局提升到 **≥4/局**（人类水平约 5-8/局），且全程不改连接组拓扑（只训权重/读出/可塑性）。

## 1. 现状（已完成，勿重复）

| 环节 | 状态 | 证据 |
|---|---|---|
| 连接组数据 | ✅ 真实完整，166,700 神经元 / 25.58M 边，哈希校验通过 | `data-reference/data-integrity.json` |
| 视觉输入 | ✅ 已通（感光→板层→全脑），但对比度被压扁（gamma 编码，42% 采样点近黑） | `brightness-stats.json` |
| 运动输出 | ✅ 已通（原作者从未接通；10 个真实读出原本 0 放电） | `validation.json`（作者自认 false） |
| 固定权重闭环 | ✅ 最佳配置 scale=1.5 / bias=6 / th=−48，kills≈1.3/207 tics | `calibrate-closed-loop.jsonl` |
| PPO 老师 | ⚠️ 30k 步训过，**退化**：argmax 630 帧全为单一动作（右转+前进+不开火），从未学会战斗 | `teacher-log.json`、`diagnose.json` |
| 蒸馏读出层 v1 | ✅ 跑通：10 秒长时 distilled kills=2.33 vs baseline=1.0（+133%），存活 8-10s vs 3s | `longeval_t600.json` |

**已验证做不通的坑（勿重复）**：
- 站桩规则老师（运动检测）：0.67 杀/局，被 8 敌冲脸围殴
- bias=0 或光自适应下：视觉驱动不动运动链（12mV 偏置是运动链激活的必要载体）
- DNa02R 任何参数近 0 放电（勿死磕）
- DNpe017 与 MN9 在 scale≥1.5 互斥
- 评估必须用 **600 tics（10 秒）以上**：210 tics（3.5s）截断会把 baseline 的"3 秒死亡"误判为"满局存活"，所有对比都失真

## 2. 已诊断的三个真实瓶颈（按影响排序）

1. **老师太弱（最大瓶颈）**：PPO 30k 步在 12 离散动作 + 稀疏击杀奖励下退化。学生只能学到老师的"活下来"部分，击杀学不到。
2. **特征通道窄**：14 个读出里只有 5 个实机放电（DNp20_L/R ~93%、MN9_L/R ~56-66%、DNa02L 50%；MDN/DNp09/DNpe017 <1.3%）。单帧特征信息量不足。
3. **一帧错位**：v1 蒸馏时"老师动作"对应上一帧画面、"果蝇特征"来自当前帧（v2 已修）。

## 3. 下一步路线（按优先级，每步都有验收标准）

### Step 1 — 重训一个真正会打的老师（预计 1-2 小时）
- **换场景/奖励**：先用敌人更密、信号更频繁的场景（如 `deadly_corridor`）训老师，再迁移回 combat_survival；或奖励整形：击杀×10、扣血×1，**加"开火时中心亮度变化"小奖励**（引导开枪）。
- **超参调整**：rollout 2048、epochs 10、熵系数 0.02（防止策略坍缩到单一动作）、lr 1e-4（后期衰减）、训练量 **100k-200k 步**（不是 30k）。
- **验收**：老师在 combat_survival 跑 600 tics，**击杀 ≥4/局** 且 argmax 动作分布至少 3 种动作占比 >10%（不再退化）。

### Step 2 — 蒸馏 v2（已写好 `code/distill_v2.py`）
- 修一帧错位 + 5 帧历史窗口（14→70 维特征）
- 收集 8-10 局（300 tics/局），学生 70→256→128→12，交叉熵，lr 1e-3，80 epochs
- **验收**：学生在未见帧上对老师动作一致率 ≥60%（v1 只有 34%）

### Step 3 — 闭环评估对比（600 tics × 3 seeds）
- 三配置：baseline（作者 BCI）/ random student / distilled student
- **成功标准**：distilled kills ≥ baseline ×2、存活 ≥ baseline ×2，健康均值为正
- 产出更新的对比图 + 实机 GIF（`code/record_gif.py`）

### Step 4 — 若 Step 1-3 击杀仍 <3：进入 doomfly-rl 原案（小时级）
- 冻结连接组拓扑，但给**每个突触一个可学习增益**（不是只训读出层）
- 二阶段：先模仿老师（行为克隆），再用奖励信号微调（PPO on readout）
- 视觉升级：加夜间暗适应增益（真实果蝇 R1-R6 有，模型缺），把感光 `30L/(0.02+L)` 改成带自适应的版本
- 参考论文：`docs/` 目录外本地还有 5 篇 PDF（MaleCNS 连接组、果蝇视觉、脑模型、ViZDoom）

## 4. 运行环境速查

- 项目根 `D:\fly\doomfly`，venv `.\.venv-neural\Scripts\python.exe`，**必须先 `$env:PYTHONPATH='D:\fly\doomfly'`**
- 大脑加载：`Brain(outputs/doom/malecns_v1/graph.npz)`，权重缩放 `brain.weight = (brain.weight * 1.5)`
- 自定义阈值内核：见 `code/calibrate_closed_loop.py` 的 `advance_th`（njit，参数 thresh）
- 读出神经元 14 个 ID：见 `data-reference/manifest.json` 的 readouts 列表
- 动作空间：12 离散 = turn[−6,0,6] × forward[0,20] × attack[0,1]

## 5. 绝对不要做的事

- 不要升 numpy（>1.26 会炸 scipy）
- 不要直接 `python doom/server.py`（必须 `python -m doom.server`）
- 不要用 210 tics 做评估（会得到假结论）
- 不要在用户没同意时 git commit / push
- 不要改连接组拓扑数据（`graph.npz` / feather 文件）——训练只在外面加层
