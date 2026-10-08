# doomfly-training

把真实果蝇全脑连接组（**MaleCNS v1.0，166,700 神经元 / 25.58M 突触**）接到 ViZDoom 上玩 Doom 的实验仓库。
基于上游 [nftechie/doomfly](https://github.com/nftechie/doomfly)（Alex Wormuth, 2026-09），本仓库在其基础上完成了：

1. **连接审计**：原版"大脑→游戏动作"层从未接通（10 个真实运动读出全 0 放电，游戏靠工程 BCI 旁路驱动——作者 `validation.json` 自认 `biologically_validated: false`）。
2. **物理连接修复**：调仿真参数（权重 scale=1.5、膜电位偏置 bias=6、放电阈值 th=−48），让真实运动神经元 MN9 从 0 放电到稳定放电并驱动开火/运动。
3. **doomfly-rl 式蒸馏**：训一个 PPO 老师玩 Doom → 每帧把"游戏画面 → 果蝇脑 14 个读出神经元放电"作特征 → 训练一个读出层模仿老师动作。

## 目录结构

```
code/                 所有实验脚本（老师/蒸馏/校准/诊断/录制）
docs/                 PROGRESS.md 权威进度档案、SNAPSHOT.md、原版参考
data-reference/        小数据（manifest/validation/评估 json/诊断结果）
models/               teacher.pt（PPO 老师）、student.pt（蒸馏读出层）
TRAINING_PLAN.md      下一步训练的完整提示词/策略（可直接喂给 agent）
```

## 大脑数据怎么复现（重要）

大脑数据分两部分：

**已在仓库内（小文件，<100MB）**：`data-reference/brain-snapshot-v0/`
- annotations.feather、neurotransmitters.feather、source.lock.json、normalized/ 下 catalog/neurons/npy/report

**大文件（超 GitHub 100MB 限制，从 Releases 下载）**：

| 文件 | 大小 | 作用 |
|---|---|---|
| `graph.npz` | 239 MB | **仿真直接加载的脑图**（Brain 类用它） |
| `edges.feather` | 1.0 GB | MaleCNS 原始连接边（上游构建） |
| `normalized/edges.arrow` | 293 MB | 归一化后边表 |

**拉取方法（一键）**：
```powershell
# 从仓库根目录跑：
.\scripts\fetch_brain.ps1                    # 拉原始 v0 脑
.\scripts\fetch_brain.ps1 -Tag brain-v1      # 拉训练后的新版本脑
```
脚本会自动建目录、跳过已存在的文件。

**版本管理约定**：
- `brain-v0-original`：训练前原始快照（已固定，不再变）
- `brain-v1`、`brain-v2`…：每次训练/大改脑结构后发新 tag，旧版本保留可回滚
- 发新版 = 在 https://github.com/zyj669915/doom/releases/new 拖文件上传（单文件限 2GB）

## 环境

- Python venv：`.venv-neural`，**numpy 必须锁 1.26.4**（scipy 1.10.1 依赖 <1.27；装包前先查依赖，勿升 numpy）
- torch 2.x CPU 版、numba、matplotlib 3.11、pillow
- 运行任何项目脚本前设 `$env:PYTHONPATH='D:\fly\doomfly'`
- ViZDoom 1.3.0 + freedoom2.wad；果蝇玩的是自建小竞技场 `doom/scenarios/combat_survival.wad`

## 已验证的关键结论（详见 docs/PROGRESS.md）

- 固定权重下"会动不会玩"是物理极限：8 轮参数扫描 + 3 轮闭环，最佳配置 kills≈1.3 / 207 tics
- 蒸馏 v1（PPO 老师 30k 步）：10 秒长时对抗下 distilled kills=2.33 vs baseline=1.0（+133%），存活 8-10 秒 vs 3 秒
- 老师退化是击杀瓶颈：PPO 30k 步后 argmax 退化为单一动作，蒸馏无米之炊 → 见 TRAINING_PLAN.md 下一步
