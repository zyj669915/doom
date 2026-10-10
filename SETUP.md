# 在新电脑上从零训练——完整复现步骤

> 目标：一台干净 Windows 电脑，clone 代码 → 拉脑图 → 跑通 PPO 老师和果蝇蒸馏。
> 预计耗时：环境安装 30 分钟（含下载），训练首轮 20 分钟。

## 0. 硬件要求

- Windows 10/11（Linux 同理，路径换成 venv/bin/）
- 8GB+ 内存（脑图 1GB + 仿真）
- 20GB 空闲磁盘（代码 + 脑图 1.5GB + 模型）
- **CPU 即可**（torch 是 cpu 版，不需要 N 卡）

## 1. 装基础工具

1. 装 Git for Windows
2. 装 Python 3.11.x（**勾选 Add to PATH**）——不要 3.12+（scipy 1.10 不兼容）
3. 装 Visual Studio Build Tools（勾选"使用 C++ 的桌面开发"）——编译神经仿真内核要用

验证：
```
git --version
python --version    # 应显示 3.11.x
```

## 2. 拉两份代码

```
# 本体引擎 + Doom 场景（上游 nftechie/doomfly）
git clone https://github.com/nftechie/doomfly.git D:\fly\doomfly

# 本实验仓库（训练脚本 + 模型 + 文档）
git clone https://github.com/zyj669915/doom.git   D:\fly\experiment
```

> 注意：上游本体和本实验仓库**分开两个目录**。训练脚本路径假设本体在 D:\fly\doomfly。

## 3. 建虚拟环境装依赖

```
cd D:\fly\doomfly
python -m venv .venv-neural
.\.venv-neural\Scripts\Activate.ps1

# 关键包（版本已踩坑锁定，照抄）
pip install numpy==1.26.4 scipy==1.10.1 numba torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib pillow vizdoom gymnasium pyyaml tqdm requests pyarrow
```

> **坑**：numpy 必须锁 1.26.4，升 1.27 scipy 直接崩。
> **坑**：torch 用 cpu 版即可，不需要 CUDA。

## 4. 编译神经仿真内核

```
cd D:\fly\doomfly
python build_kernel.py
```

## 5. 拉果蝇脑图（大文件）

```
cd D:\fly\experiment
.\scripts\fetch_brain.ps1
# 默认 tag = brain-v0-original，拉到 D:\fly\doomfly\outputs\doom\malecns_v1\
```

## 6. 跑通训练（三步）

```
cd D:\fly\doomfly
$env:PYTHONPATH='D:\fly\doomfly'   # 每次新开终端都要设
.\.venv-neural\Scripts\Activate.ps1

# ① 训 PPO 老师（17 分钟，30k 步）
python D:\fly\experiment\code\teacher_ppo.py

# ② 用老师玩游戏收集示范，蒸馏进果蝇脑
python D:\fly\experiment\code\distill_fly.py

# ③ 600 tics（10 秒）长时评估对比
python D:\fly\experiment\code\long_eval.py
```

## 7. 验收标准

- ① 老师：`data-reference/teacher-log.json` 里 reward 从 2.0 升到 3+
- ② 蒸馏：`models/student.pt` 生成，训练 acc > 40%
- ③ 评估：distilled kills ≥ 2.0，baseline kills ≤ 1.0（本机实测 2.33 vs 1.0）

## 常见坑（别重复踩）

| 坑 | 现象 | 解决 |
|---|---|---|
| numpy 版本 | scipy 崩 | 锁 1.26.4 |
| PYTHONPATH 没设 | ModuleNotFoundError: doom | `$env:PYTHONPATH='D:\fly\doomfly'` |
| 直接 python doom/server.py | 报错 | 必须 python -m doom.server |
| 评估只跑 210 tics | 数据假提升 | 必须 600 tics |
| brain.counts 是 int64 | 网络 dtype 冲突 | 喂前转 float32 |
| 规则老师 | 站桩被围殴 | 不适合当老师，用 PPO |

## 训练过程想改脑结构？

本机已存原始快照：`D:\fly\brain-snapshot\v0-original\`（1.37GB，11 文件）。
新电脑要回滚到原版：把 brain-v0-original Releases 的文件覆盖回 outputs/doom/malecns_v1/ 即可。
