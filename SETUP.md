# 本机新管线运行说明

更新：2026-10-19。当前推荐入口为 code/flyrl_cli.py；旧teacher→distill教程已撤出推荐路径，历史成绩不作新包验收。

**2026-10-19 新增多路线实验栈 `flyrl_next`**：入口 `code/flyrl_next_cli.py`（支持 `-I -B`，不依赖 PYTHONPATH），配置在 `configs_next/`，运行记录在 `runs_next/`。七条路线 A1—A7 及 A3 独立变体、suite 队列、stop/resume、冻结评估的完整命令见根目录 [果蝇闭环游戏_多路线训练与验证方案.md](../果蝇闭环游戏_多路线训练与验证方案.md) 第七至九节；最终开发验收（145 项测试通过、A4/A5/A6 微型优化+重载、source hash `8012ccd36571a5936ba3143f164342bf0ee4b3a5b22f0204f3978b70edcf1d33`）见其第十节。新栈不修改下方 `flyrl` 对应内容，二者不共用 checkpoint，不得互相冒充。

## 1. 已核验的本机环境

- Windows11，Python3.11.9：D:/fly/doomfly/.venv-gpu/Scripts/python.exe
- numpy2.4.6、numba0.68.0、torch2.11.0+cu128、vizdoom1.3.2、pyarrow26.0.0、pandas3.0.6。
- RTX4060 Laptop约8GiB，CUDA可用；**全脑仍是CPU Numba**，GPU仅加速小读出。线性小头CPU通常已足够，配置默认cpu，避免额外传输开销。
- 本包不需要编译C++ Native内核，不要求降numpy，不依赖旧scipy1.10路线。本轮未安装/升级依赖；这些是本机实测版本，不是所有新电脑都已验证的锁文件。

路径基于包位置推导D:/fly，配置相对路径以该根目录解析。新机器需完整保留doomfly与experiment并列结构，或改JSON中的输入路径。不要自动下载覆盖权威脑图；备份/clean图绝不改。

## 2. PowerShell命令（任意工作目录）

```powershell
$py = 'D:/fly/doomfly/.venv-gpu/Scripts/python.exe'
$entry = 'D:/fly/experiment/code/flyrl_cli.py'
$cfg = 'D:/fly/experiment/code/flyrl/configs/corridor.json'

# 只核对依赖与强哈希，不训练
& $py -I -B $entry doctor --config $cfg

# 短真实全脑闭环，不更新读出
& $py -I -B $entry smoke --config $cfg --steps 4

# 无标签开放环输入依赖检查；不是生理视觉验收
& $py -I -B $entry diagnose --config $cfg --frames 8

# 以下才是正式训练；本轮没有代用户执行
& $py -I -B $entry train --config $cfg
```

也可在experiment/code目录用python -m flyrl，wrapper则支持-I且不依赖PYTHONPATH。不用激活venv或运行旧teacher脚本。

8h单臂批次换配置corridor-8h.json。先走廊再战斗，不保证小时预算内学会。

```powershell
# 四臂用相同配置分别执行，不要同时跑几个全脑进程争CPU
& $py -I -B $entry train --config $cfg --arm rewired --rewire-seed 17
& $py -I -B $entry train --config $cfg --arm retinal
& $py -I -B $entry train --config $cfg --arm pixels

# 同接口PPO备选，不能与DQN结果冒充同算法消融
& $py -I -B $entry train --config $cfg --algorithm ppo
```

## 3. 冻结评测与比较

训练结束输出run和checkpoint目录；latest-checkpoint.json只保存显式模型路径，不被评测覆盖。

```powershell
# 把路径换成本次训练result.json里的checkpoint目录；不是旧model.pt
& $py -I -B $entry evaluate --checkpoint 'D:/fly/experiment/runs/<run>/checkpoint-<steps>-<id>' --split val

# 同一模型、相同冻结统计，检查输入依赖和未训练/随机基线（仅val调试）
& $py -I -B $entry evaluate --checkpoint 'D:/fly/experiment/runs/<run>/checkpoint-<steps>-<id>' --split val --intervention black
& $py -I -B $entry evaluate --checkpoint 'D:/fly/experiment/runs/<run>/checkpoint-<steps>-<id>' --split val --intervention scramble
& $py -I -B $entry evaluate --checkpoint 'D:/fly/experiment/runs/<run>/checkpoint-<steps>-<id>' --split val --policy random
& $py -I -B $entry evaluate --checkpoint 'D:/fly/experiment/runs/<run>/checkpoint-<steps>-<id>' --split val --policy untrained

# 所有选择结束后才跑test，不能反复拿test调参
& $py -I -B $entry evaluate --checkpoint 'D:/fly/experiment/runs/<run>/checkpoint-<steps>-<id>' --split test

# 只接受实际步数/布局/源码/版本/资产等可比的冻结评估result
& $py -I -B $entry compare 'D:/fly/experiment/runs/<eval-real>/result.json' 'D:/fly/experiment/runs/<eval-control>/result.json'
```

走廊默认每个唯一layout一局，不把同地图换seed当独立地图；当前默认val/test各2个布局只是pilot。combat仍是单一固定地图，不宣称未见地图泛化。

严格评测会拒绝源码/输入/版本与checkpoint不一致。继续修改代码后，旧smoke checkpoint无法通过新版本评测是预期保护；不要删校验或用不安全pickle强加载。checkpoint没有全脑/游戏动态快照，只供冻结评估，不支持精确续训。

## 4. 测试

无需安装pytest：

```powershell
& $py -I -B -c "import sys,unittest;sys.path.insert(0,'D:/fly/experiment/code');s=unittest.defaultTestLoader.discover('D:/fly/experiment/code/tests',pattern='test_flyrl_*.py');r=unittest.TextTestRunner(verbosity=1).run(s);sys.exit(not r.wasSuccessful())"
& $py -I -B 'D:/fly/experiment/code/tests/test_corridor_pipeline.py'
```

第二条旧测试脚本自行加入code路径，共17项。通过只说明覆盖的实现正确，不意味着旧采集/缓存/所有调用链已修，也不意味全脑视觉生理有效。

## 5. 输出与边界

- experiment/runs/<唯一ID>/：配置、溯源、groups.npz、calibration.json、episodes.jsonl、updates.jsonl、checkpoint和result/failure。
- 每局先重置game、brain、motion和trace；time-limit保留真实final RGB并bootstrap；真终止不bootstrap。游戏reset有10个引擎tic出生等待，不算策略动作，日志须与step区分。
- 原始图SHA256应为 f4d41f011e97e510761d011634891b1c082f61f91463eb586ee9cf8c6371b1d1。
- 不再保存每帧166700全脑counts。replay存246维float32特征，容量20000，仅obs/next_obs约39.36MB，另有少量动作/奖励数组。
- wall-time到期后不开始新优化；单个不可中断调用和最终持久化可能略超时，真实wall_s单独记录。
- 下载/解压内容视作不可信数据，放独立空目录；用-I读取；不从下载目录运行构建器，不加载不可信PT pickle。

2026-10-19本地验收证据索引：[runs/acceptance-20261019.json](runs/acceptance-20261019.json)。65项新测试、17项旧共享测试通过；最终DQN/PPO各12转移微型优化、安全快照重载和冻结评估，以及DQN四种冻结控制、combat短闭环已通过。14个结果路径存在，最终源码hash一致。微型验收每局仅2tic，不是学会游戏的证据，也不是正式训练模型。

详细方法、成功线和预算限制见TRAINING_PLAN.md及根目录分析报告第10节。保留所有历史失败材料；没有授权不得删备份、commit/push或发布项目。
