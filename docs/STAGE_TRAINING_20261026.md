# 三arm训练执行记录

更新日期：2026-10-26。本文记录本轮实际实施与启动结果，不替代历史实验结果。

## 2026-11-17 核实与路线调整（优先于下方历史状态）

原始结果已核实：10月26日三组256总tic验收均完成；real单seed累计50,000tic、12,245次优化、40个完整训练episode，0通关。0/10k/30k冻结quick-val均0/4；50k完整val完成7局均失败，第8局因RAM保护中断，`complete=false`、`success_rate=null`。队列于22:53停止，rewired/frontend正式训练未开始。下方“未开始训练”和“继续进入三组”等语句是当时快照，不是当前事实。

最新证据：[real50k训练结果](../runs_stage_20261026/real/20261026-201524-train-real-761edd2a/result.json)、[50k未完成val](../runs_stage_20261026/real/20261026-220225-eval-val-real-c91f7e60/result.json)、[队列停止记录](../runs_stage_20261026/suite-status.json)。55个checkpoint来源源码文件与当前版本一致，不代表所有输入/依赖/权重和资源条件均已通过。

**不再默认续接本队列。**新的目标、短程转角任务建议、主线/备选和停止条件见[项目目标与路线决策](reports/项目目标与路线决策_20261117.md)。旧50k补评估可作归档，不是新任务实施前置；原入口不能从中断局精确续评。不修改旧run、checkpoint、冻结源码或权威图。

## 原当前判定（早期快照）

用户已授权开展下一轮限额训练。独立协议入口已新增，已实际调用`doctor`和`suite`；**训练队列被资源门禁拒绝，尚未开始正式训练，也没有新增学习成绩。**不能把资源拒绝当作全脑方法的阴性实验，更不能称已经达到行为阶段目标。

## 2026-10-26 15:35 资源解除与队列实际推进（优先于下方旧记录）

用户授权关闭 Edge/ChatGPT/Claude/VSCode 后，可用内存由约 2.4GiB 恢复至 6.08GiB；`doctor`门禁通过（`preflight-20261026-145009-fec925b5.json`，RAM 4.61GiB、commit 11.04GiB）。

`suite`按正式配置重新启动成功（日志 `runs_stage_20261026/_suite_launch_20261026-1458.log`），已实际推进：

1. **frontend 任务辨识完成**：`frontend/20261026-145741-task-probe-frontend-d1aaecf1`，96 局，8 种预锁固定/周期动作全部失败（未触发"任一成功率75%则不投入全脑长训"提前终止），符合"任务需要学习"预期。
2. **real 256tic 资源验收通过**：`real/20261026-153001-accept-real-cab9187b`，stop=`accept_tic_limit`，255 game_tics、64 decisions、0 updates、wall 121.4s（<10分钟）、brain_ms 7285.7，原子 checkpoint `checkpoint-000000255-0cf9d274` 已保存；随机策略基线 max_progress 0.48。队列继续 rewired/frontend 验收，随后进入 real→rewired→frontend 各 10k→30k→50ktic 训练（墙钟最多12h）。

本轮尚未完成正式训练与冻结评估；这些是本轮新推进、仍待后续完成的验收项。

## 独立实现范围

- 新包：`experiment/code/flyrl_stage/`；入口：`experiment/code/flyrl_stage_cli.py`。
- 协议：`malecns-near-turn-three-arm-v1`；配置：`experiment/configs_stage/near_turn_three_arm_v1.json`。
- real、rewired、frontend均使用1536维双时间尺度signed-hash特征、同容量MLP和DoubleDQN、完整六动作、逐tic感觉处理与4tic决策。
- real/rewired保留全部原图神经元和边；rewired只在内存改变目标端点，不覆盖原图。此null只是粗筛，不保持入强度、自环和重复边数，不足以确认H2。
- frontend用相同受体RGB采样/前处理和受体ID投影，不运行脑动力学；同维度不表示三组信息相同。
- train12、val12、test24为预锁near_turn、同光照224、左右和起点/角度扰动。test未使用。
- 仅训练布局随机动作校准；冻结校准后学习，遥测不进入策略。
- 快照严格绑定源码、依赖版本、原始输入哈希和有效图签名，安全加载使用`weights_only=True`；恢复到新episode，不冒充精确游戏状态恢复。

## 队列的预定行为

单槽串行，先做8种预锁固定/周期动作的train任务辨识；任一成功率达到75%则不投入全脑长训。该检查不是穷尽性的可控性证明，也不使用遥测来选择动作。

随后三个arm分别进行最多256个总脑/感觉tic、10分钟无优化资源验收。通过后按real→rewired→frontend训练，每arm累计10k→30k→50ktic，累计训练墙钟最多12小时。每阶段真实新增交互和优化后才继续，不把同预算重载称为续训。

训练和评估分进程，不同时保留两个脑。预锁0/10k/30k使用相同4个开发val条件，50k使用完整12条件；只有共有4case可跨时点比较。0点另做随机策略对照。50k结果仅为首轮筛查，不能凭单训练seed声称可重复优势。

资源不足、STOP、数值异常、来源漂移、样本预算未完成、任务负对照过高均停止队列，不自动调参追逐阳性。队列目前没有断点恢复总调度功能；中断后须核对状态再按checkpoint续训，不能直接重跑suite冒充原试验延续。

## 实际启动记录

1. `doctor`预检报告：`experiment/runs_stage_20261026/preflight-20261026-141953-841c9516.json`。
   - RAM可用3,857,235,968 B，约3.59GiB。
   - 任务预算1.5GiB＋固定系统余量3GiB，需要至少4.5GiB。
   - RAM不足，退出码3，未导入训练大库或载入脑。
2. `suite`实际启动尝试：`experiment/runs_stage_20261026/preflight-20261026-142941-dc30caa4.json`。
   - RAM可用2,272,362,496 B，约2.12GiB；commit可用5,110,743,040 B，约4.76GiB。
   - RAM与commit均未满足4.5/5GiB启动要求，退出码3。
   - 当时完整单元回归仍在运行，可能占用部分内存；此报告仅为当时快照，不是当前可用量。

3. 完整回归结束后重新doctor：`preflight-20261026-143628-886501f7.json`，RAM约4.02GiB，仍不足。
4. 最终按正式JSON配置重新调用suite：`preflight-20261026-143853-bb7d2a24.json`。RAM可用4,123,172,864 B（约3.84GiB），距4.5GiB启动要求缺约0.66GiB；commit约8.14GiB足够。退出码3，没有留下运行中的训练任务，也未设置自动重试。

门禁在导入numpy/torch/脑之前执行，且API内部再次检查。任务预算下限不能通过配置降低。运行中每2秒采样系统资源和进程树，低于3GiB RAM或512MiB commit/采样未知时请求边界停止。进程树峰值来自离散采样，不是精确生命周期峰值；短命子进程可能漏采。

## 验证与未验事项

- 资源模块21项mock单测通过，并经独立重跑确认。
- 独立协议最终9项测试通过，覆盖配置、frontend安全状态、real/rewired微型全边图、宏转移、门禁不构建脑、原子失败以及模拟环境真实32→64tic新增优化续训/冻结评估/篡改拒绝。增补测试曾因`-I`未包含测试目录而失败一次，已修正导入并重跑通过。
- 首次非隔离全量discover失败：运行106项，53个error，首个错误为`torch c10.dll`初始化`WinError 1114`，后续模块导入连锁失败。不能将该次记为通过，也未据此安装或替换依赖。
- 随后采用`-I -B`、先导入torch、单线程再discover的隔离命令重跑，**共158项测试，157项通过、1项CUDA可用性测试跳过**，约4.98秒。当前不确定首次DLL失败是导入顺序、环境还是瞬时资源所致，不把重跑通过当成已定位根因。
- 历史checkpoint源码清单50个文件全部一致，差异0。核对工具：`experiment/code/check_stage_frozen.py`。本轮该检查不重新hash大图，正式训练provenance仍必须检查权威图强哈希。
- **未完成：**真实ViZDoom任务辨识、全脑256tic资源验收、正式三arm训练、正式冻结评估、完整suite端到端成功运行。当前纯测试不能替代这些验收。

## 本地操作

在`D:/fly`下使用现有环境，不安装依赖：

```powershell
& "D:/fly/doomfly/.venv-neural/Scripts/python.exe" -I -B "D:/fly/experiment/code/flyrl_stage_cli.py" doctor
& "D:/fly/doomfly/.venv-neural/Scripts/python.exe" -I -B "D:/fly/experiment/code/flyrl_stage_cli.py" suite --config "D:/fly/experiment/configs_stage/near_turn_three_arm_v1.json"
```

单arm训练/续训使用`train --arm real --max-game-tics 10000`，续训额外传`--resume <checkpoint目录>`并提高累计样本上限，仍受50k/12h限制。冻结评估使用`evaluate --checkpoint <目录> --quick`。

队列状态为输出目录的`suite-status.json`；各子任务日志、run、admission、resources、checkpoint和result独立保留。向输出根目录写入`STOP`可在边界停止；它不是强制杀进程。异常退出留下的锁只可在确认原PID已退出后处理，不自动清理未知锁。不要删除历史run、备份或权威图。

下一步是资源恢复后重新门禁与真实验收，不是继续增加CNN训练、降低系统余量或宣称训练已开。
