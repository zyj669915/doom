# 短程任务训练适配：实现与开发验收

日期：2026-11-17。任务：[T010](../research/tasks/T010-short-turn-adapter.md)。用户在结构整理后授权“继续执行”，本轮实施独立适配与开发测试，**没有正式训练、真实游戏、完整脑加载或confirm评估**。

## 实现范围

新增 [flyrl_turn_train](../code/flyrl_turn_train/) 及 [CLI](../code/flyrl_turn_train_cli.py)，协议malecns-short-turn-learning-v1，绑定已实现环境malecns-short-turn-v1。旧flyrl/flyrl_next/flyrl_stage/flyrl_turn及doomfly源码不改，不重写旧manifest，不恢复已有删除。

- [配置](../code/flyrl_turn_train/config.py)：标准库严格校验，读取一次快照；仅real/frontend、CPU、AGC-motion、原脑工作点、六动作/4tic/280tic/γ=.999。最多30k训练tic、12h累计墙钟，配置默认6h只是开发保护值，尚非锁定实验预算。
- [编码](../code/flyrl_turn_train/encoding.py)：继承既有stage表征并绑定新协议；real仍全图，frontend读取同图受体ID/UV但不实例化FrozenBrain。黑屏在感觉输入处先于AGC/运动处理，从episode开始，保留同一冻结校准。
- [训练/评估](../code/flyrl_turn_train/runner.py)：复用现有DoubleDQN、模型及逐tic宏步。训练/校准/初始感觉帧/脑时间分别记账，预留0/10k/30k快照请求，不自动启动评估进程。
- [存储](../code/flyrl_turn_train/storage.py)：递归源码、图/iwad资产、解释器与依赖版本、配置/state强hash；先校验再CPU weights_only=True；UUID incomplete目录原子提交、原子latest，失败保留不完整证据。
- CLI：准入发生在NumPy/torch/游戏导入前，跨output使用固定TURN_TRAIN_ACTIVE.lock；最终close、写盘和monitor退出的STOP/资源/墙钟仍否决complete。未知锁不删除。

## 学习与恢复语义

MLP1536→256→128→6；DoubleDQN＋Adam/Huber，lr3e−4、batch64、replay5000、learning_starts256决策、每次完整决策至多一次更新、target每500更新、梯度裁剪10。epsilon按累计训练tic从1线性降到.1，10k后保持.1；eval greedy。

校准默认128个随机动作宏步样本，按12个train case固定配额，各至少两个；每宏步最多4tic。每case重置感觉/脑历史，统计跨train case累积后冻结；不使用dev/confirm。校准tic与初始帧另计，不当训练预算。训练每局从12个train case均匀采样，case绑定自己的环境seed；不向策略提供case/镜像或遥测。

宏步使用R=Σγ^i r_i、D=0（真终止）否则γ^k，target=R+D·Q，不再乘γ。截断保留真实最终RGB/特征，reset不跨局。

受控边界停止后的续训只允许output、累计max_game_tics、累计wall_hours变化，其他配置/arm/seed/任务必须匹配；拒绝目标小于已有交互、无校准/校准中断或accept快照。保留target/optimizer/replay/校准/RNG/累计计数，进入新episode并重置脑与感觉历史，不是精确恢复游戏。快照保存期间主要墙钟由checkpoint_write_seconds补计；最后manifest/目录rename/latest自身I/O无法精确写入即将提交的manifest，属于已知小尾部记账局限；本次执行的停止判断仍包含整个操作。

## 命令（列出不等于本轮已执行）

已有 [real配置](../configs_turn_train/real-dev.json) / [frontend配置](../configs_turn_train/frontend-dev.json)，均为开发配置，不是已锁定正式方案。

```powershell
$py = 'D:/fly/doomfly/.venv-neural/Scripts/python.exe'
$cli = 'D:/fly/experiment/code/flyrl_turn_train_cli.py'
$cfg = 'D:/fly/experiment/configs_turn_train/real-dev.json'
& $py -I -B $cli validate --config $cfg
# 以下真实执行需对应授权、资源和来源准入；本轮未运行。
& $py -I -B $cli doctor --config $cfg
& $py -I -B $cli accept --config $cfg
& $py -I -B $cli train --config $cfg
# 从本协议快照恢复；不能传旧stage或accept快照。
& $py -I -B $cli train --config $cfg --resume '<本协议checkpoint目录>'
& $py -I -B $cli evaluate --checkpoint '<本协议checkpoint目录>' --quick
& $py -I -B $cli evaluate --checkpoint '<本协议checkpoint目录>' --policy random
& $py -I -B $cli evaluate --checkpoint '<本协议checkpoint目录>' --policy untrained
& $py -I -B $cli evaluate --checkpoint '<本协议checkpoint目录>' --intervention black
```

accept：总game_tics＋初始感觉帧最多256、最多600秒，不校准/优化/replay，不是136局环境辨识。evaluate只支持新dev：quick固定前4case、完整12case，可用两预声明sequence-negative/sequence-positive；不按镜像选序列，不开放confirm以免未锁定使用。随机每4tic保持动作；所有策略都保留同一感觉/脑更新路径，便于成本与来源审计。

输出为runs_turn_train_20261117/<独立run>/；STOP可放run/output/其父目录。单槽锁只约束新适配入口，旧CLI不共享此锁，操作前仍须确认没有其他完整脑任务，不能据该锁推断全机器空闲。

## 独立复核与修复

独立只读复核确认四项边界并在新适配中修复，未改旧源码：

1. SIGINT转换为完整转移/更新边界的controlled_stop，避免更新中途保存不完整学习器。其他异常写failure-progress，记录已完成计数、未返回宏步的已知env.tics增量、墙钟/phase/最后安全快照及source_resume；不保存污染态，拒绝从含failure/failure-progress的run自动恢复，需另行审计成本和恢复方案。
2. CLI最终STOP/资源/墙钟否决冻结评估时，同时清空success_rate，保留已完成逐局事实。
3. 评估完成前重新计算完整provenance并比对起始manifest，source_drift时complete=false、success_rate=null。每次reset的实际cfg/WAD/IWAD资产来源另写assets.jsonl。
4. 只读env代理累计宏步内最大progress，不再只取最后tic值；仍复用原宏步奖励/折扣实现。

异常成本记录不是通用谱系账本；跨run人为重选更早source checkpoint仍需外部审计，不能假装自动防止所有重复预算。进程被强杀或系统断电也不保证最后成本已落盘。此适配明确支持受控边界停止恢复，而非任意异常无损恢复。

## 本轮开发验证与未验事项

实际调用了两配置validate及--help，退出0；validate不导入NumPy/torch/ViZDoom/脑，有子进程测试确认。开发测试使用临时目录、小向量、替身游戏和合成2受体图，进行了真实DoubleDQN更新及真实安全快照save/load往返，**不是完整脑或ViZDoom训练**。

第一次关联回归在加载torch时c10.dll出现WinError1114，suite尚未执行；使用既有环境先导入torch、单线程后初版关联169项通过；四项独立复核修复新增6项回归后，最终关联175项全部通过。最终结果见 [机器可读验证](../research/t010-validation-20261117.json)，不把测试通过当G1成功。

仍需：T020新任务完整脑真实资源/吞吐与无优化链路验收；T015锁定实际seed/评估时点/总成本/确认规则；之后获训练授权才做real/frontend有限筛查。原136局实机仍是旧环境验收版本证据，本次没有重跑或修改其记录。
