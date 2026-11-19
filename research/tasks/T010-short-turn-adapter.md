# T010：独立短任务训练适配

状态由 [TASKS](../TASKS.md) 统一维护，本文不另存第二份状态。用户于2026-11-17随后授权“继续执行”，本任务据此完成独立适配及开发测试，不包含T020全脑实机或30k正式训练。实际实现与范围见 [适配验收](../../docs/SHORT_TURN_TRAIN_ADAPTER_20261117.md)。

## 要解决的问题

任务建立时已有短程环境，但原环境 CLI 无 train；旧 stage 的协议、终点、来源清单和评估队列不能通过修改配置冒充新任务。需要让现有完整脑表示／DoubleDQN 在新环境中可训练，并可用同输入前处理的 frontend 对照，不重写第三套学习算法。

## 必读与可复用输入

- [目标](../CHARTER.md)、[状态](../STATUS.md)、[环境协议](../protocols/short-turn-env-v1.md)、[学习协议草案](../protocols/short-turn-learning-v1.md)。
- [短环境](../../code/flyrl_turn/environment.py)、[配置/构建接口](../../code/flyrl_turn/config.py)、[现有环境验收](../../docs/SHORT_TURN_ACCEPTANCE_20261117.md)。
- 只读检查 [旧 stage runner](../../code/flyrl_stage/runner.py)、[学习器](../../code/flyrl_next/learning.py)、[模型](../../code/flyrl_next/models.py)、[脑](../../code/flyrl/brain.py)、[资源模块](../../code/flyrl_stage/resources.py) 和实际 manifest，确定可复用接口后再设计新增模块。

## 允许的最小实现范围（获授权后）

- 在不属于旧 checkpoint 来源清单的新文件中增加适配器、配置和测试；旧源码保持字节不变，不复制整套三 arm 框架，不放宽验证。
- real：RGB→既有受体前处理→完整冻结脑→1536 维→外部读出；frontend 明确是无脑匹配对照，无像素旁路混入 real。
- 每 tic 推进脑和感觉历史，4 tic 决策；实际长度 k 的宏转移使用 R=Σγ^i r_i、D=(非真终止)γ^k，target=R+D·Q，不再额外乘 γ。
- 正确区分 terminal/truncation，截断保留真实最终 RGB/特征；初始帧、校准、训练、评估分别记账；reset 切断跨局历史。
- 配置读取一次形成快照；manifest 绑定自身和实际只读依赖、任务、输入、资产、版本及权重哈希；原子保存与受控停止可审计。
- resume 保留读出/target/optimizer/replay/校准/RNG/累计 schedule，但重置新 episode、脑及历史；不得声称精确恢复游戏。
- 复用现有资源保护，标准库准入在重依赖/全脑加载之前；新入口不能暗中自动运行训练或评估队列。

## 交付与验收

1. 独立可解析 train/eval/短验收入口与最小 real/frontend 配置；接口按实际实现命名，不在本文编造现成命令。
2. 纯测试覆盖宏步折扣、边界、校准隔离、来源拒绝、原子 checkpoint、停止、resume、新旧协议拒绝混用；frontend 测试证明不加载脑。
3. 旧 real50k 55 源码核对不漂移；不加载未知 pickle，不改权威图、不消费 confirm。
4. 说明所有可执行命令及行为，区分测试通过、真实游戏、全脑验收与学习结果。
5. T020 的真实全脑验收单独授权并限额；T010 完成不意味着可以开始 30k 训练。

## 不做与停止条件

不换 PPO/GRU/前端，不增加脑内学习，不自动重跑 136 局环境验收，不原样续接旧队列。若必须修改旧冻结源码才能复用，先停下提出独立替代方案；若范围变成重写训练框架，收缩为接口原型再决策。纯测试不能证明整脑实机可用或 G1 成功。
