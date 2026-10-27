# 历史训练方案：冻结全脑 + 小读出在线强化学习（2026-10-19）

> **2026-10-26 当前入口：**下一轮以[主报告R3](../果蝇闭环游戏_科研复盘与下一阶段实验方案_20261025.md#r3-当前进度与收敛训练方案2026-10-26)与[最新进度](docs/PROGRESS.md)为准。保留本文作旧flyrl协议说明，下面的旧线性头、20k步上限和性能优先顺序不再是新实验默认。当前主线为A2样式完整脑＋MLP DoubleDQN的real/rewired/frontend匹配实验；A1备选、A3条件触发，暂停原A4/A5/A6扩训，暂缓A7。先补独立版本三arm与同光照短任务并验收，再做10ktic链路pilot及统一30–50ktic学习筛查。**这些功能尚未补齐，不要把旧命令当成新方案已可执行。**不改权威图、不污染冻结源码，不自动启动长训。

更新：2026-10-19。本文取代旧的“先重训 PPO 老师再蒸馏”推荐。旧脚本、模型与负结果保留作历史材料，不作为新管线缓存或可信验收成绩。

## 1. 决策与模型边界

目标优先级：闭环游戏表现；先 corridor，再 combat。当前实现是**完整 MaleCNS 冻结 reservoir + 外部线性读出**，不是脑内学习，也不是已验证的生物果蝇视觉/动作系统。

- 不修改 graph.npz、原始 feather、backup、brain-v0-clean 或权威图副本。保留全部发布边。
- RGB24 → 线性光亮度/UV → 可选人工 agc_motion → 参数化全脑 LIF → 解剖群体平均 Hz → 50/200ms trace + log1p → 训练期冻结归一化 → 小读出 → 动作。
- 原始 annotations 按整数 bodyId 对齐 source_id，用显式 somaSide，缺失时才用 rootSide；冲突/中线保留 unknown。123 固定互斥群体，246维。不是依据动作标签选择神经元。侧别是解剖注释，不等于视网膜坐标。
- 实际游戏35Hz，神经 dt=.1ms，逐帧累计余数，28.5/28.6ms交替；不能再用50ms/tic。600tics约17.14秒，210tics约6秒。
- 默认 scale=1.5、bias=6、threshold=-48 是历史工程候选，不是生理最优值；上游基线为1/12/-45。新包不调用阈值硬编码的Native内核。
- 坐标、ANGLE、生命、敌人/击杀信息不进入策略，仅用于环境奖励和审计。

## 2. 为什么主线选择 DoubleDQN

全脑主要在CPU，采样比小读出更新贵；动作仅6/12种，采用 replay 重用昂贵转移，DoubleDQN分离选动作与目标估值。它是有条件的工程选择，不是声称已比较训练成绩或保证优于PPO。

| 方法 | 本轮定位 | 原因 |
|---|---|---|
| 在线DoubleDQN | 主线 | 小离散动作、replay样本重用、显式next feature、Huber与target稳定化 |
| PPO | 同接口备选 | 少一些离策略外推，但需要不断产生新on-policy样本；样本重用有限 |
| teacher-student / DAgger | 历史诊断、未来公平对照 | ANGLE特权与闭环覆盖问题须先控制；不是原则上不能用，也不是数学上限 |
| 离线FQI | 诊断而非默认训练 | 旧探索数据覆盖不足，未覆盖动作价值可能外推失真 |
| ES | 不选 | 每候选需要昂贵全脑闭环评估 |
| 全图BPTT/EventProp/三因子 | 本轮不引入 | 信用分配、可塑性与生理验证是另一个研究阶段，不能只加奖励乘权重冒充 |

主线默认：线性头 hidden=0；Adam lr3e-4、batch64、replay20000、warmup256、每环境转移1次更新、target每500优化步同步、Huber、梯度裁剪10、epsilon1→.1（10000步）。

**gamma=.999**：35Hz长走廊约数百tic，.99的有效视野仅约100tic，过度折扣远端出口。本轮统一环境potential shaping与学习器gamma=.999；它仍是待验证工程参数，不保证最优。环境奖励必须随gamma一致改变。

PPO备选：相同线性actor，critic hidden32，rollout128、epochs4、batch64、clip.2、lambda.95、entropy.01、lr3e-4、gradclip.5、KL监测。old_logp为采样时固定值；time-limit可bootstrap但GAE不跨reset。

## 3. 任务与课程

走廊6动作：turn{-6,0,+6}×forward{0,20}；战斗再×attack{0,1}，从不强制前进。

走廊课程按**累计训练步数**：前10%出口附近直行、10–30%转角附近、余下从完整入口；每阶段包含左右镜像。课程只改变生成WAD起点，不按坐标选动作。训练/验证/测试以实际生成WAD内容哈希拒绝重复；验证/测试改变起点抖动和光照，但仍是同一L形几何家族，不是任意新地图泛化。

奖励：每tic-.001，出口一次+10，forward且低位移才撞墙-.02，potential=gamma*Phi(next)-Phi(now)，Phi=-remaining/3712*5，真终止Phi=0，截断保留Phi与最终画面。停止/原地转向不是撞墙。战斗10Δkills+.1Δhits-.02damage-.001，奖励是工程定义而非果蝇多巴胺机制。

## 4. 预算及四臂

配置：code/flyrl/configs/corridor.json（2h pilot）、corridor-8h.json（8h批次）、combat.json（8h，仅走廊有效后启用）。均max_env_steps=20000；达到步数或墙钟先到者停止。墙钟包含启动与校准；单次引擎/编码/优化调用不可中断，最终持久化额外记录。停止时可能尚未进入完整入口课程，须看真实steps和课程日志，不把课程局成功冒充完整走廊成功。

四臂：real、rewired（按源符号层置换目标端点）、retinal（同UV人工前端后固定池化）、pixels（全图像固定池化）；共同246维、trace、读出、动作、奖励、样本上限与墙钟上限。重连保入出度/每目标符号入度，不保目标入强度、自环/重复边数；统计写入rewire.json。同维不等于同信息或总计算量。

2h不能保证四臂都达到20000步。若某臂墙钟先到，compare会拒绝把实际步数不同的结果标成同预算。正式归因需选择所有臂都能完成的共同样本预算另作预注册配置，并报告墙钟效率；不要事后挑最好checkpoint/只保留成功run。

## 5. 执行顺序

具体可复制命令见SETUP.md。

1. doctor核对版本与脑图强hash；smoke仅短闭环，不优化；diagnose查黑屏/静止/闪光/左右运动/空间置换。
2. 先跑真实臂2h pilot；不用旧老师/旧counts。校准均匀分配全部训练布局上的无标签随机轨迹，冻结统计。
3. 显式checkpoint评估val；默认走廊每个唯一layout只跑一局，避免确定性seed伪重复。默认仅2个val/2个test布局，样本太少，只能pilot。
4. 若完整入口未学习，先核对课程是否实际走到、动作分布、黑屏差值、观测有效维度；不能直接扩大MLP并宣称脑提供信息。预先决定用PPO的公平同预算试验或更细空间表征研究，不能在test上选方法。
5. 完整入口val成功率预注册目标≥80%且显著优于随机/未训练头，撞墙率下降，输入干预确实损害表现后，才冻结方案跑test并迁移combat。默认2个layout无法建立稳定80%推断，须扩展独立布局后正式确认。
6. 连接组优势另需至少3个训练初始化seed、多个独立重连实例、更多未见布局。报告episode条件CI与训练seed间不确定性，不能把二者混同。

## 6. 存档与不做事项

唯一run目录在experiment/runs，含resolved config、全源码/graph/annotations/WAD/IWAD强hash、版本、groups、校准轨迹、JSONL、版本化checkpoint。torch.load仅weights_only=True；禁止加载来源不明pickle。新checkpoint只供冻结评估；没有游戏/全脑动态快照，不声称可精确resume。

旧corridor_student* / corridor_rl_q / corridor_readout_upgrade仍有已记录残留，**不推荐继续正式训练**；新入口不导入它们、不复用缓存、不解析latest。旧结果只保留为历史观察：如旧战斗distilled2.33 vs baseline1.0、走廊ANGLE老师100%、学生0%，不能作为新包可重复成绩或科学优势。

本轮仅做短smoke、纯函数/toy测试与各算法12转移的生产链路优化验收，未启动正式数小时训练。不删除文件、不commit/push，不公开部署。原作者公开上线的How it works和科学launch门槛仍是独立事项。
