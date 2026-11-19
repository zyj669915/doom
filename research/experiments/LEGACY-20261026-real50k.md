# 旧 real50k：训练完成，完整评估未完成

记录日期：2026-11-17。这是历史索引，不是新实验或继续训练授权。

- 协议：malecns-near-turn-three-arm-v1；near_turn出生，远端出口，最多1200 tic/局。
- 方法：完整冻结脑→1536维→MLP DoubleDQN；训练只改变外部读出。
- 谱系：seed41027，10k→30k→50k为同一谱系，累计50k而非90k；后两组仅资源验收。
- [最终训练result](../../runs_stage_20261026/real/20261026-201524-train-real-761edd2a/result.json)：50,000 game tics、12,500 decisions、12,245 updates、40完整episode；完整训练局成功0，另有800tic部分局。
- [完整val中断result](../../runs_stage_20261026/real/20261026-220225-eval-val-real-c91f7e60/result.json)：7局完成均失败，第8局中断，complete=false，success_rate=null；不能记为0/12。
- [训练最终checkpoint manifest](../../runs_stage_20261026/real/20261026-201524-train-real-761edd2a/checkpoint-000050000-9d0f5a7f/manifest.json) 与评估使用的同50k另一checkpoint后缀不同，不能隐去身份差异；评估绑定路径见其result。

观察不能说明拓扑无用：匹配正式三组比较未完成，行为在地板，只有单seed。旧progress以整条entry路线3712为分母，near_turn名义起点即约43.1%，不能解释成已学会43%。

决策：保留阴性与中断，不默认扩训或补齐旧队列。下一步缩短行为链，见 [当前主线](../CHARTER.md) 与 [学习草案](../protocols/short-turn-learning-v1.md)。补旧val只属可选归档，不阻塞新任务。
