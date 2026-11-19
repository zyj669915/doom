# 短程环境：原版实机与后续修复

记录日期：2026-11-17。协议：[malecns-short-turn-v1](../protocols/short-turn-env-v1.md)。问题是环境是否可达、非平凡且边界正确，**不是模型能否学会**。

## 执行与结果

1. [17:00:37首次result](../../runs_turn_20261117/20261117-170037-accept-473c861e/result.json)：资源门禁拒绝，未启动游戏或训练；失败材料保留。
2. [17:23:47实机result](../../runs_turn_20261117/20261117-172347-accept-258e2255/result.json)：136/136完成，674.375秒，通过环境门槛，graph_loaded=false、training_started=false。
3. 名义左右各108tic、0碰墙可达。train六固定动作均0/12、两周期与随机各1/12、两固定序列各6/12；所有序列均无条件施用于两侧，不把每侧最优序列组合成“学习策略”。
4. [provenance](../../runs_turn_20261117/20261117-172347-accept-258e2255/provenance.json) 与 [resources](../../runs_turn_20261117/20261117-172347-accept-258e2255/resources.json) 保存源码/版本及资源；离散采样峰值不是完整生命周期峰值。

## 修复及版本边界

实机后独立复核修复末局/监控退出STOP或超时漏判，以及门禁前后重复读配置的TOCTOU。新增4回归后，新测试26项、关联回归82项通过，CLI validate通过。[post-review-validation](../../runs_turn_20261117/post-review-validation.json) 明确 live_rerun_after_fixes=false。

环境几何、奖励和动作未变；当前acceptance/config/CLI三文件与历史实机manifest有预期差异，不能重写历史manifest伪装为修复版实机已通过。详见 [完整验收说明](../../docs/SHORT_TURN_ACCEPTANCE_20261117.md)。

## 结论与下一步

任务适合作为入门左右选择，50%开环基线必须报告并超过；不是丰富导航、视觉依赖或G1。当次环境验收没有训练适配或完整脑/冻结dev证据。后续[T010](../tasks/T010-short-turn-adapter.md)已完成独立适配开发验证；完整脑新任务吞吐/资源、真实未训练神经策略、冻结dev/confirm与学习仍待完成，不默认重复136局。
