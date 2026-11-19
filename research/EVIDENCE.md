# 结论与证据索引

核实日期：2026-11-17。只保存小型索引，不复制原始 run、权重、脑图或图像。文件存在不等于内容已重新核验；下表说明复核层级。

| ID | 观察／允许的结论 | 来源 | 证据边界 |
|---|---|---|---|
| E001 | 旧 real 累计 50k tic，同一训练谱系 | [训练 result](../runs_stage_20261026/real/20261026-201524-train-real-761edd2a/result.json) | result 本次重读；无新训练；完整训练局成功0来自既有逐局核查 |
| E002 | 旧 50k val 未完成，7 局完整失败 | [val result](../runs_stage_20261026/real/20261026-220225-eval-val-real-c91f7e60/result.json) | complete=false、success_rate=null；RAM原因见既有 [执行复盘](../docs/STAGE_TRAINING_20261026.md) |
| E003 | 旧 55 源码近期核对无漂移 | [原 manifest](../runs_stage_20261026/real/20261026-201524-train-real-761edd2a/checkpoint-000050000-9d0f5a7f/manifest.json)、[复核记录](../runs_turn_20261117/post-review-validation.json) | 不是本次重新加载模型或核验全部输入/依赖；恢复仍需完整校验 |
| E010 | 原版短任务环境 136/136 完成，通过可达性/开环验收 | [result](../runs_turn_20261117/20261117-172347-accept-258e2255/result.json)、[provenance](../runs_turn_20261117/20261117-172347-accept-258e2255/provenance.json) | result 本次重读；graph_loaded=false、training_started=false；固定序列 train 各6/12，不是学习成绩 |
| E011 | 实机第一次被资源门禁拒绝，后来重试通过 | [拒绝 result](../runs_turn_20261117/20261117-170037-accept-473c861e/result.json)、[后次资源](../runs_turn_20261117/20261117-172347-accept-258e2255/resources.json) | 保留失败路径，不把单时刻门禁概括成项目一直无法运行 |
| E012 | 两项边界修复，26项新增测试、82关联回归通过 | [post-review-validation](../runs_turn_20261117/post-review-validation.json)、[验收说明](../docs/SHORT_TURN_ACCEPTANCE_20261117.md) | 修复版未重跑实机，3源码与历史实机manifest预期不同；环境几何/奖励/动作未改 |
| E020 | 历史 A6 val6/12 等为无脑CNN记录 | [历史进度](../docs/PROGRESS.md) | 当前本机原runs缺失，仅二手档案；不当重新验证或完整脑成功 |
| E030 | 本次科研结构静态检查、10项检查器回归及旧55源码只读复核 | [结构验证记录](structure-validation-20261117.json) | 仅结构/软件与源码哈希，不加载模型、不重验脑图或游戏，不新增科学成绩 |
| E040 | T010独立训练适配、替身闭环/安全快照/冻结评估开发验证 | [T010验证](t010-validation-20261117.json)、[实现说明](../docs/SHORT_TURN_TRAIN_ADAPTER_20261117.md) | 不加载完整脑、不启动真实游戏或正式训练；正式协议仍draft，confirm未消费 |
| E041 | T020真实CLI资源准入被拒绝，accept未启动 | [T020记录](t020-validation-20261117.json)、[准入原始结果](../runs_turn_train_20261117/20261117-202401-doctor-real-eaebb7e1/result.json) | 此时RAM差约218.3MiB，未加载脑/游戏；不是方法阴性，不代表持续资源状态 |

## 证据使用规则

- 先说观察和样本/预算，再说解释；解释标注替代原因，不能倒写进原 result。
- 方法阳性、阴性、执行失败、未执行分别记录。资源停止不是方法阴性，单 seed 失败不普遍否定算法。
- 同任务/预算/条件才作匹配比较；不同版本另列。旧 near_turn 与新短门的 progress、成功率不能直接拼成学习曲线。
- 训练 seed 与条件 case 分开；同一谱系 10k→30k→50k 不算三独立 seed，不把50k累计当成额外90k。
- 软件、实机环境、全脑资源、学习、结构贡献、生理验证分级报告；不由低层验收推出高层结论。
- 新结果先存 run，再创建 [实验记录](experiments/README.md)，最后更新本表和 [STATUS](STATUS.md)。

## 来源与缺失处理

若链接指向的本地 run 在另一台机器缺失，写“原始产物未随文字同步提供”，不要生成空结果或从报告倒造 JSON。research/prompts 是可纳入版本管理的文本；runs 仍是本地大产物，文字索引可保留相对路径，但不意味着已归档到 Git 或外部备份。
