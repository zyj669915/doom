# 大脑快照 v0-original（训练前原始基准）

**这是什么**：本项目使用的果蝇大脑连接组（MaleCNS v1.0）**训练前的原始快照**。在你开始训练（可能修改权重/结构）之前，先保存这份原始数据，用于：
1. **恢复**：任何训练改动后，可从此快照恢复到原始连接组
2. **对比**：训练前后"大脑结构/权重"的差异对照基线

**位置**：`D:\fly\brain-snapshot\v0-original\malecns_v1\`
**大小**：1,369.7 MB，11 个文件
**来源**：从 `D:\fly\doomfly\connectome_data\malecns_v1\` 完整复制（robocopy 校验，0 失败）

## 文件清单

| 文件 | 大小 | 内容 |
|---|---|---|
| edges.feather | 1002.5 MB | 原始突触连接（25,582,938 条边） |
| normalized/edges.arrow | 293.4 MB | 运行时归一化边（CSR 图的源） |
| neurotransmitters.feather | 41.3 MB | 神经递质类型 |
| annotations.feather | 13.8 MB | 神经元注释 |
| normalized/catalog.feather | 7.8 MB | 神经元目录 |
| normalized/neurons.feather | 7.1 MB | 神经元本体 |
| normalized/neuron_ids.npy | 1.3 MB | 神经元 ID 数组 |
| normalized/outgoing_synapse_counts.npy | 1.3 MB | 出边计数 |
| normalized/incoming_synapse_counts.npy | 1.3 MB | 入边计数 |
| normalized/report.json | <1 MB | 归一化报告 |
| source.lock.json | <1 MB | 数据源锁定记录 |

**数据规模**：166,700 神经元 / 25,582,938 边 / 124,177,617 接触 / 10,299,701 weight_one / 101 self_edges

## 当前仿真参数（2026-09-29 修复实验后的基线）

- LIF 内核：静息/复位 −52mV、放电阈值 −45mV（修复实验最优 −48）、膜 τ20ms、突触 τ5ms、延迟 1.8ms、不应期 2.2ms、dt=0.1ms
- 权重公式：突触数 × 递质符号 × 0.275（ACh+，GABA/谷氨酸/组胺−）
- 感光驱动：`30L/(0.02+L)`；板层恒定偏置 bias（默认 12mV，修复实验最优组合 bias=6 + 权重×1.5 + 阈值 −48）
- 解码：BCI（DNp20/DNpe017）或 biological（MN9 开火+运动）

**注意**：本快照只含连接组数据（大脑结构）。仿真参数与解码方式在代码里（`D:\fly\doomfly\doom\`），训练若要改权重/结构，建议在此快照目录下另存版本（v1、v2…），保持 v0 不变。

## 快照历史

| 版本 | 时间 | 内容 |
|---|---|---|
| v0-original | 2026-09-29 | 训练前原始连接组（MaleCNS v1.0 原样） |
