# DOOMFLY 原版参考与研究总结

## 一、原版项目已下载到 D 盘

- 原版代码（上游 GitHub: `nftechie/doomfly`，最新提交 `71ecf53`）：`D:\fly\reference\doomfly-upstream`
- **你机器上的 `D:\fly\doomfly` 与原版行为完全一致**。差异只有 4 处环境适配，均不影响行为：
  1. `doom/build_kernel.py`、`doom_learning_v6/brain.py`：clang++ → g++（Windows 无 clang）
  2. `data-integrity.json`：重新审计的时间戳/哈希
  3. `manifest.json`：4 个 BCI 读出（DNp20/DNpe017）在列表中的位置挪到前面（解码按 type 过滤，顺序无关）

## 二、原版作者的“成果”到底在哪

**作者做了验证、也做了训练尝试，但两者都是负面结果**（作者自己在 README 里写明）：

> “Status: **live experimental training, not demonstrated learned survival.** The current v6 candidate **failed its visual, conditioning and survival validation gates**.”（在线实验训练中，未证明学会生存；v6 候选未通过视觉/条件化/存活验证门）

- **工程 BCI 是作者自己承认的临时方案**：README 原话 “DNp20 right-minus-left activity to turning, DNpe017 to movement and firing. **These are engineered controller assignments, not established natural motor functions.**”
- 项目里的 `validation.json` / `audit-experiments.json` 等审计文件**就是作者做的“这一步”**——他记录到：生物运动/口器读出（MN9/MDN/DNa02/DNp09）在所有条件下 0 脉冲；视觉通路未校准。
- 学习实验 v1–v6 全部是负结果（v6 未通过验证门）。

**所以“别人做的那个版本的效果”= 基线 BCI 的随机行为**，不是学会玩 Doom。

## 三、实测：提升在哪里、是否正确连接

| 测试（同帧/同 seed） | 正常视觉 | 黑屏 | 断开视网膜 |
|---|---|---|---|
| biological 解码（真实运动通路） | 0/0/不开火 | 0/0/不开火 | 0/0/不开火 |
| BCI 解码（DNp20/DNpe017） | 转向1.55/前进20/开火 | 0.29/9.0/开火 | 0.29/9.0/开火（=黑屏） |
| 闭环击杀（3 seed，6 秒神经时间） | 1 / 1 / 1 | **3 / 3 / 2** | — |

结论：**当前不是正确连接**——真实运动通路全部沉默；行为由工程 BCI + 12mV 恒定偏置驱动；且**有视觉比瞎了杀得更少**（视觉是负贡献）。提升空间 = 把“视觉伤害”变成“视觉帮助”。

## 四、果蝇玩游戏的逻辑（靠视觉吗？只用特定链路？）

**链路**：游戏帧(640×480) → 4146 个感光点采样（3335 R1-R6 亮度 + 811 R8 颜色，双线性亮度+gamma）→ 感光神经元注入 `30L/(0.02+L)` → 板层 L1/L2/L3/L5 恒定 **12mV** → 全连接组 LIF 传播（166,700 神经元/2558万边/1.8ms 延迟/-45mV 阈值）→ **只读 4 个下降神经元**：DNp20(R−L)→转向、DNpe017→前进+开火 → ViZDoom 按钮。

- **“靠视觉”但只靠亮度，不靠理解**：没有对象识别、没有瞄准、没有目标坐标。
- **只用了一条特定链路**：视网膜 →(3跳)→ DNp20/DNpe017 + 板层偏置。其余 16.6 万神经元（约 9,500 个在放电）对按钮**零贡献**——黑屏 ≡ 断开视网膜（动作完全相同）已证明。

## 五、参考文献（D:\fly\reference\papers\）

已下载 5 篇（有效 PDF 已校验）：
- Dorkenwald 2024, *Neuronal wiring diagram of an adult brain*, Nature（雌性脑接线图）
- Lappalainen 2024, *Connectome-constrained networks predict neural activity across the fly visual system*, Nature（果蝇视觉系统模型——训练视觉的关键参考）
- Shiu 2024, *A Drosophila computational brain model*, Nature（果蝇脑 LIF 模型）
- Kempka 2016, *ViZDoom*, arXiv（游戏引擎）
- eLife 视觉通路到蘑菇体论文

未下载成功（bioRxiv 对本机限流 429 / PMC 仅 HTML / Cell 403），请在浏览器打开：
- **MaleCNS 数据源论文**（本项目连接组出处）：Berg et al. *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*，bioRxiv 2025.10.09.680999 → https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/ 或 https://doi.org/10.1016/j.cell.2026.08.015
- Bates 2025, *Distributed control circuits across a brain-and-cord connectome*, bioRxiv 2025.07.31.667571 → https://www.biorxiv.org/content/10.1101/2025.07.31.667571v1

## 六、2026-09-23 连接修复实验记录（完整证据链）

**目标**：让真实运动通路（MN9/DNa02/MDN/DNp09）放电并让视觉内容调制行为。

### 动力学诊断（calibrate-diag.json）
- MN9/DNa02/MDN/DNp09 直接上游 **0 放电**（读出永远收不到输入，v 钉死 -52mV）
- DNp20/DNpe017（BCI）上游 100% 活跃，读出膜电位 **-45.17~-45.59mV**（差 0.2~3mV 跨阈值）

### 8 轮扫描关键发现
1. 权重放大：scale=2 时运动链开始放电（MN9L:249），但全图活性x5、DNa02 转向不对称（L183 vs R0）
2. 零偏置：bias=0 时黑屏静默，但视觉也驱动不动运动链
3. **感光→板层是组胺抑制（histamine，真实生物学）**：感光 140Hz 疯狂放电但板层 0——这就是"黑屏杀得多"（黑屏→无抑制→板层被 12mV 抬活）的物理根源
4. 分侧增益、时间差分视觉均无效
5. **bias=6+scale=1.5+th=-48：MN9 视觉活性 451 vs 黑屏 203（+122%）**——神经层视觉调制达成

### 闭环 A/B（3 seed，combat_survival，210 tics）
| 配置 | kills 均值 | 存活 tics | 说明 |
|---|---|---|---|
| baseline（原版 BCI） | 1.0 | 192 | 作者原版 |
| fix_v1（MN9 开火+DNpe017 前进） | 0.33 | 171 | DNpe017 在 scale>=1.5 退化 |
| fix_v2（MN9 运动+开火，DNp20 转向） | **1.33** | 207 | 行为最佳 |
| fix_v2 黑屏 | 1.33 | 194 | 视觉仍无贡献 |
| fix_v4（bias=6 视觉调制 122%） | 0.67 | 197 | 移动过多反而送死 |
| 随机连接组 | 0.67 | 194 | 与修复版不可区分 |

### 最终结论
- ✅ **物理连接达成**：真实运动神经元 MN9 从 0 脉冲 → 激活并驱动行为（开火+运动）
- ✅ **神经视觉调制达成**：视觉活性为黑屏 2.2 倍（感光抑制→板层→运动链）
- ❌ **行为层面视觉稳定正贡献未达成**：黑屏闭环表现等效甚至更好（12mV 偏置让行为通道不依赖视觉；更多移动不必然等于更多击杀）
- **判断**：固定权重 LIF + 亮度饱和视网膜 + 恒定偏置，行为提升已到物理极限。"玩得好"需要目标导向的学习/可塑性——这是作者 v1-v6 全部失败、README 声明 "no demonstrated learned survival" 的同一动力学根源

### 可复跑脚本（D:\fly\doomfly\outputs\doom\）
- audit_calibrate_diag.py（动力学诊断）/ calibrate_scan.py~calibrate_scan8.py（参数扫描）
- calibrate_closed_loop.py（闭环 A/B，增量落盘 JSONL）
- probe_scale1.5.py / probe_th48.py / probe_v3.py（读出放电探测）
- 数据：calibrate-diag.json / calibrate-scan{1..8}.json / calibrate-closed-loop.jsonl / dna02-upstream-audit.json / readout-probe-*.json / probe-v3.json
