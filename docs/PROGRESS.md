# DOOMFLY 项目进度档案

> 本文件是项目的**权威进度记录**：原版是什么样、我们每一轮做了什么、结论是什么、下一步做什么。
> 每次有实验/进展，就更新本文件（看文末"更新日志"）。

**一句话项目**：把真实果蝇大脑的连接组（MaleCNS v1.0，16.6 万神经元）接到 Doom 游戏上，让它"玩"游戏——最终目标是"真的玩得好"（靠视觉信息获得行为优势）。

**代码位置**：`D:\fly\doomfly`（上游原版：`D:\fly\reference\doomfly-upstream`）
**实验数据**：`D:\fly\doomfly\outputs\doom\`
**参考文献**：`D:\fly\reference\papers\`（5 篇 PDF）

---

## 一、三层连接状态（看懂一切的核心视角）

把"果蝇脑玩游戏"拆成三段，每段一个状态：

| 层 | 是什么 | 原版状态 | 我们修复后 | 目标 |
|---|---|---|---|---|
| **输入** | 游戏画面 → 果蝇眼睛 | ⚠️ 半通：只采 4146 点"亮度"，未光学校准；感光对大脑第一站是**抑制**的 | ✅ 通了：视觉内容确实调制大脑（放电量 2.2 倍于黑屏） | 看得清：升级为对比度/运动检测 |
| **大脑** | 真实连接组（16.6 万神经元/2558 万突触） | ✅ 数据是真的、完整加载 | ✅ 未动（数据逐字节一致） | 学得会：权重可塑性 + 奖励信号 |
| **输出** | 大脑 → 游戏按钮 | ❌ **没接通**：真实运动神经元全 0 放电，游戏靠 4 个"会放电的神经元"工程硬接（BCI 旁路） | ✅ 接通：真实运动神经元 MN9 从 0 放电 → 稳定放电并驱动开火+移动 | 动得对（已有）+ 行为优势（需训练） |

**核心结论**：连接组数据没错（是真的），错的是"大脑→游戏动作"这一层**原版从来就没接通过**——不是我们修坏的，是作者自己的记录 + 我们的复测一致。

---

## 二、原版是什么样（作者交付的东西）

作者 Alex Wormuth 基于 MaleCNS v1.0 连接组搭了"果蝇玩 Doom"：
- 输入：游戏画面 → 4146 个感光点（亮度采样）→ 注入果蝇脑
- 大脑：LIF 仿真，信号在 16.6 万神经元连接组里传播（固定权重）
- 输出：读脑内 14 个神经元 → 转成转向/前进/开火

**作者自己记录的事实（证据：`outputs\doom\validation.json`，作者亲笔）**：
- `biologically_validated: false`（未经生物学验证）
- **10 个真实运动/口器读出（MN9×2、DNa02×2、MDN×4、DNp09×2）在所有条件下 `spikes: 0, rate_hz: 0.0`** —— 一个脉冲都没有
- 作者原话（interpretation 字段）："运动/口器读出保持沉默。这不是成功游戏控制、真实视觉或学习的证据"
- 游戏动作实际由工程 BCI 驱动（DNp20 R−L→转向、DNpe017→前进+开火）
- 作者 README 原话："These are engineered controller assignments, not established natural motor functions"（工程控制器分配，非自然运动功能）；"Status: live experimental training, not demonstrated learned survival"（未证明学会生存）；v1–v6 学习实验全部失败

**结论**：原版 = 果蝇脑当"游戏手柄"用，不是"果蝇在玩"。会动，但不动脑。

---

## 三、我们做了什么（时间线工作记录）

### 阶段 0：环境与界面（已完成）
- 确认 Doom 完整：freedoom1（36 关）+ freedoom2（32 关）共 68 关都在磁盘；果蝇玩的是自建小竞技场 combat_survival.wad（768×768，最多 8 敌）
- 写人工游玩脚本 `human_play.py`（你亲自玩用，pygame 界面）
- 启动项目自带观战界面 doom-ui（localhost:3000）+ 仿真服务器（localhost:8766）

### 阶段 1：连接审计（已完成）—— "原版到底连没连上"
- 数据层 ✅：166,700 神经元 / 25,582,938 边 / 哈希校验一致
- 输入层 ⚠️：`optically_calibrated: false`（视网膜投影是推断的、未校准）；感光→板层是**组胺抑制**（真实生物学）
- 输出层 ❌：10 个真实运动读出 **0 脉冲**；游戏动作由工程 BCI 驱动
- 补证：黑屏 ≡ 断开视网膜（行为几乎相同）→ 行为主要靠 12mV 恒定偏置驱动，不靠视觉内容

### 阶段 2：连接修复（已完成）—— "先修复，直到确实连接上"
（用户选定 A 路线后的实验，全部可复跑：`outputs\doom\calibrate-*.py`）

动力学诊断：MN9/DNa02 直接上游 0 放电；BCI 神经元膜电位差 0.2~3mV 跨阈值。
8 轮参数扫描 → 找到：**权重×1.5 + 阈值 −48mV + 偏置 6mV** 时视觉调制最强。

| 轮 | 做了什么 | 结果 |
|---|---|---|
| ① 诊断 | 查清运动神经元为何 0 放电 | 上游全沉默；感光抑制板层；12mV 偏置掩盖视觉 |
| ②-⑤ 扫描 | 权重放大/去偏置/感光增益等 | 找到视觉调制组合（见上） |
| ⑥ 闭环对比 | 修复版 vs 原版 BCI vs 随机连接组（3 seed） | 真实运动神经元 MN9 驱动开火+移动 ✅ |
| ⑦ 最终闭环 | 视觉 vs 黑屏 vs 基线 | **黑屏表现≈视觉 → 固定权重下行为无法提升** |

闭环成绩（combat_survival，3 seed 均值）：

| 配置 | 击杀 | 存活 tics | 说明 |
|---|---|---|---|
| 原版 baseline（BCI） | 1.0 | 192 | 作者原版 |
| fix_v1 | 0.33 | 171 | 前进通道退化 |
| **fix_v2（MN9 驱动）** | **1.33** | **207** | 行为最佳 |
| fix_v2 黑屏 | 1.33 | 194 | 视觉无贡献 |
| fix_v4（视觉调制 122%） | 0.67 | 197 | 移动多反而送死 |
| 随机连接组 | 0.67 | 194 | 与修复版不可区分 |

### 阶段 3 结论（当前）
- ✅ **物理连接达成**：真实运动神经元 MN9 从 0 脉冲 → 激活并驱动行为（原版从未做到）
- ✅ **神经视觉调制达成**：视觉活性为黑屏 2.2 倍（经真实通路：感光→板层→运动链）
- ❌ **行为层面视觉正贡献未达成**：黑屏闭环等效；随机连接组与修复版不可区分
- **判断**：固定权重 LIF + 亮度饱和视网膜 + 恒定偏置，"会动不会玩"是物理极限 → **行为提升必须靠训练**（与作者 v1–v6 全失败的动力学根源相同，本项目独立复现）

---

## 四、关键结论汇总

1. 连接组数据是真的、没被修改（哈希一致）
2. 原版"大脑→游戏动作"从没接通（运动读出 0 放电，BCI 旁路），**这是作者自己承认并记录的事实**
3. 我们把真实运动通路接通了（MN9 驱动行为），但**固定权重下行为提升不了**
4. 参考全世界的做法：**所有让果蝇脑"会玩"的项目都是经过训练的** → 训练是必经之路

---

## 五、参考项目与方法（别人怎么让果蝇脑玩游戏）

| 项目 | 方法 | 链接 |
|---|---|---|
| **alpha-fly** | AlphaZero 风格**自对弈**训练；明确声明"没加/删连接、没翻转符号"，只在**权重层面**学习，连接组内部自己学会 | https://turlockmike.github.io/alpha-fly/ |
| **doomfly-rl（nonatofabio）** | 先用 **PPO** 训练 5 个普通 Doom 智能体当"老师"，再**蒸馏**到果蝇脑 | https://dev.to/nonatofabio_28/a-fruit-fly-brain-plays-doom-4an |
| **arXiv 2602.17997** | 果蝇脑全身运动控制：**模仿学习**（专家轨迹）+ **PPO 强化学习**两阶段训练 | https://arxiv.org/html/2602.17997 |
| **Biological Processing Units** | 昆虫连接组做骨架（GNN-BPU），训练可学习权重下棋 | https://arxiv.org/html/2507.10951v1 |
| **Digital Sphinx（Brunton）** | 线虫连接组+果蝇身体，深度 RL 训练走路；**重要警示：光有连接组不训练，不会产生行为** | https://www.thetransmitter.org/systems-neuroscience/digital-sphinx-raises-questions-about-connectome-models/ |
| **Beat Saber（domi）** | 用果蝇脑训练玩 Beat Saber（节奏游戏） | （见 TweakTown 报道） |
| 作者 Alex Wormuth（我们手头这个） | 固定权重 + BCI 旁路；奖励信号（受伤→PPL101 多巴胺）存在但 v1–v6 训练全失败 | 本机项目 |

**共同点**：全部是**训练连接组内的突触权重**（不是改连接结构），用强化学习（PPO/AlphaZero/模仿学习）等标准方法。

**已下载到 `D:\fly\reference\papers\` 的文献**（5 篇，PDF 已校验）：
- Lappalainen 2024（果蝇视觉系统连接组约束模型——**升级视觉通路的关键参考**）
- Shiu 2024（果蝇脑 LIF 模型）
- Dorkenwald 2024（雌性脑接线图）
- Kempka 2016（ViZDoom 引擎）
- eLife 视觉通路到蘑菇体论文

### 2026-09-29 补充：最前沿训练方法调研（学术）

| 方法/论文 | 做了什么 | 适配判断 |
|---|---|---|
| **Emergent Individuality in Whole-Brain Connectome Simulations of Drosophila**（Zenodo 19152238） | 首个完整果蝇脑 LIF 仿真 + **Hebbian 突触可塑性**（η=1e-4）+ NeuroMechFly 身体闭环；两只同初始脑独立演化出不同行为 | ✅ **最贴我们的路线**：直接在完整连接组上加可塑性 |
| **Adaptive Surrogate Gradients for Sequential RL in SNNs**（arXiv 2510.24461，2025-10） | SNN 强化学习新方法：代理梯度斜率自适应 + 老师策略引导（privileged guiding policy），无人机控制任务性能 ×2.1 | ✅ 训练读出/关键通路的梯度法 |
| **RL with Low-Complexity Liquid State Machines**（arXiv 1906.01695） | 随机脉冲网络（不可训练）+ **只训练读出层** + Q-learning，打 Atari | ✅ 最省算力路线：固定连接组只训读出 |
| **A dynamic connectome supports … reward-based learning**（Kappel & Maass，arXiv 1704.04238） | 理论：自发突触动力学 + 多巴胺奖励 → 网络自配置完成任务 | ✅ 训练=权重采样/可塑性 + 奖励的理论根基 |
| **Reward-modulated STDP / TD actor-critic**（Frémaux & Gerstner） | 多巴胺调节 STDP，连续时间 actor-critic 强化学习 | ✅ 果蝇 PPL101 多巴胺机制的理论对应 |
| **Controlled Forgetting Networks**（Frontiers 2020） | 多巴胺调制 STDP，持续学习防灾难性遗忘 | ✅ 训练中保持已有行为 |
| **Biological Processing Units**（arXiv 2507.10951） | 昆虫连接组做骨架（GNN），训练可学习参数下棋 | ✅ 混合路线：结构固定+外部可训练参数 |

**三条可选训练路线（按推荐序）**：
- **A. 权重可塑性（最贴果蝇生物学）**：连接组 + 奖励调节 STDP/多巴胺可塑性（项目已有 PPL101 奖励通路雏形）→ 参考 alpha-fly、Emergent Individuality、Kappel 理论
- **B. 读出层训练（最省算力）**：固定连接组 + 只训练读出层（LSM/BPU 路线）→ 先验证"能学"
- **C. 蒸馏/代理梯度（上限最高）**：PPO 训老师 → 蒸馏进果蝇脑（doomfly-rl 已验证）；或用 surrogate gradient 直接梯度训练 SNN

### 2026-09-29 补充二：权威训练方法（期刊/顶会级，含方法介绍）

**梯度类（性能上限最高，适合训练读出/关键通路）**
| 方法 | 权威出处 | 是什么 | 适配点 |
|---|---|---|---|
| **e-prop** | Bellec et al. 2020, *Nature Communications* | 递归脉冲网络在线学习（eligibility propagation），生物可行梯度下降，性能接近 BPTT，支持深度强化学习 | 让连接组里的关键通路能在线学"打 Doom" |
| **Surrogate Gradient** | Neftci, Mostafa & Zenke 2019, *IEEE Signal Processing Magazine* | 用可导"代理"替代脉冲不可导点，使梯度下降可用于 SNN 的通用框架 | 训练读出层/视觉通路的梯度工具 |
| **EventProp** | Wunderlich & Pehle 2021, *Scientific Reports* | 伴随法算脉冲网络精确梯度（不近似） | 高精度梯度训练，代价高 |
| **Hybrid ANN→SNN + STDB** | Rathi et al. ICLR 2020 | 先把普通神经网络转成 SNN 再微调 | 先用 PPO 老师蒸馏再微调 = doomfly-rl 路线 |
| **TET / DSR** | Deng et al. ICLR 2022；Meng et al. CVPR 2022 | 深度 SNN 训练 SOTA（时域/脉冲表示梯度） | 如果升级成深层视觉特征 |

**生物可塑类（最贴果蝇真实学习机制）**
| 方法 | 权威出处 | 是什么 | 适配点 |
|---|---|---|---|
| **Reward-modulated STDP** | Frémaux & Gerstner 2016, *PLoS Comput Biol* | 多巴胺奖励信号调节突触可塑性（STDP），连续时间 actor-critic | 果蝇 PPL101 多巴胺 → 权重更新的理论实现 |
| **Biologically-plausible Reward Propagation** | T. Zhang et al.（中科院）2020 | 顶向下传播"目标标签"而非误差，类似奖赏引导学习 | 比反向传播更贴脑，可用在读出层 |
| **Symmetric STDP / BP-STDP** | Hao & Xu（中科院）2018；Tavanaei & Maida 2017 | 用 STDP 近似反向传播的监督学习 | 保持"果蝇式"学习的同时有梯度精度 |
| **Dynamic connectome reward learning** | Kappel, Legenstein & Maass 2017（arXiv/理论） | 自发突触动力学+多巴胺 → 网络自配置 | 训练=权重采样+奖励的理论根基 |

**读出训练/随机骨架类（最省算力，先验证"能学"）**
| 方法 | 权威出处 | 是什么 | 适配点 |
|---|---|---|---|
| **LSM + 读出训练** | Ponghiran, Srinivasan & Roy 2019（arXiv） | 随机脉冲网络只训练读出层打 Atari | 固定连接组只训 4 个读出神经元 = 最小可行实验 |
| **BPU / 连接组骨架** | arXiv 2507.10951 | 昆虫连接组做骨架（GNN），训练外部参数下棋 | 连接组不动、外面加可训练层 |

**身体闭环/端到端 RL 类（类似任务的最权威参考）**
| 方法 | 权威出处 | 是什么 | 适配点 |
|---|---|---|---|
| **NeuroMechFly** | Lobato-Ríos et al. 2022, *Nature Methods* | 成年果蝇神经机械仿真（腿/翅/身体动力学） | "脑+身体"闭环的黄金标准框架 |
| **NeuroMechFly v2** | Wang-Chen et al.（EPFL）2023-2024 | 加入视觉+嗅觉+本体反馈，RL 训练多模态导航；含连接组约束视觉网络 | **我们任务的最接近参考：连接组视觉 + RL 导航** |
| **Whole-body fly + DRL** | Vaxenburg et al.（Janelia/DeepMind/Columbia）2024 | MuJoCo 全身果蝇模型，端到端 RL 训练飞行/走路/视觉引导 | 证明 RL 可训果蝇视觉控制（真实身体） |



---

## 六·五、大脑快照（训练前基准，用户要求保存）

- **位置**：`D:\fly\brain-snapshot\v0-original\malecns_v1\`（1,369.7 MB，11 文件，MaleCNS v1.0 原始连接组完整复制，robocopy 0 失败）
- **说明**：`D:\fly\brain-snapshot\SNAPSHOT.md`（文件清单、仿真参数、用途：训练改动后可恢复/对比）
- **意义**：用户声明后续训练可能修改大脑结构 → 先留原始基准 v0，训练后另存 v1/v2… 保持 v0 不变

## 六·六、果蝇眼中的画面（可视化）

- **图片**：`D:\fly\reference\fly-vision\fly_eye_view.png`（三栏：原游戏画面 / 果蝇感光点阵 3335 点（L 1148/R 2187）/ 大脑实际注入电流）
- **脚本**：`D:\fly\doomfly\render_fly_eye.py`（可复跑）
- **实测统计（combat_survival 一帧）**：亮度均值 0.077、亮点（>0.5）0.0%、**纯黑 42.1%** → **直观印证"视觉帮不上忙"根因：场景暗 + 亮度饱和采样 = 果蝇几乎什么都看不清**

## 六·七、"看着不暗"的验证（2026-09-29，回答用户质疑）

- **图片**：`D:\fly\reference\fly-vision\brightness_explained.png`；**数据**：`brightness-stats.json`
- **实测数字**：原画面 sRGB 均值 76/255（中位数 92）→ **人眼看着是"中等灰"，不暗（用户直觉正确）**；但**物理线性亮度均值仅 0.096**（sRGB 128 对应线性亮度只有 0.216，因为 sRGB 是 γ2.4 感知编码）；果蝇采样线性亮度 0.077（≈ sRGB 78 的中等灰），42% 采样点近黑（落在阴影/暗纹理区）
- **结论**：果蝇"能看到"（不是全黑），但**信息区分度极低**——采样值挤在 0~0.1 窄带，感光响应 `30L/(0.02+L)` 输出接近饱和前段，画面细节差异被压扁；且**模型没有真实果蝇的暗适应机制**（R1-R6 在夜间会增益上调，夜间也能飞）→ 视觉升级方向（参考 Lappalainen 2024）
- **脚本**：`D:\fly\doomfly\verify_brightness.py`



---

## 六、常见问题解答（Q&A）

**Q1：需要学习才能正常玩游戏吗？**
A：是。所有"会玩"的版本都经过训练（见第五节）。未训练的固定权重连接组 = "会动不会玩"，这是必然的。

**Q2：原版能放电吗？**
A：**运动神经元不能**（10 个读出 0 脉冲，作者自己记录）；**BCI 用的 4 个神经元能放电**（DNp20/DNpe017，被当游戏手柄用）。两者要分清。

**Q3：是否修改了果蝇大脑？**
A：**连接组数据没动**（16.6 万神经元/2558 万突触逐字节一致，git diff 行为零差异）。改的是**仿真参数**：权重缩放、放电阈值、板层偏置、解码方式（哪个神经元接哪个按钮）。类比：没改电路图，只调了电压和接线端子。

**Q4：一定要训练吗？人类为什么不用？**
A：仿真果蝇脑是"**死脑**"：固定权重快照、无学习机制、无感觉-运动闭环。人脑是"**活脑**"：进化+个体发育已把视觉/运动/奖励系统打磨好，而且**人玩游戏时脑就在实时训练**（多巴胺奖励、运动学习）。人的"不用训练"是错觉——训练从出生前就开始了。

**Q5：训练之后会改变大脑结构吗？**
A：改**突触权重**（连接强度），不改**连接组拓扑**（谁连谁）。alpha-fly 原话："没加或删连接，没翻转符号"。权重可塑性 = 学习。极端长期才涉及结构可塑（突触生成/修剪），主流方法（含所有参考项目）都是权重级。

---

## 七、下一步（训练方向，等待确认）

基于阶段 3 结论，让果蝇"玩好"的路径：
1. **视觉通路升级**：亮度饱和 → 对比度/运动检测（参考 Lappalainen 2024 PDF）
2. **权重可塑性 + 奖励信号**：参考 Shiu 脑模型、alpha-fly（自对弈）、doomfly-rl（PPO 蒸馏）
3. **评测**：闭环 A/B（视觉 vs 黑屏 vs 随机）——训练有效的判据 = 正常视觉显著优于黑屏，且优于随机连接组

---

## 八、实验脚本与数据索引（`D:\fly\doomfly\outputs\doom\`）

- 审计：`audit_calibrate_diag.py`、`audit_decoder_ab.py`、`audit_readout_inputs.py`、`audit_dna02_upstream.py`
- 扫描：`calibrate_scan.py`~`calibrate_scan8.py` → `calibrate-scan{1..8}.json`
- 闭环：`calibrate_closed_loop.py` → `calibrate-closed-loop.jsonl`
- 探测：`probe_scale1.5.py`/`probe_th48.py`/`probe_v3.py` → `readout-probe-*.json`/`probe-v3.json`
- 作者原版验证记录：`validation.json`（含 10 读出全 0 证据）

---


## 七、doomfly-rl 蒸馏训练（2026-09-29 完成首轮闭环）

### 方法（对应用户指派的 doomfly-rl 思路，本机缩小版）
1. **PPO 老师**：普通 Doom 智能体（MLP 策略，观察=24×32 降采样灰度 768 维，12 离散动作=转向3×前进2×开火2，奖励=击杀×10−扣血×1），30k 步 ≈ 17 分钟（CPU）。学习曲线：2.0 → 中段峰值 3.7，后期 1~3 波动（3 局评估方差大；小训练量上限）。
2. **蒸馏**：老师玩 8 局（1680 帧）→ 每帧跑**真实果蝇连接组**（scale=1.5, bias=6, thresh=−48）→ 取 14 个读出神经元放电计数作特征 → 训练读出层 Student(14→128→64→12) 交叉熵 60 epochs（acc≈45.7%）。
3. **闭环评估**：3 seeds × 210 tics，combat_survival，对比三配置。

### 结果（evaluation.json / three_config_compare.png）
| 配置 | 平均击杀 | 平均存活 tics/210 | 局末健康 |
|---|---|---|---|
| baseline（作者 BCI，scale1/bias12） | 1.00 | 192 | −5 |
| random（随机读出层+修复脑） | 0.00 | 205 | 8 |
| **distilled（蒸馏读出层+修复脑）** | **1.00** | **210（满局）** | **80** |

### 结论（诚实版）
- ✅ **流水线跑通**：从老师到果蝇脑特征再到读出层闭环，全链路可复现。
- ✅ **明确进步**：蒸馏果蝇脑"学会活着"——满局存活 + 健康 80 vs 原版 BCI −5；实机对比 GIF 见 outputs\doom\distill-v1\（distilled_seed41029.gif vs baseline_seed41029.gif）。
- ⚠️ **击杀未超越**：kills 1.0 与原版持平。原因：老师本身只到中等水平（小训练量）、14 维读出特征信息少（MDN/DNp09 几乎不放电，实际只有 MN9/DNp20/DNpe017 有信号）、12 动作离散粗。
- 下一步候选：老师加大训练量+动作细化、特征加历史窗口（5 帧→70 维）、蒸馏"动作分布"而非 argmax、按 doomfly-rl 二阶段继续。


### 首轮结果的复盘诊断（2026-09-29 追加，回答"真有提升吗/连接有问题吗"）
用户质疑合理：击杀层面（1.0 vs 1.0）无实质提升。诊断结论（diagnose.json，630 未见帧）：
- **物理连接通但通道窄**：14 读出中仅 5 个实机放电（DNp20_L/R ~93%、MN9_L/R ~56-66%、DNa02L 50%），其余 9 个 <1.3% 近乎哑火。
- **教学连接弱**：学生 argmax 对上老师动作仅 34%（老师自身确定性仅 44%）。
- **根因 = 老师退化**：PPO 老师 30k 步后 argmax 策略 630 帧全为单一动作（右转+前进+不开火），即"乱转活着"，从未学会战斗 → 蒸馏无米之炊。
- **方法论缺陷**：场景 episode_timeout=0（无时间上限），此前"210 tics 满局"实为 3.5s 截断，"存活 210 vs 192"差距被高估；已启动 600 tics（10s）长时重评（longeval_t600.json）。
- 规则老师尝试：运动检测 v1 0 杀；站桩扫射 v2 0.67 杀（仍被 8 敌冲脸围殴），证明该场景对"站着不动"策略不友好。
- 蒸馏 v2 已就绪（distill_v2.py）：修一帧特征-标签错位 + 5 帧历史窗口（70 维）+ 600 tics 评估。
### 产物
- 脚本：teacher_ppo.py（老师）、distill_fly.py（蒸馏）、plot_results.py、record_gif.py
- 数据：outputs\doom\teacher-v1\teacher.pt+teacher-log.json；outputs\doom\distill-v1\distill-data.npz+student.pt+evaluation.json+*.gif+*.png

---

## 更新日志

- 2026-09-29（三）：doomfly-rl 蒸馏训练首轮完成（PPO 老师 30k 步 → 果蝇脑 14 读出特征 → 读出层闭环）：distilled 满局存活+健康 80 vs baseline −5，击杀持平 1.0；实机 GIF+对比图已产出

- 2026-09-29（二）：① 权威训练方法调研完成（Nature Communications e-prop / IEEE SPM surrogate gradient / Nature Methods NeuroMechFly / ICLR-CVPR 深度 SNN / PLoS Comput Biol reward-STDP 等，含方法介绍）；② 验证"画面看着不暗 vs 果蝇看不清"（gamma 编码+采样落点，数据在 brightness-stats.json）
- 2026-09-29：① 前沿训练方法调研完成（7 篇/项目，3 条训练路线推荐）；② 保存大脑快照 v0-original（D:\fly\brain-snapshot，1.37GB）；③ 果蝇视角可视化完成（D:\fly\reference\fly-vision，实测纯黑 42.1% 印证视觉无贡献根因）；④ 修复 matplotlib 安装导致的 numpy 升级环境问题（已降回 1.26.4，scipy 兼容恢复）
- 2026-09-23：连接修复实验完成（8 轮扫描 + 3 轮闭环），结论写入；创建本进度档案（2026-09-29 会话中归档整理）
- （以后每次有进展在这里追加一行）
