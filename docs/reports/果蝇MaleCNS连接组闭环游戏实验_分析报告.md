# 果蝇 MaleCNS 全脑连接组接入 ViZDoom 闭环实验：分析与诊断报告

> 初稿为截至 2026-10-18 的只读审查；2026-10-19 已追加全链路复核、独立新训练包实现及短程工程验证，详见第 10 节。
> 第 4 节数字仍是历史仓库报告值，不是新包复现结果。已执行的单测、短全脑仿真与极短优化调用链验收，不等于正式训练、游戏成功或生物有效性验证。
> 分析对象：把果蝇全脑连接组（MaleCNS）神经仿真接入 ViZDoom/DOOM 做闭环控制的实验工程。

---

## 0. 执行摘要

**核心判断：当前不能把"闭环成绩为 0"归因于"果蝇脑不行"，因为实验管线存在一批已定位、足以使结果失效的实现 bug；同时，上游项目本身也从未证明连接组具备游戏学习能力。**

三点最重要结论：

1. **当前实际主线是"冻结完整连接组、只训练外部读出层"，而不是训练脑内权重。** MaleCNS 图（166,700 神经元、25,582,938 有向边）权重在运行时仅整体 ×1.5，不回写；训练的是独立 PPO 老师、外部 MLP/线性分类器和离线 FQI Q 网络。三份 `graph.npz` 的 SHA-256 完全一致，证明脑图未被实验改写。

2. **走廊 PPO 老师达到 100% 出口率，但历史版本额外使用了学生看不到的引擎内部 ANGLE 朝向（特权输入）；所有学生闭环（行为克隆三代 + 无教师 FQI）出口率全部为 0。** 老师成功 ≠ 老师动作可由学生的视觉/脑活动推断。

3. **历史结果受到确定实现错误和实验设计限制的共同混杂，不能唯一归因**：归一化统计、在线窗口、特征选择/验证泄漏、跨 episode 脑状态等历史问题见第 5 节 A–L；FQI 零出口样本属于覆盖限制而不是实现 bug。用户已修复部分主路径，当前残留及新路线见第 10 节。文献侧（doomfly-rl、alpha-fly 等）显示"打乱图/无连接组"并不更差、甚至更快，因此**必须做四臂消融才能真正归因**，不能下"连接组有用/无用"的结论。

**当前选定方向**：独立 `experiment/code/flyrl`，冻结完整连接组与内部权重，预注册无标签神经群体，训练线性/极小读出做在线 DoubleDQN（PPO 为同接口备选），先走廊再战斗。选择理由是 CPU 全脑采样昂贵、离散动作少、replay 可重用经验，不是已证明 DoubleDQN 普遍优于 PPO。teacher-student 保留为历史诊断或另行设计的公平基线，不是数学上限。单臂单次 2 小时作 pilot、8 小时作较长批次；不保证预算内学会，更不保证四臂多种子全部完成。

---

## 1. 项目身份与上游来源

### 1.1 两个嵌套仓库（不是单一无版本目录）

`D:\fly` 本身不是 Git 仓库，但包含两个独立仓库：

| 仓库 | 路径 | 上游 origin | 当前 HEAD |
|---|---|---|---|
| 上游 DOOMFLY | `D:\fly\doomfly` | https://github.com/nftechie/doomfly.git | `71ecf53d`（作者 nftechie = Alex Wormuth），提交信息 "Fix spectator pistol orientation and barrel-mounted flash" |
| 本地实验 | `D:\fly\experiment` | https://github.com/zyj669915/doom.git | `e87b81c2`，提交信息 "Add SETUP.md: full from-zero training guide" |

两仓库均有未提交改动。上游工作区已改：`doom/game.py`、`outputs/doom/malecns_v1/manifest.json`；未跟踪 `corridor_avoid.cfg/.json/.wad` 走廊场景文件。

### 1.2 上游 DOOMFLY 到底做了什么

上游 README 声明的链路：ViZDoom RGB 帧 → 3,335 个 R1–R6 亮度输入（v6 另加 811 个 R8 颜色代理）→ 在 MaleCNS 图上跑近似神经动力学 → 固定工程接口解码动作（DNp20 左右差转向、DNpe017 rate 前进 + spike 开火）→ 受伤后向 PPL101 多巴胺细胞注入人工厌恶刺激 → 在 4,184 条 KC→MBON11 边上尝试实验性可塑性。

**上游自己明确划定边界**（这一点很重要，不能过度解读）：
- 动作映射是 engineered controller assignments，不是已证实的天然运动功能；
- 不是字面意义上的"活果蝇脑重建"；
- v6 候选**没有通过**视觉、条件化和生存验证；
- 原始生物运动读出（DNa02/MDN/MN9）在基线验证中保持沉默，`biologically_validated: false`。

因此对项目最准确的描述是：**一个保留 MaleCNS 图结构、用近似 LIF 动力学、靠人工神经元→游戏按钮映射控制 ViZDoom 的闭环实验；不是已验证的果蝇视觉/自然动作/成功学习系统。**

### 1.3 对应的基础论文与数据

| 名称 | 出处 | 在本项目中的角色 |
|---|---|---|
| MaleCNS v1.0 连接组 | Berg et al., *Cell* 189(18):5504–5526.e15, 2026. DOI 10.1016/j.cell.2026.08.015 | 结构与注释来源（166,700 神经元、11,710 类型）。提供结构，不提供 LIF 参数、视觉几何、动作语义或学习规则 |
| 全脑 LIF 模型 | Shiu et al., *Nature* 634:210–219, 2024. DOI 10.1038/s41586-024-07763-9；代码 github.com/philshiu/Drosophila_brain_model | LIF 框架与部分参数的科学参考；原论文验证的是进食/触角梳理回路，不是视觉/Doom/RL，不能把其验证直接转移 |
| 视觉系统连接组约束模型 | Lappalainen et al. (FlyVis), *Nature* 634:1132–1140, 2024. DOI 10.1038/s41586-024-07939-3；代码 github.com/TuragaLab/flyvis | 视觉前端升级与生理验证的首要参考；其消融证明"连接约束 + 任务优化"两者都重要，仅有连接图不足以自然产生正确的 T4/T5 方向选择性 |
| 多巴胺记忆动力学 | Huang et al., *Nature* 634:1141–1149, 2024. DOI 10.1038/s41586-024-07819-w | v6 记忆规则与部分目标放电率的来源；属适配，不是完整复现，更不代表原作者认可 |

---

## 2. 结构与完整闭环调用链

### 2.1 顶层规模

- `D:\fly\doomfly` ≈ 7.15 GiB（含两个虚拟环境 `.venv-gpu` 4.49 GiB、`.venv-neural` 1.03 GiB）；
- `D:\fly\experiment` ≈ 7.65 GiB，其中 `code/outputs` 神经活动中间数组约 7.5 GiB；
- `D:\fly\backup` ≈ 1.57 GiB，MaleCNS 原始脑图哈希固定基线。
- 体积主要来自虚拟环境、多份脑图副本和全脑 spike 数组，不是源代码。

### 2.2 规范闭环链（上游基线）

```
ViZDoom 游戏帧 (640×480 RGB)
  → Game.pixels()
  → retinal_samples(frame, uv)        # 3,335 个 UV 双线性采样，sRGB→线性亮度
  → NativeBrain.step() / C++ kernel  # 全脑 LIF 推进
  → NeuralControls.decode(counts)   # bci: DNp20 左右差转向 / DNpe017 前进+开火
  → Game.act()                       # 三个按钮：转向、前进、开火
  → Game.observation()               # health/kills/ammo/hitcount/score 仅用于观察审计，严禁进入解码
```

关键边界：动作解码只接收 spike count 和时间，**不接收坐标、敌人、生命值或奖励**。

### 2.3 本地实验的三条训练路线（须与上游 v6 生物可塑性区分）

1. **PPO 老师（纯 ANN，不经过果蝇脑）**：`teacher_ppo.py`（战斗）、`corridor_teacher.py`（走廊）。输入 24×32 灰度+帧差，走廊老师额外拼接引擎内部 ANGLE 的 cos/sin（共 1538 维）。
2. **蒸馏/行为克隆**：`distill_v2.py`、`corridor_student*.py`。同帧双路——老师出动作标签，视网膜→全脑→选 readout 神经元→窗口→外部 MLP 模仿老师动作，再闭环。
3. **无教师离线 FQI**：`corridor_rl_q.py`。随机探索采集 transition → 无监督 top-K 神经元 → 窗口+差分 → Fitted Q Iteration → greedy Q 动作闭环。

---

## 3. 连接组、仿真参数与输入输出通路

### 3.1 连接组规模（确定性事实）

| 项目 | 数量 |
|---|---|
| 神经元 | 166,700 |
| 有向边 | 25,582,938 |
| 突触接触 | 124,177,617 |
| 成功映射到屏幕的 R1–R6 | 3,335（候选 3,377，未映射 42） |
| 递质符号不确定神经元 | 3,718 |

三份 `graph.npz` 均为 250,157,214 字节，SHA-256 =
`f4d41f011e97e510761d011634891b1c082f61f91463eb586ee9cf8c6371b1d1`，
位于 backup、brain-v0-clean、doomfly/outputs 三处，逐字节一致。

### 3.2 LIF 动力学参数

上游默认：dt=0.1ms、静息 −52mV、阈值 −45mV、膜时间常数 20ms、突触时间常数 5ms、延迟 1.8ms、不应期 2.2ms、板层偏置 12、感光驱动 `30L/(0.02+L)`。

本地实验校准值：运行时权重 ×1.5（SCALE=1.5）、板层偏置 6、阈值 −48mV、每游戏帧推进 50ms 神经时间。注意：graph.npz 原始权重**没有被写回**，只是运行时缩放。

### 3.3 视觉前端

当前走廊前端 `v4_agc_motion`：
```
lum_t
  → motion = clip(lum_t + 1.2·(lum_t − lum_{t-1}), 0, 1)   # 人工帧差增强
  → AGC = 0.5 + 0.5·tanh((x − mean)/std/1.5)               # 每帧全局归一化
  → 感光驱动 30L/(0.02+L) → 全脑 LIF 50ms
```
这是明显的工程前端，不是校准后的果蝇视觉。

**输入诊断实测**：静止 top32 平均 spikes 2.35、标准差 2.51（静止响应本身不稳定，std>mean）；左右方向分类 0.625；looming 相关仅 0.291；自转污染比 0.971；**黑屏 200 帧仍产生 28,080,393 个 spikes**（恒定偏置带来大量与视觉无关的背景活动）。六种旧前端离线可分性都很低（≤0.15）。

### 3.4 读出层

- 上游固定解码：biological 模式（DNa02 转向 / DNp09-MDN 前进 / MN9 开火）与 bci 工程模式（DNp20 左右差转向 / DNpe017 前进+开火）。
- 当前走廊**不再**用上述 14 个神经元直接驱动按钮，而是：保存全部 166,700 神经元每帧 spike count → 按时间方差+动作 ANOVA 选 top32/top512 → 拼单帧或 5 帧窗口（可加差分）→ 外部线性/MLP/QNet 输出左/直/右三类转向 → 前进强制 20。
- 即：**当前"读出层"是外部人工网络，不是连接组中的自然运动输出。**

---

## 4. 历史实验结果（不代表 2026-10-19 新管线成绩）

### 4.1 走廊 PPO 老师：成功，但信息特权

- 观测 1538 维 = 24×32 灰度 + 帧差 + **ANGLE cos/sin**（来自 ViZDoom 内部状态，学生不可见）；
- 终验 5 seeds × 20 episodes：**100/100 出口**、平均 565.2 tics、wall rate 0.0007；
- 训练曲线：10,240 步出口率 0.2 → 30,720 步 0.9 → 40,960 步起稳定 1.0。

### 4.2 所有学生闭环：出口率 0

| 学生方案 | 出口率 | 单帧最大进度 |
|---|---|---|
| top32 单帧 | 0 | 0.039–0.168 |
| top512 五帧+差分 | 0 | 0.003–0.014 |
| epsilon 混合数据（一半随机动作） | 0 | 0.003–0.009（700 tics 耗尽，wall rate=0 是因为几乎没前进，不是表现好） |
| 无教师 FQI | 0 | 0.026–0.144 |

离线精度却不低：top512 五帧学生离线平衡准确率 0.531–0.565；epsilon 学生训练集拟合到 1.000。**离线精度改善没有转化为闭环能力。**

### 4.3 走廊消融（离线，受泄漏污染）

| 臂 | 线性平衡准确率 | MLP 平衡准确率 |
|---|---|---|
| 真实脑 top32 | 0.546 ± 0.126 | 0.489 ± 0.090 |
| PCA 像素 32 | 0.480 ± 0.028 | 0.524 ± 0.038 |
| 打乱脑 top32 | 0.342 ± 0.026 | 0.415 ± 0.046 |

表面看真实脑相对打乱脑线性 +0.204，但由于下文的选择泄漏、非独立拆分和不对称比较，**不能视为显著的连接组增益**。

### 4.4 FQI 数据：从未见过出口

- 4 seeds × 500 帧共 2,000 帧，**exit_frames = 0**，总奖励 −33.1；
- 模型未见 +10 出口奖励，仍可学习被覆盖状态中的局部奖励关系；但未覆盖状态/动作的 Q 值及长程奖励传播没有可靠保证，存在离线外推风险。零成功样本不是理论上绝对无法改善的证明。

### 4.5 输出体积

corridor-teacher-v6 35.31 MiB；corridor-rl-q 1276.99 MiB；corridor-student 1302.36 MiB；corridor-student-eps 1302.34 MiB；corridor-ablation 642.27 MiB。其中单个 `counts-real.npy`（2000×166700 float32）就约 1.33 GiB。

---

## 5. 问题诊断（核心章节）

### 5.1 已定位的 12 个确定实现问题（A–L）

> 下列是初稿定位的历史问题，不应当作当前每条仍原样存在的断言。2026-10-19 复核确认用户的共享原语和部分主调用路径已有修复，但全链路并未全部修好；见第 10.1 节。旧 checkpoint 和旧结果不会因修源码自动变成有效证据。

- **A. 归一化统计存错**：窗口学生先对特征做了 `zscore_local()`，再对**已经标准化后**的特征算 mu/sd（结果≈0/1），并把这组统计存进 checkpoint；闭环时却用它去归一化**原始** spike 窗口特征。结果：训练模型看到的是 z-score 特征，闭环看到的基本是原始 count。同样模式存在于 `corridor_student.py` / `corridor_student_eps.py`。这足以解释"离线 0.565、训练集 1.0、闭环几乎不动"。
- **B. 在线窗口索引错误**：闭环维护一个最多 WIN+1 长度的局部 hist，却用从 episode 开始的**绝对 tics** 去索引这个短列表；当 tics 超过历史长度，所有越界索引被夹到 `len(hist)-1`，运行后期五个窗口槽位大量重复当前帧，差分校验失效。在线窗口与离线 `build()` 不一致。同见于 `corridor_rl_q.py`。
- **C. topK 特征选择泄漏标签**：用**全部** 2,000 帧和全部动作标签（时间方差 + 动作 ANOVA）选 topK 神经元，**之后**才随机 80/20 拆分——验证集标签已经参与了特征选择。
- **D. 标准化/PCA 在全数据上拟合**：`z-score`、PCA 都对全数据（含验证集）拟合后才拆分，进一步污染验证。
- **E. 时间序列随机逐帧拆分**：相邻 5 帧窗口共享 4/5 数据，随机拆分会让高度重叠的窗口同时落入训练和验证，CV 不是独立轨迹泛化。
- **F. "5 个评测 seed"实为同批数据 5 次重洗**：`EVAL_SEEDS=[0..4]` 只控制同数据的随机拆分和网络初始化，没有产生新游戏轨迹；均值±标准差不能当作 5 次独立复现。
- **G. 消融比较不对称**：top32 只按真实脑活动+真实脑动作相关性筛选，相同索引直接套用到打乱脑，天然利好真实脑臂。
- **H. 指标口径混用**：balanced accuracy 实际是各"出现过类别"的平均 recall（三类随机基线应≈1/3），日志却把它与多数类比例 0.482 并列比较，二者不可比；场景声明 6 类但实际只有 3 类出现（老师从不选停止前进）。
- **I. 老师用了学生不可见的 ANGLE**：`corridor_teacher.py` 直接读 `GameVariable.ANGLE`，但场景清单又声称 observation boundary 为 pixels-only，配置与叙述自相矛盾。老师动作对学生输入可能根本不是可判定函数。
- **J. FQI 跨 seed 边界拼接虚假转移**：4 个采集文件直接 concatenate，各 seed 最后一帧未标 terminal，会产生"seed41027 最后一帧 → seed41028 第一帧"这种不存在的转移；5 帧窗口也跨采集段拼接。
- **K. episode 间脑状态未重置**：闭环函数在 episode 循环外只构造一次 Brain，新 episode 开始时膜电位、突触电导、不应期、延迟队列、低通亮度仍从上一局延续，每一局不独立。
- **L. 结果原地覆盖、无 provenance**：corridor-student 的 JSON 被基础学生和窗口学生共用；FQI 的 CPU/GPU 运行共用 `model.pt`/`train-log.jsonl`/`closed-loop-results.json`，时间戳显示 GPU 重跑覆盖了更早 CPU 结果；checkpoint 未保存源码哈希、WAD 哈希、输入数据哈希、命令行、包版本。

**另发现**：最新脚本 `input_response_demo.py` 把 `brain.step()` 返回的元组 `(counts, elapsed)` 直接当 counts 用，且未传 `bias=6 / threshold=-48`，即使修好返回值错误，结果也不代表注释声称的校准配置。

### 5.2 确定的实验设计限制

- FQI 数据 exit_frames=0（见 4.4）；
- 行为克隆数据几乎全是成功老师轨迹：左转前 964、直前 88、右转前 948，停止前进类全为 0——学生一旦偏离进入训练分布外，无停止/恢复动作可学；
- 样本/维度失衡：2,000 样本 vs 5,120 维输入，MLP 首层约 130 万参数，高容量低样本强时间相关，训练集 100% 毫不奇怪；
- 走廊是固定 L 形、固定起点朝向，引擎 seed 差异不产生有意义的视觉多样性，即使老师多种子成功也不等于泛化；
- 脑增益消融只有 2 条采集 seed、每条 250 帧、同一固定老师、5 次随机拆分而非独立轨迹，样本不足以谈统计显著。

### 5.3 生物/模型层面的已知限制（上游自己记录）

- 所有 T4/T5 和 Kenyon cells 在早期候选中沉默；
- 视觉条件化缺少 cue specificity；模拟多巴胺存在不受控内源活动；
- v6 刺激后出现持续高 KC/PPL 活性、恢复不稳定；
- survival pilot 的"学习"效果既不稳定也可能有害（held-out 死亡时间甚至短于黑屏/memory-erased 对照）。

**结论：在 5.1 的实现问题修复前，离线 0.565 不是可靠泛化成绩，闭环 0% 也不能干净地归因于"脑内信息不足"，真实脑 vs 打乱脑的 +0.204 同样不是可靠显著结论。**

---

## 6. 文献对照：连接组玩游戏到底有没有优势

| 项目 | 做法 | 关键结果/警示 |
|---|---|---|
| **doomfly-rl** (nonatofabio) | SB3 PPO 老师 → 蒸馏到连接组约束学生（卷积编码器+连接组递归+大读出） | MaleCNS / 度保持打乱连接 / **完全移除连接组**三者在五场景成绩都在噪声范围内；无连接组版**快 22 倍**；MAP01 GRPO 后移除连接组回报还略高。→ 没有可测的果蝇特异连接优势 |
| **shreyash-sharma/doomFly** | MaleCNS 全图冻结 + PPO 只训小读出 | **纯视觉路径失败后，直接把敌人方向注入 LC4/LPLC2、敌人存在注入 Tk-FruM**——结果不能作为"果蝇视觉理解 Doom"的证据 |
| **alpha-fly** (turlockmike) | MaleCNS 结构做 AlphaZero 风格国际象棋 | 度数/符号保持但**重连的图在更短预算内学得同样快**，未建立果蝇特异布线优势 |
| **fly-hero** (bsgelman) | 连接组子图（6,900 神经元）三轨节奏游戏，多巴胺可塑性 | 报告命中从 33%→99%，但每个音符是独立三选一、奖励即时、可塑性工程化，非长时程部分可观测控制，无独立复现 |
| **FlyGM** (arXiv 2602.17997) | 全脑连接组图策略 + DRL 控生物力学身体 | 证明连接组约束策略可通过 RL 做闭环运动控制，但不是 LIF 全脑，也不证明内部动力学对应真实果蝇 |
| **Digital Sphinx** (Brunton et al. 2026, DOI 10.64898/2026.03.20.713233) | 故意用**线虫连接组**接果蝇身体，RL 训练运动接口 | 仍产出逼真行走——**"行为逼真 ≠ 生物正确"**，是对所有连接组-身体仿真的警示 |
| **NeuroMechFly v2** (Wang-Chen et al., *Nature Methods* 2024, DOI 10.1038/s41592-024-02497-y) | 加视觉/嗅觉/本体反馈，RL 导航 | 真闭环必须纳入身体、接触、姿态和感官反馈，而不仅是"脑活动→一个按钮" |
| **Eon embodied fly** (2026) | FlyWire + Shiu LIF + FlyVis + NeuroMechFly v2 集成 | 团队自承：内部活动尚未通过头方向吸引子等生物基准、视觉对行为作用有限、只读取极少数下行神经元 |
| **LSM + Q 读出** (Ponghiran et al. 2019, arXiv 1906.01695) | 随机脉冲 reservoir 冻结，只读层 Q-learning 打 Atari | "固定 reservoir + 训练读出"路线的算法先例（但用 Atari RAM 非像素、非实测果蝇连接组） |

**方法学文献**：DAgger（Ross/Gordon/Bagnell 2011, PMLR 15:627）针对闭环分布漂移；Policy Distillation（Rusu 2015, arXiv 1511.06295）与 Hinton 蒸馏（arXiv 1503.02531）解决策略压缩；e-prop（Bellec et al., *Nat Commun* 2020, DOI 10.1038/s41467-020-17236-y）、EventProp（Wunderlich & Pehle 2021, DOI 10.1038/s41598-021-91786-z）、三因子规则（Frémaux & Gerstner 2016, DOI 10.3389/fncir.2015.00085）、代理梯度（Neftci et al. 2019, arXiv 1901.09948）。

---

## 7. 为什么当前不把 teacher-student 选为主训练路线

teacher-student 是合法的训练方法，并非原则上不能用于本项目；但当前信息不等价、覆盖与评价问题尚未建立充分控制，不能仅凭蒸馏结果作脑机制或连接组优势结论：

1. **闭环分布漂移**：学生一旦动作偏离，后续画面就进入老师数据未覆盖的分布，误差累积。离线动作准确率/交叉熵不能代替闭环评估（DAgger 正是为此而提出）。
2. **师生可见信息不等价**：本地老师用了学生看不到的 ANGLE，老师动作在学生观测下可能本就不可辨识——此时蒸馏失败是信息论上缺条件信息，不是优化失败。
3. **读出层可能才是真正的游戏策略**：大 MLP/CNN 读出可自行学会敌人识别与时序记忆，连接组退化为"昂贵的随机特征变换"。doomfly-rl 的打乱图/无连接组结果正是这一风险的实证。
4. **高分老师 ≠ 可蒸馏老师**：combat 高击杀老师可能场景特化、动作塌缩；"可蒸馏性"要看动作熵、状态覆盖和对学生观测的条件可预测性。
5. **蒸馏回答的是策略压缩，不是脑内学习**：它不回答连接组能否经奖赏自学、多巴胺可塑性是否有效、生物视觉是否提供任务表征。

**合理保留方式**：受限可解码性探针（固定脑活动下指定线性读出能预测多少老师动作，不是严格上限）、数据质量诊断、DAgger 闭环基线、软标签基线；且必须与无教师 RL、像素读出、打乱图、无脑对照并列报告。

---

## 8. 建议的复现与训练路径

### 8.1 总原则

把主目标改写为：**在完全闭环、无特权游戏状态、无教师动作标签的条件下，检验 MaleCNS 神经活动能否为一个受限且可解释的动作读出提供优于像素、打乱连接和无脑基线的学习增益。** 闭环游戏表现优先，但每个成绩都要能归因。

路线已拍板：**先走廊 corridor_avoid 证明闭环学习，再迁移 combat_survival**；单次正式训练预算 2–8 小时；GPU/显存实施时自动检测；工程准备做完，正式训练命令留给用户执行。

### 8.2 分阶段

- **阶段 0 忠实复现与改进分离**：冻结记录 MaleCNS 文件哈希、上游 LIF 方程、视觉映射、固定解码、v6 规则，先精确重现上游报告（含其负结果）；任何改动（新视觉几何、新读出、RL 规则、新奖赏）都另起新模型，不冒充"原论文复现"。
- **阶段 1 先复现视觉模块（开放环，不进战斗）**：参考 FlyVis/T4-T5 文献，做亮度阶跃/闪光响应、ON/OFF 极性、四方向运动、光流旋转扩张、潜伏期与时间常数，配黑屏/静止/相位打乱/空间打乱对照。通过后才把视觉活动用作游戏 RL 状态。
- **阶段 2 最小闭环游戏**：光流转向 → 走廊避障 → looming 闪避 → 简化射击 → 最后才多敌人战斗。所有动作只能来自渲染帧驱动的神经活动，禁止注入敌人位置/bearing/标签/深度/地图。
- **阶段 3 神经活动表征**：预注册视觉投射/中央复合体/下行/运动群体，50–200ms 多时间窗，每群体记 spike count、左右差、总活动、最近脉冲时间；投影只在训练集拟合；不要用"最能预测老师动作"的方式选神经元（选择偏差）。
- **阶段 4 动作读出**：低维动作空间（左/直/右；停/前进；后加开火），左右竞争群体差值控转向、前向群体控速，加迟滞与最短动作保持；从线性策略/极小 MLP 起步。
- **阶段 5 主训练 = 无教师闭环 RL**：冻结 MaleCNS 拓扑与内部权重、冻结已验证感觉编码，**只训小型可解释读出**，直接用闭环任务奖励做在线 DoubleDQN，PPO 为备选，不用老师标签，从走廊/loom 密集奖励起步。读出 RL 证明神经状态有价值后，再引入预注册的三因子/资格迹可塑性（限 MB compartment 或感觉-动作接口）。

**不建议一开始做**：对 2,558 万条边直接 BPTT/全图 EventProp、无约束训每条突触增益、大 CNN/MLP 读出、把敌人方位直接注入 LC/LPLC、用完整 Doom 战斗当首个学习任务、只报一次最好 checkpoint。

### 8.3 必须有的四臂核心对照

相同输入、相同训练预算、相同读出结构、相同随机种子下比较：
1. 真实 MaleCNS 图；
2. 运行时按突触前符号分层置换目标端点的重连图：保 CSR 出度、入度、每目标正负入度、源权重位置；不保证目标入强度、自环/重复边数或 simple graph；
3. retinal：相同 UV 人工视网膜前端后，固定空间池化，不经过连接组；
4. pixels：完整渲染像素的固定空间池化，不用可训练 CNN，也不是已移植 FlyVis。

四臂共同 50/200ms trace、246 维及同容量读出；同维不等于同信息或同计算量，必须同时报告各臂有效非恒定维度与速度。

若真实图不能稳定优于 2–4，不得声称连接组有功能优势。另加因果对照：黑屏、帧序打乱、空间打乱、yoked reward、时间打乱奖励、可塑性开/关、学习后权重 vs 初始权重、关键视觉类型消融、读出神经元置零、跨场景/未见种子测试。

### 8.4 评估与防伪进步原则

- 闭环优先指标：任务成功率、生存时间、撞墙率、击杀率、回报分布（不只均值）、样本效率、多种子置信区间、未见场景泛化、学习后冻结权重保留测试、黑屏/打乱输入差值；离线动作准确率/teacher agreement 只作诊断。
- 训练/模型选择/最终测试种子严格分离，超参不得在测试集上调，预注册成功线与停止条件，报告所有失败 run。
- 记录策略熵防动作塌缩；区分游戏时间/神经时间/墙钟时间；每个动作对应记录输入帧、神经窗、策略输出。

### 8.5 修管线的静态检查清单（动手训练前必过）

1. 归一化 mu/sd 只在训练子集原始特征上算，checkpoint 自足保存，禁止"先 zscore 再存 0/1"。
2. 离线与在线窗口用同一个 `FeatureWindow` 实现逐元素相等；禁止用绝对 tics 索引短列表；每个 episode 开头显式清空窗口/新建 Brain。
3. 按完整 episode/seed 分组划分；topK、PCA、z-score 只在训练组拟合，test seed 完全不参与选 K/选模型/选超参。
4. 统一指标口径：balanced accuracy 对比 1/C，ordinary accuracy 对比多数类，并报 confusion matrix 与每类 recall。
5. 明确 ANGLE 是特权老师输入还是状态泄漏；至少保留"带 ANGLE"与"纯像素"两套老师 checkpoint 对照。
6. FQI 最后一帧必须 done=True 或丢弃，窗口不跨 episode/seed。
7. 每次运行写唯一 run ID，保存命令行、源码/WAD/输入哈希、包版本、完整配置、种子、起止时间；禁止 CPU/GPU 或不同学生共用同一 results.json。
8. 纯函数单测先行：离线/在线窗口一致性、归一化往返、episode 边界、指标口径——不启动 ViZDoom 就能暴露当前大部分 bug。

---

## 9. 一句话总结

这个项目已经完成了大量真实工程工作（完整 MaleCNS 加载与哈希备份、视觉采样、全脑 LIF、走廊场景、PPO 老师 100% 出口、数 GB 神经活动数据），但就"果蝇连接组带来行为提升"这一核心主张，当前证据不足：师生输入不等价、闭环学生有确定实现 bug、离线验证有泄漏、对照样本与独立性不足、FQI 无成功轨迹、神经动力学本身也未通过上游科学门槛。**下一步不是加训练轮数或扩大 topK，而是先建立严格一致、可测试、无泄漏的管线，再用分组 held-out 轨迹和四臂消融重新归因。**

---

## 10. 2026-10-19 全链路复核与实际工程修订

### 10.1 “修了12 bug”与“全链路可用”不是一回事

当前旧脚本复核结论：**部分修复，不是全修，也不是所有历史错误仍在原处**。

| 环节 | 当前核实状态 |
|---|---|
| A 原始训练统计、B 共享窗口 | 主学生路径已接入正确原语；旧模型不自动修复 |
| C/D 训练组topK、PCA/归一化 | 主学生/消融已改训练组fit；旧readout_upgrade仍有全数据选择/随机帧拆分，且导入不存在的linear3_acc |
| E/F 分组与独立评测 | 默认train/val/eval seed分开；旧监督采集没有真实episode元数据，固定地图换seed也不证明泛化 |
| G/H 对照与指标 | 主消融可各自选特征，指标口径已统一 |
| I 师生信息 | 新老师可用pixel1536维；历史ANGLE1538维模型仍是特权版本，不能混同 |
| J/K 边界与脑状态 | 主闭环reset/FQI组末mask已改善；旧离线监督仿真仍跨seed/episode延续brain/motion；RL几何exit与实际reset条件仍不一致 |
| L 结果与溯源 | 唯一run已有；旧缓存未绑定完整输入依赖，老师变化可产生新标签+旧counts错配；latest模型/评测指针仍可能指错，>50MiB只记size/mtime不够 |

另有旧group构建按np.unique排序后拼接、乱序组会导致特征/标签错位；epsilon分支把随机执行动作当老师标签；旧模型没有特征schema拒绝机制。**本轮没有逐个修复所有历史脚本，而是将它们保留为历史材料，新推荐入口不导入、不训练、不复用其缓存。**17项旧共享原语测试已实际通过，但不覆盖这些生产采集链缺口。

### 10.2 现在实际实现的推荐调用链

新增包：`experiment/code/flyrl/`；入口：`experiment/code/flyrl_cli.py`，支持 doctor / smoke / diagnose / train / evaluate / compare。

```
reset游戏（出生等待10引擎tic，不是策略动作）
  → reset脑、motion和trace
  → RGB24
  → Vision：raw 或人工agc_motion
  → FrozenBrain：完整全图、参数化Numba、dt=.1ms
  → 每游戏tic请求1000/35ms，累计余数得到28.5/28.6ms
  → 无动作标签解剖分组：每组平均Hz，按实际时长归一
  → 50/200ms低通 + log1p，246维
  → 全训练布局随机轨迹校准，统计冻结，裁剪[-10,10]
  → 线性/hidden32小读出，在线DoubleDQN或PPO
  → 6/12离散动作，每step恰好一个策略tic
```

初始RGB也需一个神经编码窗，此后每个next RGB只编码一次；每局重新建立因果状态。黑屏不是“删除偏置”，因此黑屏仍有活动可作为重要诊断。

侧别补全是本次全链审查实际纠正的关键问题：normalized表没有侧别，直接池化会把左右信息合并。现使用本地原始 `annotations.feather`（SHA256 `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2`），按整数bodyId精确对齐全166700个graph.ids；somaSide明确L/R优先，仅缺失时用rootSide，冲突/中线unknown，不根据类型名猜测。得到L81822、R83935、unknown943，一例冲突1054369160。123个预注册互斥组中32个空组（修正前缺侧别时83空组），仍保持246维；这是粗解剖池化，不能宣称保留了完整柱状/垂直空间信息。空/恒定维度和每臂校准统计均需报告。

### 10.3 训练方式比较后的选择，而不是只凭流行度换算法

| 方法 | 本轮决策 | 条件与限制 |
|---|---|---|
| 在线DoubleDQN | 默认主线 | CPU全脑采样贵，replay能重用小特征；Double改动减少选动作/估值耦合的过估计，不把replay本身归功于Double论文 |
| PPO | 已实现同接口备选 | 正确存采样old_logp，GAE区分终止/截断；on-policy样本重用受限 |
| teacher-student / DAgger | 不继续作默认主线 | 不是无效方法，但当前师生信息/覆盖问题混杂；公平设计后可另作对照 |
| 离线FQI | 保留诊断 | 没有环境新数据去纠正未覆盖动作价值，旧缓存又不可靠 |
| ES | 不选 | 每候选昂贵闭环仿真，数小时预算不利 |
| 全图突触训练/三因子/e-prop | 本阶段不实现 | 需要明确可塑性、资格迹、信用分配及独立生理验证，不应一次改完再无法归因 |

默认线性头：246×6+6=1482参数（战斗246×12+12=2964）；只训外部头，不改图权重。DQN lr3e-4、batch64、replay20000、warmup256、target每500优化步同步、Huber、gradclip10、每转移1次优化、epsilon1→.1/10000步。PPO备选rollout128/epochs4/batch64/clip.2/lambda.95/entropy.01，critic hidden32单列计数。

统一gamma=.999：35Hz下完整走廊约数百tic，.99对远端出口折扣过强，potential与学习器必须同gamma。这是预注册工程候选，不是实测最优。不能声称已跑足预算比较两算法胜负。

### 10.4 游戏、奖励、数据与评估的具体修正

- 走廊动作turn{-6,0,+6}×forward{0,20}；战斗再×attack{0,1}；不再恒前进。可显式max_turn15，但所有对照必须一致。
- 走廊生成独立WAD：修正front/right winding与双面portal，三课程起点exit/near_turn/entry、左右镜像、光照/起点角度抖动。只改起点/地图，不按坐标控制动作。
- 按计划步数前10%出口附近、10–30%转角、余下完整入口。墙钟先停可能尚未覆盖完整课程，课程成功不可冒充完整任务成功。
- 奖励gamma*Phi(next)-Phi(now)，真终止Phi=0；出口奖励只一次且必须reset；只有forward低位移才算撞墙。战斗按击杀/命中/损血增量，均为工程RL奖励。
- time-limit保留reset前真实final RGB：DQN存显式next_feature；终止不bootstrap、截断可bootstrap；PPO GAE不跨reset。引擎已结束无图时明确标记terminal，不用reset图代替。
- train/val/test seed分开；布局按**实际WAD内容hash**排重，避免默认字段省略或不同seed生成同WAD漏过检查。走廊每唯一layout只评一次，不能把确定性重复局当独立样本；当前默认val/test各2布局仅作pilot。同一L形家族并非任意地图泛化，combat尚无未见地图实验。
- 四臂同维同actor/trace/任务/预算；retinal与pixels为不同固定池化路径，不用可训练CNN。rewired仅运行时派生，不回写；保源权重与符号度，不保目标入强度/simple graph，记录自环/重复边变化。
- 冻结评估支持learned/random/untrained策略，以及none/black/scramble输入干预，复用训练冻结统计；不重新校准、不更新策略，明确记录conditions，不把不同干预混为同条件四臂比较。
- 新run唯一ID、强SHA256、resolved config、所有本包/上游源码、graph/annotations/IWAD/WAD/配置及版本；评测显式checkpoint，拒绝源码/版本/输入不一致，torch.load仅weights_only=True。分组索引另加内容签名；不兼容旧模型。
- 最终PPO尾rollout先更新再保存新快照，避免周期checkpoint恰好在末步而返回旧模型。新checkpoint不含全脑/游戏动态，不声称精确resume。
- replay只存246维特征，obs+next_obs在20000容量约39.36MB，不再默认记录每帧166700全脑数组。

### 10.5 实际验收与尚未证明的事项

已实际执行：65项新包单元/假环境集成测试通过；17项旧共享原语测试通过。新测试包括真实CUDA小头更新、safe checkpoint跨设备恢复、Numba参考内核一致、时钟余数、精确大整数ID、重连守恒、reset、真实生产train尾快照、唯一评测组合、侧别冲突/中线规则等。首次全套合跑出现“其他测试预先import torch导致副作用断言误报”，已改独立解释器测试并通过；diagnose入口首次分发错误也已修复，不掩盖失败记录。

已执行真实ViZDoom短验证：走廊六起点×镜像组合与combat可加载，静止无隐藏移动、动作实际改变角度/位置、截断保留final RGB、reset图像可复现。真实全图与运行时重连、retinal、pixels短闭环和safe checkpoint均已跑通，graph强hash未变。短真实全脑DQN/PPO各12转移、12帧校准、冻结val调用链已完成；这是**开发验收微型优化，不是正式训练**，其游戏成绩没有统计意义。

2026-10-19初次四步短窗（非长期benchmark）真实脑神经推进约0.24–0.36s/帧，重连约0.46–0.48s/帧；启动、重连构图、环境reset、优化、长程状态均有额外成本，不能直接乘算保证2h完成20000步。2h只作单臂pilot，8h为较长批次；正式预算以实际日志为准。墙钟包含初始化/校准，单次调用不可中断，最终持久化另计。compare拒绝不同实际步数/不同墙钟上限/版本/任务/资产的混杂汇总。

新增diagnose做黑屏/静止/闪光/左右运动/受体空间置换。最终双侧分组8帧/条件实测：黑屏524847脉冲，静止651710，闪光593371，左右运动653444/655445，空间置换651395；左右特征距离约9.03，权重未变。只说明存在背景活动和某种输入依赖，不证明方向选择性，也不能拿总发放量大小当任务有效信息。该诊断仅检查数值有限性、背景活动与输入依赖，不代表T4/T5方向选择性、ON/OFF生理、光流机制或FlyVis已复现。**生物视觉验证仍未通过；工程就绪不等于生物正确。**用户当前优先闭环表现，因此允许以明确标注的非生物工程模型开展pilot，而不是冒充通过第8.2阶段1的生理门槛。

最终持久化验收索引：[experiment/runs/acceptance-20261019.json](experiment/runs/acceptance-20261019.json)，其中14个结果路径已逐一核对存在，最终源码hash与DQN冻结评估记录一致。最后一轮DQN/PPO各12转移、12帧校准、每局2tic的微型验收成功，安全重载最终快照确认分别有11次/2次优化；DQN的black、scramble、random、untrained四种冻结控制全部执行完成。DQN/PPO运行分别为 `20261019-133945-integration-test-real-ce90c8e053` / `20261019-134033-integration-test-real-550e4e6229`，对应冻结val为 `20261019-134007-eval-val-real-9c2e716c58` / `20261019-134052-eval-val-real-488026f98d`。最终combat短闭环为 `20261019-133955-smoke-real-f8ffaf6e9d`，4个动作、含reset共5次观测，约10.49秒，graph_unchanged与checkpoint_reload均为true。这些都是接线、优化与持久化验收，不是游戏学习成绩。较早四臂smoke/diagnose在最后评估控制补丁前生成，保留为当时证据，不声称其快照能通过当前源码一致性校验。

正式训练命令、验收阈值与四臂/多种子运行规则见 [experiment/SETUP.md](experiment/SETUP.md) 和 [experiment/TRAINING_PLAN.md](experiment/TRAINING_PLAN.md)。本轮不再删除文件、不commit/push、不部署，不改变任何权威脑图/备份。尚未做数小时正式训练、完整走廊成功率、战斗提升、多训练seed/多重连实例、充分未见布局测试，**不能预先保证学会或声称连接组优势**。

---

## 附录：主要来源 URL

**上游与作者**
- https://github.com/nftechie/doomfly
- https://male-cns.janelia.org/download/
- https://doi.org/10.1016/j.cell.2026.08.015

**基础模型论文与代码**
- Shiu 全脑 LIF：https://www.nature.com/articles/s41586-024-07763-9 ；https://github.com/philshiu/Drosophila_brain_model
- FlyVis 视觉模型：https://www.nature.com/articles/s41586-024-07939-3 ；https://github.com/TuragaLab/flyvis
- Huang 多巴胺记忆：https://www.nature.com/articles/s41586-024-07819-w

**连接组游戏/控制对照项目**
- doomfly-rl：https://github.com/nonatofabio/doomfly-rl ；https://nonatofabio.github.io/blog/posts/doomfly_autonomous.html
- shreyash-sharma/doomFly：https://github.com/shreyash-sharma/doomFly
- fly-hero：https://github.com/bsgelman/fly-hero
- alpha-fly：https://github.com/turlockmike/alpha-fly
- FlyGM：https://arxiv.org/abs/2602.17997
- BPU 棋类：https://arxiv.org/abs/2507.10951

**身体闭环与接口警示**
- NeuroMechFly v2：https://www.nature.com/articles/s41592-024-02497-y
- Eon embodied fly：https://eon.systems/updates/embodied-brain-emulation
- Digital Sphinx：https://doi.org/10.64898/2026.03.20.713233 ；https://elifesciences.org/reviewed-preprints/111516 ；https://github.com/Brunton-Lab/DigitalSphinx2026

**方法学**
- DoubleDQN（van Hasselt/Guez/Silver，2015预印本/AAAI2016）：https://arxiv.org/abs/1509.06461
- PPO（Schulman等，2017）：https://arxiv.org/abs/1707.06347
- DRQN（部分观测与记忆的背景参考，并未在本包实现LSTM）：https://arxiv.org/abs/1507.06527
- DAgger：https://proceedings.mlr.press/v15/ross11a
- Policy Distillation：https://arxiv.org/abs/1511.06295
- Hinton 蒸馏：https://arxiv.org/abs/1503.02531
- LSM+Q 读出：https://arxiv.org/abs/1906.01695
- e-prop：https://www.nature.com/articles/s41467-020-17236-y
- EventProp：https://www.nature.com/articles/s41598-021-91786-z
- 三因子规则：https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00085/full
- 代理梯度：https://arxiv.org/abs/1901.09948
