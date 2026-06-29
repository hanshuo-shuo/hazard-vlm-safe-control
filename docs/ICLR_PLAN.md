# ICLR Plan — 把 hazard / Path-B 做成可发表论文

日期：2026-06-29
依据：对全仓代码 + 现有结果的复核，加上一次 9-agent 竞品/可发表性调研（~45 篇真实论文交叉核对，0 伪造引用）。
配套文档：[`PROJECT_PLAN.md`](PROJECT_PLAN.md)（旧路线）· [`RESULTS_SEMANTIC.md`](RESULTS_SEMANTIC.md)（主结果 pilot）· [`WHY_VLM.md`](WHY_VLM.md) · [`FAILURE_MODE_ANALYSIS.md`](FAILURE_MODE_ANALYSIS.md)

---

## 0. 一句话诊断（你为什么卡住）

你现在卖的是 **"VLM 能避开几何 cost 写不出的语义禁区"** 这个*现象*。问题是：**这个现象已经被人在真机上、更大规模地证明过了**（LaC IROS'25、CoNVOI IROS'24、ZeST、CATNAV，以及最致命的 **CORE 2026**）。所以审稿人一句话就能毙：*"现象已知、PIVOT 机制是别人的、几何盲是构造出来的同义反复，n=5、单模型、2D toy。"*

**但你手里有一个别人都没有、且确实没人占的东西**：一套**可审计的"无泄漏 / 不可测约束"诊断协议**。把论文从"系统/能力"重定位成"**测量 + 协议 + benchmark**"，是唯一能过 ICLR 的路。下面是怎么做。

---

## 1. 现状诚实评估（你已经有什么）

资产（真的有价值）：
- **干净的 4 臂因果设计** C1（几何盲）/ C2（手编 oracle 上界）/ B（VLM 选 subgoal）/ B+（VLM 自己的感知喂回控制器）。把 VLM 夹在"不可能下界"和"人工标注上界"之间——这个*实验模板*本身可发表。
- **无泄漏纪律 + 原始 transcript 审计 + Wilson CI + matched seeds**。这是你区别于所有竞品的地方。
- **"不可测 by construction" 的约束**：禁区不进 obs、不终止 episode、prompt 不命名——几何规划器是*结构性*盲，不是"perception 没调好"。这把 C1-vs-B 从"感知质量比拼"升级成"可识别性 (identifiability) 结果"。
- **自发命名证据**：VLM 在 free-text 里自己说 "water/mud/grass"（prompt 从没提）。这是 grounding 的可审计证据。
- 一堆现成基础设施：`subgoal_pivot_hazard.py` 四臂、`mpc_expert.py` 软/硬 zone、direct-VLA baseline（`direct_vla_hazard.py`）、PointPush 接触环境、离线 heuristic pilot 通道。

短板（审稿人会直接打）：
- **全是 n=5 pilot、单一闭源模型（gemini-3-flash）、单一极小 2D arena**。统计上 B/B+ 与 C1/C2 不可分。
- **三处会在 5 分钟内被代码审出来的硬伤**（见 §3，必须先修）。
- 多份文档/图其实是**同一个 5-seed realization 的重复渲染**，读起来像凑篇幅。

---

## 2. 竞品版图（必须正面对位的）

> 全部经过标题+arXiv ID+描述交叉核对，真实存在。**红色 = 直接威胁你 novelty 的。**

| 论文 | venue/年 | 做了什么 | 对你的威胁 / 你保留的差异 |
|---|---|---|---|
| **CORE** [2602.19983](https://arxiv.org/abs/2602.19983) | 2026 (UPenn/CMU) | VLM 从视觉推断 context-dependent 安全 → grounding 到 safe set → CBF 强制，**概率安全保证**，5 个 VLM、3 sim、真 Spot，几乎相同的 Oracle/No-Context/Geometric 臂结构 | **头号 kill paper。** 它把你想做的"系统/能力"版本以更大规模做完了。**唯一活路：把 delta 钉在"它告诉/传入 VLM 要避什么、不审计泄漏；你从不命名、可审计"，并尽量以 concurrent work 处理（它是 2026-02）** |
| **Language as Cost (LaC)** [2508.03138](https://arxiv.org/abs/2508.03138) | IROS 2025 | VLM 把非几何 hazard（湿地/禁入/会开的门）变成连续软 cost field 喂 MPPI，真机 | **B+ 的机制别人先发了**（VLM→软语义 cost→planner，zero-shot）。但 LaC 用显式分割器、prompt 直接命名 hazard、无泄漏审计 |
| **CoNVOI** [2403.15637](https://arxiv.org/abs/2403.15637) | IROS 2024 | PIVOT 式编号 marker，VLM 选 waypoint，"avoid grass / 走人行道"，真机 Spot/Turtlebot | **几乎等于你的 B 臂。** 差异：它的 marker 只放在 lidar 可通行格（几何预过滤=部分泄漏）；无 oracle 上界、无泄漏审计、无统计 |
| **ZeST / CATNAV / VLM-GroNav** [2508.19131](https://arxiv.org/abs/2508.19131) ·  [2409.20445](https://arxiv.org/abs/2409.20445) | 2024-25 | VLM 造 traversability costmap（含 mud/坡） | 你的"VLM 造 cost"主张的若干实例。都需要分割器/本体感知，非 zero hand-coded perception，无审计 |
| **HazardArena** [2604.12447](https://arxiv.org/abs/2604.12447) | 2026 | safe/unsafe 双胞胎场景，显式担心"安全行为可能只是保守控制的混淆，不是理解" | **抢了你"honesty moat"的科学框架**（在 manipulation 域）。既是威胁也是盟友：用它来 legitimize 你的 confound 担忧，但把 delta 钉在 closed-loop nav + 泄漏审计 + oracle 三元组 |
| **PIVOT** [2402.07872](https://arxiv.org/abs/2402.07872) | ICML 2024 | 编号候选 VQA 选择 | 你 B/B+ 的**方法母体**。只能引用为 method origin，不能当贡献 |
| **VLSBench** [2411.19939](https://arxiv.org/abs/2411.19939) | ACL 2025 | 把"视觉安全信息泄漏"作为一类方法学问题（但只在 VLM 越狱/拒答） | **你的方法学血统**：社区已承认"无泄漏"是真问题；没人把它带进 traversability/控制——这是你的白空间 |
| **STPR "Don't Do That!"** [2506.04500](https://arxiv.org/abs/2506.04500) | 2025 | LLM 把"不要做什么"翻成可执行 Python 约束给 A*/RRT* | "几何 cost 写不出这类约束"这句话是它的 motivation。但它约束**用文字给定**；你的禁区**latent 在像素里、从不命名** |
| **"Safety Not Found (404)"**（原名 Before We Trust Them）[2601.05529](https://arxiv.org/abs/2601.05529) | 2026 | 以"对 LLM/VLM 机器人决策做严格审计/质疑"为贡献 | **最接近你的 reframing 本身**。投稿前必须读全文，明确 delta |
| **Verifier Tax** [2603.19328](https://arxiv.org/abs/2603.19328) | 2026 | verifier 提安全、税掉成功率（抽象 tool agent） | 可作"软 keep-out 胜硬约束"的*概念*动机，**不能当证明**（域不同，见 §3 注意） |
| **Towards Zero-Shot Traversability** [2508.01715](https://arxiv.org/abs/2508.01715) | 2025 | 实测 VLM 地形 F1≤0.51、对 prompt 敏感 | 直接威胁你 n=5 头条数字的可靠性——也正是你需要做多 backbone/多 seed 的理由 |

**结论**：现象、机制、几何盲论证、oracle-matching、泄漏洞见——**每一块单独都已发表**。唯一没被发表的是**它们在严格泄漏审计 + 隐式地形诊断下的特定融合**。所以贡献必须打包卖，且必须是"协议/测量"而非"系统/能力"。

---

## 3. 投任何 paid run 之前，必须先修的三个 kill-shots（代码已核实）

1. **稻草人 oracle（最伤信誉）。** `subgoal_pivot_hazard.py:543` 的 C2 把 zone 当 hazard 拼进 avoid 集 → 吃 `mpc_expert.py:162` 的**硬** `collision_penalty=1e4`；而 B+ 用**软** halo（`soft_zone_weight=30`）。所以"B+ 100% 反超 oracle 60%"是 **soft-vs-hard 的不公平对比**，而且 `RESULTS_SEMANTIC.md §12.4` / `RESULTS_AMPLIFY.md §4` **你自己已经承认了**，但头条表还写"out-robusts"。审稿人会在仓库里找到你自己写好的反驳。
   - **修法**：新增一个 **soft-cost oracle**（真 zone 几何 + 与 B+ 相同的 `soft_zone_weight`）。诚实对比是"VLM 估计的软 zone vs 人标注的软 zone"。把所有"out-robusts the oracle"改成有条件表述。
2. **缺最该有的 baseline。** 没有 **open-vocab 检测/分割器**（GroundingDINO+SAM / CLIP-text "water" / OWL-ViT）→ mask → 喂同一个软 cost 控制器。在干净渲染 toy 上它很可能追平甚至超过 VLM，且无需逐场景人工、确定性、便宜 ~100×。**没有它，你只证明了"VLM 打败被故意蒙眼的规划器"，而不是"打败 roboticist 第一反应会用的感知栈"。**
3. **Tuning-on-test。** `RESULTS_SEMANTIC.md §11.2`：B+ 的三个设计选择是针对 eval 种子（45/47）的观察"逼出来"的。**修法**：在一组 disjoint 的 tuning seeds 上锁定 `vlm_zone_radius/core/soft_weight`，再到 held-out seeds 评估；论文里明说。

附带必修：**L2（prompt 完全不提地形）从没真跑**（现在的"implicit"仍给了类别提示"避开看起来不安全的地形"=instruction-following，不是 commonsense）。这是审稿人第一刀。

---

## 4. 重定位：从"VLM 打败 classical" → "无泄漏的可替代性研究 + 诊断 benchmark"

### 4.1 新 thesis（英文，paper-facing）

> **"Does a VLM high-level planner gain its robustness from genuine semantic perception, or from leaked safety cues in the prompt and evaluation? We build a leakage-controlled, closed-loop testbed where the semantic keep-out is provably absent from the observation and never named in the prompt, and we measure — across models and seeds, with audited transcripts — how much of a VLM's apparent constraint-following survives when nothing is leaked, and how much human per-scene labeling it can replace."**

注意三处措辞克制（直接拆掉审稿人的刀）：
- 不说 "a geometric cost CANNOT express this"（你自己的 pipeline 把它聚类成一个软 disk 就反驳了）。改说 **"who supplies the keep-out — a human label per scene, or the VLM zero-shot"**（substitution claim）。
- 不说 "VLM beats classical"。改说 **"VLM-perception substitutes for per-scene human labeling at near-oracle quality with zero hand-coded detectors"**。
- 不说 "commonsense" 除非 L2 真跑出来且有 false-positive 控制。

### 4.2 三大贡献支柱（只能打包卖）

1. **泄漏审计协议 + 可发布的诊断 benchmark（核心、最 ICLR-flavored）。**
   一个公开的受控 testbed：可测几何 hazard 与"结构性不可见"语义禁区共存；严格"prompt/image/obs 零 clearance/label/score/coord/in-zone-flag"；原始 transcript 审计；matched seeds + Wilson CI；**外加一个"泄漏注入旋钮"**（见支柱 2）。这是 Safety-Gymnasium / CORE / LaC 都没有的 artifact。"CLEVR for safe control"。
2. **把泄漏当作被操纵的自变量（dose-response，独有结果）。**
   系统地往 prompt 注入越来越多安全信息（label → score → coordinates → which-candidate-in-zone），画出"报告的 VLM 鲁棒性如何随泄漏膨胀"的曲线。**这把"honesty moat"从一句声明变成一个机制结果**，并直接呼应你被撤稿的前史。没人对机器人规划做过这个。
3. **可替代性 / 经济学曲线（"一个 VLM 顶 N 个手写检测器"，量化版）。**
   扫 terrain 种类数 N：画 violation + 工程成本（检测器数/LOC）随 N 的曲线，含**一个 held-out 未见地形**（手写栈漏、VLM zero-shot 接住）。配 §3 的公平 soft-soft oracle 与 open-vocab segmenter 作为两个真竞争者。

辅助但增色：**impossibility-bracketed 因果模板**（C1 不可能下界 + C2 人标上界把 VLM 夹住）+ **自发命名审计**（量化 naming 频率与"命名↔避开"相关性）+ **校准/弃权角**（VLM 何时该把决定让回安全控制器）。

### 4.3 关于 agent + MCP（你的探索方向）——明确结论

**作为论文主框架：放弃。** 8 个 agent 一致 + 文献佐证：ICLR 级会议里 MCP 只作为 benchmark 基质（MCP-Universe / MCP-Bench）或安全对象（Log-To-Leak）出现，**从不是贡献**；唯一的机器人 MCP 论文（ROSBag-MCP）自称 plumbing；ROSA / RAI 已经把"LLM agent + 感知/仿真/控制工具 + 安全校验"做成工程了。而且**你的代码现在根本没有 agent loop、没有 tool-routing**——控制流是固定 pipeline（VLM 选 subgoal/avoid-array → MPC 执行）。把它说成"embodied agent orchestrating tools over MCP"= 描述不存在的代码 = 重蹈上次撤稿的 over-claim。

> MCP 最多保留为一句实现说明："modules are exposed as standardized tools so each can be ablated/swapped"（复现性脚注），**绝不进 contribution list**。

**唯一能让"agent/tool 编排"变成科学的版本**（如果你真想做这条线）：让**调用哪个工具、何时调用**成为一个*被学习/被触发的决策*——一个在"昂贵的 VLM 感知工具 vs 便宜的几何控制器"之间权衡的 **perception-cost-vs-safety budget agent**。代码已有钩子（`subgoal_horizon` 重查询间隔、已记 VLM calls/ep）。把固定间隔变成"按不确定性/风险触发的调用预算"，并证明它在 violation/成本前沿上占优——**这才是一个真 claim，但实验现在不存在**。可作为 §7 的 reach / 第二篇。

另外两个把 MCP 变成*真研究*的小众角度（都属于"另一篇论文"，不是这篇）：
- **泄漏攻击面**：如果工具经 MCP 暴露，研究"工具返回值能否把安全 label 偷偷塞回 context"——把 MCP 变成 security/robustness 分析对象（接 Log-To-Leak / VLSBench 线）。

---

## 5. 实验矩阵（达到 ICLR bar）

### 5.1 Minimal-Viable-Paper（最小可信集，全部基于现有代码 + 适度新增）

1. **主结果**：PointHazard implicit-zone 四臂 C1/C2/B/B+ **n≥100 matched seeds**，Wilson CI + **paired McNemar**（B vs C1、B+ vs B），Holm-Bonferroni 校正。
2. **机制阶梯 L0→L1→L2**：L0 显式 amber-X 命名 → L1 类别提示（现状）→ **L2 完全不提地形**（"reach the goal sensibly"）。"violation vs prompt-specificity"曲线是全论文最干净的图，也是真 thesis。带 false-positive naming 控制 + look-alike 对抗纹理（如蓝色但安全的地面）。
3. **公平 oracle（kill-shot #1 修复）**：soft-cost true-geometry oracle，与 B+ 并列。
4. **该有的 baselines（kill-shot #2 修复）**：① open-vocab segmenter→同控制器；② prompt-only de-leaked PIVOT（证明安全控制器是 load-bearing）；③ caption-then-classical（VLM 描述场景→另一 LLM 抽坐标→喂 C2，"先描述再经典规划"的稻草人）。
5. **≥2 个 VLM backbone**（同 100 seeds）：一个别家 frontier + 一个强开源（Qwen2.5-VL-72B / InternVL 级）。杀掉"单模型"。
6. **可替代性扫描**：手写检测器栈 vs terrain 数，含 1 个 held-out 未见地形 + open-vocab segmenter 对比。
7. **泄漏负控制 + 命名审计**：occlude-zone / shuffle-numbering / no-zone 三个负控制；自动统计自发命名率与"命名↔避开"相关。
8. **全指标套件**（每臂）：violation rate、in-zone dwell、success、hazard、timeout、min-clearance、path-efficiency、VLM calls/ep、fb%/parse-fail、B+ 感知质量（估计 centroid 误差 / IoU vs 真 zone）。

### 5.2 Reach（把 workshop/borderline 推成 strong / main-track）

- **泄漏 dose-response 曲线**（支柱 2 的核心结果，强烈建议至少做到这个）。
- **第 3 个 backbone（含一个 small/cheap 模型）**画 cost/accuracy 前沿；视觉 prompt 设计鲁棒性（分辨率/marker 样式）。
- **PointPushHazard 加语义禁区**，四臂 n=100，候选升级成 push-pose/contact-side——第二个物理难点族，破"单环境"。
- **第三方 arena 交叉验证**：Safety-Gymnasium SafetyPointGoal/PointPush，加一个"渲染出来但从 cost/obs 里移除"的语义 keep-out，跑同一协议——破"bespoke toy"。
- **non-toy / 真实感**：Habitat 或 CARLA 的语义 keep-out 变体，或一个**极简真机桌面**（顶视相机 + ArUco + 贴纸禁区，~20 episodes）。这是最强的单点说服力，也直接预防"为什么只有 2D 点质量"。
- **end-to-end VLA**（same-backbone Qwen2-VL，regression+flow+LoRA，已实现）在 oracle-avoiding demo 上训练，success/violation vs demo 数——量化"把感知烤进权重"的数据/调参成本。
- **self-consistency / temperature-vote**：多数投票能否补上 B 到 B+ 的残差。

### 5.3 统计纪律（必须照做）

- 所有臂**同一组 seeds**（paired）。主图 n=100；**B+ vs B（20%→0%）这个对比可能需要 n≥200**（效应薄，靠 corner-cut 种子出现的频次）。
- 连续指标用 paired bootstrap CI + Wilcoxon signed-rank。多 backbone 把 backbone 当 random factor 报 averaged effect。
- **付费前先用 `--pilot_mode heuristic` 把全 pipeline 在 n≥100 上离线验证**；prompt 在 paid run 前**锁死**（不许按结果调 prompt，否则泄漏 moat 破功）。预注册 seed 范围/prompt/指标定义。每条 transcript 存档。

---

## 6. 时间线 + 投稿策略

文献现实（来自调研）：**当前规模（2D / n=5 / 单模型）= workshop 论文**。要上 ICLR main track 需要 §4 的 reframe **加** §5.1 的 scale。CoRL/RSS 对"经验/系统"框架更友好，但 CORE 让它们也变挤。

**推荐路径（求稳又抢旗）：**
1. **现在 → 4 周**：修三个 kill-shots（§3）+ 跑 §5.1 的 1/2/3/4/7（主结果、L2 阶梯、公平 oracle、核心 baselines、负控制）。**先离线、再小额付费**。
2. **第 5–6 周**：投一个 **workshop**（ICLR/NeurIPS/CoRL 的 embodied 或 safe-ML workshop），**把"泄漏审计协议 + 隐式地形诊断 + benchmark"的旗插在 CORE 衍生工作之前**。Workshop 版只需 1 个强 backbone + L2 + 公平 oracle。
3. **并行 / 第 2–4 月**：补 §5.1 剩余（多 backbone、可替代性扫描）+ §5.2 至少 1–2 项（dose-response 必做；PointPush 或 Safety-Gymnasium 二选一；可选真机/Habitat）。
4. **投 ICLR main track 或 CoRL**：带"协议+测量+benchmark+release"的完整故事，正面对位 CORE / LaC / HazardArena（以 concurrent 处理）。

> 注：所有 §2 的 2026 威胁论文（CORE/HazardArena/404）都是 2–4 月。**投稿前逐一核对目标会议的 concurrent-work cutoff**——很可能你能主张 concurrent 而非被 scoop，这会实质改变 framing。

---

## 7. 风险与开放决策

- **CORE 是否已经把你想说的都说了**：投稿前必须读 CORE 全文，逐条核对它的 "No-Context" 臂到底是不是你的 B、它有没有做泄漏审计。delta 段落写错=被批"连最近的相关工作都没读懂"。
- **公平 oracle 修好后，B+ 还赢不赢**：如果 soft-soft true-geometry oracle 反超 B+，故事必须彻底转向"成本/可扩展性"（一个 VLM vs N 检测器），而不是"性能反超"。**这是必须先知道答案的实验**。
- **L2 跑出来可能是负结果**（不给类别提示，VLM 不避了）：那也是诚实且可发表的——"commonsense keep-out 需要类别级提示"本身是结论。但它决定你能不能用 "commonsense" 这个词。
- **API 预算**：多 backbone × ~100–200 seeds × 多条件是这套计划的主要开销。先确认范围再付费（见 [[hazard-experiment-rigor]] 的纪律）。
- **要不要做非 toy 环境**：这是工作量的最大乘数。Workshop 不需要；ICLR main track 几乎肯定需要至少一个（Habitat/CARLA 语义 keep-out，或极简真机）。

---

## 8. 立即下一步（按顺序）

1. 写 **soft-cost true-geometry oracle** 臂（改 `subgoal_pivot_hazard.py` 的 C2 分支 + `mpc_expert.set_soft_zones` 复用），离线 heuristic 验证。
2. 实现 **open-vocab segmenter baseline**（CLIP-text 或 GroundingDINO+SAM → mask → 软 zone → 同控制器）。
3. 实现 **L2 prompt 条件** + false-positive naming 控制纹理；把 L0/L1/L2 做成一个 `--prompt_level` 开关。
4. 把 B+ 三旋钮迁到 **disjoint tuning seeds** 上锁定，eval 用 held-out。
5. 离线 n≥100 跑通全 pipeline → 小额付费跑 1 个 backbone 的主结果 + L2 + 公平 oracle → 看 §7 的两个"必须先知道答案"的实验结果，再决定 framing 与是否扩 scale。
