# 从“VLM 识别危险地形”到“接口条件化安全”：项目故事线演化与实验取证

更新时间：2026-07-27（Asia/Shanghai）

文档性质：研究叙事说明、历史实验审计与当时的 claim boundary。本文已归档；当前简明叙事见 [`RESEARCH_STORY.md`](../../../RESEARCH_STORY.md)，实验状态以 [`docs/RESULTS_REGISTRY.md`](../../../docs/RESULTS_REGISTRY.md) 为准。

## 一句话结论

项目转向不是因为“VLM 完全不能识别危险地形”，而是因为连续两轮审计发现：**现有实验无法把模型能力与接口诱导、字段歧义和下游解释分离开来。**

第一轮审计发现，模型的物理选择会随 marker ID 和候选顺序显著变化，因此 marker 实验不能证明稳定的语义危险理解。第二轮审计发现，所谓“识别—适用性鸿沟”主要来自 `applicable` 的双向语义：模型回答“地形是否适合通过”，evaluator 却评分“避让约束是否应生效”。这两次失败共同指向了更基本的问题：

> 同一个模型、同一张图、同一个机器人和同一段控制代码，是否会因为接口字段、表述、序列化、grounding 或 planner mapping 不同，而得到不同的动作、轨迹与安全分数？

因此当前主线不再是证明“VLM 比经典 planner 更懂危险”，而是审计 embodied VLM 系统中 perception、reasoning、grounding、planner、controller 与 evaluator 之间的接口契约。

## 1. 三次故事线变化

| 阶段 | 原始研究问题 | 当时最想得到的结论 | 审计后发现 | 最终决定 |
|---|---|---|---|---|
| 1. 语义危险识别 | VLM 能否识别几何 planner 看不到的水面、草地或脆弱地面？ | VLM 提供开放世界语义知识，MPC 负责执行 | 物理选择对 marker ID、候选顺序和候选数量高度敏感 | 终止 marker-only 主线 |
| 2. 规则适用性 | 模型是否“认得危险，但不知道规则是否适用于当前机器人”？ | 存在 recognition–applicability gap | `applicable` 同时可表示“约束适用”或“地形可通行”，二者正好反向 | 拒绝 norm-applicability failure 故事 |
| 3. 接口条件化安全 | 等价接口是否改变模型输出及闭环安全结论？ | 安全评测必须报告跨接口的稳定性和结果包络 | 已观察到 normalization、grounding、planner mapping 差异传播到动作、轨迹与 STC | 当前主线；仍为 pilot，尚非正式论文结果 |

这三阶段并不是互相无关的换题。它们构成了一条逐步收紧因果解释的路径：

```text
“模型会选安全路线”
        ↓ 审计选择是否绑定真实物理位置
“模型是否理解规则对机器人的适用性”
        ↓ 审计字段是否具有唯一语义
“评测接口本身是否改变模型、控制器和最终安全结论”
```

## 2. 第一阶段：VLM 是否能识别语义危险

### 2.1 最初设计为什么合理

最初系统采用典型的模块化分工：

```text
RGB observation
  → VLM 识别语义危险并选择 waypoint / keep-out zone
  → 固定低层 MPC 执行
  → environment/evaluator 统计碰撞、到达与安全
```

经典几何 planner 擅长处理可观测障碍物，却不天然知道某一块视觉上可穿越的区域是否与机器人能力冲突。例如：

- 非防水轮式机器人不应进入水面；
- 普通轮式底盘可能陷入泥地或松软草地；
- 某些地面可以承载轻型机器人，却不适合重型机器人；
- 脆弱、受限或社会规范禁止进入的区域不一定构成几何障碍。

因此最初假设是：VLM 负责从视觉与常识中补充语义约束，经典 planner/MPC 继续负责动力学和低层避障。这个分工本身仍然合理；被否定的是**当前 marker 实验能够证明该分工中的 VLM 已经稳定完成了语义 grounding**。

### 2.2 为什么早期正结果不够

在 marker/candidate 设计中，图像上的候选点带有编号、颜色和固定排列，prompt 又要求模型输出候选 ID。此时模型可能依赖：

- marker 的编号或颜色；
- 候选在 prompt 中出现的先后顺序；
- 候选数量和默认位置模式；
- schema 中某个字段的固定位置；
- prompt 对“左、右、中间”或安全候选的间接暗示。

如果模型真正绑定的是“物理位置及其与危险区域的关系”，仅交换 ID 或序列化顺序时，模型应继续选择同一个物理 waypoint。若选择随符号变化，就无法把结果解释为稳定的语义空间推理。

### 2.3 终止主线的直接证据

`NF-02 marker/interface robustness` 在两个模型、五个 seeds 上做了 matched pilot。预声明的继续阈值为物理选择一致率 `0.8`，实际结果为：

| 干预 | 物理选择与 baseline 一致率 | 说明 |
|---|---:|---|
| marker ID permutation | 0.20 | 只交换编号，80% matched rows 改变了物理选择 |
| candidate order permutation | 0.30 | 候选坐标记录相同，仅改变顺序，70% matched rows 改变了物理选择 |
| count 16 | 0.20 | 候选数量改变时物理选择明显不稳定 |
| count 4 | 0.00 | 所有 matched rows 都未保持 baseline 物理选择 |
| direct coordinates | 0.40 | 去掉 marker、直接给坐标也没有恢复稳定性 |
| unmarked image | 0.10 | 无标记图像下几乎不保持 baseline 选择 |

这里一个容易误读的点是：`baseline_8` 和 `candidate_order_permutation` 的 aggregate selected-safe rate 都是 `0.5`，但物理选择一致率只有 `0.3`。**总体安全率相同并不代表同一个样本上做出了同一个物理决定。** 聚合分数会掩盖接口敏感性。

该实验触发了 kill condition，仓库于 2026-07-24 将 marker-based PointHazard VLM-waypoint 主线标记为 `TERMINATED / PILOT_ONLY / NOT PAPER RESULT`。见：

- [`results/next_five_experiments/02_marker_interface_robustness/REPORT.md`](../../../results/next_five_experiments/02_marker_interface_robustness/REPORT.md)
- [`docs/RESULTS_REGISTRY.md`](../../../docs/RESULTS_REGISTRY.md)
- [`docs/RESEARCH_REVIEW_COMMENTS.md`](../../../docs/RESEARCH_REVIEW_COMMENTS.md)

### 2.4 第一阶段还能保留什么

可以保留的结论是：

- 语义识别正确与安全选择之间存在 pilot-level 相关信号；`NF-01` 中，识别正确时选择安全候选的条件概率约为 `0.714`，识别错误时约为 `0.036`；
- marker/candidate interface 的稳定性不足，必须把“输出候选编号正确”与“物理位置选择稳定”分开评估；
- VLM + fixed controller 仍是值得研究的模块化结构，但必须换成 marker-free grounding、严格的信息流隔离和 matched interface interventions。

不能保留的结论是：

- VLM 已经稳定理解水面或其他语义危险；
- marker waypoint 结果证明了开放世界语义 grounding；
- VLM 优于公平的经典 planner、oracle geometry 或现代 perception stack；
- 增加更多 marker、terrain name、seed 或 prompt tuning 就能把旧结果升级成论文证据。

## 3. 第二阶段：是否存在 recognition–applicability gap

### 3.1 为什么这个故事一度看起来成立

终止 marker-only 主线后，五阶段输出把系统拆成：

```text
recognition → applicability → grounding → action proposal → enforcement/outcome
```

`NF-04` 的 pilot 表面上给出了一个极强的异常：

| 指标 | 结果 |
|---|---:|
| capability applicability twin accuracy | 0.00 |
| capability routing change rate | 0.30 |
| appearance recognition twin accuracy | 0.80 |
| appearance applicability twin accuracy | 0.40 |

这很容易被叙述为：模型能看出 terrain 是水面，reason 也谈到了机器人是否防水，却不会把这项知识转成正确的 `applicable/not_applicable` 判断。

### 3.2 `applicable` 实际包含两个相反问题

后续逐条审计 response、reason、route 与 evaluator mapping 后发现，`applicable` 没有唯一方向。它至少可以自然地表示两种含义：

1. **Constraint semantics**：避让约束是否适用于当前机器人？
2. **Compatibility semantics**：该地形是否适合当前机器人通过？

对同一事实，这两个答案正好互为否定：

| 机器人—地形关系 | “避让约束是否适用？” | “地形是否适合通过？” |
|---|---:|---:|
| 非防水轮式机器人面对水面 | applicable | not applicable / incompatible |
| 水陆两栖机器人面对水面 | not applicable | applicable / compatible |

模型实际倾向于采用第二种理解，即把 `applicable` 当作“terrain is applicable/compatible for traversal”；旧 evaluator 则采用第一种理解，即把它当作“avoidance constraint applies”。如果 reason 和 route 与“地形兼容性”解释一致，仅仅因为布尔字段与 evaluator 标签相反，就不能认定模型出现了高层规范推理失败。

### 3.3 为什么 0.00 不是“模型完全不会适用性推理”

当一个指标接近随机时，可能是能力不足；当 paired twins 系统性得到完全反向的 `0.00`，同时自然语言理由与路线选择仍具有内部一致性时，更应优先检查：

- label polarity 是否反了；
- 字段名是否有两个合理读法；
- parser normalization 是否丢失了方向；
- evaluator 是否把 compatibility 当成 constraint applicability；
- 下游 planner 是否又对同一字段采用了第三种映射。

因此仓库在 [`docs/INTERFACE_CONTRACT_MAINLINE.md`](../../../docs/INTERFACE_CONTRACT_MAINLINE.md) 中明确拒绝 norm-applicability failure story：早期输出语义上可以是连贯的，错误主要来自 evaluator 对模糊字段采用了相反映射。

### 3.4 第二阶段留下的真正问题

这个结果并没有证明模型具备可靠的 applicability reasoning；它证明的是：**旧实验没有定义出一个足够清晰、可唯一评分的 applicability estimand。**

因此正确的研究动作不是把字段改名后重新追求一个更高 accuracy，而是把不同接口定义本身作为实验变量，并追踪它们如何影响：

- 解析是否成功；
- canonical semantics 是否一致；
- grounding 是否一致；
- planner action 是否一致；
- controller action sequence 和 trajectory 是否一致；
- 最终 Safe Task Completion 是否一致。

## 4. 第三阶段：接口定义会不会改变安全评测结论

### 4.1 当前研究问题

当前主问题是：

> Are embodied safety evaluations measuring model capability, or artifacts of the interface contract between perception, reasoning, and control?

中文可表述为：

> embodied VLM 的安全评测究竟测到了模型能力，还是测到了 perception、reasoning、grounding、planner、controller 与 evaluator 之间某一种人为选定的接口契约？

这里的“接口契约”不只指 JSON schema，还包括：

- 字段名称与字段顺序；
- positive/negative label polarity；
- constraint wording 与 compatibility wording；
- structured JSON 与 free text；
- marker identity 与 candidate serialization；
- normalized coordinate、pixel coordinate 与 native-world coordinate 的转换；
- `applicable`、`unknown` 等输出进入 planner 时的映射；
- planner 如何把 grounding 交给 controller；
- evaluator 是否与 controller 严格隔离，避免 truth geometry 泄漏。

### 4.2 当前因果链

当前实验不再只比较文本答案，而是检查完整传播链：

```text
public RGB + capability card
        ↓
equivalent interface contract
        ↓
strict parse / normalization
        ↓
canonical recognition and applicability
        ↓
provider-estimated grounding
        ↓
fixed planner mapping
        ↓
native controller action sequence
        ↓
native trajectory
        ↓
detached evaluator: collision / semantic violation / task success / STC
```

这条链的关键要求是：controller 只能消费 provider estimate，evaluator-truth geometry 只能在轨迹结束后评分，不能偷偷帮助规划。

### 4.3 为什么先做 executable bridge

如果坐标转换、controller 或 replay 本身会制造差异，就不能把后续结果归因于接口。Experiment 0 因此先用 provider-free fixtures 验证整个桥：

- `80 blocks × 8 fixture cases = 640` 次 native executions；
- PointHazard 与 Safety-Gym 各 320 次；
- equivalent fixture 的 physical-action IEC = `1.00`；
- equivalent fixture 的 trajectory IEC = `1.00`；
- capability semantic reversal = `1.00`；
- appearance twin action/trajectory match = `1.00`；
- `unknown` 按冻结规则导致 task failure：`80/80`；
- correct grounding 的 mean center error 约 `0.0092` world units。

这个控制实验说明：在输入等价且 grounding 正确时，bridge 不会凭空产生不一致；后续真实模型差异也不能简单归咎于投影代码失效。

### 4.4 早期 interface recovery pilot 告诉了我们什么

较早的五-seed recovery pilot 包含三个有效模型、两个 task 和两个 renderer/environment contracts，共 330 个有效 rows。它仍是 `PILOT_ONLY`，且 Safety-Gym 当时是 headless contract render，不是 native replication。

主要 matched consistency 为：

| 干预 | 一致率 |
|---|---:|
| JSON field order | 23/30 = 0.767 |
| structured vs free text | 21/30 = 0.700 |
| constraint label polarity | 27/30 = 0.900 |
| compatibility label polarity | 28/30 = 0.933 |
| constraint vs compatibility wording | 29/30 = 0.967 |
| explicit vs ambiguous applicability | 29/30 = 0.967 |
| marker-ID physical choice | 23/30 = 0.767 |
| candidate-order physical choice | 18/30 = 0.600 |

更重要的是，同一批 ambiguous applicability outputs 通过不同但都可辩护的 planner mappings 后，aggregate executed STC 可从 `0.50` 变化到 `0.80`；constraint interpretation 为 `0.70`。这正面证明了为什么不能把一个模糊字段配上唯一 evaluator mapping 后，就把结果称为模型的“安全能力分数”。

该 pilot 还观察到 Gemini 与 Qwen 在某些接口条件下发生相对排名反转，而 Mistral 是较稳定的异质性对照。不过它没有通过预注册的双环境语义效应门槛，因此不能声称普遍接口脆弱性或 native cross-environment replication。

### 4.5 当前最强的 native closed-loop scout 证据

冻结的 120-call scout 使用两个模型、两个 native environments、五个 paid seeds 和一个 `direct_path_intersection` family。120/120 response 都严格 parse 成功，随后以零 provider calls replay 到两个原生 controller。

| 层级 | 一致性 / IEC |
|---|---:|
| Parse consistency | 1.00 |
| Canonical semantic consistency | 0.70 |
| Grounding consistency | 0.35 |
| Planner-action IEC | 0.75 |
| Exact native action-sequence IEC | 0.55 |
| Exact native trajectory IEC | 0.55 |

这组数字支持三个重要判断：

1. 差异不是 parser failure 主导，因为 parse consistency 为 `1.00`；
2. 只评 `avoid/traverse/unknown` 会过度乐观，因为 planner IEC 为 `0.75`，真实 action/trajectory IEC 只有 `0.55`；
3. grounding 是关键风险面：semantic consistency 为 `0.70`，grounding consistency 进一步降到 `0.35`。

20 个 equivalent pairs 中，有 9 对改变了 exact action sequence 与 trajectory。按 pair 类型分解：

| Equivalent pair | Semantic | Grounding | Planner IEC | Native-action IEC | Trajectory IEC |
|---|---:|---:|---:|---:|---:|
| constraint ↔ compatibility | 0.25 | 0.25 | 0.25 | 0.00 | 0.00 |
| constraint polarity | 0.75 | 0.50 | 0.75 | 0.50 | 0.50 |
| structured ↔ free text | 1.00 | 0.00 | 1.00 | 0.25 | 0.25 |
| field order | 0.75 | 0.25 | 1.00 | 1.00 | 1.00 |
| compatibility polarity | 0.75 | 0.75 | 0.75 | 1.00 | 1.00 |

`structured ↔ free text` 是一个很清楚的案例：canonical semantics 与 planner action 完全相同，但 grounding 完全不一致，最后 native-action IEC 只有 `0.25`。这说明“文本答案正确”不足以代表 embodied execution 等价。

### 4.6 闭环安全结果

| Environment | Task success | STC | Collision | Semantic violation | Native-action IEC |
|---|---:|---:|---:|---:|---:|
| PointHazard native | 0.9000 | 0.6833 | 0.0000 | 0.2167 | 0.50 |
| Safety-Gym native | 0.9833 | 0.5333 | 0.4667 | 0.4500 | 0.60 |

Safety-Gym 的 task success 更高，但 STC 更低、碰撞和语义违规更多。这再次说明“到达 goal”不能替代安全指标。

当前 Contract-Induced Safety Range 为：

- `CISR-EQ = 0.15`：等价接口本身造成的平均 STC range；
- `CISR-MAP = 0.30`：ambiguous output 经过不同 planner interpretation 后造成的平均 STC range。

两者都越过预注册的 `0.10` continuation threshold。五阶段分析中，14 个不一致 equivalent pairs 的最早差异有 7 个出现在 grounding、6 个出现在 normalization、1 个出现在 planner。controller 的 executor rescue rate 为 `0.4717`：controller 有时能挽救上游错误，但不会稳定消除接口差异。

这些数字见：

- [`results/interface_contract_scout_native_analysis/REPORT.md`](../../../results/interface_contract_scout_native_analysis/REPORT.md)
- [归档的中文图文报告](INTERFACE_CONTRACT_SCOUT_REPORT_ZH.md)
- [`results/interface_contract_scout_native_analysis/ANALYSIS.json`](../../../results/interface_contract_scout_native_analysis/ANALYSIS.json)

## 5. `results/` 与历史 `outputs/` 中还有哪些旧实验

### 5.1 可用于当前故事线的 pilot / infrastructure evidence

| 目录 | 状态 | 仍可支持什么 | 不能支持什么 |
|---|---|---|---|
| `results/next_five_experiments/01_core_information_audit/` | PILOT_ONLY | recognition correctness 与 selected safety 有条件关联；P2 不单调 | 不能证明 privilege monotonicity 或跨模型泛化 |
| `results/next_five_experiments/02_marker_interface_robustness/` | PILOT_ONLY / TERMINATED | marker ID 与 candidate order 会改变物理选择；kill condition 已触发 | 不能恢复 marker-only 论文 |
| `results/next_five_experiments/03_modern_perception_substitution/` | PILOT_ONLY | 单个 operating point 上 detector geometry 减少 semantic violation，但降低 completion | 不能证明 detector 优于 VLM |
| `results/next_five_experiments/04_five_stage_twins/` | PILOT_ONLY | 暴露 applicability interface 问题，推动五阶段归因 | 不能证明 capability-applicability 是模型瓶颈 |
| `results/next_five_experiments/05_second_environment/` | INTEGRATION_EVIDENCE | Safety-Gym headless adapter 可运行，native cost 与 semantic channels 可分离 | 不能称为 Safety-Gym replication |
| `results/interface_contract_experiment_0/` | INFRA / PROVIDER_FREE | 两环境 executable bridge 可执行、可复现且不制造 fixture 差异 | 不是模型能力实验 |
| `results/interface_contract_combined_analysis/` | PILOT_ONLY | interface consistency、planner mapping range、局部 ranking reversal | 不是 native 双环境正式结果 |
| `results/interface_contract_scout_paid/` | PILOT_ONLY / COMPLETE | 保存冻结的 120 次真实模型调用、cache 与 ledger | 不能单独代表闭环结果 |
| `results/interface_contract_scout_native_analysis/` | PILOT_ONLY / CONTINUATION GATE PASSED | 当前最强的 native action、trajectory、STC 和五阶段传播证据 | 尚不是 VALIDATED 论文结果 |

### 5.2 已 invalidated 或 archived 的旧语义实验

以下产物仍在仓库中，但只能用于取证、回归或理解研究历史：

| 产物 | 状态 | 主要问题 |
|---|---|---|
| `outputs/semantic_pilot*` | INVALIDATED | semantic-zone fallback 可与 red hazard 重叠；旧 estimand 混杂 |
| `outputs/semantic_implicit_pilot*` | INVALIDATED | sampler 问题；instruction 与 commonsense 混用 |
| `outputs/semantic_bplus_*` | INVALIDATED | sampler 问题；B/B+ 与 oracle 的 router/enforcement 不正交 |
| `outputs/semantic_hetero*` | INVALIDATED | sampler 问题、n=5、缺现代 perception baseline |
| `outputs/vlm5_single_l1*`、`outputs/vlm5_l2*` | INVALIDATED | sampler 问题；L2 同时删除任务要求，并非纯 leakage 干预 |
| `outputs/invalidated/2026-07-11/offline_n100_*` | INVALIDATED | heuristic plumbing、VLM calls=0、sampler 问题 |
| `outputs/invalidated/2026-07-11/vlm20_*_heldout*` | INVALIDATED | held-out 20 中 7 个 zone-hazard overlap；router confound；输入 provenance 不完整 |
| old PIVOT/B+ results and figures | ARCHIVED | safety information leakage、router/enforcement/scene confounds |
| `zone_detector.py` self-test | ARCHIVED | 只验证 renderer-palette plumbing，不是现代 perception baseline |
| PointPush / direct VLA scaffolds | ARCHIVED | 只有环境、训练或执行 scaffold，不是当前科学主线 |

这些旧实验还有三种合法用途：

1. 验证 API、JSON parser、fallback 与 replay 管道；
2. 保存原始 “recognize-but-cross” response，供接口审计；
3. 作为 provenance/logger 和 regression tests 的固定输入。

它们不能被换标题、换图表或追加少量 seed 后重新包装为论文数字。

### 5.3 旧实验失效不只有 marker 问题

仓库记录了至少四个独立的失效源：

1. **Layout sampler**：旧 semantic-zone fallback 跳过合法性检查；复算 200 seeds 时有 61 个 zone 与 red hazard 重叠，held-out 100–119 中为 7/20。
2. **Router confound**：经典条件直接 plan to goal，B/B+ 使用 VLM subgoal 和不同的 MPC restart；success 差异不能只归因于 semantic source。
3. **Task-specification confound**：旧 L1→L2 同时删除“应该避开这种地形”的任务要求，测到的是 instruction removal，不是单一 privileged-information dose。
4. **Incomplete provenance**：旧 transcript 只有输出，缺 exact prompt、input image/hash、candidate metadata、model revision 与完整调用记录。

因此即使 marker 稳定性没有触发 kill condition，旧实验也仍需因 sampler、对照和 provenance 问题失效。

## 6. 当前论文贡献应该如何表述

### 6.1 不再使用的主张

以下表述应从摘要、introduction、图标题和实验结论中删除：

- “VLM reliably recognizes semantic hazards that classical planners cannot see.”
- “The model recognizes hazards but fails to determine norm applicability.”
- “Our VLM planner is safer/better than classical planning.”
- “A correct terrain name or rationale demonstrates grounded understanding.”
- “One canonical JSON schema yields the model's true safety score.”

### 6.2 当前可用的核心主张

在严格标注为 pilot 的前提下，当前证据可以支持：

1. 语义等价或近等价的 embodied interface contracts 会改变部分模型的 canonical decision 与 grounding；
2. grounding 差异即使不改变高层 `avoid/traverse` 标签，也会传播到 native action sequence 和 trajectory；
3. 模糊字段的合理 planner mappings 可以显著改变 executed STC；
4. controller 有时会 rescue 上游错误，但不会稳定吸收接口差异；
5. 单一 wrapper 下的模型分数可能掩盖 interface-conditioned uncertainty，应报告 consistency 与 safety envelope；
6. 效应具有模型、环境和接口类型异质性，稳健模型是重要对照，而不是反例。

### 6.3 方法论贡献

相较于“prompt sensitivity in robots”，更有价值的贡献是：

- 用 matched causal interventions 修改接口，而不修改场景物理事实；
- 将 parse、semantic normalization、grounding、planner action、native action、trajectory 与 outcome 分层；
- 用 detached evaluator 防止 truth geometry 泄漏到 controller；
- 报告 Interface Equivalence Consistency（IEC）；
- 报告 Contract-Induced Safety Range（CISR）；
- 报告 Ranking Stability Envelope（RSE），而不是为每个模型挑最有利 wrapper；
- 保存 exact image、prompt、schema、response、hash、provider revision、planner mapping 和 execution trace，使每个差异可以定位。

## 7. 当前证据边界与下一步 gate

### 7.1 当前状态

- marker-only 主线：`TERMINATED`；
- norm-applicability failure 故事：`REJECTED`；
- interface-conditioned safety：`PROMISING / PILOT_ONLY`；
- 120-call native scout：`COMPLETE`，科学 continuation criteria 已满足；
- 第二个 paid family：未授权；
- 完整 1,440-call design：未授权；
- semantic-safety `VALIDATED` results：当前为 0。

### 7.2 升级为正式论文主结果所需的核心条件

依据当前计划，至少需要：

1. 一个非 marker 的 semantic/output effect 在两个 native environments、两个模型上复现，IEC < `0.80`；
2. executed controller 上 CISR ≥ `0.15`，且 `unknown` 不能被暗中解释为 traverse；
3. rank reversal 或 rank-correlation drop 在 parse-compliant analysis 中仍成立；
4. 结果不能只由单一 provider、renderer 或 API schema rejection 驱动；
5. stage-wise attribution 能定位差异首先进入 perception、normalization、grounding、planner 还是 enforcement；
6. 扩展到第二个 preregistered scenario family，之后才考虑更大的正式 split；
7. 保持 provider-estimate/controller 与 evaluator truth 的权限隔离。

如果效应最终只剩 marker/order、只在 headless render 出现、被 native execution 完全吸收，或只改变 parse compliance 而不改变 normalized physical action，则应停止主会叙事并转为 workshop/technical report。

## 8. 如何理解这次转向

### 8.1 这是不是说明 VLM 没有语义价值

不是。当前结果只说明旧实验没有把“语义能力”从“接口依赖”中识别出来。VLM 仍可能为经典 planner 提供开放世界知识，但该主张必须通过 marker-free grounding、capability twins、现代 perception baselines、固定 planner/controller 和跨接口稳定性实验来验证。

### 8.2 这是不是单纯的 prompt sensitivity 论文

不应当是。字段顺序或输出格式影响语言模型并不新颖。这里必须证明的是闭环因果传播：

```text
equivalent contract
  → normalization / grounding divergence
  → planner/native action divergence
  → trajectory / semantic violation / STC divergence
```

如果只能展示文本输出变化，而不能展示物理执行和安全结果变化，当前主线的研究价值会明显下降。

### 8.3 为什么当前故事比“VLM 比经典 planner 好”更成熟

原故事依赖一个很强且难以排除 shortcut 的能力主张。当前故事把研究对象从单个模型的表面 accuracy，提升为整个 embodied safety pipeline 的可识别性与测量有效性。它既能解释旧失败，也能产生可复用的实验方法：以后无论比较 VLM、VLA、open-vocabulary detector 还是 classical planner，都必须先证明接口没有替被测系统决定答案。

## 9. 证据索引

建议按以下顺序阅读：

1. [`README.md`](../../../README.md)：仓库当前主线和总 claim boundary；
2. [`docs/RESULTS_REGISTRY.md`](../../../docs/RESULTS_REGISTRY.md)：所有结果的有效性状态；
3. [`results/next_five_experiments/02_marker_interface_robustness/REPORT.md`](../../../results/next_five_experiments/02_marker_interface_robustness/REPORT.md)：marker-only kill condition；
4. [`results/next_five_experiments/04_five_stage_twins/REPORT.md`](../../../results/next_five_experiments/04_five_stage_twins/REPORT.md)：旧 applicability 异常；
5. [`docs/INTERFACE_CONTRACT_MAINLINE.md`](../../../docs/INTERFACE_CONTRACT_MAINLINE.md)：拒绝旧故事与当前研究问题；
6. [`results/interface_contract_combined_analysis/REPORT.md`](../../../results/interface_contract_combined_analysis/REPORT.md)：早期 interface recovery pilot；
7. [`results/interface_contract_experiment_0/SUMMARY.json`](../../../results/interface_contract_experiment_0/SUMMARY.json)：双原生环境的 provider-free executable bridge；
8. [归档的中文图文报告](INTERFACE_CONTRACT_SCOUT_REPORT_ZH.md)：当时最完整的图文结果；
9. [归档的中文交接记录](INTERFACE_CONTRACT_HANDOFF_ZH.md)：当时的执行状态与授权边界；
10. [`legacy/docs/LEAKAGE_AND_LIMITATIONS.md`](../LEAKAGE_AND_LIMITATIONS.md)：更早 PIVOT 阶段的信息泄漏审计。

## 10. 最终研究叙事

可以把整个项目压缩成下面这段：

> 我们最初希望用 VLM 识别经典几何 planner 无法表达的语义危险，再由低层 MPC 安全执行。但 matched marker-ID 与 candidate-order 干预表明，模型的物理选择对符号接口高度敏感，因而旧结果不能证明稳定的语义 grounding。随后我们尝试把失败解释为 recognition–applicability gap，却发现 `applicable` 同时可表示“约束适用”与“地形可通行”，模型和 evaluator 采用了方向相反但都合理的解释。由此我们把研究问题推进到 embodied safety evaluation 的测量层：当视觉证据和物理场景不变时，字段语义、序列化、grounding 和 planner mapping 是否会改变 native action、trajectory 与 Safe Task Completion？当前 provider-free bridge 和 120-call native scout 已证明这种差异可以从 normalization/grounding 传播到闭环执行，但所有结果仍是 pilot；下一步必须在新的 preregistered family、更多模型与真正独立的 native task 中复现，才能形成正式论文主结果。
