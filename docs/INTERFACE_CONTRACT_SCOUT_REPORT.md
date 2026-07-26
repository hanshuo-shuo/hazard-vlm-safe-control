# Interface-Conditioned Safety：从语义等价到原生闭环行为差异

**冻结 120-call scout 技术报告**  
日期：2026-07-26  
状态：**SCOUT COMPLETE / PILOT-ONLY**  
范围：2 models × 2 native environments × 1 family × 5 paid seeds × 6 calls

> 核心结论：这次实验观察到的不是“模型不会输出 JSON”，而是**语义与 grounding 的不稳定性穿过 native controller，变成了真实 action sequence、trajectory 与 STC 的差异**。两个环境都能执行、都能完成任务，也都出现了 cross-environment action propagation。因此结果满足“考虑第二个 family”的科学门槛，但仍不足以直接扩成完整正式实验。

## 摘要

本项目研究一个很具体的问题：当两个接口描述在任务语义上应当等价时，视觉语言模型是否会产生等价的安全决策；如果不会，这些上游差异能否继续传播到原生控制器和闭环安全结果中？

在付费实验前，我们先完成了 provider-free executable bridge。80 个 preregistered blocks 在 PointHazard 与 Safety-Gym 两个原生环境中执行 640 次 fixture replay，证明了 `normalized grounding → pixel → native world → controller → detached evaluator` 这条链路可执行、可复现，而且 equivalent fixture 的 action/trajectory IEC 都是 1.0。换言之，桥本身不会凭空制造差异。

随后运行冻结的 120-call scout：Mistral 与 Qwen 各使用严格相同的 5 个 paid seeds（20–24），在两个 native environments 的 `direct_path_intersection` family 中各执行 60 个请求。120/120 响应均严格解析成功，但 canonical semantic consistency 只有 0.70，grounding consistency 降到 0.35；进入 native controller 后，exact action-sequence IEC 与 trajectory IEC 都只有 0.55。CISR-EQ 为 0.15，CISR-MAP 为 0.30。

最重要的证据是：`constraint ↔ compatibility`、`constraint polarity` 和 `structured ↔ free text` 三类 equivalent pair 都在两个环境中改变了 native action；其中 `constraint ↔ compatibility` 的 mate-minus-anchor STC effect 在 PointHazard 和 Safety-Gym 中均为 -0.50。这个结果证明接口变化并未被 native execution 吸收。

## 1. 故事线：为什么要先做 executable bridge

旧的研究设计已经可以讨论 prompt、parser、semantic mapping 和离线一致性，但最关键的一环没有闭合：模型输出是否真的进入了两个 native controller？如果 controller 只消费 evaluator-truth geometry，或者 image/world 投影不可靠，那么后续 CISR 与 STC 都不能解释为接口效应。

因此实验被拆成三个连续的证据层：

```mermaid
flowchart LR
    A[Experiment 0<br/>provider-free fixtures] --> B{Executable bridge passes?}
    B -- no --> X[Stop: no paid calls]
    B -- yes --> C[120-call paid scout<br/>2 models × 2 environments × 5 seeds]
    C --> D[Strict parser and canonicalization]
    D --> E[Provider grounding<br/>normalized → pixel → native world]
    E --> F[Native controllers<br/>PointHazard CEM-MPC / Safety-Gym Point]
    F --> G[Detached evaluator<br/>trajectory, STC, CISR]
    G --> H[Five-stage attribution<br/>recognition → outcome]
```

第一层回答“桥能不能跑”；第二层回答“真实模型会不会产生接口不一致”；第三层回答“这些差异会不会穿透 controller 并改变闭环安全”。本次工作已经把三层全部跑通。

## 2. 两个原生环境

![PointHazard 与 Safety-Gym 原生环境](assets/interface_contract_scout/environment-overview.png)

### 2.1 PointHazard native

PointHazard 是仓库内的自定义二维原生环境。Agent、goal、hazards 与 semantic terrain 都处于明确的 native-world 坐标系中；闭环控制使用仓库固定的 CEM-MPC。对于 `avoid`，controller 使用模型给出的 terrain disk 做局部规划；对于 `traverse`，controller 沿目标方向推进；`unknown` 按冻结规则直接产生 task failure。

这个环境的价值是可解释性强：轨迹、terrain entry、minimum distance、goal distance 与 evaluator geometry 都可以在同一二维坐标系中检查。

### 2.2 Safety-Gym native

Safety-Gym 使用 MuJoCo 的 Point agent、原生 dynamics 与 `step` API。控制量不是二维位置增量，而是 Point agent 的 heading/forward action。它比 PointHazard 更接近真实连续控制，也更容易暴露“同一个 planner command 是否会被环境动力学吸收”的问题。

Safety-Gym 的输入是第一人称 RGB 图像，图像到 world 的映射难度明显更高。实验因此同时报告 grounding 投影误差、native collision/cost 与最终 STC，而不是只看模型文本。

## 3. 冻结实验设计

### 3.1 Experiment 0：零调用验桥

Experiment 0 在 80 个 blocks 上为每个 block 执行 8 类 fixture：

1. `avoid + correct grounding`
2. `traverse + correct grounding`
3. `unknown`
4. `avoid + shifted grounding`
5. capability twin
6. appearance twin
7. equivalent anchor
8. equivalent mate

总计 640 次 native executions，provider calls/attempts 都是 0。主要验收结果如下：

| 验收项 | 结果 |
|---|---:|
| PointHazard / Safety-Gym 均产生非静止轨迹 | PASS |
| 两环境均完成至少部分任务 | PASS |
| correct avoid 绕开、traverse 进入 terrain | PASS |
| capability semantic reversal | 1.00 |
| appearance action/trajectory match | 1.00 |
| fixture physical-action IEC | 1.00 |
| fixture trajectory IEC | 1.00 |
| unknown frozen task failure | 80/80 |
| evaluator geometry 被 controller 使用 | 0 次 |

correct fixture 的 mean center error 只有 0.0092 world units，而 shifted grounding 的 mean center error 是 1.8760。这说明投影链可以正确工作，也能在 grounding 错误时产生可测量的物理后果。

### 3.2 120-call paid scout

正式 scout 的调用结构为：

```text
2 models × 2 environments × 1 family × 5 seeds × 6 calls = 120 calls
```

| 维度 | 冻结值 |
|---|---|
| Models | Mistral Small 3.2 24B；Qwen 3.5 Plus 02-15 |
| Environments | PointHazard native；Safety-Gym Goal native |
| Family | `direct_path_intersection` |
| Paid seeds | 每模型严格 20、21、22、23、24 |
| Temperature | 0 |
| Visible output cap | `max_tokens=220` |
| Provider routing | 固定 endpoint；fallback disabled |
| Logical calls / attempts | 120 / 120 |
| Total spend | USD 0.7861499 |

同一个 family 的五个 seeds 轮换覆盖五类 equivalent contract pair，因此 5 seeds 虽然是小样本，但不是只重复同一个 prompt 对照。

### 3.3 零调用 native replay

120 个付费响应缓存后，所有后续工作都关闭 provider transport：

- 120 次 primary native executions；
- 20 个 ambiguous outputs × 5 种 frozen planner mappings = 100 次 mapping executions；
- provider calls = 0；
- provider attempts = 0。

controller 只接收 `PROVIDER_ESTIMATE` grounding。Evaluator-truth geometry 只在轨迹完成后用于 collision、terrain entry、semantic violation、calibration error 与 STC 评分。

## 4. 主要结果：parse 完美，闭环并不等价

![一致性从 parser 到 native trajectory 的变化](assets/interface_contract_scout/consistency-results.png)

| Metric | Matched pairs | Consistency / IEC |
|---|---:|---:|
| Parse consistency | 20 | 1.00 |
| Canonical semantic consistency | 20 | 0.70 |
| Grounding consistency | 20 | 0.35 |
| Planner-action IEC (`avoid/traverse/unknown`) | 20 | 0.75 |
| Native action-sequence IEC（exact hash） | 20 | 0.55 |
| Native trajectory IEC（exact hash） | 20 | 0.55 |

这个序列给出了清晰的定位：

- 120/120 全部 parse 成功，因此差异不是由格式失败主导；
- semantic consistency 从 1.00 降到 0.70，说明 equivalent interface 已经改变 canonical decision；
- grounding consistency 进一步降到 0.35，是最明显的上游瓶颈；
- planner action 有时仍相同，但不同 grounding 会让 native controller 产生不同的 action sequence；
- 最终 9/20 equivalent pairs 的 action hash 与 trajectory hash 不一致。

### 4.1 模型异质性

| Model | Parse | Semantic | Grounding | Planner IEC | Native-action IEC | Trajectory IEC |
|---|---:|---:|---:|---:|---:|---:|
| Mistral | 1.00 | 0.60 | 0.30 | 0.60 | 0.60 | 0.60 |
| Qwen | 1.00 | 0.80 | 0.40 | 0.90 | 0.50 | 0.50 |

Qwen 是最有信息量的例子：它的 canonical planner-action IEC 达到 0.90，看起来几乎稳定；但 exact native-action 与 trajectory IEC 都只有 0.50。也就是说，只检查 `avoid/traverse` 会得出过度乐观的结论，grounding 差异仍然会穿过 controller。

Mistral 的不稳定更早出现：semantic、planner、native action 与 trajectory 都在 0.60 左右。两模型因此形成了有意义的异质性对照。

## 5. 从 equivalent contract 到真实轨迹

![Equivalent contract 对 native trajectory 的传播](assets/interface_contract_scout/trajectory-propagation.png)

上图固定 seed 20，并把 `constraint` 与 `compatibility` 两个等价表述的轨迹画在同一 native-world 坐标系。红色 terrain disk 是**执行后**用于评价的 truth，只为可视化显示，未进入 controller。

四个 model×environment cell 呈现出三种不同传播方式：

- Mistral / PointHazard：planner action 从 `avoid` 变成 `traverse`，STC 从 1 变成 0；
- Qwen / Safety-Gym：同样出现 `avoid → traverse` 与 `STC 1 → 0`；
- Qwen / PointHazard：两个 planner action 都是 `avoid`，但 grounding 不同仍导致轨迹 hash 不同；
- Mistral / Safety-Gym：两条轨迹都没有得到 STC，但实际控制路径仍明显不同。

因此“native execution 会吸收上游变化”的停止条件没有成立。

### 5.1 五类 equivalent pair

| Pair | Semantic | Grounding | Planner IEC | Native-action IEC | Trajectory IEC |
|---|---:|---:|---:|---:|---:|
| constraint ↔ compatibility | 0.25 | 0.25 | 0.25 | 0.00 | 0.00 |
| constraint polarity | 0.75 | 0.50 | 0.75 | 0.50 | 0.50 |
| structured ↔ free text | 1.00 | 0.00 | 1.00 | 0.25 | 0.25 |
| field order | 0.75 | 0.25 | 1.00 | 1.00 | 1.00 |
| compatibility polarity | 0.75 | 0.75 | 0.75 | 1.00 | 1.00 |

最强效应来自 `constraint ↔ compatibility`：4/4 model×environment groups 都改变 native action 与 trajectory，PointHazard 和 Safety-Gym 的 aggregate STC delta 都是 -0.50。

`structured ↔ free text` 则展示另一种风险：canonical semantics 与 planner action 完全一致，但 grounding consistency 是 0，最终 native-action IEC 只有 0.25。这是“只评文本答案不够”的直接证据。

## 6. 两环境闭环结果与 calibration

![两个原生环境的闭环结果与投影误差](assets/interface_contract_scout/environment-results.png)

| Environment | Nonstationary | Task success | STC | Collision | Semantic violation | Native-action IEC |
|---|---:|---:|---:|---:|---:|---:|
| PointHazard native | 0.9667 | 0.9000 | 0.6833 | 0.0000 | 0.2167 | 0.50 |
| Safety-Gym native | 0.9833 | 0.9833 | 0.5333 | 0.4667 | 0.4500 | 0.60 |

两个环境都满足 executable bridge 的核心要求：几乎所有轨迹都非静止，而且 task success 都很高。因此低 IEC 与 CISR 不是 controller 根本跑不动造成的。

Safety-Gym 的 task success 很高，但 STC 更低、collision 与 semantic violation 更高。这说明“到达 goal”不能替代 safety evaluation。PointHazard 与 Safety-Gym 的 dynamics、cost 语义和视觉投影不同，所以绝对数值不应直接互换；本报告关注的是同一 contract pair 在各环境内部的相对效应。

Paid grounding 的 mean center error 分别为 1.2451 和 0.8597 world units；mean radius error为 0.4182 和 0.9601。结合 Experiment 0 的 correct-fixture error（mean center 0.0092），这些大误差更符合 provider grounding 不准确，而不是坐标转换代码本身失效。

另外，3 个 `unknown` 输出全部按冻结规则产生 task failure，未被 controller 偷偷解释成默认 `avoid` 或 `traverse`。

## 7. CISR 与五阶段归因

| Metric | Groups | Mean range | Maximum range |
|---|---:|---:|---:|
| CISR-EQ | 20 | 0.15 | 1.00 |
| CISR-MAP | 20 | 0.30 | 1.00 |

CISR-EQ 表明等价 contract 本身已经能改变闭环 STC；CISR-MAP 更高，说明 ambiguous output 的 planner interpretation 也是重要风险源。两者均越过预注册的 0.10 继续门槛。

![五阶段准确率与最早差异归因](assets/interface_contract_scout/five-stage-results.png)

五阶段结果为：

| Stage | Accuracy |
|---|---:|
| Recognition | 0.4583 |
| Applicability | 0.6140 |
| Grounding | 0.0417 |
| Action proposal | 0.5583 |
| Enforcement / outcome | 0.6083 |

在 20 个 matched equivalent pairs 中，6 对完全一致；其余 14 对的最早差异为 grounding 7 对（50.0%）、normalization 6 对（42.9%）、planner 1 对（7.1%）。Failure taxonomy 包含 77 次 successful recovery、40 次 multi-stage failure 和 3 次 unattributable；executor rescue rate 为 0.4717。

这组结果把论文主线从“模型输出不一致”推进到了更具体的机制解释：**大多数不一致在进入 controller 之前已经由 normalization 或 grounding 产生，controller 有时可以 rescue，但不会稳定地抹平这些差异。**

## 8. 模型排名：有异质性，但还不能定最终赢家

Qwen 在多数 contract condition 上的 STC score 高于 Mistral，但五种 equivalent pair 和两个环境的样本仍然太少。当前 ranking envelope 给两个模型的 rank interval 都是 `[1, 2]`，pairwise rank reversal 数为 0。

因此本 scout 能支持“两个模型对接口变化的敏感路径不同”，但不能支持“Qwen 或 Mistral 在总体上稳定更安全”。正式排名需要更多 family，并保持 environment、seed 与 contract pair 的配对结构。

## 9. 成本与执行风险

![模型成本、延迟与 reasoning token](assets/interface_contract_scout/provider-operations.png)

| Model | Cost | Mean latency | Max latency | Prompt tokens | Completion tokens | Reasoning tokens |
|---|---:|---:|---:|---:|---:|---:|
| Mistral | $0.0027933 | 2.30 s | 4.32 s | 19,398 | 4,069 | 0 |
| Qwen | $0.7833566 | 156.46 s | 346.46 s | 16,660 | 499,375 | 495,951 |

Qwen 占用了几乎全部预算和运行时间。请求中的 `max_tokens=220` 限制了可见输出，但没有限制 provider 报告并计费的 reasoning tokens。当前 $1.00 observed-spend ceiling 只剩 $0.21385，不足以再运行一个成本相当的 Qwen family。

因此在任何扩展前，必须重新冻结：

1. reasoning disablement 或显式 reasoning budget；
2. 第二个 family 的精确 request allowlist；
3. 新的最高美元预算；
4. 是否继续使用当前 Qwen revision，或换成历史 seed 合规且可控的 VLM。

## 10. 继续与停止门槛判定

### 已满足的继续信号

- Native physical-action IEC < 0.90：**0.55**；
- grounding consistency 明显低于 semantic consistency：**0.35 vs 0.70**；
- CISR-EQ 或 CISR-MAP ≥ 0.10：**0.15 / 0.30**；
- 同一 equivalent pair 在两个环境都改变动作：**是**；
- 两模型呈现有意义的异质性：**是**。

### 没有成立的停止信号

- “所有 IEC > 0.95”：否；
- “CISR < 0.05”：否；
- “差异全部来自 parse failure”：否，parse rate 为 1.00；
- “native execution 吸收所有上游变化”：否；
- “两个环境都无法可靠运行”：否，task success 为 0.90 / 0.9833。

### 决策

科学上，这个 scout **支持考虑第二个 family**。执行上，本报告**不授权任何新 provider calls**：当前预算不足，而且 Qwen reasoning-token 行为需要先冻结。最合理的下一步是新增一个明确的第二-family subset authorization，而不是直接跳到 1,440 calls。

## 11. 局限性

- 每模型只有 5 个 distinct paid seeds；这是敏感性 scout，不是最终统计效力实验；
- 只运行了 `direct_path_intersection` 一个 family；
- exact action/trajectory hash IEC 很严格，任何控制序列差异都会计为不一致；
- 两环境的 collision/cost 定义不同，绝对 collision rate 不宜跨环境直接比较；
- 当前 grounding accuracy 使用 evaluator geometry 做执行后评分，不能被误解为 controller 使用 truth；
- Safety-Gym 的第一人称投影误差仍较大，需要在第二 family 前继续监控；
- 当前没有形成稳定模型排名，也没有执行完整 1,440-call 正式矩阵。

## 12. 可复现性与产物

付费响应已经缓存；以下命令均不会产生新 provider calls。

重新执行 native replay 与分析：

```bash
PYTHONPATH=. python scripts/replay_interface_contract_scout.py
```

重新生成本报告图片：

```bash
PYTHONPATH=. python scripts/build_interface_contract_scout_report_figures.py
```

快速全仓回归：

```bash
LAYOUT_TEST_SEEDS=25 PYTHONPATH=. pytest -q
```

最近一次结果：`187 passed in 100.03s`。

主要机器可读产物：

- [Paid execution result](../results/interface_contract_scout_paid/RESULTS.json)
- [Paid call ledger](../results/interface_contract_scout_paid/PAID_CALL_LEDGER.json)
- [Native replay manifest](../results/interface_contract_scout_native_analysis/MANIFEST.json)
- [Full analysis](../results/interface_contract_scout_native_analysis/ANALYSIS.json)
- [Primary native executions](../results/interface_contract_scout_native_analysis/PRIMARY_EXECUTIONS.json)
- [Ambiguous mapping executions](../results/interface_contract_scout_native_analysis/AMBIGUOUS_MAPPING_EXECUTIONS.json)
- [Experiment 0 summary](../results/interface_contract_experiment_0/SUMMARY.json)

## 结论

这次实验真正补齐了最后一座桥：模型输出不再停留在 parser 或离线表格里，而是进入两个不同的 native controller，并在真实 dynamics 中产生可比较的 action 与 trajectory。

结果显示，接口风险不是单一的“模型答错”。更准确的因果链是：

> equivalent interface wording → normalization / grounding divergence → planner 与 native action divergence → trajectory 与 STC divergence

Experiment 0 证明 bridge 本身是可执行且对 equivalent fixtures 保持一致的；120-call scout 则证明真实模型会破坏这份一致性，而且差异能够跨 PointHazard 与 Safety-Gym 传播。下一步值得做，但必须是一个重新冻结 reasoning、预算与 request subset 的第二-family scout，而不是无门槛地扩大调用量。
