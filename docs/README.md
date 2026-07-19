# docs/ 索引 — 先读这个

更新于 2026-07-17。这个仓库经历了多次方向纠偏；下表区分当前战略、待重跑结果和历史材料。

项目当前一句话：

> 对 VLM-guided control 做闭环安全信息归因，区分合法任务规格、视觉识别、空间 grounding、特权提示、routing 与低层执行分别贡献了多少安全性。

---

## 🟢 ACTIVE — 当前真相源

| 文档 | 是什么 |
|---|---|
| [PROTOCOL.md](PROTOCOL.md) | **实验语义真相源（v1）。** 冻结信息类型、capability twins、task spec、P0–P4、factor schema、evaluator、replay 切换语义与 seed split；实现和正式 run 必须服从它。 |
| [RESEARCH_REVIEW_COMMENTS.md](RESEARCH_REVIEW_COMMENTS.md) | **修改清单。** 按 BLOCKER/MAJOR review comment 写明位置、问题、要求修改与验收条件。先处理 B01–B07，未解决前暂停 paid scaling。 |
| [ICLR_PLAN.md](ICLR_PLAN.md) | **新战略主文档。** 覆盖 2026-06-29 旧计划；主线改为 Safety Accounting / causal attribution，包含 protocol、factorial design、实验矩阵、go/no-go gate、ICLR/CoRL 分流与 definition of done。 |
| [CLAUDE_PLAN.md](CLAUDE_PLAN.md) | **30 天执行计划（2026-07-13 → 2026-08-10）。** 把 ICLR_PLAN Phase 0–3 展开成按周/按日、带验收条件的工作包；含"继续 vs pivot"的战略结论、预算上限与 Day-30 决策树。 |
| [MEMO_01.md](MEMO_01.md) | **W1 gate memo #1（2026-07-19）。** 记录 protocol v1.2.1 冻结、因子正交 snapshot 与测试验收。 |
| [RESULTS_REGISTRY.md](RESULTS_REGISTRY.md) | **结果状态真相源。** 区分 VALIDATED / PILOT_ONLY / INVALIDATED / ARCHIVED，记录失效原因、归档位置和新结果准入条件。 |

**一句话现状：** 现有代码是强 pilot，但 semantic-zone sampler、MPC safety claim 和 router baseline 存在阻断性问题；旧 semantic n=5/n=20 只能作 debugging evidence，修复 protocol 前不再扩量。

---

## 🟠 待重跑结果（不要作为论文数字）

| 文档 | 状态 |
|---|---|
| [RESULTS_FAIR_BASELINES.md](RESULTS_FAIR_BASELINES.md) | held-out n=20 与 transcript 对调试很有价值，但场景 sampler 会生成 zone-hazard overlap，且 C2/CV 与 B+ 的 router 不一致。结果必须在 B01–B07 修复后重跑。 |
| [RESULTS_SEMANTIC.md](RESULTS_SEMANTIC.md) | 早期 semantic pilot；旧 oracle/B+ headline 已被推翻，且场景生成器问题会影响统计。 |
| [RESULTS_AMPLIFY.md](RESULTS_AMPLIFY.md) | 旧“放大 VLM 优势”实验；不能继续引用 oracle 对比。 |
| [RESULTS_MONTH1.md](RESULTS_MONTH1.md) | pure-geometry 早期信心实验，只能作历史 pilot。 |

---

## 🟡 已被新计划取代

| 文档 | 状态 |
|---|---|
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | 2026-06-18 的旧六个月路线，已被新版 ICLR_PLAN 取代。 |
| [WHY_VLM.md](WHY_VLM.md) | 旧对外叙事，仍含过强的安全与 VLM 优势表述。修改要求见 review comments。 |
| [FAILURE_MODE_ANALYSIS.md](FAILURE_MODE_ANALYSIS.md) | 旧 learned-physics 阶段的泄漏发现，对当前路线仍有 motivation 价值。 |
| [EXPERIMENT_NOTES.md](EXPERIMENT_NOTES.md) | 旧 PIVOT 实验记录，结论受泄漏影响。 |
| [VLM_PHYSICS_LITERATURE_REVIEW.md](VLM_PHYSICS_LITERATURE_REVIEW.md) | learned-physics 阶段文献调研；当前竞品与路线以新版 ICLR_PLAN 为准。 |

---

## 代码与历史

- 代码地图：[../STRUCTURE.md](../STRUCTURE.md)
- 当前主 harness：[../subgoal_pivot_hazard.py](../subgoal_pivot_hazard.py)
- 环境：[../env_pointhazard.py](../env_pointhazard.py)、[../env_pointpushhazard.py](../env_pointpushhazard.py)
- 低层控制器：[../mpc_expert.py](../mpc_expert.py)
- 历史归档：[../legacy/](../legacy/)

---

## 推荐阅读顺序

1. PROTOCOL：先看当前实验语义与不可更改的判定规则；
2. RESEARCH_REVIEW_COMMENTS：再看什么必须修；
3. ICLR_PLAN：理解新研究问题和完整路线；
4. CLAUDE_PLAN：本月每天做什么、做到什么算完成；
5. RESULTS_REGISTRY：确认哪些数字还能用；
6. RESULTS_FAIR_BASELINES：理解旧 pilot 暴露了哪些机制，但不要把数字当最终结果；
7. STRUCTURE：定位需要修改的代码。
