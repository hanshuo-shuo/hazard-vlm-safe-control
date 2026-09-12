# Documentation Map

Updated: 2026-09-12.

## Current reading

- [`RELLIS_REVISION_PROGRESS_2026-09-12.md`](RELLIS_REVISION_PROGRESS_2026-09-12.md)：有预算上限的接口修订、240 原始扫描核查；独立误差证据仍不足，在新 48 帧前提前停止；
- [`RELLIS_EXTERNAL_PROGRESS_2026-09-12.md`](RELLIS_EXTERNAL_PROGRESS_2026-09-12.md)：完整诊断论文重写与首轮 24 帧真实数据几何资格检查；当前接口未过门槛，未进入外部训练；
- [`../paper/visual_risk_diagnosis/manuscript.md`](../paper/visual_risk_diagnosis/manuscript.md)：当前完整英文正文；
- [`PRO_DECISION_ROUND_PROGRESS_2026-09-12.md`](PRO_DECISION_ROUND_PROGRESS_2026-09-12.md)：最新 Pro 三实验推进，测量/后处理诊断、20 个匹配模型与独立风险认证；
- [`../research/pro_decision_round_20260912/protocol.json`](../research/pro_decision_round_20260912/protocol.json)：本轮冻结的风险目标、数据分工和停止条件；
- [`C3_CONFIRMATION_PROGRESS_2026-09-12.md`](C3_CONFIRMATION_PROGRESS_2026-09-12.md)：最新 15 模型随机布局确认实验、五种子统计、零覆盖校准瓶颈及下一步；
- [`../paper/compositional_visual_risk/CURRENT.md`](../paper/compositional_visual_risk/CURRENT.md)：论文当前材料入口，区分原始草稿和确认实验 addendum；
- [`C3_PAPER_PROGRESS_2026-09-12.md`](C3_PAPER_PROGRESS_2026-09-12.md)：新增六模型配对实验、位置依赖的修复、公式对照与英文论文草稿；
- [`../paper/compositional_visual_risk/manuscript.md`](../paper/compositional_visual_risk/manuscript.md)：完整英文研究草稿，明确保留开发诊断与正式证据的边界；
- [`C3_QUEST_PROGRESS_2026-09-12.md`](C3_QUEST_PROGRESS_2026-09-12.md)：Quest 资源、真实 VLM 检查、当前标签问题及远端分支补充进展；
- [`C3_QUEST_RESULTS_2026-09-12.json`](C3_QUEST_RESULTS_2026-09-12.json)：模型版本、作业、131 次调用和旧 Quest 结果的数值索引；
- [`C3_FOUNDATION_PROGRESS.md`](C3_FOUNDATION_PROGRESS.md)：本轮已执行工作、实际结果、复现命令与下一步资源；
- [`C3_FOUNDATION_RESULTS_2026-09-09.json`](C3_FOUNDATION_RESULTS_2026-09-09.json)：小型数值证据与本地完整产物的 hash；
- [`../RESEARCH_STORY.md`](../RESEARCH_STORY.md)：C³-Safe 的研究故事，以及被保留的旧 pilot 证据；
- [`C3_SAFE_MAINLINE.md`](C3_SAFE_MAINLINE.md)：主线定义、数据契约、实验矩阵、baseline 和 gates；
- [`../configs/c3_safe_mainline_manifest.json`](../configs/c3_safe_mainline_manifest.json)：planned scope；
- [`REPOSITORY_STATUS_2026-08-13.md`](REPOSITORY_STATUS_2026-08-13.md)：仓库整理和 archive boundary；
- [`RESULTS_REGISTRY.md`](RESULTS_REGISTRY.md)：所有结果/基础设施 artifact 的状态真相源。

## Truth sources

| Document | Authority |
|---|---|
| `../research/pro_decision_round_20260912/protocol.json` | 本轮三实验的执行定义与统计筛选；不等同于原 C³-Safe 正式方法 gate |
| `C3_SAFE_MAINLINE.md` | C³-Safe 当前 protocol 和 go/no-go gates |
| `../configs/c3_safe_mainline_manifest.json` | C³-Safe planned machine-readable scope |
| `RESULTS_REGISTRY.md` | `VALIDATED` / `PILOT_ONLY` / `INVALIDATED` / `ARCHIVED` 状态 |
| `C3_FOUNDATION_PROGRESS.md` | 已实现范围与开发运行结果；不改变科学 gate 阈值 |
| `review_status_registry.json` | 旧 review item 的 machine-readable status |

## Reusable foundation

- `../c3_safe/`：当前空间场、运动暴露、能力/规则代价、数据契约、oracle controller 与小 CNN；
- `../scripts/*c3*.py`：几何检查、配对数据采集、无 VLM 空间模型训练和对照；
- `../env_pointhazard.py`、`../hazard_renderer.py`：PointHazard dynamics/rendering；
- `../envs/`：policy/evaluator separation 与 native adapters；
- `../evaluation/capability_twins.py`、`semantic_evaluator.py`、`outcomes.py`、`schemas.py`：
  twin truth、评测与 provenance；
- `../evaluation/geometry_calibration.py`、`semantic_geometry.py`：空间需求场投影、
  swept-footprint rasterization 与运动暴露 truth；
- `../safe_expert.py`、`../mpc_expert.py`：transition coverage/reference baseline。

## Retained storyline evidence

旧 interface-contract 报告不再是主线，但保留为动机与失败分析：

- native scout：[`../results/interface_contract_scout_native_analysis/REPORT.md`](../results/interface_contract_scout_native_analysis/REPORT.md)；
- combined pilot：[`../results/interface_contract_combined_analysis/REPORT.md`](../results/interface_contract_combined_analysis/REPORT.md)；
- provider-free bridge：[`../results/interface_contract_experiment_0/SUMMARY.json`](../results/interface_contract_experiment_0/SUMMARY.json)；
- marker termination：[`../results/next_five_experiments/02_marker_interface_robustness/REPORT.md`](../results/next_five_experiments/02_marker_interface_robustness/REPORT.md)；
- 完整中文历史叙事：[`../legacy/docs/advisor_rewrite_2026-07/RESEARCH_STORYLINE_EVOLUTION_ZH.md`](../legacy/docs/advisor_rewrite_2026-07/RESEARCH_STORYLINE_EVOLUTION_ZH.md)。

所有这些 artifact 的状态都不是 `VALIDATED`，引用数字时必须同时写状态和实验用途。

## Frozen historical documents

`PROTOCOL.md`、`ICLR_PLAN.md`、`INTERFACE_CONTRACT_MAINLINE.md` 和
`RESEARCH_REVIEW_COMMENTS.md` 保留在原路径，因为旧 manifest/test 可能依赖其内容或 hash。
它们是历史协议记录，不是 C³-Safe 的入口。

## Verification

核心环境和旧 provider-free replay 测试仍按仓库根目录 README 中的命令运行；这些测试
验证历史基础设施，不等价于 C³-Safe 训练或科学 gate。

新增几何/成本、分组泄漏防护与 checkpoint 测试在 `../tests/test_c3_*.py`。
训练可选依赖为 `../requirements-learning.txt`；本轮的真实检查和测试范围见进度文档。
