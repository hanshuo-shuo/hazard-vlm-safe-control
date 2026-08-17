# Documentation Map

Updated: 2026-08-13.

## Current reading

- [`../RESEARCH_STORY.md`](../RESEARCH_STORY.md)：C³-Safe 的研究故事，以及被保留的旧 pilot 证据；
- [`C3_SAFE_MAINLINE.md`](C3_SAFE_MAINLINE.md)：主线定义、数据契约、实验矩阵、baseline 和 gates；
- [`../configs/c3_safe_mainline_manifest.json`](../configs/c3_safe_mainline_manifest.json)：planned scope；
- [`REPOSITORY_STATUS_2026-08-13.md`](REPOSITORY_STATUS_2026-08-13.md)：仓库整理和 archive boundary；
- [`RESULTS_REGISTRY.md`](RESULTS_REGISTRY.md)：所有结果/基础设施 artifact 的状态真相源。

## Truth sources

| Document | Authority |
|---|---|
| `C3_SAFE_MAINLINE.md` | C³-Safe 当前 protocol 和 go/no-go gates |
| `../configs/c3_safe_mainline_manifest.json` | C³-Safe planned machine-readable scope |
| `RESULTS_REGISTRY.md` | `VALIDATED` / `PILOT_ONLY` / `INVALIDATED` / `ARCHIVED` 状态 |
| `review_status_registry.json` | 旧 review item 的 machine-readable status |

## Reusable foundation

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
