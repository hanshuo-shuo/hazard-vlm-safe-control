# Repository status — 2026-08-13

> 本文保留 2026-08-13 的仓库快照。2026-09-09 已完成的基础实现、开发数据和小 CNN
> 对照见 [`C3_FOUNDATION_PROGRESS.md`](C3_FOUNDATION_PROGRESS.md)；下文的未实现状态
> 描述当时情况。

这是一份仓库入口和 artifact 处置索引，不是第二个结果注册表。结果状态唯一以
[`RESULTS_REGISTRY.md`](RESULTS_REGISTRY.md) 为准，C³-Safe 的冻结范围以
[`C3_SAFE_MAINLINE.md`](C3_SAFE_MAINLINE.md) 和
[`../configs/c3_safe_mainline_manifest.json`](../configs/c3_safe_mainline_manifest.json) 为准。

## 当前主线

**C³-Safe** 已选为唯一继续开发的科学主线，但目前是 `PLANNED / NOT RUN`。
仓库暂无可用于主线结论的 checkpoint、teacher-label dataset、policy learner 或
VALIDATED semantic-safety result。

## 保留的 storyline 基础证据

这些结果保留，是为了说明研究问题为什么从“VLM 识别/选择危险”收紧到“训练一个可组合的
空间需求—运动暴露—能力/规则关系安全策略”；它们都带有明确的 pilot/invalidated 边界：

| 证据 | 保留用途 | 状态 |
|---|---|---|
| marker ID / candidate order instability | 说明旧 waypoint interface 不能支撑稳定 semantic grounding | `PILOT_ONLY / TERMINATED` |
| `9/20` equivalent interface pairs 改变 exact native trajectory | 说明 text/interface 差异可穿透闭环，作为 motivation/failure analysis | `PILOT_ONLY` |
| PointHazard/Safety-Gym pilot 的 success–STC 分离 | 说明不能只报告到达率 | `PILOT_ONLY` |
| provider-free executable bridge | 说明 replay/evaluator 链路可回放，不是算法结果 | `INFRA` |

入口叙事见 [`../RESEARCH_STORY.md`](../RESEARCH_STORY.md)，细节和原始 artifact 路径见
[`RESULTS_REGISTRY.md`](RESULTS_REGISTRY.md)。

## 物理路径策略

本次整理不移动、不删除已有结果和旧源码。原因是多个 manifest、测试和结果 JSON 记录了
原始路径与 hash；改名会破坏历史 provenance。冻结路线通过文档边界表达，而不是通过把
旧文件伪装成新的主线代码表达。

### 复用但必须重跑

- `env_pointhazard.py`、`hazard_renderer.py`；
- `envs/protocol_env.py`、`envs/point_hazard_adapter.py`、`envs/safety_gym_goal_adapter.py`；
- `evaluation/capability_twins.py`、`evaluation/semantic_evaluator.py`、`evaluation/outcomes.py`、
  `evaluation/schemas.py`；
- `evaluation/geometry_calibration.py`、`evaluation/semantic_geometry.py`；
- `safe_expert.py`、`mpc_expert.py`。

### 冻结，仅作历史/回归

- `evaluation/interface_contract*`、`interface_execution.py`、`interface_metrics.py`、
  `five_stage_audit.py`；
- `pivot_vlm.py`、`subgoal_pivot_hazard.py`、PointPush 与 direct-VLA scaffolds；
- 旧 `outputs/`、`outputs/invalidated/`、`results/interface_contract_*` 与旧 figures。

## 下一步边界

先完成 Gate 0 的 layout、adapter、geometry、swept-footprint/exposure truth、provenance
和 Safety-Gym semantic step 修复，再新增 spatial teacher-label schema 与学生训练代码。
当前不新增付费 VLM run，不把旧 pilot 数字升级成 baseline，也不安装
Torch/Transformers/SB3 等尚未需要的训练依赖。
