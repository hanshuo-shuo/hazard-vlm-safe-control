# C³-Safe: Capability–Constraint Counterfactual Semantic Safety Distillation

当前唯一继续开发的科学主线是 **C³-Safe**。它把 VLM 限定为训练期、action-free 的
环境需求 teacher；测试时不加载 VLM，动作只由小型 constrained policy 输出。

当前状态：**`PLANNED / NOT RUN`**。仓库没有可继承的 VALIDATED semantic-safety 数字，
也没有现成的 C³-Safe checkpoint、teacher-label dataset 或通用 policy learner。

先读：

- [`RESEARCH_STORY.md`](RESEARCH_STORY.md)：新的 storyline，以及保留的旧 pilot 证据；
- [`docs/C3_SAFE_MAINLINE.md`](docs/C3_SAFE_MAINLINE.md)：协议、矩阵、baseline 和 Go/No-Go gates；
- [`docs/RESULTS_REGISTRY.md`](docs/RESULTS_REGISTRY.md)：所有 artifact 的状态真相源；
- [`docs/REPOSITORY_STATUS_2026-08-13.md`](docs/REPOSITORY_STATUS_2026-08-13.md)：保留/冻结/复用边界。

## 一句话 thesis

VLM 更适合告诉小模型“这个环境要求机器人具备什么”，而不是告诉机器人“下一步怎么走”。
先学习像素级空间需求场，再用 swept footprint 计算当前运动暴露；把能力不兼容和任务
规则分别计价，并用 motion twins 与 same-transition capability/rule counterfactuals
监督独立 semantic critic，可以学习一个无需 VLM 的组合泛化安全策略。

## 保留的 storyline 基础结果

旧结果只用于 motivation 和 failure analysis，不能当作 C³-Safe 的 baseline 或论文
headline：

| 证据 | 结果 | 状态 | 用途 |
|---|---:|---|---|
| marker-ID / candidate-order physical-choice consistency | 0.20 / 0.30 | `PILOT_ONLY / TERMINATED` | 说明旧 waypoint interface 不稳定 |
| 120-call scout：exact native trajectory consistency | 0.55；9/20 matched pairs 改变轨迹 | `PILOT_ONLY` | 说明 interface/grounding 差异可穿透闭环 |
| PointHazard pilot：success / STC | 0.90 / 0.68 | `PILOT_ONLY` | 说明到达率不等于安全 |
| Safety-Gym pilot：success / STC | 0.98 / 0.53 | `PILOT_ONLY` | 说明必须分开报告 native cost 与 semantic safety |

完整数字和限制见 [`RESEARCH_STORY.md`](RESEARCH_STORY.md) 与
[`docs/RESULTS_REGISTRY.md`](docs/RESULTS_REGISTRY.md)。

## 仓库分层

| 区域 | 当前角色 |
|---|---|
| `env_pointhazard.py`, `hazard_renderer.py`, `envs/` | C³-Safe 可复用的环境、观测边界和 adapter |
| `evaluation/capability_twins.py`, `semantic_evaluator.py`, `outcomes.py`, `schemas.py` | capability/rule truth、评测和 provenance 基础 |
| `evaluation/geometry_calibration.py`, `semantic_geometry.py` | 需求场投影、swept-footprint rasterization 与暴露 truth 的几何基础 |
| `safe_expert.py`, `mpc_expert.py` | transition 覆盖与 oracle/reference baseline |
| `docs/C3_SAFE_MAINLINE.md` | 空间需求场—运动暴露—关系代价主线的唯一实验协议 |
| `configs/c3_safe_mainline_manifest.json` | 机器可读的 planned scope；不是结果证明 |
| `results/`, `outputs/` | 历史/基础设施 artifact，状态以 registry 为准 |
| `legacy/` 和冻结的旧 top-level modules | provenance、回归和历史路线 |

旧 interface-contract 文件、旧 marker/PIVOT、PointPush 和 direct-VLA 不再接受新的
科学开发；因为历史 manifest/test 仍引用原路径，本次整理不移动、不删除它们。

## 当前实施边界

本次整理只完成主线协议、manifest、入口叙事和 artifact 边界；尚未实现训练器，也不新增
Torch/Transformers/SB3 等依赖。下一步先完成 Gate 0：layout invariants、两个 adapter、
expert coverage、projection calibration、swept-footprint/exposure truth、Safety-Gym
semantic step/evaluator 和完整 provenance。

## 旧回归测试

核心环境和历史 interface-contract 的 provider-free 回归仍可按原方式运行：

```bash
python -m pytest -q \
  tests/test_condition_contract.py \
  tests/test_interface_contract_protocol_v1.py \
  tests/test_interface_execution_bridge_v2.py \
  tests/test_interface_contract_scout.py
```

这些测试验证旧协议/回放链路，不等价于 C³-Safe 的科学验证。
