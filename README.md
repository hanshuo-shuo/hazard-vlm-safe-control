# C³-Safe: Capability–Constraint Counterfactual Semantic Safety Distillation

当前唯一继续开发的科学主线是 **C³-Safe**。它把 VLM 限定为训练期、action-free 的
环境需求 teacher；测试时不加载 VLM，动作只由小型 constrained policy 输出。

当前状态（2026-09-09）：**基础实现与 simulator-only 视觉学生已运行；VLM 蒸馏与策略学习尚未运行。**
已实现统一安全记账、两个环境的运动足迹评测、投影校准、开发数据采集和小 CNN 训练。
现有两个 checkpoint 是无 VLM 的空间识别对照；尚无 C³-Safe policy checkpoint、真实
teacher labels 或 `VALIDATED` semantic-safety 结果。

先读：

- [`docs/C3_FOUNDATION_PROGRESS.md`](docs/C3_FOUNDATION_PROGRESS.md)：本轮完成项、实际结果、复现命令和下一步；
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
| PointHazard pilot：success / 严格 STC | 54/60 / 41/60 | `PILOT_ONLY` | 说明到达率不等于安全 |
| Safety-Gym pilot：success / 严格 STC | 59/60 / 14/60 | `PILOT_ONLY / 口径重算` | 原 32/60 是 semantic-only completion，未排除 native cost |

完整数字和限制见 [`RESEARCH_STORY.md`](RESEARCH_STORY.md) 与
[`docs/RESULTS_REGISTRY.md`](docs/RESULTS_REGISTRY.md)。

## 仓库分层

| 区域 | 当前角色 |
|---|---|
| `c3_safe/` | 新的空间场、swept footprint、能力/规则代价、数据契约、oracle controller 和视觉学生 |
| `scripts/*c3*.py`, `configs/c3_safe_foundation.json` | 已运行的开发检查、采集、训练和对照入口 |
| `env_pointhazard.py`, `hazard_renderer.py`, `envs/` | C³-Safe 可复用的环境、观测边界和 adapter |
| `evaluation/capability_twins.py`, `semantic_evaluator.py`, `outcomes.py`, `schemas.py` | capability/rule truth、评测和 provenance 基础 |
| `evaluation/geometry_calibration.py`, `semantic_geometry.py` | 保留的历史几何审计；新运动暴露实现在 `c3_safe/geometry.py` |
| `safe_expert.py`, `mpc_expert.py` | transition 覆盖与 oracle/reference baseline |
| `docs/C3_SAFE_MAINLINE.md` | 空间需求场—运动暴露—关系代价主线的唯一实验协议 |
| `configs/c3_safe_mainline_manifest.json` | 正式方法的目标范围及实现状态；不是结果证明 |
| `results/`, `outputs/` | 历史/基础设施 artifact；新 `results/c3_*/` 本地保留，摘要与图进入 `docs/` |
| `legacy/` 和冻结的旧 top-level modules | provenance、回归和历史路线 |

旧 interface-contract 文件、旧 marker/PIVOT、PointPush 和 direct-VLA 不再接受新的
科学开发；因为历史 manifest/test 仍引用原路径，本次整理不移动、不删除它们。

## 当前实施边界

开发检查覆盖 12 个几何/关系案例、400 个随机运动和真实 MuJoCo 相机标记校准。
PointHazard 的 20 组同场景对照中，使用相同 A* + MPC 的盲规划 arm 与 oracle arm
都到达 20/20，严格 STC 分别为 0/20 和 20/20。这是数据与控制器检查，不是学习策略结果。

两组 11,739 参数的 RGB-only CNN 已用 simulator masks 训练。加入运动暴露监督后，
开发测试集 exposure MAE 从 0.839 降到 0.096，但 water IoU 从 0.475 降到 0.319，
且固定阈值下有 1/5 个危险样本漏检。数据只有水域、单训练 seed；不能宣称 Gate 1/2
通过、组合泛化或闭环安全。完整 Gate 0 仍缺真实 teacher provenance 与多地形数据覆盖。

运行入口（每次使用新的输出目录）：

```bash
python3 scripts/run_c3_foundation_checks.py --native --output results/c3_checks_local
python3 scripts/collect_c3_oracle_data.py --output results/c3_data_local
python3 scripts/train_c3_spatial_baseline.py --dataset results/c3_data_local \
  --exposure-weight 0 --output results/c3_student_local
```

环境依赖见 `requirements.lock`；训练额外使用 `requirements-learning.txt` 中的 PyTorch。
本机已有 Torch 2.5.1，本轮未下载 VLM 权重、未发起 provider calls。

## 回归测试

本轮完整测试 208 项通过，后续新增的 3 项视觉学生测试与 8 项数据回归也通过，合计覆盖
211 个不同测试。开发时可运行：

```bash
LAYOUT_TEST_SEEDS=25 python3 -m pytest -q tests
```

这里的 layout seeds 是快速回归规模，不替代正式大规模 sampler sweep。真实 native RGB
测试需要可用的图形上下文；测试通过不等价于科学 gate 通过。

核心环境和历史 interface-contract 的 provider-free 回归仍可按原方式运行：

```bash
python -m pytest -q \
  tests/test_condition_contract.py \
  tests/test_interface_contract_protocol_v1.py \
  tests/test_interface_execution_bridge_v2.py \
  tests/test_interface_contract_scout.py
```

这些测试验证旧协议/回放链路，不等价于 C³-Safe 的科学验证。
