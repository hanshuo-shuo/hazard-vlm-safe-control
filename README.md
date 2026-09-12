# C³-Safe research: visual risk measurement and safe acceptance

当前研究重点是**视觉风险的测量、后处理和安全接受**：按照 Pro 的三实验建议，判断为何
排序泛化改善没有转化为可用的风险接口。原始 C³-Safe 的 VLM 蒸馏与策略学习设想保留，
本轮不以加入 VLM、关系头或 RL 模块作为贡献目标。

最新推进（2026-09-12）：**唯一同信息重建对照完成，九页 ICLR 2027 正文工作稿已成稿。**
同一粗场下，固定 PLIC 恢复完美 target64 的 44/54 个丢失安全机会，同时新增 16 个危险接受。
五模型轨道恢复 83 个 model–scene 决策、新增 43 个危险接受；接受中危险率 1.03%→1.56%。
这说明当前均匀聚合未利用完粗信息，也说明重建不能自动获得安全保证。
追加比较只消费已经用过的数据集，没有第三批确认、新训练或新认证。完整偏移开发曲线、
有害事件及 25/43 个具体抵消模式保留；不宣称 PLIC 在相同风险下优于任意标量修正。
稿件为 9 页正文、13 页总计；论文源码、PDF 与图表均在 Quest 编译、验证并逐页检查。
作者名单、导师审稿及完整匿名实验补充包仍待落实；尚未提交 OpenReview。

上一轮：**同源 oracle 聚合误差分解与一次新 IID 确认完成。**
固定同一 512 参考足迹，在已消费的 4,000 场景发现实际决策信号，再冻结 2,000 新场景验证。
新确认中，完美 coarse oracle 丢失 63/2,000 安全机会，完美实际监督目标 oracle 仍丢失
54/2,000；模型固定动作的池化标签变化 3.22%，同足迹数值敏感性 0.15%。保守方向和
边界交叠/小裕度富集得到预先指定检验支持。没有新增训练、重做认证或搜索新数据集。
论文主线转向“空间目标到动作暴露的接口误差”，不把基本协方差恒等式称为新理论。

RELLIS 外部扩展保持关闭：240 扫描审计后提前停止，新 48 帧从未打开，不是 0/48。
这部分已移入附录；当前证据仍限于受控合成任务。

当前状态（2026-09-12）：**Pro 要求的三实验已全部在 Quest 完成**：五个旧模型无训练审计、
20 个固定预算的几何×栅格模型，以及独立校准/测试的固定选择器认证。撤销 −2 后处理后的
固定阈值基线已经得到统计认证；新 IID 测试接受危险率 1.07%，安全机会利用率 96.93%。
整对残差上界仍零覆盖。当前结论支持简单修复和已有风险控制的恰当应用，不支持新安全算法主张。

全部新计算放在 Quest，数值、源码冻结和权重均保留。本轮使用共同高分辨率 mean-exposure
参考，不能与旧 low16 结果或三通道/q95 主线混算。此前的位置捷径审计、基础实现、安全记账、
两个环境的运动足迹检查与小 CNN 对照仍保留；Qwen3-VL-8B 的 126 份真实标注仍全部 unknown-only。
尚无 C³-Safe policy checkpoint、合格 teacher 监督集或 `VALIDATED` semantic-safety 结果。

先读：

- [`output/pdf/perfect_coarse_predictions_iclr2027.pdf`](output/pdf/perfect_coarse_predictions_iclr2027.pdf)：九页正文的匿名工作稿（未投稿）；
- [`docs/SAME_INFORMATION_PROGRESS_2026-09-12.md`](docs/SAME_INFORMATION_PROGRESS_2026-09-12.md)：最终补充对照、机会恢复与新增危险、论文交付；
- [`paper/visual_risk_diagnosis/iclr2027/README.md`](paper/visual_risk_diagnosis/iclr2027/README.md)：官方格式 LaTeX 稿和独立编译说明；
- [`paper/visual_risk_diagnosis/manuscript.md`](paper/visual_risk_diagnosis/manuscript.md)：重写的英文正文，以空间目标到动作暴露为主线；
- [`docs/ORACLE_AGGREGATION_PROGRESS_2026-09-12.md`](docs/ORACLE_AGGREGATION_PROGRESS_2026-09-12.md)：上一轮 oracle 发现/新确认结果与冻结记录；
- [`docs/RELLIS_REVISION_PROGRESS_2026-09-12.md`](docs/RELLIS_REVISION_PROGRESS_2026-09-12.md)：一次有预算上限的修订、240 扫描证据和新样本前的停止决定；
- [`docs/RELLIS_EXTERNAL_PROGRESS_2026-09-12.md`](docs/RELLIS_EXTERNAL_PROGRESS_2026-09-12.md)：RELLIS 几何资格结果、独立复核与停止决定；
- [`docs/PRO_DECISION_ROUND_PROGRESS_2026-09-12.md`](docs/PRO_DECISION_ROUND_PROGRESS_2026-09-12.md)：最新三实验结果、固定基线认证与停止/收窄决定；
- [`research/pro_decision_round_20260912/README.md`](research/pro_decision_round_20260912/README.md)：冻结协议、Quest 复现与论文补充稿；
- [`docs/C3_CONFIRMATION_PROGRESS_2026-09-12.md`](docs/C3_CONFIRMATION_PROGRESS_2026-09-12.md)：最新随机布局确认实验、风险校准失败模式与下一步；
- [`paper/compositional_visual_risk/CURRENT.md`](paper/compositional_visual_risk/CURRENT.md)：论文草稿与本轮补充材料入口；
- [`docs/C3_PAPER_PROGRESS_2026-09-12.md`](docs/C3_PAPER_PROGRESS_2026-09-12.md)：上一轮六模型配对审计和位置平衡修复；
- [`docs/C3_QUEST_PROGRESS_2026-09-12.md`](docs/C3_QUEST_PROGRESS_2026-09-12.md)：已接入的资源、真实 teacher 失败诊断及补发现的远端研究进展；
- [`docs/C3_FOUNDATION_PROGRESS.md`](docs/C3_FOUNDATION_PROGRESS.md)：本轮完成项、实际结果、复现命令和下一步；
- [`RESEARCH_STORY.md`](RESEARCH_STORY.md)：新的 storyline，以及保留的旧 pilot 证据；
- [`docs/C3_SAFE_MAINLINE.md`](docs/C3_SAFE_MAINLINE.md)：协议、矩阵、baseline 和 Go/No-Go gates；
- [`docs/RESULTS_REGISTRY.md`](docs/RESULTS_REGISTRY.md)：所有 artifact 的状态真相源；
- [`docs/REPOSITORY_STATUS_2026-08-13.md`](docs/REPOSITORY_STATUS_2026-08-13.md)：保留/冻结/复用边界。

## 一句话 thesis

以下保留本分支的原方法假设；Quest 的另一条路线已弱化 CF auxiliary loss 的贡献主张。
两条路线的协议需对齐，不能把假设或已消费开发集上的改进当成已经成立的科学结论。

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
本地 simulator-only 学生使用已有 Torch 2.5.1。后续 Quest 接入复用了独立 Qwen 环境，
下载了官方 8B 权重并完成真实本地 VLM 调用；所有这些运行的付费 provider calls 均为 0。

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
