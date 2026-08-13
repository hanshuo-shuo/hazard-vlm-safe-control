# C³-Safe 主线协议

决策日期：2026-08-13
状态：**PLANNED / NOT RUN**
工作标题：**Teach Requirements, Not Actions: Capability–Constraint Counterfactual Distillation for Semantic-Safe Control**

## 1. 当前决策

仓库主线切换到 **C³-Safe（Capability–Constraint Counterfactual Semantic Safety Distillation）**。
当前没有任何已运行的 C³-Safe 结果，也没有任何旧 semantic-safety 数字可以直接继承为
论文结果。旧的 marker、interface-contract、semantic pilot、PIVOT 和 direct-VLA
资产仅保留 provenance、motivation、failure analysis 或回归用途。

核心 thesis 是：

> VLM 负责告诉学生模型“环境要求机器人具备什么”，不负责告诉机器人“下一步怎么走”；
> 环境需求、机器人能力和任务规则分离后，用同一 transition 的反事实风险翻转监督，
> 训练一个测试时完全不依赖 VLM 的安全策略。

这条主线不是 VLM action proposal、candidate ranking、waypoint selection、trajectory
scoring 或在线 VLM safety shield。测试阶段必须删除 VLM runtime，动作只能由小型
Gaussian SAC actor 输出。

## 2. 信息流与学习对象

```text
RGB observation I ──> frozen/offline VLM teacher ──> action-free requirement label
        │                                              (training only)
        └────────────> compact requirement encoder dθ(I)

(dθ(I), capability κ, rule q, state/action)
        └────────────> factorized semantic cost critic Qc
                              + reward critic Qr
                              + independent SAC-Lagrangian actor
```

Teacher 不得接收 action、candidate、waypoint、trajectory proposal、planner output、
evaluator truth 或 candidate safe/unsafe label。模拟器提供 native physical cost；
capability/rule twin 只改变能力或规则，不改变同一 scene、同一 observation 和同一
state transition。

结构化语义 cost 的目标形式为：

\[
\hat c_{sem}(I,\kappa,q)=
\sigma\left(b+\sum_j q_jw_j[d_{\theta,j}(I)-\kappa_j]_+\right).
\]

训练至少包含四类约束：

\[
\begin{aligned}
L_{req} &= BCE(d_\theta(I),\tilde d),\\
L_{cf} &= [m-y_{ab}(\hat c_a-\hat c_b)]_+,\\
L_{inv} &= \|h_\theta(I)-h_\theta(aug(I))\|_2^2,\\
L_{Q_c} &= TD\ loss\ with\ native\ physical\ cost + \hat c_{sem}.
\end{aligned}
\]

actor/dual update 使用 cost critic 做 CMDP 约束；这里的 counterfactual loss 是必须
单独消融的算法因素，不得把它混在普通 late concatenation conditioning 里。

## 3. 最小数据与 provenance 契约

每个 teacher label、transition 和 twin 必须能通过 manifest 重建。最小字段如下：

| 字段 | 含义 | 来源/权限 |
|---|---|---|
| `scene_id`, `episode_id`, `t` | 场景和 transition 身份 | 公开 manifest |
| `image_path`, `image_sha256` | 输入 RGB 与 hash | SENSOR |
| `state_hash`, `next_state_hash` | 同一 transition 配对 | public state interface |
| `geometry_seed`, `appearance_id`, `geometry_hash` | split 与 twin identity | manifest |
| `capability_card`, `capability_vector` | 当前机器人能力 | CAPABILITY |
| `rule_card`, `rule_vector` | 当前任务规则 | TASK_SPEC |
| `teacher_model`, `teacher_revision` | 离线 teacher 身份 | provenance |
| `prompt_sha256`, `response_sha256` | 完整 teacher 请求/响应 | provenance |
| `requirement_label`, `label_schema_version` | action-free 环境需求 | TEACHER_OUTPUT |
| `native_physical_cost` | 模拟器物理 cost | EVAL/transition source |
| `twin_group_id`, `intervention` | matched counterfactual 关系 | experiment manifest |
| `git_sha`, `dirty`, `protocol_version` | 可重放版本 | provenance |

禁止用 `COMPLETE` manifest 状态代替科学有效性；只有通过结果注册表和 Gate 0–2 的
artifact 才能进入 `VALIDATED`。

## 4. 冻结的最小实验矩阵

| 维度 | 计划值 |
|---|---|
| 任务 | PointHazard；修复后的 `SafetyPointGoal1-v0` semantic variant |
| 主 teacher | `Qwen3-VL-8B-Instruct`；只做离线 action-free labeling |
| teacher sensitivity | `InternVL3-8B` 仅占 10% label，不作为核心结果 |
| student | ResNet18/小 CNN requirement encoder + capability/rule MLP + Gaussian SAC actor |
| capability | `waterproof`、`mud/rough-terrain mobility`；建议加入 simulator 锚定的 footprint/clearance 轴 |
| rules | `avoid-water`、`avoid-mud`、`protect-fragile-terrain` |
| appearance split | 纹理/色彩模板与 geometry seed train/validation/test 完全隔离 |
| capability split | 二元能力轴的四个组合中训练三个，测试未见组合 |
| joint OOD | 未见 appearance × 未见 capability × 未见 rule combination |
| 规模 | 每任务约 2k–5k teacher-labelled frames；至少 5 seeds；每 split/seed 至少 100 episodes |
| metrics | STC、episode violation、violation steps、native cost、success、return、AUROC/AUPRC、FNR、ECE、latency、参数量 |

Safety-Gym 在进入训练前必须完成：world-coordinate overlay、逐步 semantic violation、
center/footprint/swept-segment 判定选择并冻结，以及 native cost 与 semantic cost 分离。

## 5. 必须比较的 baseline 与消融

### Baselines

1. Reward-only SAC。
2. Oracle semantic-cost SAC-Lagrange（仅作上界）。
3. No-VLM：同 backbone + domain randomization 或闭集 simulator labels。
4. Scalar-VLM-Cost：VLM 直接给固定 capability 的 scalar risk。
5. LateConcat-Critic：普通 capability/rule 拼接，无 counterfactual/monotonic loss。
6. DGC-like generic representation distillation。

不在最小版本复现完整 Dreamer VLM-SAFE 或 PROCO world model；先用 scalar、late-concat
和 simulator-only 对照隔离 C³-Safe 的真正新增因素。

### 必须消融

1. 去掉 matched capability/rule counterfactual loss。
2. factorized requirement–capability interaction 改为 late concatenation/direct scalar。
3. 去掉 VLM requirement supervision，替换为 DINO/CLIP frozen feature、domain randomization
   或 simulator-only label。

## 6. Go / No-Go gates

### Gate 0 — 基础设施

- train/validation/test 的 seed、geometry hash、RGB hash 重叠为 0；
- Safety-Gym overlay 与 evaluator 的随机点 agreement ≥ 98%；
- 新依赖、data manifest、teacher prompt/image/response hash 齐全；
- native physical cost 与 semantic violation 不互相泄漏。

任一项失败，不进入训练结论。

### Gate 1 — teacher 与 requirement student

在至少 200 张人工审查、类别平衡的 held-out 图像上：

- teacher environment-requirement macro-F1 ≥ 0.80；
- capability/rule risk-flip accuracy ≥ 0.80；
- student held-out appearance AUROC ≥ 0.85、ECE ≤ 0.10；
- student 比 frozen DINO/CLIP 或 no-VLM baseline 至少高 5 AUROC points。

失败即 No-Go：无法回答为什么需要 VLM。

### Gate 2 — 算法贡献

在 joint OOD 上相对最强 non-VLM/late-concat baseline：

- episode semantic violation 相对下降 ≥ 25%；
- STC 提升 ≥ 8 个百分点；
- success 下降不超过 5 个百分点，或 return 下降不超过 10%；
- native physical collision 不恶化超过 2 个百分点；
- 去掉 counterfactual loss 后 violation 相对恶化 ≥ 10%；
- 至少 5 seeds，主要 STC/violation 差异的 95% CI 不跨 0。

若 counterfactual ablation 没有明显影响，即使总体策略表现良好，也 No-Go。

## 7. 仓库分层

### 继续复用（先修复/重跑后才可成为新证据）

- `env_pointhazard.py`、`hazard_renderer.py`；
- `envs/protocol_env.py`、`envs/point_hazard_adapter.py`、`envs/safety_gym_goal_adapter.py`；
- `evaluation/capability_twins.py`、`semantic_evaluator.py`、`outcomes.py`、`schemas.py`；
- `evaluation/geometry_calibration.py`、`evaluation/semantic_geometry.py`；
- `safe_expert.py`、`mpc_expert.py`，仅用于数据覆盖和 oracle baseline。

### storyline 保留（只能作 motivation / failure analysis）

- `results/interface_contract_scout_native_analysis/`：9/20 equivalent pairs 改变 exact native trajectory；
- `results/interface_contract_combined_analysis/`：field/order、grounding 与 ranking instability 的 pilot 审计；
- `results/interface_contract_execution_bridge/`：provider-free replay bridge；
- `results/next_five_experiments/02_marker_interface_robustness/`：marker/candidate stability kill evidence；
- `docs/assets/interface_contract_scout/` 与 `legacy/docs/advisor_rewrite_2026-07/`：历史图和完整叙事。

上述全部保持原路径和原 hash，状态仍由 `docs/RESULTS_REGISTRY.md` 管理，不能包装成
C³-Safe baseline 或 validated headline。

### 停止科学开发但保留 provenance

`pivot_vlm.py`、`subgoal_pivot_hazard.py`、PointPush/direct-VLA scaffolds、
`evaluation/interface_contract*`、`interface_execution.py`、`interface_metrics.py`、
`five_stage_audit.py` 以及旧 marker/waypoint/PIVOT figures。由于历史脚本、测试和 manifest
仍引用这些路径，当前不物理移动、不删除，只在文档中明确冻结。

## 8. 实施顺序

1. 运行 layout invariants、两个 adapter、expert coverage、policy baseline 和 geometry calibration；
2. 新增 teacher-label schema、manifest、prompt/image/response hash logger；
3. 修复并冻结 Safety-Gym semantic step/evaluator；
4. 采集 transition，生成 capability/rule twins，完成人工审查与 teacher batch labeling；
5. 实现 compact requirement encoder、factorized cost head 和 SAC-Lagrangian trainer；
6. 只读回放和 Gate 0–1 通过后，运行 baselines、5-seed 主实验与三项消融；
7. Gate 2 通过后才生成论文图表和新的 `VALIDATED` 注册记录。

当前这次仓库整理只完成协议、入口和状态边界，不声称完成上述训练实现。
