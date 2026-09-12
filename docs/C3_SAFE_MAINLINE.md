# C³-Safe 主线协议

决策日期：2026-08-13；方法修订：2026-08-17
实现状态（2026-09-12）：**FOUNDATION IMPLEMENTED / METHOD NOT VALIDATED**
工作标题：**Teach Spatial Requirements, Measure Motion Exposure: Capability–Constraint Counterfactual Distillation for Semantic-Safe Control**

## 1. 当前决策

仓库主线切换到 **C³-Safe（Capability–Constraint Counterfactual Semantic Safety Distillation）**。
几何/评测/开发数据管道已实现，并运行了 simulator-only 空间学生对照，详见
[`C3_FOUNDATION_PROGRESS.md`](C3_FOUNDATION_PROGRESS.md)。真实 VLM 空间蒸馏、独立
semantic/physical critics 和 SAC actor 尚未运行，完整 Gate 0–2 均未宣告通过。
没有任何旧 semantic-safety 数字可以直接继承为论文结果。
旧的 marker、interface-contract、semantic pilot、PIVOT 和 direct-VLA
资产仅保留 provenance、motivation、failure analysis 或回归用途。

2026-09-12 已完成真实 Quest teacher 接入诊断，但 126 个回答只有 unknown，无属性正标注。
同时发现另一条远端 C³-Safe 路线已经转向 factorized composition，并有较完整的开发结果；
详见 [`C3_QUEST_PROGRESS_2026-09-12.md`](C3_QUEST_PROGRESS_2026-09-12.md)。本文件保留
此分支的目标协议，尚未将不同分支的贡献定义或 gate 合并。

以下矩阵与阈值保留为正式方法的目标协议；开发检查与小数据对照不降低这些要求。
当前执行细节（q95、圆形足迹、开发 seeds）单独冻结在
`configs/c3_safe_foundation.json`，不把开发集冒充正式 held-out 集。

核心 thesis 是：

> VLM 负责告诉学生模型“图像中每个位置要求机器人具备什么”，不负责告诉机器人
> “下一步怎么走”；学生把空间需求场与 transition 的 swept footprint 相交得到运动暴露，
> 再分别计算能力不兼容代价和任务规则代价，训练一个测试时完全不依赖 VLM 的安全策略。

这条主线不是 VLM action proposal、candidate ranking、waypoint selection、trajectory
scoring 或在线 VLM safety shield。测试阶段必须删除 VLM runtime，动作只能由小型
Gaussian SAC actor 输出。

## 2. 信息流与学习对象

```text
RGB observation I ──> frozen/offline VLM teacher ──> action-free spatial requirement label
        │                                              R~(I), training only
        └────────────> spatial requirement field Rθ(I) ∈ [0,1]^(H×W×K)

(state, action) ──> predicted short motion / training transition ──> swept footprint F(τ)
Rθ(I) + projection ΠI(F(τ)) ──> exposure e(τ,I)
e + capability κ ──> capability-incompatibility cost ccap
e + task rule q ──> normative rule cost crule
ccap + crule ──> semantic cost csem ──> semantic critic Qsem
native simulator cost ──────────────────────────────> physical critic Qphys
(Qr, Qsem, Qphys) ──> independently constrained SAC-Lagrangian actor
```

Teacher 不得接收 action、candidate、waypoint、trajectory proposal、planner output、
evaluator truth 或 candidate safe/unsafe label。模拟器提供 native physical cost；
capability/rule twin 只改变能力或规则，不改变同一 scene、observation、空间需求场、
state transition 和 swept footprint。另设 motion twin：保持图像、capability 和 rule
不变，只让一条短轨迹穿过目标区域、另一条绕开，用来识别“看见水”等同于“进入水”的
presence-only shortcut。再设 visual twin：保持 capability、rule 与相对运动不变，只把
目标区域移入或移出 swept footprint，用来识别“只读 capability/rule 卡片、不看图像”的
card-only shortcut。

### 2.1 空间需求场

学生预测：

\[
R_\theta(I)\in[0,1]^{H\times W\times K},
\]

其中 channel 是局部环境属性或需求，例如 `water_ingress_demand`、
`low_traction_demand` 和 `fragile_surface_property`。teacher 只提供 action-free 的 mask、
soft field 或带空间坐标的区域标注；不得根据候选动作改变场值。一个整图 requirement
vector 只能由空间场池化得到并用于辅助分类，不得作为 semantic cost 的直接输入。

### 2.2 运动暴露

对 transition 或短轨迹 \(\tau=(s,a,s')\)，由机器人几何和运动路径构造 swept footprint
\(\mathcal F(\tau)\)，并通过冻结的相机/世界变换投影到需求场：

\[
e_j(\tau,I)=
\operatorname{Pool}_{p\in\Pi_I(\mathcal F(\tau))}R_{\theta,j}(I,p).
\]

`Pool` 的默认实现冻结为 footprint 内的高分位池化；实验同时报告 max/mean sensitivity，
但不得按结果挑选。越界、遮挡或投影无效必须显式产生 `unknown_projection`，不能悄悄当成
零暴露。训练 one-step target 可以用已发生 transition 的 \(s'\) 构造真实 swept footprint；
部署时 actor/critic 不能读取未来 \(s'\)，只能使用当前 \((s,a)\) 及冻结运动学模型预测的
\(\tilde\tau(s,a)\)，或直接由 critic 学习其期望累计代价。

### 2.3 关系代价

能力不兼容和规范性任务规则是两种不同关系：

\[
c_{\mathrm{cap}}(\tau)=
\sum_{j\in\mathcal C}w_j[e_j(\tau,I)-\kappa_j]_+,
\]

\[
c_{\mathrm{rule}}(\tau)=
\sum_{\ell\in\mathcal Q}v_\ell q_\ell e_{\pi(\ell)}(\tau,I),
\]

\[
c_{\mathrm{sem}}(\tau)=
1-\exp[-(c_{\mathrm{cap}}(\tau)+c_{\mathrm{rule}}(\tau))].
\]

\(\pi(\ell)\) 把规则映射到对应空间属性 channel。`q=0` 只能关闭相应规范性规则，不能
关闭机器人的物理不兼容：例如未启用 `avoid_water` 任务规则时，非防水机器人进入水域
仍有 \(c_{\mathrm{cap}}>0\)。`protect_fragile_surface` 则通过 \(c_{\mathrm{rule}}\) 激活，
不被伪装成机器人能力。所有 \(w_j,v_\ell\ge 0\)，从而能力提高不能增加
\(c_{\mathrm{cap}}\)，规则激活也不能降低对应 \(c_{\mathrm{rule}}\)。

模拟器原生碰撞、动力学失稳等 \(c_{\mathrm{native}}\) 始终独立。方法使用独立的
\(Q_{\mathrm{sem}}\) 与 \(Q_{\mathrm{phys}}\)、独立 TD target、预算和日志；actor update
可同时受两项 Lagrange multiplier 约束，但任何训练表或指标都不得先把二者求和再报告。

训练至少包含：

\[
\begin{aligned}
L_{field} &= BCE/Dice(R_\theta(I),\tilde R),\\
L_{exp} &= \ell(e_\theta(\tau,I),\tilde e(\tau,I)),\\
L_{cf} &= [m-y_{ab}(c_a-c_b)]_+,\\
L_{eq} &= \|R_\theta(TI)-T R_\theta(I)\|_1,\\
L_{Q_{sem}} &= TD(c_{sem}),\qquad L_{Q_{phys}}=TD(c_{native}).
\end{aligned}
\]

几何增强使用空间等变损失 \(L_{eq}\)，不能错误使用整图 embedding invariance 抹掉位置；
仅亮度、纹理等不改变几何的增强使用逐像素不变性。counterfactual 和 motion-exposure
loss 都必须单独消融，不能混在普通 late concatenation conditioning 里。

## 3. 最小数据与 provenance 契约

每个 teacher label、transition 和 twin 必须能通过 manifest 重建。最小字段如下：

| 字段 | 含义 | 来源/权限 |
|---|---|---|
| `scene_id`, `episode_id`, `t` | 场景和 transition 身份 | 公开 manifest |
| `image_path`, `image_sha256` | 输入 RGB 与 hash | SENSOR |
| `requirement_field_path`, `field_sha256`, `field_schema_version` | action-free 空间需求标注 | TEACHER_OUTPUT |
| `state_hash`, `next_state_hash` | 同一 transition 配对 | public state interface |
| `robot_pose`, `footprint_geometry`, `motion_samples` | swept footprint 重建 | public state/robot model |
| `camera_calibration_hash`, `projection_version` | image/world 投影 | frozen calibration |
| `swept_footprint_hash`, `pooling_operator` | 暴露区域和池化定义 | derived public artifact |
| `exposure_vector`, `exposure_validity` | transition 的实际空间暴露 | derived label |
| `geometry_seed`, `appearance_id`, `geometry_hash` | split 与 twin identity | manifest |
| `capability_card`, `capability_vector` | 当前机器人能力 | CAPABILITY |
| `rule_card`, `rule_vector` | 当前任务规则 | TASK_SPEC |
| `teacher_model`, `teacher_revision` | 离线 teacher 身份 | provenance |
| `prompt_sha256`, `response_sha256` | 完整 teacher 请求/响应 | provenance |
| `global_requirement_label` | 仅辅助分类的整图标签，不直接产生 cost | TEACHER_OUTPUT |
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
| student | ResNet18/小 CNN spatial encoder-decoder + swept-footprint exposure module + capability/rule relation heads + Gaussian SAC actor |
| capability | `waterproof`、`mud/rough-terrain mobility`；建议加入 simulator 锚定的 footprint/clearance 轴 |
| rules | `avoid-water`、`avoid-mud`、`protect-fragile-terrain` |
| appearance split | 纹理/色彩模板与 geometry seed train/validation/test 完全隔离 |
| capability split | 二元能力轴的四个组合中训练三个，测试未见组合 |
| joint OOD | 未见 appearance × 未见 capability × 未见 rule combination |
| 规模 | 每任务约 2k–5k teacher-labelled frames；至少 5 seeds；每 split/seed 至少 100 episodes |
| metrics | field mIoU/AUPRC、exposure MAE/AUROC、cross-vs-bypass ranking、semantic STC/violation、native cost、success、return、critic AUROC/AUPRC/FNR/ECE、latency、参数量 |

Safety-Gym 在进入训练前必须完成：world-coordinate overlay、逐步 semantic violation、
swept-footprint rasterization 与投影冻结，以及 native cost 与 semantic cost 分离。只用
robot center 或只判断画面中是否存在某类地形都不能作为主方法。

## 5. 必须比较的 baseline 与消融

### Baselines

1. Reward-only SAC。
2. Oracle semantic-cost SAC-Lagrange（仅作上界）。
3. No-VLM：同 backbone + domain randomization 或闭集 simulator labels。
4. Global-Requirement：旧的整图 \(d_\theta(I)\) + capability/rule scalar head，无运动暴露。
5. Presence-Only-Field：学习空间场，但整图池化后直接算 cost，不与 footprint 相交。
6. Cards-Only-Critic：只看 capability/rule/action，不看 RGB，显式测量 card shortcut。
7. Scalar-VLM-Cost：VLM 直接给固定 capability 的 scalar risk。
8. LateConcat-Critic：普通 image/capability/rule/action 拼接，无显式 exposure 或关系头。
9. Oracle-Field/Exposure SAC-Lagrange：使用 simulator mask 与真实 swept footprint 的诊断上界。
10. DGC-like generic representation distillation。

不在最小版本复现完整 Dreamer VLM-SAFE 或 PROCO world model；先用 scalar、late-concat
和 simulator-only 对照隔离 C³-Safe 的真正新增因素。

### 必须消融

1. 去掉 matched capability/rule counterfactual loss。
2. 去掉 swept-footprint exposure，改为 global pooling/presence-only risk。
3. swept footprint 改为 robot center 或 endpoint，量化漏检穿越区域的代价。
4. 分开的 \(c_{cap}\)/\(c_{rule}\) 改回 \(q_j[e_j-\kappa_j]_+\)，检查规则是否错误关闭能力风险。
5. factorized relation head 改为 late concatenation/direct scalar。
6. 去掉 VLM spatial supervision，替换为 DINO/CLIP、domain randomization 或 simulator-only mask。

## 6. Go / No-Go gates

### Gate 0 — 基础设施

- train/validation/test 的 seed、geometry hash、RGB hash 重叠为 0；
- Safety-Gym overlay 与 evaluator 的随机点 agreement ≥ 98%；
- swept-footprint rasterizer 与 simulator oracle 的随机 transition agreement ≥ 98%；
- 同图 cross/bypass motion twin 的 exposure truth 必须不同，零交集轨迹的 exposure false-positive rate ≤ 2%；
- visual twin 保持 cards/motion 不变时，区域移入/移出 footprint 必须改变 exposure truth；
- 新依赖、data manifest、teacher prompt/image/response hash 齐全；
- native physical cost 与 semantic violation 不互相泄漏。

任一项失败，不进入训练结论。

### Gate 1 — teacher 与 requirement student

在至少 200 张人工审查、类别平衡的 held-out 图像上：

- teacher spatial requirement macro-F1 ≥ 0.80，region mIoU ≥ 0.60；
- student held-out spatial-field AUPRC ≥ 0.80，exposure AUROC ≥ 0.85、ECE ≤ 0.10；
- 同图 cross-vs-bypass exposure/risk ranking accuracy ≥ 0.90；
- visual-twin exposure/risk ranking accuracy ≥ 0.90，Cards-Only 不得通过该 gate；
- capability/rule risk-flip accuracy ≥ 0.80，且 `q=0` 不得关闭 capability incompatibility；
- student 比 frozen DINO/CLIP 或 no-VLM baseline 至少高 5 AUROC points。

失败即 No-Go：无法回答为什么需要 VLM。

### Gate 2 — 算法贡献

在 joint OOD 上相对最强 non-VLM/late-concat baseline：

- episode semantic violation 相对下降 ≥ 25%；
- STC 提升 ≥ 8 个百分点；
- success 下降不超过 5 个百分点，或 return 下降不超过 10%；
- native physical collision 不恶化超过 2 个百分点；
- 去掉 counterfactual loss 后 violation 相对恶化 ≥ 10%；
- 去掉 motion exposure 后 cross/bypass ranking 至少下降 10 points 或 violation 相对恶化 ≥ 10%；
- 至少 5 seeds，主要 STC/violation 差异的 95% CI 不跨 0。

若 counterfactual ablation 没有明显影响，即使总体策略表现良好，也 No-Go。

## 7. 仓库分层

### 继续复用（先修复/重跑后才可成为新证据）

- `env_pointhazard.py`、`hazard_renderer.py`；
- `envs/protocol_env.py`、`envs/point_hazard_adapter.py`、`envs/safety_gym_goal_adapter.py`；
- `evaluation/capability_twins.py`、`semantic_evaluator.py`、`outcomes.py`、`schemas.py`；
- `evaluation/geometry_calibration.py`、`evaluation/semantic_geometry.py`，扩展为需求场投影、
  swept-footprint rasterization 与 exposure truth；
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
2. 冻结 footprint 几何、transition 采样、image/world 投影、rasterizer 和 exposure pooling；
3. 新增 spatial teacher-label schema、manifest、prompt/image/response/field hash logger；
4. 修复并冻结 Safety-Gym semantic step/evaluator，证明 native/semantic cost 分离；
5. 采集 transition，生成 motion twins 与 capability/rule twins，完成人工空间标注审查；
6. 实现 spatial encoder-decoder、exposure module、分离的 relation cost heads、
   \(Q_{sem}/Q_{phys}\) 和 SAC-Lagrangian trainer；
7. 只读回放和 Gate 0–1 通过后，运行 baselines、5-seed 主实验与六项消融；
8. Gate 2 通过后才生成论文图表和新的 `VALIDATED` 注册记录。

当前这次仓库整理只完成协议、入口和状态边界，不声称完成上述训练实现。
