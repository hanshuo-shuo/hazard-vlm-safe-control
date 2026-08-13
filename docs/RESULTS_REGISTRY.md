# Results Registry

更新：2026-08-13

本文件是所有实验产物的状态真相源。结果只能处于以下状态之一：

- **VALIDATED**：环境、协议、对照、统计和 artifact 均满足当前计划，可作为论文证据；
- **PILOT_ONLY**：只验证现象或管道，不能作为正式数字；
- **INVALIDATED**：发现 sampler、泄漏、对照或 estimand 问题，只能作取证/回归记录；
- **ARCHIVED**：属于已放弃路线，仅用于保存研究历史。

当前没有任何 semantic-safety 结果达到 `VALIDATED`。C³-Safe 主线目前是
`PLANNED / NOT RUN`，不存在可继承的主线数字、checkpoint 或 teacher-label dataset。

## C³-Safe current mainline

- Decision date: `2026-08-13`
- Evidence class: `PLANNED / NOT RUN`
- Protocol: [`docs/C3_SAFE_MAINLINE.md`](C3_SAFE_MAINLINE.md)
- Scope manifest: `configs/c3_safe_mainline_manifest.json`
- Test-time VLM: **forbidden**
- Teacher role: offline, action-free environment-requirement labeling only
- Current artifacts: none

The old pilot numbers below are not inherited as C³-Safe baselines. A new result may enter
`VALIDATED` only after the C³-Safe Gate 0–2 checks, complete provenance, and an independent
artifact record. A manifest or run marked `COMPLETE` only means execution completed; it does
not mean the scientific result is valid.

## Retained storyline — interface-contract provider-free infrastructure

- Decision date: `2026-07-26`
- Evidence class: `INFRA / BLOCKED / PROVIDER_FREE`
- Protocol: `docs/INTERFACE_CONTRACT_PROTOCOL.md`
- Source manifest: `configs/interface_contract_pilot_manifest.json`
- Artifact: `results/interface_contract_provider_free_dry_run/`
- Matrix: 80 explicit blocks, 1,440 logical request rows, 480/model
- Paid seeds frozen per model: `[20,21,22,23,24]`
- Provider calls/attempts in dry run: `0 / 0`
- Native gates: PointHazard native RGB/dynamics/reward/cost/termination and
  Safety-Gymnasium `SafetyPointGoal1-v0` native RGB/dynamics/reward/cost/termination
- Formal planner: evaluator-truth grounding rejected; oracle geometry restricted
  to the separate `oracle_upper_bound` arm
- Authorization: **NOT AUTHORIZED**

The three-layer paid guard caps each canonical model identity at five distinct
paid seeds, 480 new logical calls, three attempts per request, and 1,440 total
attempts. Failed, timed-out, empty, schema-invalid, and model-mismatched attempts
are reserved before transport and remain in the ledger. The compatibility smoke
is the first formal matrix cell and cannot be repeated.

Under the literal global seed ceiling, Gemini 2.5 Flash-Lite and
Qwen3-VL-30B-A3B-Instruct are `PAID_INELIGIBLE`: historical evidence contains
paid seeds `0–4` and `20–24`. Their prior artifacts remain replayable without new
provider calls. Mistral's historical paid seeds are `20–24`; its revision is not
yet frozen. Two additional model slots and the maximum spend remain unresolved.

## Retained storyline — interface-contract recovery pilot

- Decision date: `2026-07-26`
- Evidence class: `PILOT_ONLY / NOT PAPER RESULT`
- Mainline decision: `PROMISING_METHOD_AND_PHENOMENON_BUT_NOT_YET_AN_ICLR_MAIN_RESULT`
- Seeds per API model: `[20,21,22,23,24]` (hard ceiling of five distinct seeds)
- Valid science models: Gemini 2.5 Flash-Lite, Qwen3-VL-30B-A3B-Instruct,
  Mistral Small 3.2 24B
- Valid crossed rows: 330
- Total calls including retained invalid GPT arm: 440
- Total provider cost: `$0.051349465`
- Combined artifact: `results/interface_contract_combined_analysis/`

The audit covers JSON field ordering, structured/free-text output, positive and
negative label semantics, constraint/compatibility wording, ambiguous
applicability, marker IDs, candidate ordering, downstream planner mappings and
model ranking across two tasks and two renderer/environment contracts.

Allowed pilot claims:

- field-order physical-action consistency is 23/30 and
  structured/free-text consistency is 21/30;
- marker-ID physical-choice consistency is 23/30 and candidate-order
  consistency is 18/30;
- plausible mappings of the same ambiguous outputs produce executed STC from
  0.50 to 0.80 through fixed PointHazard MPC and headless Safety-Gym dynamics;
- Gemini/Qwen relative rank reverses under some interface conditions;
- Mistral is a stable counterexample/control, so the effect is heterogeneous.

Blocking evidence and forbidden promotion:

- semantic-contract sensitivity did not pass the predeclared two-environment
  replication gate;
- Safety-Gym evidence uses a headless contract render, not native RGB/dynamics;
- the second closed-loop arm still uses headless rather than native Safety-Gym;
- the result must not be promoted to ICLR-grade, universal fragility, or native
  cross-environment replication.

The GPT-5-mini arm is retained but excluded from capability ranking: only 8/110
rows parsed, 87 returned empty content, and 15 were otherwise truncated or
invalid under the frozen request. The separately registered Mistral replacement
changes only model identity; all 110 prompt/image hashes match the parent.

The formal recovery protocol and kill conditions are in
[INTERFACE_CONTRACT_MAINLINE.md](INTERFACE_CONTRACT_MAINLINE.md).

## Retained storyline — marker mainline termination

- Decision: `TERMINATED`
- Termination date: `2026-07-24`（pilot report generation date）
- Evidence class: `PILOT_ONLY / NOT PAPER RESULT`
- Reason: marker/candidate interface instability triggered the predeclared kill
  condition.
- Follow-up: no seed expansion, prompt rescue, marker renumbering, candidate
  count selection, or paid rerun. Retain only as negative interface evidence,
  regression, historical pilot and case study.

### Five-pilot registry records

All five records were produced on branch `safety`, commit
`adcce013721ef7d9bc05893d54167950354f566e`, seeds `[0,1,2,3,4]`.
Models are Gemini 2.5 Flash Lite and Qwen3-VL-30B-A3B-Instruct except experiment
5, which has no model. Provider counts below describe requests represented in
the evidence; the final replay execution had zero new provider calls.

| Experiment ID | Requests | Tokens | Cost USD | Status | Kill condition | Artifact |
|---|---:|---:|---:|---|---|---|
| NF-01 core information audit | 70 | 34,438 | 0.005164 | `PILOT_ONLY` | supporting evidence: P2 not monotonic; no rank reversal | `results/next_five_experiments/01_core_information_audit/` |
| NF-02 marker interface robustness | 70 | 35,856 | 0.005813 | `PILOT_ONLY / TERMINATED` | **TRIGGERED**: marker 20%, order 30% | `results/next_five_experiments/02_marker_interface_robustness/` |
| NF-03 modern perception substitution | 10 | 3,611 | 0.000674 | `PILOT_ONLY` | not the marker kill test | `results/next_five_experiments/03_modern_perception_substitution/` |
| NF-04 five-stage twins | 30 | 13,731 | 0.002333 | `PILOT_ONLY` | not the marker kill test | `results/next_five_experiments/04_five_stage_twins/` |
| NF-05 second environment smoke | 0 | 0 | 0 | `INTEGRATION_EVIDENCE` | not applicable | `results/next_five_experiments/05_second_environment/` |

Allowed claims:

- NF-01: recognition-conditioned selected safety was a strong pilot signal;
  P2 did not outperform P0; no model rank reversal was observed.
- NF-02: marker/candidate interface was unstable and the kill condition fired.
- NF-03: at one pilot operating point detector geometry traded completion for
  fewer semantic violations; this does not establish detector superiority.
- NF-04: capability applicability failed in the pilot and motivates, but does
  not validate, five-stage attribution.
- NF-05: the headless Safety-Gym adapter ran and preserved twin geometry/native
  cost separation.

Forbidden claims:

- stable semantic spatial reasoning from marker results;
- privilege monotonicity or cross-model/environment generalization;
- detector superiority, a proven applicability bottleneck, or Safety-Gym
  replication;
- formal safety, ICLR-grade evidence, or any unrun calibration improvement.

Follow-up action for NF-01/02 is termination and regression-only retention.
NF-03/04 proceed only through cached marker-free geometry and capability audits.
NF-05 proceeds as provider-free infrastructure until a new protocol is frozen.

WP-2.0 状态边界：`PROTOCOL.md` 1.2.1 的 numeric/evaluator 语义保持冻结；
1.2.2 只登记 M04 structured prompt amendment。本月不再增加 JSON lexical、
binary64 或 serialization 细节，也不在 W2 实现完整标准解释器。所有付费 VLM/OpenRouter/API
run 暂停。

---

## 当前状态总表

| 结果/产物 | 状态 | 原因 | 是否保留 |
|---|---|---|---|
| C³-Safe mainline | PLANNED / NOT RUN | 尚未实现 teacher-label schema、factorized critic 或 SAC-Lagrange trainer | 是，当前唯一科学主线 |
| outputs/month1_confidence* | PILOT_ONLY | pure-geometry n=5；不受 semantic-zone sampler 影响，但规模小、单模型 | 是，历史信心实验 |
| outputs/semantic_pilot* | INVALIDATED | semantic-zone fallback 可与 red hazard 重叠；旧 C1/C2/B estimand | 是，历史 |
| outputs/semantic_implicit_pilot* | INVALIDATED | 同上；L1 instruction 与 commonsense 混用 | 是，历史 |
| outputs/semantic_bplus_* | INVALIDATED | sampler 问题；B/B+ 与 oracle 的 router/enforcement 不正交 | 是，历史 |
| outputs/semantic_hetero* | INVALIDATED | sampler 问题；n=5；缺现代 perception baseline | 是，历史 |
| outputs/vlm5_single_l1* | INVALIDATED | sampler 问题；单模型 n=5 | 是，历史 |
| outputs/vlm5_l2* | INVALIDATED | sampler 问题；L2 删除任务约束，不是纯 leakage 干预 | 是，历史 |
| offline_n100_l1/l2 | INVALIDATED | heuristic plumbing run，VLM calls=0；sampler 问题 | 是，移入 invalidated archive |
| vlm20_l1/l2_heldout* | INVALIDATED | 7/20 zone-hazard overlap；router confound；输入 artifact 不完整 | 是，移入 invalidated archive |
| vlm2_smoke* | INVALIDATED | 被 n=20 覆盖且使用同一无效 sampler | 是，移入 invalidated archive |
| zone_detector self-test | ARCHIVED | 只证明 renderer-palette plumbing；明确不作为 perception baseline，不再扩展 | 是，历史 |
| PointPush expert smoke | ARCHIVED | 只验证环境/专家可运行；PointPush 不进入当前科学主线 | 是，历史 |
| direct VLA / PointPush VLA scaffolds | ARCHIVED | 保留代码和旧图作 provenance；当前阶段停止训练、调参和性能比较 | 是，历史 |
| old PIVOT/B+ figures and replays | ARCHIVED | 旧 router/enforcement/scene confounds；不得包装为当前 baseline 或新结果 | 是，历史 |
| outputs/*.png 旧结果图 | ARCHIVED | 图片用于理解历史路线，但不能支撑当前 claim | 是，文档需标注状态 |

## 历史 backend 与运行状态

这些是仓库/基础设施状态，不是论文结果状态：

| Backend / run gate | 状态 | 说明 |
|---|---|---|
| Safety-Gymnasium native Goal adapter | `INFRA` | `SafetyPointGoal1-v0` native vertical slice 可复用，但尚无论文级结果。 |
| Safety-Gymnasium semantic variant | `BLOCKED` | 语义 variant 的 protocol gate 与 evidence 尚未完成；不进入当前 MVF。 |
| B01 sampler | **RESOLVED** | checked final-attempt grid fallback 保留正常 resample/golden 序列；四种配置各 10,000 seeds 的正式 invariant sweep 于 2026-07-22 通过。 |
| Unified direct/replay harness | **INFRA / PASSED** | `direct/replay × none/oracle` vertical slice、shared enforcement、artifact reconstruction 和 STC audit 已通过专项测试。 |
| Formal paid VLM/OpenRouter/API runs | **PAUSED** | 14-condition offline matrix、machine release manifest 与 core reproducibility lock 已通过；manifest 仍为 `BLOCKED / NOT AUTHORIZED`，model list、pilot subset、预算、日期和 operator 未冻结。2026-07-26 separately scoped interface micro-pilot 不解锁该 manifest。 |

---

## 2026-07-01 held-out n=20 取证记录

原始产物归档到 outputs/invalidated/2026-07-11/。

这些文件仍有三类价值：

1. 验证真实 VLM API、JSON parsing、fallback=0 的闭环通路；
2. 保存 “recognize-but-cross” 的原始 response；
3. 用作未来 artifact/provenance logger 的回归输入。

不能用来声称：

- B/B+ 优于公平 oracle 或现代感知栈；
- 已建立 success–safety Pareto frontier；
- L2 证明 commonsense avoidance；
- tuning-disjoint 已经使结果达到 paper-grade；
- 当前 MPC 采用 safety-oriented sampling MPC（empirical）表述，不含形式化安全性质。

---

## 失效原因

### I01 — Layout sampler

env_pointhazard.py 的 semantic-zone fallback 跳过合法性检查。默认配置复算 200 seeds，61 个 zone 与 red hazard 重叠；held-out 100–119 中为 7/20。

### I02 — Router confound

C2/CV 直接 plan 到 goal；B/B+ 使用 VLM subgoal 和周期性 MPC restart。success 差异不能只归因给 zone source。

### I03 — Task specification confound

旧 L1→L2 同时删除 “应该避开这种地形” 的任务要求，测到的是 instruction removal，不是纯 privileged-information dose。

### I04 — Incomplete audit artifact

旧 transcripts 只保存输出，没有 exact prompt、input image/hash、candidate metadata、model revision 和完整 provenance。

---

## 历史 semantic-safety 结果进入 VALIDATED 的条件

以下条目记录的是旧 interface/semantic 路线的冻结门槛，不是 C³-Safe 的替代协议；
当前 C³-Safe 的门槛以文档末尾的记录纪律和 `C3_SAFE_MAINLINE.md` 为准。

必须同时满足：

1. [review_status_registry.json](review_status_registry.json) 中 B01–B07 全部为 `DONE`；
2. layout invariant tests 通过；
3. task/capability/appearance/privileged cue 正交；
4. router 与 zone source 分离；
5. 主指标使用 Safe Task Completion；
6. exact prompt/image/candidate/provenance 可重放；
7. 至少一个现代 open-vocabulary baseline；
8. 预先冻结 seed split、protocol version、prompt 和统计；
9. 结果跨至少两个模型和两个环境复现。

---

## 记录纪律

- 不覆盖旧 JSON；新 run 使用独立 run ID；
- 每个结果文件必须包含 protocol_version、git SHA、dirty flag 和 split；
- 无效结果移动到 outputs/invalidated/YYYY-MM-DD/，并在本表记录；
- 文档引用结果时必须同时写状态；
- 负结果和错误实验可以保留，但不得通过改标题重新包装成有效证据。

---

## C³-Safe 记录纪律（2026-08-13）

- 主线新 run 使用独立的 C³-Safe protocol version 和 run ID，不覆盖旧 JSON；
- teacher 必须是 action-free requirement label，保存 model/revision、prompt/image/response hash；
- twin 必须固定 scene、observation 和 state transition，只改变 capability 或 rule；
- native physical cost、semantic requirement label 和 evaluator truth 分开存储并带 provenance；
- train/validation/test 的 seed、geometry hash、RGB hash overlap 必须为 0；
- 旧 interface-contract/marker/PIVOT/PointPush/direct-VLA artifact 只能标记为 motivation、
  failure analysis、INFRA 或历史记录，不得改 status 冒充 C³-Safe evidence；
- Gate 0–2 未通过前，不生成论文 headline，不声明 `VALIDATED`，不启动大规模 paid run。

实现顺序、baseline、ablation 和数值门槛唯一见
[`C3_SAFE_MAINLINE.md`](C3_SAFE_MAINLINE.md)。
