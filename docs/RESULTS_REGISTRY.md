# Results Registry

更新：2026-07-23

本文件是所有实验产物的状态真相源。结果只能处于以下状态之一：

- **VALIDATED**：环境、协议、对照、统计和 artifact 均满足当前计划，可作为论文证据；
- **PILOT_ONLY**：只验证现象或管道，不能作为正式数字；
- **INVALIDATED**：发现 sampler、泄漏、对照或 estimand 问题，只能作取证/回归记录；
- **ARCHIVED**：属于已放弃路线，仅用于保存研究历史。

当前没有任何 semantic-safety 结果达到 VALIDATED。

WP-2.0 状态边界：`PROTOCOL.md` 1.2.1 的 numeric/evaluator 语义保持冻结；
1.2.2 只登记 M04 structured prompt amendment。本月不再增加 JSON lexical、
binary64 或 serialization 细节，也不在 W2 实现完整标准解释器。所有付费 VLM/OpenRouter/API
run 暂停。

---

## 当前状态总表

| 结果/产物 | 状态 | 原因 | 是否保留 |
|---|---|---|---|
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
| zone_detector self-test | PILOT_ONLY | 证明 renderer-palette detector plumbing；不代表 open-vocabulary baseline | 可重新生成 |
| PointPush expert smoke | PILOT_ONLY | 只验证环境/专家可运行，尚未接入新 protocol | 可重新生成 |
| outputs/*.png 旧结果图 | ARCHIVED | 图片用于理解历史路线，但不能支撑当前 claim | 是，文档需标注状态 |

## 当前 backend 与运行状态

这些是仓库/基础设施状态，不是论文结果状态：

| Backend / run gate | 状态 | 说明 |
|---|---|---|
| Safety-Gymnasium native Goal adapter | `INFRA` | `SafetyPointGoal1-v0` native vertical slice 可复用，但尚无论文级结果。 |
| Safety-Gymnasium semantic variant | `BLOCKED` | 语义 variant 的 protocol gate 与 evidence 尚未完成；不进入当前 MVF。 |
| B01 sampler | **RESOLVED** | checked final-attempt grid fallback 保留正常 resample/golden 序列；四种配置各 10,000 seeds 的正式 invariant sweep 于 2026-07-22 通过。 |
| Unified direct/replay harness | **INFRA / PASSED** | `direct/replay × none/oracle` vertical slice、shared enforcement、artifact reconstruction 和 STC audit 已通过专项测试。 |
| Paid VLM/OpenRouter/API runs | **PAUSED** | 14-condition offline matrix 已通过；`PAID_RUN_RELEASE.md` 仍为 `BLOCKED / NOT AUTHORIZED`，model list、pilot subset、预算、日期和 operator 未冻结。 |

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

## 新结果进入 VALIDATED 的条件

必须同时满足：

1. RESEARCH_REVIEW_COMMENTS 的 B01–B07 全部 RESOLVED；
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
