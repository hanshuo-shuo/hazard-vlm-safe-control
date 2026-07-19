# Research Review Comments — 顶会路线修改清单

日期：2026-07-11

适用范围：当前 PointHazard / semantic-zone / VLM subgoal 主线

审查结论：**Request major revision。当前代码是有价值的研究原型，但现有语义结果不能作为论文数字，现有系统叙事不足以支撑 CoRL/ICLR 主会。**

本文件采用类似 PR review comment 的格式。每条 comment 都包含位置、问题、要求修改和验收条件。状态只允许：

- OPEN：尚未修改；
- IN PROGRESS：正在修改，但验收条件未全部满足；
- RESOLVED：代码、测试和新结果均满足验收条件；
- WONTFIX：明确放弃对应 claim，并在所有文档中删除。

---

## 总览

| ID | 严重度 | 主题 | 当前状态 | 是否阻止 paid run |
|---|---|---|---|---|
| B01 | BLOCKER | semantic-zone fallback 产生无效场景 | RESOLVED | 是 |
| B02 | BLOCKER | MPC 强安全措辞与实现不符 | RESOLVED（降级路线） | 否 |
| B03 | BLOCKER | C2/CV 与 B+ 的 planner/router 不一致 | OPEN | 是 |
| B04 | BLOCKER | 现有 semantic n=5/n=20 结果必须重跑 | OPEN | 是 |
| B05 | BLOCKER | task specification 与 leakage 混为一谈 | OPEN | 是 |
| B06 | BLOCKER | L0/L1/L2 不是受控的单变量干预 | OPEN | 是 |
| B07 | BLOCKER | audit artifact 没保存输入侧证据 | OPEN | 是 |
| M01 | MAJOR | 指标没有以 safe task completion 为主 | OPEN | 否 |
| M02 | MAJOR | “one VLM vs N detectors” 没有实验 | OPEN | 否 |
| M03 | MAJOR | 缺现代 open-vocabulary perception baseline | OPEN | 否 |
| M04 | MAJOR | free-text naming 不能当 grounded understanding | OPEN | 否 |
| M05 | MAJOR | policy interface 未实现信息流隔离 | OPEN | 否 |
| M06 | MAJOR | 协作/保证类术语过强 | OPEN | 否 |
| M07 | MAJOR | run manifest 与仓库复现性不完整 | OPEN | 否 |
| M08 | MAJOR | PointPush/VLA 目前只是 scaffold | OPEN | 否 |

---

## Blocking comments

### [B01] semantic-zone fallback 跳过有效性检查

**状态：RESOLVED（2026-07-17）**

WP-1.1 已移除未经检查的 zone fallback，完整 layout 失败时改为确定性子流重采样，并在 `reset()` 末尾执行独立 validity assertion。WP-1.2 新增 `tests/test_layout_invariants.py`，覆盖单区 implicit/三区 hetero 与 corridor on/off 四种配置，逐类断言 zone-hazard、zone-zone、zone-start、zone-goal 重叠数为零，并加入固定 seed 的 golden snapshot。依赖环境记录在 `requirements.lock`。

**位置**

- env_pointhazard.py:217–297，尤其是 281–289；
- env_pointhazard.py:183–185 的 hazard pair separation 公式。

**问题**

正常 rejection sampling 失败后，fallback 直接把语义区放到 start→goal 走廊位置，不再检查它是否与 start、goal、red hazard 或已有 zone 重叠。默认单区配置下复算 200 seeds，有 61 个 zone 与 red hazard 重叠；held-out seeds 100–119 中有 7/20 重叠。

这 7 个 held-out 重叠场景恰好是 blind MPC 没有 semantic violation 的全部 7 个场景；剩余 13 个有效场景里，blind MPC 是 13/13 违规。因此当前 C1=65% 不是目标分布的有效估计。

**要求修改**

1. 禁止未经检查的 fallback；放置失败时应重新采样 layout、扩大合法搜索区域，或显式 raise。
2. 将 zone validity 写成独立函数，并在 reset 后 assert。
3. 核对 min_hazard_pair_sep 的实现与注释；当前公式包含 “-1.0”，与配置语义不一致。
4. 增加大规模 deterministic layout test。

**验收条件**

- 至少 10,000 个固定 seeds 中，start/goal/hazard/zone 非法重叠均为 0；
- 每个生成场景都有 layout_valid=true 和 placement_attempts 记录；
- 旧 semantic JSON 全部标为 invalidated，不与新结果混用。

---

### [B02] MPC 采用 empirical safety-oriented sampling MPC 表述

**状态：RESOLVED（降级路线，2026-07-17）**

已将代码、docstring、README、结构说明和当前研究计划中的强安全表述统一降级为
**safety-oriented sampling MPC（empirical）**。MPC 的安全表现只作为 sampled rollout
上的 hazard、semantic violation、success 和 clearance 等 empirical metrics 报告；当前实现
没有 feasibility shield、backup policy、recursive feasibility 或 terminal invariant set，
因此不建立形式化安全性质。

**位置**

- mpc_expert.py:105–109；
- mpc_expert.py:153–166；
- mpc_expert.py:191–215；
- README.md、STRUCTURE.md、subgoal_pivot_hazard.py 以及相关 docstring 中的强安全文案。

**问题**

当前 CEM-MPC 对预测碰撞只加有限 penalty，没有 feasibility mask、无安全样本检测、backup action、recursive feasibility 或 terminal invariant set。即使所有 sampled rollout 都碰撞，act() 仍会执行最低 cost 序列的第一个动作。环境和 MPC 也只检查离散时刻碰撞。

**已执行与验收**

- 统一为 exact-model, safety-oriented sampling MPC（empirical）；
- collision、semantic violation、success 和 clearance 只作为 sampled rollout 的 empirical
  metrics 报告；
- 未引入形式化 safety shield，因此不作形式化安全声称；
- 当前仓库目标词 grep 仅剩明确标为 INVALIDATED 的历史结果原文。

---

### [B03] C2/CV 与 B+ 不是只差 zone source

**位置**

- subgoal_pivot_hazard.py:555–601；
- subgoal_pivot_hazard.py:644–668；
- subgoal_pivot_hazard.py:859–944；
- mpc_expert.py:88–95。

**问题**

C2-soft/CV 一次性直接 plan 到真实 goal；B+ 每隔若干步由 VLM 选择局部 subgoal，并在每次 plan 时清空 MPC nominal。三者的 halo 半径来源也不同。因此当前 “B+ 100% success vs C2/CV 90%” 可能来自 router、MPC restart 或 cost geometry，而不是感知来源。

**要求修改**

实现显式全因子设计：

- Router：direct-goal / fixed classical waypoint / VLM waypoint；
- Zone source：none / oracle / open-vocab detector / VLM；
- Enforcement：同一 hard-core/soft-halo 参数族。

主对比必须固定 router 和 enforcement，只改变 zone source。可额外用 replayed subgoal sequence 做 paired counterfactual。

**验收条件**

- 每个 headline comparison 只有一个被操纵变量；
- 结果表包含 router × zone-source 的交互项；
- 不再把单个 operating point 称作 Pareto frontier。

---

### [B04] 当前 semantic 主结果不得继续扩量

**位置**

- outputs/semantic_*.json；
- outputs/vlm20_l1_heldout.json；
- outputs/vlm20_l2_heldout.json；
- docs/RESULTS_SEMANTIC.md；
- docs/RESULTS_AMPLIFY.md；
- docs/RESULTS_FAIR_BASELINES.md。

**问题**

现有结果同时受到 B01 的无效场景、B03 的 planner confound，以及 B05/B06 的任务定义问题影响。n=100 只会让错误 estimand 的置信区间更窄。

**要求修改**

- paid run 暂停；
- 旧结果只保留为 historical/debugging evidence；
- 修完 B01–B07、锁定 protocol、通过 offline assertions 后重新生成正式 seed block。

**验收条件**

- 新结果使用新 run ID、git SHA、protocol version 和不重叠的 seed split；
- 论文表格不混用 invalidated 与新结果。

---

### [B05] 合法任务规格与泄漏必须分开

**位置**

- subgoal_pivot_hazard.py:268–378；
- subgoal_pivot_hazard.py:1091–1099；
- 旧 ICLR plan 的 L0/L1/L2 解释。

**问题**

“避开会损坏轮式机器人的地形”是合法 task specification；“水在坐标 (x,y)”“candidate 7 不安全”“clearance=0.2”才是 privileged leakage。当前 L1→L2 同时删除了安全任务要求，因此测到的是 instruction removal，不是单纯的 leakage dose。

L2 中 prompt 没要求避水却在 evaluator 中惩罚穿水，还缺少机器人 capability 定义；对两栖或防水机器人，这个 label 可能相反。

**要求修改**

固定以下公开信息：

- robot embodiment/capability card；
- generic safety constitution；
- task goal。

只操纵 scene-specific privileged information：

1. raw pixels；
2. semantic class list；
3. mask/coordinates；
4. candidate safe/unsafe flag；
5. clearance/score/ranking。

**验收条件**

- 所有 leakage conditions 的任务目标完全一致；
- 对同一图像加入 capability twin，例如 wheeled/non-waterproof vs amphibious；
- evaluator label 可由 task+capability 唯一推出。

---

### [B06] L0/L1/L2 必须改成正交因子

**位置**

- subgoal_pivot_hazard.py:268–298；
- subgoal_pivot_hazard.py:1091–1121。

**问题**

renderer appearance 由 zone_semantics 控制，prompt text 由 prompt_level 控制。默认 L0→L1 会同时把 amber-X 换成 water；若固定 water 再指定 L0，prompt 又会错误描述 amber-X。当前实现不能生成严格受控的完整 prompt ladder。

**要求修改**

将以下因子完全解耦：

- visual appearance；
- task specification；
- privileged cue level；
- candidate annotation；
- evaluator semantics。

**验收条件**

- 生成器能对同一底图只改一项文本；
- 每次 run 保存 factor vector；
- 自动 snapshot test 验证非目标因子逐字节不变。

---

### [B07] transcript audit 缺输入侧证据

**位置**

- subgoal_pivot_hazard.py:471–487；
- subgoal_pivot_hazard.py:1201–1289；
- outputs/*.transcripts.json。

**问题**

当前只保存 response/raw/choice/parse flags，没有实际 prompt、图片或 hash、candidate world/image coordinates、请求 payload、provider/model revision、完整超参和代码版本。无法仅凭 artifact 复核输入侧是否泄漏。

**要求修改**

每次 VLM call 保存：

- exact prompt；
- image file 或 cryptographic hash；
- candidate metadata；
- allowed information tags；
- model/provider/version/request ID；
- temperature、seed（若支持）、timestamp、latency、cost；
- parser output 与 fallback；
- git SHA、dirty flag、完整 CLI/config。

**验收条件**

- 单个 episode artifact 足以离线重放 VLM 输入和 policy decision；
- audit script 能自动证明无 EVAL_ONLY/PRIVILEGED 字段进入 policy payload。

---

## Major comments

### [M01] 主指标改成 safe task completion

当前 success 和 semantic violation 分开报告，会产生“成功但违规”的错觉。主指标应为：

> safe task completion = reached goal AND zero semantic violation AND zero physical collision

并同时报告 violation severity、dwell/exposure、false intervention、timeout、path efficiency 和 calls/cost。

**验收条件**：所有 headline table 第一列是 safe task completion；边际指标只用于诊断。

---

### [M02] “one VLM vs N detectors” 目前只是计划

最新 n=20 只有单 water zone；heterogeneous 结果只有旧 n=5，且缺公平 oracle/CV/open-vocab baseline。

**要求修改**：在没有 terrain-count curve、held-out terrain 和工程成本定义前，删除该 headline claim。

---

### [M03] 颜色检测器不是现代 perception baseline

renderer-palette detector 是 sanity check，不足以代表当前机器人感知栈。

**要求修改**：至少加入一个 open-vocabulary detector/segmenter → 相同 cost map → 相同 planner，并记录 latency/compute。

---

### [M04] free-text naming 不能证明理解

“说出 water/blue”可能是 post-hoc rationale。现有 L2 中多次出现模型明确提到 blue/water 但仍选择穿越。

**要求修改**：用 forced-choice recognition、norm applicability、mask/marker grounding 和 action choice 四个机器可判阶段；free-text 只作定性例子。

---

### [M05] policy interface 需要最小权限

policy.reset 当前接收含 true semantic_zones 的 info，policy.act 接收完整 env。虽然真实 VLM 分支暂未读取这些字段，但系统级 non-interference 不能靠“目前没用”保证。

**要求修改**：为 policy 提供最小 observation/render interface；privileged evaluator state 放到独立对象。

---

### [M06] 统一术语

- 没有人类 joystick/operator intent/user study，不称人机协作系统；
- 推荐称 hierarchical VLM-guided autonomy；
- 不再使用 oracle 对齐、常识理解、保证安全等未经结果支持的词。

---

### [M07] 固化复现环境

增加依赖锁定、测试入口、run manifest、统计脚本、protocol version 和 README 中的有效/无效结果标记。闭源模型 alias 必须记录日期和 provider revision，并用开放权重模型作为主要复现路径之一。

---

### [M08] PointPush/VLA 不计入当前论文证据

PointPush 环境、专家和 direct-VLA 代码是可复用资产，但还没有新 protocol 下的 semantic twins、factorial baselines 或正式结果。除非完成这些内容，否则只在 future work/implementation assets 中提及。

---

## 允许恢复 paid run 的门槛

只有以下条件全部满足后才允许开始正式 API 实验：

- B01–B07 全部 RESOLVED；
- protocol、prompt、factor levels、seed split 和主指标冻结；
- 至少一个 open-vocabulary baseline 跑通；
- router × zone-source 全因子离线测试通过；
- 100+ seeds 的 layout/assertion/metric regression tests 通过；
- 完整 call artifact 可离线审计；
- 先用 3 个模型、2 个环境做小规模 go/no-go pilot，确认机制效应不是单模型/单 toy 特例。

---

## 建议的 review resolution 顺序

1. B01 → B02：先保证环境与 claim 正确；
2. B05 → B06 → B07：再定义真正要测的 estimand；
3. B03 → M01：重构公平实验和指标；
4. M03 → M04 → M05：加入强 baseline 和可判阶段；
5. 最后处理 M02/M07/M08，决定 ICLR measurement 路线或 CoRL method 路线。
