# Execution status

本文件仍是主线切换的验收 prompt。执行状态更新于 2026-07-25：

- marker mainline：`TERMINATED`（termination date `2026-07-24`）；
- provider policy：本轮 `0` 个真实请求；历史 runner 默认 cache-only；
- future real-key invariant：每个 API key 最多 `5` 个 distinct scene seeds；
- active implementation：marker-free geometry、five-stage audit、cached
  detector calibration、Safety-Gym capability twins；
- scientific results：尚无新增；未运行数字必须保持 `TBD`。

---

你现在接手仓库：

`hanshuo-shuo/hazard-vlm-safe-control`

工作分支：

`safety`

请先完整阅读当前分支中的以下文件，不要立即修改代码：

- `README.md`
- `STRUCTURE.md`
- `docs/ICLR_PLAN.md`
- `docs/CLAUDE_PLAN.md`
- `docs/PROTOCOL.md`
- `docs/RESULTS_REGISTRY.md`
- `docs/RESEARCH_REVIEW_COMMENTS.md`
- `docs/PAID_RUN_RELEASE.md`
- 最新的 `next five experiments` pilot 总结
- 五个实验目录中的全部 `REPORT.md`
- `MANIFEST.json`
- `scripts/run_next_five_experiments.py`
- 与 PointHazard、VLM router、marker candidates、policy interface、semantic evaluator、detector adapter、MPC、Safety-Gymnasium adapter、artifact provenance 和 STC 有关的源码与测试

## 一、当前研究决定

最新五种子 pilot 已经触发预先设定的 kill condition：

- marker-ID permutation 下，物理选择一致率只有 20%；
- candidate coordinate order permutation 下，一致率只有 30%；
- direct-coordinate control 一致率 40%；
- unmarked-image control 一致率 10%；
- P0/P1/P2 selected-safe rate 为 50% / 60% / 40%；
- P2 没有优于 P0；
- 没有观察到模型排名反转；
- candidate 数量变化会显著改变 safe-choice rate。

因此，正式终止：

`marker-based PointHazard VLM waypoint ICLR mainline`

不要通过增加 seeds、修改 prompt、重新编号 marker、调整 candidate 数量或挑选有利参数来挽救这条主线。

Marker-based 实验只能作为：

1. interface instability 的负面证据；
2. regression test；
3. historical pilot；
4. 说明 marker/candidate interface 会污染语义安全结论的 case study。

新的优先研究方向是：

1. marker-free semantic geometry interface；
2. detector/segmenter → shared geometry → fixed planner；
3. capability-conditioned norm applicability；
4. perception error 与 downstream enforcement 的交互；
5. PointHazard 保留为因果单元测试；
6. Safety-Gymnasium 作为下一阶段主要验证环境。

本次工作不得启动新的付费 provider 实验。除已有缓存 replay 外，不允许调用外部模型 API。

---

# 二、本轮总体目标

完成一次严格、可审计的研究主线切换，使仓库达到以下状态：

1. marker 主线在文档、registry、CLI 和测试中被正式冻结；
2. 建立统一的 marker-free semantic geometry contract；
3. 将 detector、oracle、fixture 和未来 VLM grounding 输出统一映射到同一个 geometry schema；
4. 建立 detector geometry decomposition 与 planner calibration 的离线实验框架；
5. 在 Safety-Gymnasium 中建立 capability twins 的正式协议与可运行 harness；
6. 五阶段 audit 输出被明确拆分为 recognition、applicability、grounding、action 和 enforcement；
7. 所有新设计均有测试、artifact、hash、manifest 和明确的 paper-evidence status；
8. 不制造任何尚未运行的科学结果。

请把本轮工作视为一个较大的 repository refactor + experiment infrastructure package，连续完成，不要只改一两个文件后停止。

---

# 三、工作包 A：正式冻结 marker-based 主线

## A1. 更新研究决策文档

修改：

- `README.md`
- `STRUCTURE.md`
- `docs/ICLR_PLAN.md`
- `docs/CLAUDE_PLAN.md`
- `docs/RESULTS_REGISTRY.md`
- `docs/RESEARCH_REVIEW_COMMENTS.md`
- `docs/PAID_RUN_RELEASE.md`

明确写入：

- marker-based PointHazard ICLR mainline：`TERMINATED`
- evidence status：`PILOT_ONLY / NOT PAPER RESULT`
- termination date：使用 pilot report 的生成日期
- termination reason：marker/interface instability
- 不得扩 seed
- 不得基于 marker experiments 声称稳定 semantic spatial reasoning
- P0/P1/P2 pilot 未支持 privilege monotonicity 或 rank reversal
- 当前最重要的正向线索是：
  - recognition-conditioned safety；
  - capability/norm applicability failure；
  - detector safety–completion trade-off；
  - Safety-Gymnasium port feasibility。

确保各文档状态一致，不要再出现同一任务在一个文件中标为 DONE、另一个文件中标为 OPEN 的情况。

## A2. Registry 规范

在 `RESULTS_REGISTRY.md` 中为五个 pilot 建立正式记录，至少包括：

- experiment ID；
- branch；
- commit SHA；
- seeds；
- models；
- provider request count；
- token count；
- cost；
- status；
- claim allowed；
- claim forbidden；
- artifact path；
- kill condition 是否触发；
- follow-up action。

明确区分：

- integration evidence；
- pilot scientific evidence；
- formal paper evidence；
- invalidated/historical evidence。

## A3. CLI 防误用

检查所有可能运行 marker waypoint 大规模实验的 CLI。

增加防护：

- 默认拒绝 marker-based scale-up；
- 只有显式 `--allow-terminated-marker-pilot` 或同等清晰的 override 才能运行；
- override 时必须打印醒目警告；
- artifact 中必须记录 override；
- 不允许该 override 与 paid-provider release 默认同时启用。

不要删除旧代码，以保证复现性。

---

# 四、工作包 B：建立 marker-free semantic geometry contract

设计一个统一、最小、不可泄漏 evaluator truth 的 geometry contract。

建议建立新的模块，例如：

`evaluation/semantic_geometry.py`

具体命名可根据仓库结构调整，但必须保持职责单一。

## B1. 核心数据结构

至少定义以下对象：

### SemanticRegion

包含：

- `region_id`
- `semantic_class`
- `geometry_type`
- `geometry`
- `coordinate_frame`
- `confidence`
- `source`
- `source_model`
- `source_artifact_hash`
- `applicability`
- `applicability_confidence`
- `provenance`
- `adapter_version`

支持的 geometry 至少包括：

- disk；
- axis-aligned bounding box；
- polygon；
- mask reference 或 raster mask；
- empty/no-region。

不要把 evaluator-only truth 放入 policy-visible schema。

### ApplicabilityDecision

至少包含：

- region ID；
- applicable / not applicable / unknown；
- robot capability依据；
- task依据；
- confidence；
- structured reason code；
- free-text explanation 仅作为非计分辅助字段。

### SemanticGeometryPayload

包含：

- scene/request identity；
- image hash；
- coordinate-frame metadata；
- image-to-world transform version；
- regions；
- parser status；
- adapter status；
- fallback status；
- provenance；
- schema version。

## B2. 坐标系统

必须明确区分：

- image pixel coordinates；
- normalized image coordinates；
- provider-native coordinates，例如 0–1000；
- world coordinates；
- simulator coordinates；
- planner cost-map coordinates。

实现并测试：

- provider-native → pixel；
- normalized → pixel；
- pixel → world；
- world → pixel；
- round-trip tolerance；
- out-of-bounds rejection；
- malformed geometry rejection；
- image resize / letterbox handling；
- coordinate-frame versioning。

禁止使用未经记录的隐式缩放。

## B3. 数据源适配器

为以下来源实现统一 adapter：

1. `none`
2. `oracle`
3. `fixture`
4. `detector`
5. 未来的 `vlm_grounding`

当前不需要真实调用新的 VLM，但要为未来接口预留严格 schema。

所有 adapter 必须输出同一 `SemanticGeometryPayload`。

## B4. 权限审计

更新 `PolicyInput` 和 forbidden-field audit，确保：

- P0 看不到 semantic region；
- detector 只能看到公开 RGB/task/capability；
- oracle 只能用于明确的 oracle condition；
- evaluator truth 不能通过 dataclass、closure、env reference、metadata、debug field 或 serialized object 泄漏；
- artifact 中记录实际暴露给 policy 的 payload bytes/hash。

---

# 五、工作包 C：重构 detector → planner 路线

当前 detector pilot 使用 disk adapter，结果是 0% semantic violation、40% success。下一步要判断损失来自哪里，而不是直接盲调 MPC。

## C1. Geometry decomposition framework

建立可离线 replay 的实验框架，至少支持以下 arm：

1. detector full geometry；
2. detector center + detector radius；
3. detector center + oracle radius；
4. oracle center + detector radius；
5. oracle center + oracle radius；
6. detector bounding box；
7. detector polygon/convex hull；
8. detector mask；
9. blind；
10. oracle。

若当前 detector artifact 只提供 box 或 center，应以不伪造数据为原则：

- 可以从已有 box 得到 disk、box、polygon；
- 不得凭空生成真实 segmentation mask；
- synthetic mask 只能作为 adapter sensitivity，并明确标记。

## C2. Geometry metrics

每个 scene 至少输出：

- center error；
- radius error；
- area error；
- IoU；
- false-positive area；
- false-negative area；
- Hausdorff 或 boundary distance，若实现合理；
- planner minimum clearance；
- path length；
- success；
- semantic violation；
- physical collision；
- timeout；
- STC；
- intervention count；
- replan count。

## C3. Dev/test split

建立冻结的 scene-family split：

- dev/calibration seeds；
- pilot/report seeds；
- future formal test seeds。

不得使用当前五个 pilot seeds 反复选择参数。

在协议中明确：

- dev seeds 可用于 adapter 和 planner calibration；
- test seeds 在参数冻结后只运行一次；
- 所有参数选择写入 release artifact。

## C4. Planner operating curve

在不改变 detector 输出的前提下，支持单因素或受控网格 sweep：

- hard radius inflation；
- soft halo；
- semantic penalty；
- collision penalty；
- replanning interval；
- target arrival tolerance。

不要一次无约束搜索所有参数。

优先实现：

- 固定 geometry adapter；
- 每次只改变一个参数族；
- 输出 success–violation Pareto curve；
- 输出 STC、path efficiency 和 conservatism；
- 保存完整 config hash。

## C5. 防止 benchmark overfitting

加入：

- scene family holdout；
- seed holdout；
- parameter budget；
- maximum number of calibration trials；
- frozen-selection record；
- no peeking test；
- test-run sentinel。

---

# 六、工作包 D：重构五阶段 audit protocol

当前五阶段需要从“报告字段”升级为可计分、可审计的协议。

建议统一为：

1. `recognition`
2. `applicability`
3. `grounding`
4. `action_proposal`
5. `enforcement_outcome`

## D1. Recognition

结构化输出：

- recognized classes；
- class confidence；
- region references；
- unknown/uncertain。

评价：

- class precision/recall；
- region-conditioned recognition；
- appearance-twin consistency。

## D2. Applicability

结构化输出：

- applicable / not applicable / unknown；
- capability reference；
- task reference；
- reason code。

评价：

- applicability accuracy；
- capability-twin reversal accuracy；
- false incompatibility；
- missed incompatibility；
- recognition-correct but applicability-wrong rate。

这里必须明确：

`recognition correct != applicability correct`

## D3. Grounding

结构化输出：

- region geometry；
- region ID；
- uncertainty；
- coordinate frame。

评价：

- center/area/IoU；
- applicable-region grounding accuracy；
- wrong-region grounding；
- correct recognition + correct applicability + wrong grounding。

## D4. Action proposal

结构化输出：

- target/route；
- expected terrain interaction；
- whether avoidance is intended；
- feasibility status。

评价：

- proposed path intersects applicable unsafe region or not；
- route reversal in capability twins；
- proposal feasibility；
- unnecessary detour。

## D5. Enforcement outcome

结构化输出：

- accepted；
- modified；
- blocked；
- fallback；
- executed；
- final trajectory；
- violation；
- task success；
- STC。

评价：

- upstream error absorbed；
- upstream error amplified；
- correct proposal damaged by enforcement；
- unsafe proposal rescued；
- over-conservative block。

## D6. Error taxonomy

实现自动 row-level taxonomy，例如：

- `RECOGNITION_FAILURE`
- `APPLICABILITY_FAILURE`
- `GROUNDING_FAILURE`
- `ACTION_SELECTION_FAILURE`
- `ENFORCEMENT_FAILURE`
- `OVERCONSERVATIVE_ENFORCEMENT`
- `MULTI_STAGE_FAILURE`
- `SUCCESSFUL_RECOVERY`
- `UNATTRIBUTABLE`

不得强迫每个失败只属于一个阶段；同时保留：

- earliest detectable failure；
- contributing failures；
- final failure mode。

---

# 七、工作包 E：Safety-Gymnasium capability twins

当前 `SafetyPointGoal1-v0` port 已通过 headless integration smoke。现在建立正式 capability-twin infrastructure，但本轮不做 paid provider experiment。

## E1. Twin contract

每一对 twin 必须保持：

- identical scene seed；
- identical geometry；
- identical RGB rendering，除非实验是 appearance twin；
- identical goal；
- identical initial state；
- identical native Safety-Gym cost；
- identical candidate/geometry interface；
- identical planner config；
- 只改变 capability card 或 applicability rule。

必须有自动 test 验证 twin invariance。

## E2. 首批 capability families

至少实现协议与 fixture：

### Family 1: Water

- non-waterproof robot：water applicable unsafe；
- amphibious robot：water not applicable unsafe。

### Family 2: Mud / rough terrain

- ordinary wheeled robot：mud/rough terrain incompatible；
- tracked/all-terrain robot：compatible。

如果 Safety-Gym native environment 不支持可视化 mud，可先作为 registered semantic overlay，但必须与 native cost 严格分离。

### Family 3: Clearance or footprint

- large robot：narrow passage incompatible；
- compact robot：compatible。

如果 dynamics 尚未建模 robot size，本轮可以只建立 protocol fixture 和 evaluator contract，但不得虚构实际 physical feasibility result。

## E3. Appearance twins

至少支持：

- visually water-like but semantically safe；
- visually ordinary but semantically unsafe；
- same applicability, different texture；
- same texture, different evaluator semantics。

Appearance twin 不得与 capability twin 混在同一主效应实验里。

## E4. Metrics

实现：

- applicability accuracy；
- capability reversal consistency；
- route reversal consistency；
- STC reversal consistency；
- false conservative avoidance；
- unsafe non-reversal；
- recognition/applicability gap；
- grounding/applicability gap；
- executor rescue rate。

## E5. Smoke baselines

本轮只需支持无需外部 provider 的 baseline：

- blind；
- oracle；
- fixture structured outputs；
- cached detector replay；
- zero-action smoke；
- safe scripted baseline；
- shortest-path/native planner baseline，若环境允许。

明确标注 zero-action 不是 performance baseline。

---

# 八、工作包 F：统一 runner、artifact 和 manifest

不要为每个实验写互不兼容的脚本。

建立或重构一个统一 runner，例如：

`scripts/run_semantic_geometry_audit.py`

支持：

- environment；
- seed range；
- scene family；
- router；
- geometry source；
- applicability source；
- enforcement config；
- capability profile；
- appearance profile；
- cached provider replay；
- no-provider mode；
- dev/test split；
- dry run；
- artifact output；
- deterministic hash；
- resume；
- failure recovery。

## Artifact 至少保存

- command；
- git SHA；
- dirty status；
- Python/package versions；
- simulator version；
- OS；
- scene seed；
- environment；
- task card；
- capability card；
- image bytes/hash；
- model-visible payload bytes/hash；
- geometry payload；
- applicability payload；
- adapter version；
- planner config；
- raw/cached provider response；
- parser status；
- fallback；
- trajectory；
- per-stage output；
- evaluator output；
- STC decomposition；
- token/cost，若适用；
- timestamps；
- result status；
- evidence class。

## Manifest

自动生成：

- experiment-level `MANIFEST.json`；
- row count；
- failed rows；
- resumed rows；
- duplicate request hashes；
- missing artifact audit；
- schema version；
- config hash；
- result digest；
- test status。

---

# 九、测试要求

请补充系统性测试，而不是只写 happy path。

至少包括：

## Geometry

- coordinate conversion；
- malformed box/polygon；
- out-of-range provider coordinates；
- letterbox/resize；
- world↔image round trip；
- empty detections；
- multiple detections；
- confidence ties；
- deterministic region ordering；
- adapter version hash。

## Policy boundary

- forbidden evaluator fields；
- hidden env references；
- nested objects；
- debug metadata leakage；
- serialized truth leakage；
- P0 no-privilege；
- oracle only in oracle arm。

## Twins

- identical geometry across capability twins；
- identical rendering across capability twins；
- only capability payload differs；
- appearance twin only changes designated appearance fields；
- native cost remains unchanged；
- semantic evaluator changes only where declared。

## Five-stage audit

- recognition correct / applicability wrong；
- applicability correct / grounding wrong；
- unsafe proposal rescued；
- safe proposal blocked；
- multi-stage failure；
- parser failure；
- fallback；
- unknown state；
- deterministic taxonomy。

## Runner/artifacts

- resume；
- duplicate rows；
- hash stability；
- partial run recovery；
- manifest verification；
- no API key leakage；
- no authorization header leakage；
- terminated marker override warning；
- dev/test sentinel。

---

# 十、文档输出

新增一份清晰的执行文档，例如：

`docs/MARKER_FREE_GEOMETRY_PLAN.md`

内容包括：

1. why marker mainline was terminated；
2. current scientific hypotheses；
3. marker-free architecture；
4. geometry contract；
5. five-stage protocol；
6. detector decomposition；
7. Safety-Gym capability twins；
8. dev/test discipline；
9. allowed claims；
10. forbidden claims；
11. experiment gates；
12. ICLR / CoRL / workshop decision criteria。

再新增：

`docs/NEXT_EXECUTION_MEMO.md`

要求写成真实项目 memo，包含：

- 完成了什么；
- 未完成什么；
- 当前 blockers；
- 哪些是 science；
- 哪些只是 infrastructure；
- 下一批无需 provider 的命令；
- 未来 paid run 前置条件；
- stop criteria。

---

# 十一、当前允许和禁止的结论

## 允许写入文档的结论

- marker/candidate interface 在五种子 pilot 中高度不稳定；
- 预声明 kill condition 已触发；
- P2 未显示优于 P0；
- pilot 未观察到模型 rank reversal；
- recognition correctness 与 selected-safe 强相关；
- capability applicability 出现明显失败信号；
- detector 在 pilot 中消除了 semantic violation，但任务完成率较低；
- Safety-Gymnasium minimal port 可运行。

## 禁止写入的结论

- VLM 普遍不能进行 norm reasoning；
- detector 优于 VLM；
- five-stage audit 已经证明主要瓶颈是 applicability；
- Safety-Gym 中现象已经复现；
- geometry calibration 一定会提高 STC；
- 本方法具有 formal safety guarantee；
- 当前结果达到 ICLR paper evidence；
- 任意跨模型、跨环境泛化结论。

所有未运行的数字必须写为 `TBD`，不得补造。

---

# 十二、验收标准

本轮完成后应满足：

1. marker 主线在全仓库状态一致；
2. 旧结果仍可复现，但默认不能误运行 scale-up；
3. marker-free geometry schema 完整；
4. oracle/fixture/detector 使用同一 contract；
5. provider/native/pixel/world 坐标转换有测试；
6. detector geometry decomposition 可通过 cached replay 离线运行；
7. planner calibration 有 dev/test 隔离；
8. Safety-Gym capability twins 可 headless 运行；
9. 五阶段输出可自动计分和归因；
10. artifacts 和 manifest 完整；
11. 无 API key 或 header 泄漏；
12. 所有测试通过；
13. 文档不夸大结果；
14. 无新的付费 provider 请求。

---

# 十三、执行方式

请按照以下顺序连续工作：

1. 仓库审计与状态梳理；
2. 写简短 implementation plan；
3. 冻结 marker 主线；
4. 实现 geometry schema；
5. 实现 adapters 和 coordinate transforms；
6. 重构 harness；
7. 实现 five-stage audit；
8. 实现 detector decomposition runner；
9. 实现 Safety-Gym twins；
10. 补测试；
11. 更新文档；
12. 运行完整验收；
13. 给出最终执行 memo。

不要中途因为某个非关键问题停下来询问。遇到局部歧义时，优先选择：

- 最小实现；
- 不夸大科学结论；
- 保留向后兼容；
- 明确标记 TODO；
- 通过测试固定行为。

不要重写整个仓库，不要大规模移动无关文件，不要修改历史结果内容，不要删除已有 artifacts。

---

# 十四、最终回复格式

完成后请按以下结构汇报：

## 1. Executive summary

说明主线切换是否完成。

## 2. Files changed

按模块分类列出关键文件及作用。

## 3. Architecture after refactor

用简洁数据流说明：

`public observation → recognition/applicability/geometry → fixed planner → enforcement → STC`

## 4. Marker termination enforcement

说明 CLI、registry、docs 和 tests 如何防止误用。

## 5. Geometry contract

说明 schema、coordinate transforms 和 adapters。

## 6. Five-stage audit

说明各阶段字段、metrics 和 taxonomy。

## 7. Safety-Gym twins

说明已实现的 twin families、invariants 和 smoke results。

## 8. Detector calibration framework

说明已支持的 geometry arms、metrics、dev/test split 和 operating curves。

## 9. Tests

列出实际执行命令、通过数量、跳过项和失败项。

## 10. Reproducibility

说明 manifests、hash、resume 和 secret audit。

## 11. Remaining blockers

分为：

- science blockers；
- engineering blockers；
- paid-run blockers。

## 12. Exact next commands

给出下一位执行者可以直接复制运行的无需 provider 命令。

## 13. Claim boundary

列出现在可以声称和不能声称的内容。

## 14. Final recommendation

明确给出当前状态：

- ready for cached offline calibration；
- ready/not ready for Safety-Gym provider pilot；
- ready/not ready for paid formal experiment；
- ICLR mainline status。
