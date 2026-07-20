你现在是这个项目的高级研究工程师。请直接在当前仓库中完成 Safety-Gymnasium Goal 环境的主线接入，不要只给建议或伪代码，要检查代码、修改文件、运行测试并汇报结果。

仓库与分支：

- Repository: `hanshuo-shuo/hazard-vlm-safe-control`
- Target branch: `agent/wp-1-1-semantic-zone`
- Target Safety-Gymnasium environment: `SafetyPointGoal1-v0`
- 本次不要接入 Push、Car、Ant 或其他环境。
- 本次禁止产生任何收费的 VLM/OpenRouter/API 调用。

---

# 1. 总目标

将 `SafetyPointGoal1-v0` 作为当前 safety-accounting 主线的第二个环境 backend 接入。

它不能替代现有 PointHazard，而应与 PointHazard 共用统一的：

- policy-facing environment interface；
- router / zone source / enforcement 分层；
- evaluator；
- episode artifact；
- seed 和 provenance 记录；
- CLI 入口；
- replay-ready 数据格式。

最终应能通过一个明确的 CLI 参数，在以下 backend 之间切换：

```text
point_hazard
safety_gym_goal
```

本次完成的不是完整论文实验，而是一个经过测试的、无权限泄漏的 vertical slice。

---

# 2. 开始前必须阅读

先仔细阅读并总结以下文件的约束，再开始修改：

```text
README.md
docs/CLAUDE_PLAN.md
docs/PROTOCOL.md
docs/MEMO_01.md
docs/RESULTS_REGISTRY.md
docs/RESEARCH_REVIEW_COMMENTS.md
docs/ICLR_PLAN.md
env_pointhazard.py
subgoal_pivot_hazard.py
tests/test_layout_invariants.py
tests/test_factor_orthogonality.py
```

重点确认：

1. 当前 Protocol 的 policy-visible 和 evaluator-only 信息边界；
2. 当前 policy 是否仍通过 `env`、`info` 或 `semantic_zones` 获取 privileged state；
3. 当前 router、zone source、enforcement 是否仍然耦合；
4. 当前 artifact 缺少哪些 provenance 字段；
5. Safety-Gymnasium 应该接入哪一层，而不是把旧的 monolithic harness 再复制一份。

先输出一个简短实施计划，然后直接执行，不需要等待确认。

---

# 3. 研究语义约束

必须严格区分两类安全信息。

## 3.1 Physical hazards

Safety-Gymnasium 原生 hazards 属于 physical constraints：

- 使用原生 `cost`；
- 对所有 capability 都构成约束；
- 可以通过 Safety-Gymnasium 的合法 public observation 被感知；
- evaluator 可以保存其真实 simulator state；
- policy 不得直接读取 master coordinates。

## 3.2 Semantic terrain

Semantic terrain 是本项目新增的 contextual rule：

- 同一块 terrain 是否违规取决于 capability；
- 例如 water 对普通 wheeled robot 违规，对 amphibious robot 不违规；
- semantic terrain 不得直接产生 Safety-Gymnasium native cost；
- semantic terrain 不得作为精确坐标、半径、terrain class 注入 policy observation；
- semantic terrain ground truth 只能进入 evaluator context；
- policy 只能通过允许的 RGB、prompt、detector output 或 replay source 获得相关信息。

不要把原生 hazard 政名为 semantic hazard。这样会破坏 capability twin 和论文 estimand。

---

# 4. 强制架构要求

优先增加新的小模块，不要继续把逻辑堆进 `subgoal_pivot_hazard.py`。

可以根据仓库现有结构调整具体目录，但逻辑上至少需要以下组件：

```text
envs/
    protocol_env.py
    point_hazard_adapter.py
    safety_gym_goal_adapter.py

evaluation/
    schemas.py
    semantic_evaluator.py

scripts/
    smoke_safety_gym_goal.py
```

若仓库当前不适合建立这些目录，可以使用等价结构，但必须保持职责分离。

## 4.1 统一环境接口

定义清晰的 protocol-facing environment abstraction，至少包括：

```python
reset(seed)
step(action)
public_observation()
render_public_rgb()
evaluator_context()
scene_manifest()
close()
```

其中：

- `public_observation()`：只能返回 policy 合法可见的信息；
- `render_public_rgb()`：返回实际可发送给 VLM 的 RGB，不允许带 debug ground-truth overlay；
- `evaluator_context()`：保存 simulator truth，但不得传给 policy；
- `scene_manifest()`：用于 artifact 和 replay；
- policy 不允许接收裸 `env` 对象。

不要让 policy 通过下面任何路径读取 privileged information：

```python
env.semantic_zones
env.unwrapped
env.model
env.data
reset_info["semantic_zones"]
step_info["semantic_zones"]
evaluator_context
scene_manifest 中的 ground-truth geometry
```

如果现有 PointHazard policy interface 仍存在这些违规，本次应做最小必要重构，使两个 backend 都走统一接口。

---

# 5. 分阶段实施

严格按 Gate 顺序完成。前一阶段测试未通过时，不要继续扩大范围。

## Gate A：Safety-Gymnasium 基础适配

先完成标准 `SafetyPointGoal1-v0` adapter，不添加 semantic terrain。

必须支持：

- deterministic `reset(seed=...)`；
- Safety-Gymnasium 六元 step API：

```python
obs, reward, cost, terminated, truncated, info
```

- `rgb_array` rendering；
- action space 和 observation space；
- native reward；
- native cost；
- terminated / truncated；
- episode length；
- scene/environment metadata；
- clean close；
- headless execution。

不要假设固定使用 OSMesa。支持通过环境变量选择 MuJoCo backend，并在文档中说明常见选项，例如 EGL 或 OSMesa。

依赖应作为 optional dependency 或独立 requirements 文件加入，避免在未安装 Safety-Gymnasium 时破坏 PointHazard。

测试中可以使用：

```python
pytest.importorskip("safety_gymnasium")
```

但必须额外提供一个安装依赖后的真实 smoke test。

---

## Gate B：统一 artifact

每个 Safety-Gymnasium episode 至少记录：

```json
{
  "protocol_version": "...",
  "environment_backend": "safety_gym_goal",
  "environment_id": "SafetyPointGoal1-v0",
  "environment_version": "...",
  "seed": 0,
  "initial_observation_summary": {},
  "actions": [],
  "rewards": [],
  "native_costs": [],
  "terminated": false,
  "truncated": false,
  "termination_reason": "...",
  "trajectory": [],
  "scene_manifest": {},
  "git_sha": "...",
  "git_dirty": false,
  "dependency_versions": {}
}
```

要求：

- JSON serializable；
-每个 episode 独立保存；
- artifact schema 尽量复用 PointHazard；
-不能把 policy 不应看到的信息重新注入 policy；
-记录 Safety-Gymnasium、Gymnasium、MuJoCo、Python 版本；
-记录当前 git SHA 和 dirty state。

如果已有 artifact schema，扩展或复用它，不要创建第二套互不兼容的格式。

---

## Gate C：最小主线 vertical slice

让 `SafetyPointGoal1-v0` 能进入当前主线 CLI。

至少支持一个无 VLM、无 semantic terrain 的最小运行：

```text
environment_backend = safety_gym_goal
router = direct_goal 或等价 deterministic router
zone_source = none
enforcement = none
```

要求：

- 不调用 OpenRouter/VLM；
- 能运行至少 2 个 deterministic episodes；
- 同一个 seed 重复执行时初始场景和前若干步结果一致；
- 输出 artifact；
- 输出 success、return、native cost、episode length；
- 不声称产生任何论文级 semantic-safety result。

如果当前 CLI 仍使用 `L0/L1/L2`，不要继续扩展旧 prompt-level 设计。环境接入应使用 Protocol 中的新 factor abstraction，或者至少保持与未来 P0–P4 兼容。

---

## Gate D：Semantic terrain extension

只有 Gate A、B、C 全部通过后，才实现 semantic terrain。

创建 Safety-Gymnasium Goal 的最小 semantic variant，例如：

```text
SemanticSafetyPointGoal1-v0
```

或者等价的内部 task class。

第一版只需要一种 terrain：

```text
water
```

以及两种 capability：

```text
wheeled_non_waterproof
amphibious
```

必须满足：

1. terrain 在 public RGB 中可见；
2. terrain 是 non-colliding visual region；
3. terrain 不计入 native Safety-Gymnasium cost；
4. terrain ground-truth geometry 和 class 只进入 evaluator；
5. policy observation 不包含 terrain 的精确中心、半径或类别；
6. evaluator 根据 trajectory、terrain、capability 独立判断 semantic violation；
7. 同一个 scene seed 能生成 capability twin；
8. capability 改变时，物理布局、terrain 外观、goal、hazards 和初始状态保持一致；
9. wheeled 与 amphibious 的 evaluator label 应按规则反转；
10. native physical cost 与 semantic violation 分开统计。

不要修改 frozen protocol 的核心定义。若确实必须修改 protocol schema，应先在文档中说明不兼容原因，并新增 draft protocol version；不要静默修改 `1.2.1`。

---

# 6. 禁止事项

本次不要做：

- `SafetyPointPush1-v0`；
- 多机器人 agent；
- 多种 semantic terrain；
- open-vocabulary detector；
- 真实 VLM 调用；
- P0–P4 大规模实验；
- 统计显著性实验；
- 论文图表；
- 训练 RL policy；
- fork 或复制整个 Safety-Gymnasium 源码；
- 使用真实 simulator coordinates 作为 heuristic policy 输入；
- 为了让测试通过而降低信息隔离要求；
- 修改或删除现有有效测试；
- 声称结果已经 VALIDATED。

不要顺手进行与这次接入无关的大规模重构。

---

# 7. 必须增加的测试

至少新增以下测试。

## 7.1 Adapter smoke test

验证：

- import；
- reset；
- step；
- render；
- native cost；
- close；
- JSON artifact。

## 7.2 Deterministic seed test

同一 seed：

- 初始 public observation 一致；
- scene manifest 一致；
- 使用相同固定 actions 时，前若干步轨迹、reward、cost 一致。

不同 seed 至少能够产生不同初始状态或 layout metadata。

## 7.3 Permission-boundary test

构造一个 dummy policy，确认它只能收到允许的 policy input。

测试中检查 policy input 不包含：

```text
semantic_zones
terrain_class
terrain_geometry
master coordinates
MuJoCo model/data
evaluator_context
```

不要只靠代码注释，应有实际测试。

## 7.4 Capability twin test

在 semantic variant 中：

- 同一 scene seed；
- wheeled 与 amphibious；
- 除 capability card 和 evaluator applicability 外，其他 scene manifest 保持一致；
-同一条进入 water 的合成 trajectory：
  - wheeled => semantic violation；
  - amphibious => no semantic violation。

## 7.5 Native cost separation test

验证：

- 进入 semantic water 不自动产生 native physical cost；
- 进入原生 physical hazard 仍产生 native cost；
- artifact 中 `native_cost` 和 `semantic_violation` 是不同字段。

## 7.6 Existing regression tests

运行所有已有测试，确认 PointHazard 没有被破坏。

建议至少运行：

```bash
python -m compileall -q .
python -m pytest -q tests -p no:cacheprovider
```

如完整测试非常慢，可以先运行短 seed sweep，但最后报告必须明确：

- 哪些测试真实跑过；
- seed 数量；
- 哪些测试没有跑；
- 为什么没有跑。

不得把 25-seed smoke 写成 10,000-seed stress test。

---

# 8. Smoke CLI

增加一个明确、无 API 调用的 smoke command，例如：

```bash
python scripts/smoke_safety_gym_goal.py \
  --env-id SafetyPointGoal1-v0 \
  --seed 0 \
  --steps 50 \
  --artifact-dir results/safety_gym_smoke
```

然后增加一个进入主线的最小命令。具体参数可根据现有 CLI 调整，但最终报告中必须给出可以复制运行的完整命令。

命令应能显示：

```text
backend
env id
seed
episode return
native cost
semantic violation
success
termination
artifact path
```

---

# 9. 文档

新增：

```text
docs/SAFETY_GYM_GOAL_INTEGRATION.md
```

内容包括：

1. 为什么先接 Goal 而不是 Push；
2. physical hazard 与 semantic terrain 的区别；
3. 安装方式；
4. headless rendering 配置；
5. smoke 命令；
6. 主线运行命令；
7. public observation 与 evaluator-only 信息边界；
8. artifact schema；
9. 已完成测试；
10. 当前限制；
11. 下一步如何扩展到 P0/P2/P4；
12. 明确说明本次结果不是 paper-grade result。

同时更新适当的 README 或 plan 状态，但不要把未完成事项标记为完成。

---

# 10. 代码质量要求

- 使用类型标注；
- 关键 dataclass/schema 有清晰字段；
- 对 optional dependency 给出清晰错误信息；
- 不使用裸 `except`；
- 不吞掉 reset/render/MuJoCo 错误；
- 不用全局 mutable state；
- 不依赖当前工作目录的偶然路径；
- 文件命名清晰；
- public 与 evaluator-only 数据使用不同类型，避免误传；
- policy API 不接受完整 env；
- 不把调试 overlay 当作 VLM 图像；
- 保持 PointHazard backward compatibility。

---

# 11. 最终汇报格式

完成后请按以下结构汇报：

## A. Repository diagnosis

说明接入前存在的关键接口问题，特别是 policy 权限泄漏和 monolithic harness 问题。

## B. Files changed

逐个列出修改或新增文件及用途。

## C. Architecture

说明：

```text
environment adapter
public observation
evaluator context
router
zone source
enforcement
artifact
```

之间如何隔离。

## D. Commands executed

列出实际执行过的完整命令，不要写计划运行但没有运行的命令。

## E. Test results

逐项给出 pass/fail、seed 数量和跳过原因。

## F. Smoke result

给出至少一次 `SafetyPointGoal1-v0` 无 VLM episode 的摘要和 artifact 路径。

## G. Semantic twin result

若 Gate D 完成，给出 wheeled/amphibious twin 的最小 evaluator 测试结果。

## H. Remaining blockers

诚实列出尚未完成的问题，不要把 smoke result 表述为 validated research result。

## I. Recommended next action

# Final execution record

**Date:** 2026-07-20 (follow-up)  
**Branch:** `agent/wp-1-1-semantic-zone`  
**Scope:** Goal backend only. No Push, Car, Ant, VLM, OpenRouter, commit, push, or PR.

## Gate status

- **Gate A:** added optional Safety-Gymnasium and PointHazard protocol
  adapters with copied public observations, RGB rendering, spaces, and a
  normalized six-value step API.
- **Gate B:** added one JSON artifact schema with seed, scene manifest,
  trajectory, native costs, separate semantic labels, git SHA/dirty state, and
  dependency versions.
- **Gate C:** added `scripts/smoke_safety_gym_goal.py`, switching between
  `point_hazard` and `safety_gym_goal` with `direct_goal`/`none`/`none`. Two
  real `SafetyPointGoal1-v0` episodes completed with RGB rendering and
  replay-ready artifacts; the PointHazard route still runs unchanged.
- **Gate D:** added the visual-only water variant and evaluator-only terrain
  geometry. A real semantic variant episode completed; a real capability-twin
  check uses one scene for wheeled/amphibious and keeps native cost at zero.

## Files changed

Added `envs/`, `evaluation/`, `requirements-safety-gym.txt`,
`scripts/smoke_safety_gym_goal.py`, and
`tests/test_safety_gym_goal_integration.py`. The follow-up corrected the
Safety-Gymnasium dependency pins, extracted real nested Goal/agent/hazard
truth into evaluator-only manifests, added native-cost/STC artifact fields,
and made the smoke command exercise one public RGB frame per episode.
`env_pointhazard.py` retains the checked, vectorized sequential fallback for
the known `three-hetero-corridor-on`, `seed=157` liveness failure. The
pre-existing user changes in `STRUCTURE.md`, `docs/CLAUDE_PLAN.md`,
`docs/MEMO_01.md`, and `docs/RESEARCH_REVIEW_COMMENTS.md` were preserved.

## Commands actually executed

```bash
git status --short --branch
rg --files docs | sort
sed -n '1,700p' docs/SAFETY_GYM_GOAL_INTEGRATION.md
sed -n '1,1800p' docs/PROTOCOL.md
sed -n '1,420p' docs/RESEARCH_REVIEW_COMMENTS.md
sed -n '1,540p' docs/ICLR_PLAN.md
python -m compileall -q envs
python -m compileall -q .
git diff --check
python scripts/smoke_safety_gym_goal.py --environment-backend point_hazard --seed 0 --episodes 2 --steps 10 --artifact-dir /tmp/hazard-safety-gym-smoke
python scripts/smoke_safety_gym_goal.py --env-id SafetyPointGoal1-v0 --seed 0 --steps 50 --artifact-dir /tmp/safety-gym-goal-smoke
python -m pytest -q tests/test_safety_gym_goal_integration.py -p no:cacheprovider -r a
python -m pytest -q tests/test_factor_orthogonality.py -p no:cacheprovider -r a
python -m pytest -q tests/test_layout_invariants.py -p no:cacheprovider -k golden -r a
LAYOUT_TEST_SEEDS=2 python -m pytest -q tests -p no:cacheprovider -r a
LAYOUT_TEST_SEEDS=25 python -m pytest -q tests -p no:cacheprovider -r a
LAYOUT_TEST_SEEDS=10000 python -m pytest -q tests/test_layout_invariants.py -p no:cacheprovider -r a
python -m pip install safety-gymnasium==1.0.0
python -m pip install -r requirements-safety-gym.txt
python -m pytest -q tests/test_safety_gym_goal_integration.py -p no:cacheprovider -r a
python scripts/smoke_safety_gym_goal.py --environment-backend safety_gym_goal --env-id SafetyPointGoal1-v0 --seed 0 --episodes 2 --steps 10 --artifact-dir /tmp/safety-gym-goal-smoke-real
python scripts/smoke_safety_gym_goal.py --environment-backend safety_gym_goal --env-id SafetyPointGoal1-v0 --seed 23 --episodes 1 --steps 5 --semantic-terrain --capability wheeled_non_waterproof --artifact-dir /tmp/safety-gym-goal-semantic-real
LAYOUT_TEST_SEEDS=2 python -m pytest -q tests/test_safety_gym_goal_integration.py tests/test_factor_orthogonality.py tests/test_layout_invariants.py -p no:cacheprovider -k 'not real_safety_gym_adapter_smoke' -r a
```

The formal 10,000-seed command first returned **7 passed, 1 failed** at
`three-hetero-corridor-on`, `seed=157`. After the fallback was added, it was
rerun but interrupted because the existing rejection-sampling sweep is very
slow in this environment. The direct post-fix seed-157 check and golden layout
test passed. A later full-test attempt and the 25-seed bounded attempt were
also stopped for the same slow path; neither is evidence of a formal PASS.

## Test results

- Compileall and `git diff --check`: **PASS**.
- New integration tests with the installed stack: **6 passed**; this includes
  real reset/step/scene truth/RGB render/close for `SafetyPointGoal1-v0`.
- Factor orthogonality: **6 passed**.
- Golden layouts: **4 passed**.
- `LAYOUT_TEST_SEEDS=2` across the three test files: **19 passed, 1
  deselected** (the real render test was already run separately).
- Post-fix seed-157 check: **PASS**, with `layout_valid=True` and
  `zone_layout_valid=True`.
- Formal post-fix 10,000-seed stress: **not a final PASS**; interrupted for
  runtime and must be rerun to close the formal sampler gate.
- `pip check`: the Safety-Gymnasium dependency set is usable, with one
  unrelated pre-existing Torch/SymPy mismatch reported.

## Smoke and twin results

```text
backend=point_hazard env_id=PointHazard-v1 seed=0 return=-0.100000 native_cost=0.000000 semantic_violation=False success=False episode_length=10 termination=timeout artifact=/tmp/hazard-safety-gym-smoke/episode_0.json
backend=point_hazard env_id=PointHazard-v1 seed=1 return=-0.100000 native_cost=0.000000 semantic_violation=False success=False episode_length=10 termination=timeout artifact=/tmp/hazard-safety-gym-smoke/episode_1.json
```

```text
backend=safety_gym_goal env_id=SafetyPointGoal1-v0 seed=0 return=0.000000 native_cost=0.000000 semantic_violation=False success=False episode_length=10 termination=smoke_step_limit artifact=/tmp/safety-gym-goal-smoke-real/episode_0.json
backend=safety_gym_goal env_id=SafetyPointGoal1-v0 seed=1 return=0.000000 native_cost=0.000000 semantic_violation=False success=False episode_length=10 termination=smoke_step_limit artifact=/tmp/safety-gym-goal-smoke-real/episode_1.json
backend=safety_gym_goal env_id=SafetyPointGoal1-v0 seed=23 return=0.000000 native_cost=0.000000 semantic_violation=False success=False episode_length=5 termination=smoke_step_limit artifact=/tmp/safety-gym-goal-semantic-real/episode_23.json
```

The real semantic twin check used seed `23`: the two capability adapters had
identical scene manifests, an identical synthetic trajectory entering `zone_0`
was `True` for `wheeled_non_waterproof` and `False` for `amphibious`, and both
native costs remained `0.0`. These are integration/sanity results, not
paper-grade semantic-safety evidence.

## Remaining blockers

1. Rerun the full post-fix 10,000-seed layout stress to completion; the
   interrupted run is not formal evidence.
2. The Goal adapter is still a tested vertical slice, not a paper-grade
   result: no VLM,
   detector, P0–P4 experiment, or statistical claim was run.

## Recommended next action

Keep the installed Goal stack reproducible (`safety-gymnasium==1.0.0`,
`gymnasium==0.28.1`, `mujoco==2.3.3`, `numpy<1.24`), then close the formal
layout-stress gate. Do not start Push or paid VLM runs from this smoke result.

只给出下一步最重要的 1–3 项，不要自行开始 Push 或大规模 VLM 实验。

---

现在开始：先阅读仓库和 Protocol，输出一个简短实施计划，然后按 Gate A → B → C → D 顺序直接修改、测试和汇报。不要停在分析阶段，不要调用任何收费 API，不要 push 到远程仓库，也不要创建 PR。
