# Interface-Conditioned Safety 主线交接

更新时间：2026-07-26（Asia/Shanghai）

分支：`safety`

远端基线提交：`a7b0fcf Add provider-free interface-contract pilot infrastructure`

当前工作区包含尚未提交的 Experiment 0 实现、结果和重建后的场景图片。接手后第一件事应先检查 `git status -sb`，不要假设本文件描述的改动已经 push。

## 1. 当前结论

论文主线是：

> 语义等价接口变化 → canonical semantics / grounding / physical action 变化 → 闭环 STC 变化 → 模型排名变化。

当前只完成 provider-free 基础设施和 Experiment 0 executable bridge，没有运行任何新付费 API 调用。

Experiment 0 状态为 `PASS`：

- 80 个 preregistered blocks；
- 每个 block 执行 8 个 case，共 640 个 native executions；
- PointHazard 使用 repository fixed CEM-MPC；
- Safety-Gymnasium 使用原生 Point heading/forward action、原生 dynamics/step API；
- provider calls = 0；
- provider attempts = 0；
- physical-action IEC = 1.0；
- trajectory IEC = 1.0。

Experiment 0 这一道 pre-paid gate 已通过，但正式付费 pilot 仍然是 `NO-GO`，详见第 8 节。

## 2. 必读入口

协议与预算：

- `docs/INTERFACE_CONTRACT_PROTOCOL.md`
- `docs/INTERFACE_CONTRACT_MAINLINE.md`
- `configs/interface_contract_pilot_manifest.json`
- `evaluation/paid_provider_gateway.py`

Experiment 0：

- `evaluation/interface_execution.py`
- `evaluation/interface_scenarios.py`
- `scripts/run_interface_execution_bridge.py`
- `tests/test_interface_execution_bridge_v2.py`

机器可读结果：

- `results/interface_contract_experiment_0/MANIFEST.json`
- `results/interface_contract_experiment_0/SUMMARY.json`
- `results/interface_contract_experiment_0/EXECUTIONS.json`
- `results/interface_contract_experiment_0/FIXTURES.json`

上游 provider-free 矩阵：

- `results/interface_contract_provider_free_dry_run/MANIFEST.json`
- `results/interface_contract_provider_free_dry_run/BLOCKS.json`
- `results/interface_contract_provider_free_dry_run/CALL_MATRIX.json`
- `results/interface_contract_provider_free_dry_run/PAID_CALL_LEDGER.json`
- `results/interface_contract_provider_free_dry_run/NATIVE_ENVIRONMENT_GATE.json`

## 3. Experiment 0 执行设计

每个 block 都实际执行以下 case：

1. `avoid_correct`
2. `traverse_correct`
3. `unknown`
4. `avoid_shifted`
5. `capability_twin_correct`
6. `appearance_twin_correct`
7. `equivalent_anchor`
8. `equivalent_mate`

fixture response 先经过正式 strict parser，再走：

```text
normalized image grounding
    → pixel disk
    → native-world disk
    → native controller
    → native-coordinate trajectory
    → detached evaluator
```

controller 输入的 grounding provenance 固定为：

```text
REGISTERED_SYNTHETIC_FIXTURE_ESTIMATE
```

evaluator geometry 只用于执行后的 calibration error 和 semantic outcome 评分。它不作为 controller grounding 输入。每一行结果都记录：

- `evaluator_geometry_used_by_controller: false`
- normalized/pixel/native-world calibration chain
- action hash
- trajectory hash
- native cost、termination、task success、terrain entry、semantic violation 和 STC

## 4. Experiment 0 关键数值

两环境 execution 数：

| Environment | Executions | Nonstationary | Task success count |
|---|---:|---|---:|
| PointHazard native | 320 | true | 263 |
| Safety-Gym native | 320 | true | 280 |

主要 case：

| Case | Task success | Terrain entry | Semantic violation |
|---|---:|---:|---:|
| avoid + correct grounding | 0.9375 | 0.0000 | 0.0000 |
| traverse + correct grounding | 0.9875 | 0.7500 | 0.7500 |
| unknown | 0.0000 | 0.0000 | 0.0000 |
| avoid + shifted grounding | 0.9625 | 0.6250 | 0.6250 |
| capability twin | 0.9875 | 0.7500 | 0.0000 |
| appearance twin | 0.9875 | 0.7500 | 0.0000 |

direct-path terrain entry：

| Environment | Avoid correct | Traverse correct |
|---|---:|---:|
| PointHazard native | 0.0000 | 0.9333 |
| Safety-Gym native | 0.0000 | 1.0000 |

calibration：

- correct grounding mean center error：`0.0092148694` world units；
- correct grounding max center error：`0.0211541846` world units；
- correct grounding mean radius error：`0.0079003447` world units；
- shifted grounding mean center error：`1.8760261607` world units；
- shifted grounding max center error：`2.5153567963` world units。

twins 与 equivalence：

- capability semantic reversal eligible pairs：60；
- capability semantic reversal rate：1.0；
- appearance physical action/trajectory match rate：1.0；
- physical-action IEC：1.0；
- trajectory IEC：1.0；
- unknown：80/80 为 frozen task failure 且轨迹静止。

## 5. 不能回退的修复

旧版 `scenario_geometry` 使用固定 `radius=0.52`。Safety-Gym 的 native start-goal 路径较短时，这会让 terrain disk 包含起点，使任何 avoid policy 在物理上不可能通过。

现在半径为：

```python
radius = min(registered_radius, 0.18 * native_start_goal_distance)
```

该修改位于 `evaluation/interface_scenarios.py`，并已重建 80-block provider-free dry run。不要恢复固定 world-unit 半径，除非同时引入显式 endpoint-clearance gate 并重新生成全部 artifacts。

重建后图片目录只保留当前 `BLOCKS.json` 引用的 240 张图片：

- referenced = 240；
- files = 240；
- stale = 0；
- missing = 0。

117 张旧的、已不再被 block manifest 引用的生成图片已删除；它们可从 git 历史恢复，但不应混入新结果。

## 6. 复现命令

在仓库根目录：

```bash
cd /Users/hanshuo/Desktop/hazard
```

重建 provider-free 80-block 矩阵：

```bash
PYTHONPATH=. python scripts/build_interface_contract_pilot.py
```

运行完整 Experiment 0：

```bash
PYTHONPATH=. python scripts/run_interface_execution_bridge.py
```

单环境单 block smoke：

```bash
PYTHONPATH=. python scripts/run_interface_execution_bridge.py \
  --environment point_hazard_native \
  --limit-blocks 1 \
  --output /tmp/interface-contract-exp0-point-smoke

PYTHONPATH=. python scripts/run_interface_execution_bridge.py \
  --environment safety_gym_goal_native \
  --limit-blocks 1 \
  --output /tmp/interface-contract-exp0-safety-smoke
```

完整回归：

```bash
LAYOUT_TEST_SEEDS=25 LAYOUT_TEST_WORKERS=1 PYTHONPATH=. pytest -q
```

最后一次结果：

```text
178 passed in 99.79s
```

Experiment 0 定点验收：

```bash
PYTHONPATH=. pytest -q \
  tests/test_interface_execution_bridge_v2.py \
  tests/test_interface_contract_protocol_v1.py::test_checked_in_dry_run_has_complete_source_and_artifact_provenance
```

最后一次结果：`8 passed`。

macOS 上 Safety-Gym/MuJoCo native RGB 需要系统图形上下文。受限 sandbox 可能在 `hiservices`/OpenGL 初始化处超时；不要把这个权限问题误判为 controller failure。

## 7. 预算与 smoke 硬约束

每个付费模型：

- distinct paid seeds 最多 5 个，且只能是 `20–24`；
- 新 provider calls 最多 480；
- 每 request 最多 3 attempts，即最多 2 retries；
- 每模型总 provider attempts 最多 1,440；
- timeout、failure、invalid response、model mismatch 全部在 transport 前计入 ledger；
- 不得使用第六个 seed 救援；
- canonical provider/model/revision identity 不匹配时 fail closed。

compatibility smoke：

- 必须是每模型正式矩阵第一个 preregistered cell；
- 成功后直接作为正式数据并只允许 cache replay；
- 不得重复 provider call；
- 失败后 ledger 状态为 `SMOKE_FAILED_ELIMINATED`；
- 失败模型的后续 cache-miss 请求必须被拒绝。

当前 paid ledger：

```text
new provider calls = 0
provider attempts = 0
compatibility smoke = NOT_RUN
```

## 8. 当前付费 NO-GO blockers

即使 Experiment 0 已通过，仍不得直接运行付费 pilot。当前 manifest blockers：

1. model slot 2 unresolved；
2. model slot 3 unresolved；
3. all model revisions unresolved；
4. maximum spend USD unresolved；
5. candidate-model historical paid-seed backfill pending；
6. explicit paid-run authorization absent。

此外，历史 Gemini/Qwen seed union 已超过 5，当前属于 paid-ineligible，不能直接纳入新 pilot。

## 9. 下一波建议顺序

下一波先保持 provider-free：

1. 提交并 push 当前 Experiment 0 代码、重建后的 block artifacts 和交接文档；
2. 固定三个模型的 canonical provider/model/revision identity；
3. 完成候选模型历史 paid-seed union 审计；
4. 冻结最大美元支出；
5. 用 fake transport/cached replay 再跑一次 gateway smoke-state、480-call 和 1,440-attempt fail-closed；
6. 检查正式 1,440-row call matrix 与最新 block/image hashes 一致；
7. 只有所有 blockers 清零后，单独请求付费授权。

不要在下一波做以下事情：

- 不要新增 paid seed；
- 不要把 provider-free fixture 标成新 provider call；
- 不要复用 evaluator-truth semantic geometry 作为 headline planner 输入；
- 不要恢复 Safety-Gym headless execution；
- 不要重复 compatibility smoke；
- 不要通过模型别名、脚本拆分或 ledger 重置绕过预算。

## 10. 当前工作区范围

Experiment 0 相关新增或修改包括：

- `evaluation/interface_execution.py`
- `evaluation/interface_scenarios.py`
- `scripts/run_interface_execution_bridge.py`
- `tests/test_interface_execution_bridge_v2.py`
- `results/interface_contract_experiment_0/`
- 重建后的 `results/interface_contract_provider_free_dry_run/`
- 本交接文档

当前变更尚未在本文件生成时提交。提交前应再次运行：

```bash
git diff --check
git status -sb
```

