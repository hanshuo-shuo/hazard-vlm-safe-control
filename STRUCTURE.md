# 项目结构与研究状态

最后核对：2026-07-11。

本仓库的代码保持顶层平铺，因为现有脚本使用 `from env_pointhazard import ...`
一类导入。文档和历史代码分别放在 `docs/`、`legacy/`；无效实验产物不删除，按日期归档。

当前论文方向不是“VLM 打败经典规划”，而是 **Safety Accounting for VLM-guided
control**：把感知、语义判断、路由和执行层的安全责任拆开测量。PointHazard 是协议与
归因的 unit test；PointPush 是后续接触任务 scaffold。投稿路线见
[`docs/ICLR_PLAN.md`](docs/ICLR_PLAN.md)，代码与证据问题见
[`docs/RESEARCH_REVIEW_COMMENTS.md`](docs/RESEARCH_REVIEW_COMMENTS.md)，所有结果状态见
[`docs/RESULTS_REGISTRY.md`](docs/RESULTS_REGISTRY.md)。

状态标签：

- `ACTIVE`：当前研究路线直接使用。
- `INFRA`：可复用基础设施，但不等于已有论文证据。
- `BLOCKED`：存在会影响结论的已知问题，修复前不能产出正式数字。
- `BASELINE`：需要保留或补强的对照。
- `EVAL`：测试、画图或分析工具。
- `ARCHIVED`：只保留历史和复现价值，不作为当前证据。

## 1. 环境与渲染

| 文件 | 状态 | 当前用途与限制 |
|---|---|---|
| `env_pointhazard.py` | `ACTIVE · BLOCKED B01` | 惯性点质量、红色 hazard、可选语义区域。语义区 fallback 未重新检查与 hazard 的重叠，已污染既有 semantic runs；修复并加 invariant test 后才能重跑。观测使用绝对坐标。 |
| `hazard_renderer.py` | `INFRA` | PointHazard 的 PIL renderer，支持 restricted/water/mud/grass 外观。是像素感知实验的一部分，修改 palette 时必须同步检查 detector 与数据 provenance。 |
| `env_pointpushhazard.py` | `INFRA` | 接触推物环境。当前是下一阶段 scaffold，尚无经验证的主结果。 |
| `pointpush_hazard_renderer.py` | `INFRA` | PointPush renderer。 |

## 2. 控制器与专家

| 文件 | 状态 | 当前用途与限制 |
|---|---|---|
| `mpc_expert.py` | `ACTIVE · BLOCKED B02` | 基于已知环境动力学的 CEM-MPC。它使用有限采样和有限碰撞/软区代价，**没有**可行性屏蔽、backup policy 或形式化安全保证；不得再写成“硬拒绝后只执行无碰撞动作”或“provably safe”。 |
| `safe_expert.py` | `BASELINE` | A* + PD 的历史低层基线。grid path 安全不代表惯性跟踪轨迹安全，保留作比较。 |
| `pointpush_expert.py` | `INFRA` | PointPush demo/expert scaffold，需独立验证其成功率和碰撞率。 |

## 3. VLM 方法与感知基线

| 文件 | 状态 | 当前用途与限制 |
|---|---|---|
| `subgoal_pivot_hazard.py` | `ACTIVE · BLOCKED B03–B07` | 当前实验 harness：direct、subgoal、oracle、detector 与 B+ 路线。它是做 safety-accounting 消融的载体，**不是已经成立的新方法贡献**。正式运行前需统一 router/controller、任务定义、统计和 provenance。 |
| `pivot_vlm.py` | `INFRA` | 候选生成、标注、VLM 调用与解析的共享工具。原始响应应继续记录。 |
| `zone_detector.py` | `BASELINE` | 针对 renderer palette 的颜色 blob sanity baseline；不能代表完整 classical perception。还需 open-vocabulary/learned segmentation 基线。 |
| `direct_vla_hazard.py` | `BASELINE` | PointHazard 端到端 VLA scaffold。历史训练数字不自动成为当前公平基线。 |
| `direct_vla_pointpush.py` | `BASELINE` | PointPush 端到端 VLA scaffold。 |
| `shared_autonomy_pointpush_hazard_learned_physics_pivot.py` | `ARCHIVED/REUSE` | PointPush learned-physics PIVOT 原型。受旧 prompt-leakage 结论影响，只复用代码，不复用 headline claims。 |

## 4. 测试、评估与脚本

| 路径 | 状态 | 说明 |
|---|---|---|
| `evaluate_pointpush_expert.py` | `EVAL` | PointPush expert 评估入口。 |
| `test_push_render.py` | `EVAL` | PointPush renderer smoke test。 |
| `scripts/validate_pointpushhazard.py` | `EVAL` | PointPush 环境验证。 |
| `scripts/make_*_figures.py` | `EVAL` | 历史结果作图。图是否可用于论文由 registry 决定，不由脚本存在与否决定。 |
| 其余 `scripts/` | `EVAL/ARCHIVED` | 训练日志、VLA sweep 和历史作图工具；运行前检查硬编码路径与对应结果状态。 |

## 5. 文档真相源

| 文件 | 角色 |
|---|---|
| `docs/RESEARCH_REVIEW_COMMENTS.md` | 逐项审稿式问题清单；blocker 未关闭前不要扩大实验。 |
| `docs/ICLR_PLAN.md` | 覆盖旧路线的当前研究与投稿计划。 |
| `docs/RESULTS_REGISTRY.md` | 每份结果的 VALIDATED / PILOT_ONLY / INVALIDATED / ARCHIVED 状态。 |
| `docs/README.md` | 文档入口与推荐阅读顺序。 |
| `docs/RESULTS_*.md` | 历史实验记录。保留原始数字，但顶部状态优先于正文当时的解释。 |
| `docs/PROJECT_PLAN.md`、`WHY_VLM.md` 等 | 已 superseded/archived 的思考过程，不是当前路线。 |

## 6. 数据与产物

| 目录 | 内容与纪律 |
|---|---|
| `outputs/` | 小型结果、图和 transcript。新结果必须登记配置、commit、seed、模型标识和状态。 |
| `outputs/invalidated/2026-07-11/` | n=20 held-out、offline n=100 和 smoke runs 的原始记录；因 B01 与比较混杂而失效，保留作 forensic/debug，不作论文数字。 |
| `hazard/` | 本地权重、logs、GIF，受 `.gitignore` 保护。脚本仍有默认路径依赖，勿随意改名；需另行备份。 |
| `data/` | demo/checkpoint 数据，受 `.gitignore` 保护。 |
| `legacy/` | 旧 PIVOT、SAC/diffusion、gym wrapper、历史文档与展示 GIF。只作档案。 |

## 7. 当前 blockers

编号与详细复现见 `docs/RESEARCH_REVIEW_COMMENTS.md`：

- `B01`：semantic-zone fallback 可能与 red hazard 重叠，破坏任务语义和因果解释。
- `B02`：MPC 没有论文中曾暗示的形式化/硬安全保证。
- `B03`：C2/CV 与 B+ 不只改变感知源，还改变 router，比较不公平。
- `B04`：success、hazard、semantic violation 的任务口径需统一并报告联合失败。
- `B05`：旧统计规模小，且 paired design 没有完整使用 paired inference。
- `B06`：模型/provider、prompt、代码 commit 与输出 provenance 不完整。
- `B07`：当前 toy 的颜色 detector 太弱，不能支撑“一个 VLM 替代 N 个 detector”。

在 `B01–B07` 关闭前，`docs/RESULTS_REGISTRY.md` 中没有 semantic 结果可标为
`VALIDATED`。

## 8. 整理与研究历史

- **2026-06-18**：首次整理；死代码和旧笔记移入 `legacy/`、`docs/`，启动 Path B。
- **2026-06-21**：跑通 n=5 pure-geometry 信心实验。该结果仅为 `PILOT_ONLY`。
- **2026-06-28**：加入 semantic zone、implicit/heterogeneous 外观与 B+ 感知回传；当时形成的
  “追平/反超 oracle”“一个 VLM 顶 N 个 detector”解释，现均为历史假设，不能引用为结论。
- **2026-06-29**：加入 fair-soft oracle、颜色 detector、L2 prompt ladder 与 held-out seed 支持。
- **2026-07-01**：完成 n=20 held-out runs；后来发现其中 7/20 semantic zones 与 red hazard
  重叠，且 policy routing 混杂，因此这些数字已 `INVALIDATED`，原始文件保存在
  `outputs/invalidated/2026-07-11/`。
- **2026-07-11**：完成代码/方法审计，建立结果 registry，路线改为 Safety Accounting；清理纯
  cache，但保留所有研究记录和无效结果 provenance。

原则：**不删除失败、不覆盖旧数字、不让旧解释冒充当前结论。** 新实验先登记，再运行；发现
问题时更新状态并归档原始文件。
