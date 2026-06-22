# 项目结构地图

最后整理：2026-06-18。代码层保持**平铺**（所有 import 都是 `from env_pointhazard import ...` 这种顶层导入，搬进子目录会批量搞断 import），只隔离了文档和死代码。

状态标签是相对**新方向 Path B**（VLM 选高层 subgoal → 安全低层控制器执行）而言的：

- `★ CORE` 主线必用，且是 Path B 的关键资产
- `KEEP` 基础设施，继续用
- `REUSE` 代码可复用，但**实验结论受泄漏污染**（见 `docs/FAILURE_MODE_ANALYSIS.md`），结果要重跑
- `BASELINE` 论文需要的对照基线
- `EVAL` 评估/可视化工具

---

## 1. 环境 Environments
| 文件 | 状态 | 说明 |
|---|---|---|
| `env_pointhazard.py` | ★ CORE | 点质量 + 随机 hazard + 随机 goal。惯性导航主环境。✅ `make_env` 里坏的渲染器 import 已修（改成 `from hazard_renderer import`）。obs 是**绝对坐标**（goal/hazard 中心），不是相对。 |
| `env_pointpushhazard.py` | KEEP | 接触推物环境，Path B 第 3 月的接触主环境。 |

## 2. 渲染器 Renderers
| 文件 | 状态 | 说明 |
|---|---|---|
| `hazard_renderer.py` | ★ CORE | PointHazard 的 PIL 渲染器。**不能改名**（env 里有 latent import 引用 + 脚本依赖）。 |
| `pointpush_hazard_renderer.py` | KEEP | PointPush 渲染器。 |

## 3. 低层控制器 / 专家 Controllers（★ Path B 的半成品）
| 文件 | 状态 | 说明 |
|---|---|---|
| `mpc_expert.py` | ★★ CORE | **CEM-MPC，按 env 的精确动力学（含惯性/drag/反弹）rollout，硬拒绝撞 hazard 的轨迹 → 只 commit 无碰撞动作。比 A\*+PD 更强也更诚实（纯几何 toy 上 100%/0 碰撞 vs safe_expert 90–95%）。是 Path B 的默认低层控制器。** drop-in API：`plan`/`reset_from_obs`/`act`，与 SafeExpert 同签名。 |
| `safe_expert.py` | ★ CORE | A\*（grid 路径）+ PD 跟踪。**注释说 provably safe 但只对 grid 路径成立，跟踪轨迹受惯性影响会偶尔擦到 hazard（~5–10%）**，所以降级让位给 `mpc_expert.py`，保留作对照低层。✅ `reset_from_obs` 已从相对坐标修成绝对坐标，匹配当前 env。 |
| `pointpush_expert.py` | KEEP | PointPush 的专家控制器，用于生成 demo / 当低层。 |

## 4. PIVOT / VLM 共享核心
| 文件 | 状态 | 说明 |
|---|---|---|
| `pivot_vlm.py` | ★ CORE | 候选生成 `generate_candidates`、`annotate_candidates`、VLM 选择/解析 `_call_vlm_select` / `_parse_pivot_selection`。所有 method 脚本都依赖它。 |

## 5. PIVOT 方法脚本（entry points）
| 文件 | 状态 | 说明 |
|---|---|---|
| `subgoal_pivot_hazard.py` | ★ CORE | **第 1 月信心实验主脚本**：matched-seed 三方对照 `direct`（VLM 直接选低层 force，去泄漏）vs `subgoal`（VLM 选离散 subgoal → 低层控制器执行）vs 纯低层控制器（`mpc` 默认 / `safe_expert`，`--low_level` 切换；B 内部也用同一个）。prompt 严格不含 clearance/safe-label/score。`--pilot_mode heuristic` 可无 API key 离线冒烟；`--pilot_mode vlm` 走 OpenRouter。输出 success/hazard/timeout/min-clearance/VLM-calls 对照表。冒烟结果：mpc & subgoal 100%/0 碰撞，direct ~70%/~30% 碰撞 —— 正是 Path B 第一张图。 |
| `shared_autonomy_hazard_learned_physics_pivot.py` | REUSE | 在线 learned-physics PIVOT（本地 VLM）。**结论受泄漏污染，需重跑零泄漏版。** |
| `shared_autonomy_hazard_learned_physics_pivot_openrouter.py` | REUSE | 同上，OpenRouter VLM 版。 |
| `shared_autonomy_hazard_dual_mlp_pivot_openrouter.py` | REUSE | dual-MLP physics 变体。 |
| `shared_autonomy_pointpush_hazard_learned_physics_pivot.py` | REUSE | PointPush 版 learned-physics PIVOT（96KB，最大）。 |

> ⚠️ 这 4 个脚本都要做**泄漏审计**：prompt 里有没有混入 clearance / safe-unsafe label / score（参见 `docs/FAILURE_MODE_ANALYSIS.md` 第 8 节的审计表）。

## 6. Direct VLA 基线
| 文件 | 状态 | 说明 |
|---|---|---|
| `direct_vla_hazard.py` | BASELINE | same-backbone 端到端 VLA 对照（regression / flow head）。 |
| `direct_vla_pointpush.py` | BASELINE | PointPush 版（import 了 `direct_vla_hazard`）。 |

## 7. 评估 / 测试
| 文件 | 状态 | 说明 |
|---|---|---|
| `evaluate_pointpush_expert.py` | EVAL | PointPush 专家评估。 |
| `test_push_render.py` | EVAL | PointPush 渲染冒烟测试。 |

---

## 8. 数据 / 产物目录（不是代码，别改名）
| 目录 | 内容 | 注意 |
|---|---|---|
| `hazard/` | 模型权重 `learned_physics_agent.pt` / `dual_mlp_agent.pt`、logs、gifs | **不能改名**：脚本 `--log_dir`/`--gif_dir`/`--physics_ckpt`/`--ckpt`/`--out` 默认值全硬编码成 `hazard/...`。 |
| `data/` | `augmented_hazard_demos.npz`、`diffusion_hazard/` | 训练数据。 |
| `outputs/` | 论文图、render 输出 png | 结果图。 |
| `scripts/` | 分析/画图脚本 + `.sh` 运行脚本 | 保持原位（内部有相对路径）。 |
| `docs/` | 三份研究笔记（失败分析 / 实验记录 / 文献综述） | 本次从 root 移入。 |
| `legacy/` | 旧版 PIVOT（momentum/primitive/self-iterating）、diffusion/SAC 训练、`env_hazard_gym.py`（本次新增，已坏的 gym wrapper） | 归档，不再用。 |

---

## 9. 整理历史
**2026-06-18（首次）**
- `env_hazard_gym.py` → `legacy/`（唯一的死代码：import 不存在的 `pointmaze_copilot` 包，无人引用）。
- 三份 `*.md` 笔记 → `docs/`。
- 删掉散落的 `.DS_Store`。
- 代码层**未动**，所有 import 保持可用。

**2026-06-18（Path B 启动）**
- 新建 `subgoal_pivot_hazard.py`（第 1 月信心实验主脚本，见 §5）。
- 新建 `mpc_expert.py`（CEM-MPC 低层控制器，见 §3）；设为默认低层，替下偶尔擦 hazard 的 `safe_expert`。
- 修 `env_pointhazard.make_env`：`from hazard.hazard_renderer` → `from hazard_renderer`（坏 import）。
- 修 `safe_expert.reset_from_obs`：相对坐标 → 绝对坐标，匹配当前 env（之前纯 SafeExpert 从 obs 重规划会规划到错误位置）。
- 清掉 root `__pycache__/`。

## 10. Path B 下一步（§5 的 `subgoal_pivot_hazard.py` 已落地）
- 跑 `--pilot_mode vlm` 的真实三方对照，存进 `outputs/`，作为转向的第一张实证图。
- 第 2–3 月：给环境加 A\* 写不出 cost 的语义/语言约束（偏好侧、禁区语义、子任务顺序），让 VLM 高层真正赢过纯经典规划。
</content>
