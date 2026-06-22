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
| `subgoal_pivot_hazard.py` | ★ CORE | **第 1 月信心实验主脚本（已跑通真实 VLM，见 `docs/RESULTS_MONTH1.md`）**：matched-seed 三方对照 `direct`（VLM 直接选低层 force，去泄漏）vs `subgoal`（VLM 选离散 subgoal → 低层控制器执行）vs 纯低层控制器（`mpc` 默认 / `safe_expert`）。prompt + 图像标注严格不含 clearance/safe-label/score；硬化项：去掉静默 heuristic 兜底（改 `--vlm_fallback hold` 中性 no-op，并记 fb%）、Wilson 95% CI、`--log_transcripts` 原始响应审计、retry/API-error 区分、temp 默认 0。`--pilot_mode heuristic` 离线冒烟；`--pilot_mode vlm` 走 OpenRouter。n=5 真实结果：mpc & subgoal 100%/0 碰撞、direct 80%/20% 碰撞、fb%=0 —— 正是 Path B 第一张图（n=5 仅验证链路，正式图需 ~100 seeds）。 |
| `shared_autonomy_pointpush_hazard_learned_physics_pivot.py` | REUSE | PointPush 版 learned-physics PIVOT（96KB，最大）。结论受泄漏污染，代码留作第 3–4 月接触环境复用。`test_push_render.py` 依赖它。 |

> ⚠️ 泄漏审计：prompt 里有没有混入 clearance / safe-unsafe label / score（参见 `docs/FAILURE_MODE_ANALYSIS.md` 第 8 节的审计表）。
> 📦 三个 PointHazard learned-physics/dual-MLP PIVOT 脚本（`shared_autonomy_hazard_learned_physics_pivot[_openrouter].py`、`shared_autonomy_hazard_dual_mlp_pivot_openrouter.py`）**已归档到 `legacy/`**：结论受泄漏污染，且其 PointHazard 环境已被干净的 `subgoal_pivot_hazard.py` 取代，无人 import。

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

**2026-06-21（Month-1 跑通 + 整理）**
- `subgoal_pivot_hazard.py` 硬化到可发表标准（去静默兜底 / Wilson CI / transcript 审计 / temp0 / hold 兜底）。
- 跑通真实 VLM 三方对照（n=5，gemini-3-flash），结果记到 `docs/RESULTS_MONTH1.md`。
- 归档 3 个泄漏污染的 PointHazard PIVOT 脚本到 `legacy/`（见 §5）。

## 10. Path B 下一步
- **扩到 ~100 matched seeds（+可选 2–3 个 VLM backbone）**，把 `docs/RESULTS_MONTH1.md` 的 n=5 pilot 升级成正式第一张图。
- 第 2–3 月（**主结果，5-seed pilot 已赢，见 `docs/RESULTS_SEMANTIC.md`**）：给环境加 A\* 写不出 cost 的语义约束，让 VLM 高层真正赢过纯经典规划。
  - **n=5 真 VLM（gemini-3-flash）结果**：sem_viol C1 80% / C2 0% / **B 20%**，goal 全 100%、0 碰撞、fb%=0。VLM 把违规砍到 1/4、逼近 oracle，零手工感知 → 方向性赢。唯一一次违规（seed 47）是"自信但空间不精"——transcript 里它两次声称在绕开 amber 区却仍擦进去（同 Month-1 低层失败模式）。n=5 CI 重叠大，正式图需 ~100 seeds。
  - **图**：`scripts/make_semantic_figures.py` 生成 `outputs/fig_vlm_view_seed*.png`（VLM 实际看到的输入）+ `outputs/fig_traj_seed*.png`（C1 直穿 / C2 绕 / B 绕 三连轨迹）。seed44 是 win 图、seed47 是失败图。已嵌入 `docs/RESULTS_SEMANTIC.md`。
  - **⚠️ 复现性待修（100-seed 正式跑前）**：`mpc_expert.MPCExpert` 的 CEM `rng` 没播种（`default_rng()` 无 seed）→ 低层不是逐位可复现（seed47 B 在表里 7 步、图里 5 步）。正式跑前要按 episode seed 给控制器 rng 播种，让 matched-seed 表精确可复现。
  - **已实现「禁区语义」轴**：`env_pointhazard.py` 加了 `semantic` keep-out 区（amber ✕ 圆，`--n_semantic_zones>=1`）——**不终止、不进 obs**，故纯几何 cost 写不出它；采样在 start→goal 走廊上，保证几何规划器会直穿。`hazard_renderer.py` 画它，VLM 从图里自己看见。
  - `subgoal_pivot_hazard.py` 三方对照：**C1** `mpc`/`safe_expert`（几何盲，直穿）vs **C2** `mpc_oracle`/`*_oracle`（把禁区手工喂成障碍 = 上界）vs **B** `subgoal`（VLM 看图绕行，底层控制器仍纯几何 → 唯一懂禁区的是 VLM）。新指标 `sem_viol`（进过禁区的 episode 比例，Wilson CI）。
  - 防泄漏延续：prompt 只给「避开 amber 禁区」这种**语言/任务约束**（类比「到绿色 goal」），绝不给坐标/哪个候选在区内/clearance/score。
  - **离线 heuristic 验证已通过**（n=20）：C1 sem_viol 95%、C2 0%、B（盲 stand-in）85% —— 指标与 oracle 都对。**赢的判据**：真 VLM 跑时 B 的 sem_viol 从 ~85% 塌向 C2 的 ~0%，C1 仍 ~95%。
  - 复现：`python subgoal_pivot_hazard.py --pilot_mode vlm --n_semantic_zones 1 --episodes 5 --seed 43 --model google/gemini-3-flash-preview --temperature 0 --vlm_fallback hold --log_transcripts --out outputs/semantic_pilot.json`
  - 若真 VLM 想绕却绕不开（底层控制器在两次 subgoal 之间抄近路穿区），调 `--subgoal_radius`（调小=更细）和 `--subgoal_horizon`（调小=更勤重选）。
  - 后续轴（未做）：偏好侧、子任务顺序。子任务顺序要改 episode 结构与 success 定义，放禁区跑通之后。
</content>
