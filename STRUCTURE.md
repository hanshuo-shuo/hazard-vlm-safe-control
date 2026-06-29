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
| `docs/` | 研究笔记 + `WHY_VLM.md`（简明英文图解：VLM 优势在哪，先读）+ `RESULTS_AMPLIFY.md`（**带图详细分析：杠杆 1 B+ + 杠杆 2 异质区**）+ 结果记录（Month1 / Semantic §11/§12）/ 失败分析 / 文献综述 | 本次从 root 移入。 |
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

**2026-06-28（杠杆 2：异质多语义区——一个 VLM 顶 N 个手写检测器）**
- `hazard_renderer.py`：抽出 `_draw_semantic_zone(odraw,cx,cy,pr,style)`，新增 **mud（棕色斑块）/ grass（绿色草叶）** 两种外观（+原 water/restricted）。`render()` 加 `semantic_styles` per-zone 参数。
- `env_pointhazard.py`：config 加 `semantic_styles` 池（round-robin 分配，空=同质，**单区同质 RNG 序列不变**，已验证 90/0/85/35 逐位一致）；多区时沿走廊 t-band 铺开（[0.18,0.82]），fallback 也按 zi 分散防重叠。`render()` 透传 styles。
- `subgoal_pivot_hazard.py`：CLI `--semantic_styles water,mud,grass`。B+ 的质心聚类**天然支持多区**（每区一簇），无需改。
- **真 VLM n=5 结果**（3 区 water/mud/grass，implicit）：**C1 100% / C2 oracle 0%(但 success 仅 60%) / B 40% / B+ 20%**，B/B+ 全 100% success、0 碰撞。两个亮点：① B+ 比 B 砍半（40→20）、比盲经典砍 5×（100→20）；② **B+ 100% success 反超 oracle 的 60%**——oracle 硬避 3 区+8 hazard 过约束超时，B+ 软感知区永远有可行路 → 多约束下感知路线比手编更鲁棒。
- **决定性证据**：transcript 里 VLM **自发**点名三种地形 water 28×/mud 18×/grass 9×（"water, mud, and tall grass"、"brown cratered area"），prompt 从没提过 → 一个模型零样本顶替 N 个手写检测器。详见 `docs/RESULTS_SEMANTIC.md` §12。
- 图：`outputs/fig_hetero_seed{43,44,45}.png`。数据：`outputs/semantic_hetero.json`(+transcripts)。

**2026-06-28（杠杆 1：B+ 把 VLM 感知喂给低层，逼近 oracle）**
- `subgoal_pivot_hazard.py`：新增第四臂 **B+ `subgoal_perceive`**。VLM 在同一次调用里额外输出 `"avoid":[markers]`（纯感知，绝不告诉它答案 → 防泄漏不变），脚本把 flagged markers **按质心聚类**估计禁区，喂给低层当 **hard core（≈真区尺度 0.8）+ soft halo（1.5，软代价永不成墙）**。低层只通过 VLM 的感知得知禁区，仍零手工标注（这是 B+ 与 oracle C2 的本质区别）。
  - 新 flag：`--vlm_zone_radius`（halo/聚类半径）、`--vlm_zone_core`（hard core）、`--vlm_soft_weight`（soft 权重）。`mpc_expert.py` 加 `set_soft_zones()` + `soft_zone_weight`（per-step 软代价，非硬拒绝）。
  - **真 VLM n=5 结果**：**implicit B+ = 0% sem_viol / 100% goal**（追平 oracle，赢过 B 的 20%），仅 +0.4 VLM 调用/ep；explicit B+ 0% on 43–46，但 seed47 是「选择失败」（VLM 自信选错 waypoint，喂感知救不了）故与 B 并列 20%。详见 `docs/RESULTS_SEMANTIC.md` §11。
  - 三个设计坑（都由实测逼出，见 §11.2）：① 散布 disk→质心聚类（否则封死 seed45 gauntlet 活锁）；② hard+soft 分层（软代价防活锁）；③ prompt 只标地形不标 hazard（否则 success 崩到 40%）。
  - 语义任务**默认策略集**改为四臂 C1/C2/B/B+。数据：`outputs/semantic_bplus_{explicit,implicit}.json`(+`.transcripts.json`)。
  - n=5 caveat：explicit 下 B+ 与 B 不可统计区分（唯一违规是选择失败种子）；机制证据来自离线 stand-in（完美感知+盲选）把 corner-cut 违规 85%→35%。真正分离需 ~100 seeds（杠杆 4）。

**2026-06-28（隐式语义轴）**
- `hazard_renderer.py`：加 `semantic_style="water"`（水洼外观，默认仍 `"restricted"` amber-X）。
- `subgoal_pivot_hazard.py`：加 `--zone_semantics {explicit,implicit}` + `SemanticSpec`（render 样式 + prompt clause 配对，默认 explicit 向后兼容）。
- `scripts/make_semantic_figures.py`：加 `--zone_semantics`，implicit 图加后缀防覆盖 explicit 主结果图。
- 跑通 5-seed implicit 真 VLM：B 仍 20%（赢），transcript 自发认出"water"。详见 §10 与 `RESULTS_SEMANTIC.md` §10。

**2026-06-29（kill-shots：公平 baseline + L2 常识 + 投稿 reframe）**
- 竞品/可发表性调研 + 代码核实 → `docs/ICLR_PLAN.md`（重定位为「无泄漏可替代性研究 + 诊断 benchmark」，弃 MCP 主框架；CORE/LaC/HazardArena 为主要威胁）。
- `subgoal_pivot_hazard.py`：新增 **C2-soft `mpc_oracle_soft`**（真禁区几何 + 与 B+ 同款软机制的**公平** oracle，`--oracle_soft_mode bplus|pure`）；新增 **`--prompt_level L0/L1/L2`**（prompt 文本与渲染外观解耦，L2=完全不提地形=真常识测试，L2 自动去掉 B+）；新增 **`--seed_list`**（tuning/eval 种子互斥，防 tuning-on-test）。语义默认臂集扩成 C1/C2-hard/C2-soft/detector/B/B+。
- **`zone_detector.py`**（新）：颜色 blob 检测器 baseline（每地形一个颜色规则，IoU≈0.9），从像素读禁区喂同一控制器 → **`mpc_detector`** 臂（「为什么不直接写检测器」的对照）。
- `scripts/make_killshot_figures.py`（新）：bar/轨迹/检测器叠加/L2 图，复用 replay。
- **5-seed 真 VLM 结果**（见 `docs/RESULTS_FAIR_BASELINES.md`）：单区 B+ 0% 但**公平 soft oracle 与 CV 检测器同样 0%**——「B+ 反超 oracle」作废；检测器在干净 toy 上≈oracle ⇒ VLM 优势不在性能；**L2 无提示 B 违规 20%→40%**（仍 < C1 60%），但 VLM 仍自发叫它 "water hazard/blue obstacle" ⇒ 常识真实但 L1 一半是 instruction-following。

## 10. Path B 下一步
- **扩到 ~100 matched seeds（+可选 2–3 个 VLM backbone）**，把 `docs/RESULTS_MONTH1.md` 的 n=5 pilot 升级成正式第一张图。
- 第 2–3 月（**主结果，5-seed pilot 已赢，见 `docs/RESULTS_SEMANTIC.md`**）：给环境加 A\* 写不出 cost 的语义约束，让 VLM 高层真正赢过纯经典规划。
  - **n=5 真 VLM（gemini-3-flash，可复现版）结果**：sem_viol C1 60% / C2 0% / **B 20%**，goal 全 100%、0 碰撞、fb%=0。VLM 把违规砍到 1/3、逼近 oracle，零手工感知 → 方向性赢。per-seed：C1 在 44/46/47 进区、B 只在 47、C2 从不。唯一违规（seed 47）是"自信但空间不精"——transcript 里它三次都选 waypoint 8、声称在绕开 amber 区却仍擦进去（同 Month-1 低层失败模式）。n=5 CI 重叠大，正式图需 ~100 seeds。
  - **图**：`scripts/make_semantic_figures.py` 生成 `outputs/fig_vlm_view_seed*.png`（VLM 实际看到的输入）+ `outputs/fig_traj_seed*.png`（C1 直穿 / C2 绕 / B 绕 三连轨迹）。seed44 是 win 图、seed47 是失败图。计数与表精确一致。已嵌入 `docs/RESULTS_SEMANTIC.md`。
  - **✅ 复现性已修**：`mpc_expert.MPCExpert` / `safe_expert.SafeExpert` 现在在 `policy.reset` 按 episode seed 给各自 `rng` 播种（SafeExpert 的全局 `np.random` 也换成自带 rng）→ 整条 matched-seed 链逐位可复现（两次同配置离线跑结果完全一致）。唯一残余非确定性是 VLM provider 在 temp0 下本身的抖动（外部，管不了）。
  - **已实现「禁区语义」轴**：`env_pointhazard.py` 加了 `semantic` keep-out 区（amber ✕ 圆，`--n_semantic_zones>=1`）——**不终止、不进 obs**，故纯几何 cost 写不出它；采样在 start→goal 走廊上，保证几何规划器会直穿。`hazard_renderer.py` 画它，VLM 从图里自己看见。
  - `subgoal_pivot_hazard.py` 三方对照：**C1** `mpc`/`safe_expert`（几何盲，直穿）vs **C2** `mpc_oracle`/`*_oracle`（把禁区手工喂成障碍 = 上界）vs **B** `subgoal`（VLM 看图绕行，底层控制器仍纯几何 → 唯一懂禁区的是 VLM）。新指标 `sem_viol`（进过禁区的 episode 比例，Wilson CI）。
  - 防泄漏延续：prompt 只给「避开 amber 禁区」这种**语言/任务约束**（类比「到绿色 goal」），绝不给坐标/哪个候选在区内/clearance/score。
  - **离线 heuristic 验证已通过**（n=20）：C1 sem_viol 95%、C2 0%、B（盲 stand-in）85% —— 指标与 oracle 都对。**赢的判据**：真 VLM 跑时 B 的 sem_viol 从 ~85% 塌向 C2 的 ~0%，C1 仍 ~95%。
  - 复现：`python subgoal_pivot_hazard.py --pilot_mode vlm --n_semantic_zones 1 --episodes 5 --seed 43 --model google/gemini-3-flash-preview --temperature 0 --vlm_fallback hold --log_transcripts --out outputs/semantic_pilot.json`
  - 若真 VLM 想绕却绕不开（底层控制器在两次 subgoal 之间抄近路穿区），调 `--subgoal_radius`（调小=更细）和 `--subgoal_horizon`（调小=更勤重选）。
  - **✅ 隐式语义升级已跑通（2026-06-28，见 `RESULTS_SEMANTIC.md` §10）**：新 flag `--zone_semantics {explicit,implicit}`。`implicit` 把禁区从「amber ✕ 抽象禁止符 + prompt 点名」换成「**水洼外观**（`hazard_renderer.semantic_style="water"`，青碧半透明 + 波纹，无符号无标签）+ prompt 只给类别提示『避开看起来不该开过去的地形』，绝不提 water/teal/X/坐标/哪个候选」。把 instruction-following 升级成真常识。
    - **结果与 explicit 完全一致**：C1 60% / C2 0% / **B 20%**，全 100% goal / 0 碰撞 / fb%=0。去掉标签没让 VLM 掉链子。
    - **决定性证据**：transcript 里 VLM **自发**把它叫 "blue water-like unsafe terrain" / "unsafe water terrain"——prompt 从没说过 water/blue → 是认出来的，不是被告知。
    - 图：`fig_vlm_view_implicit_seed44.png`（waypoint 7 落在水塘里）、`fig_traj_implicit_seed{44,47}.png`（44 win：C1 直穿 / C2 左绕 / B 右绕；47 fail：B 仍擦水塘下缘）。explicit 的图用无后缀名,不被覆盖。
    - 数据：`outputs/semantic_implicit_pilot.json` + `.transcripts.json`。
    - 仍是 L1（类别提示）；L2（prompt 完全不提地形）未做。n=5 caveat 同 explicit。
  - 后续轴（未做）：L2 全隐式、偏好侧、子任务顺序。子任务顺序要改 episode 结构与 success 定义，放禁区跑通之后。
</content>
