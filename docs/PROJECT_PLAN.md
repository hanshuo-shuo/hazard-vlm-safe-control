# 项目路线 Plan（博士论文主线）

日期：2026-06-18
状态：从"learned-physics PIVOT"转向 **Path B**。半年内出主结果。

相关文档：
- `WHY_VLM.md` —— **简明英文图解：VLM 的优势到底在哪**（给导师/论文 intro，先读这个）
- `FAILURE_MODE_ANALYSIS.md` —— 泄漏发现 + Path A/B 对比（动机来源）
- `EXPERIMENT_NOTES.md` —— 旧实验记录（结论受泄漏污染，代码可复用）
- `VLM_PHYSICS_LITERATURE_REVIEW.md` —— 相关工作版图
- `../STRUCTURE.md` —— 代码资产清单与状态

---

## 0. 一句话定位

> VLM 选**语义/策略层 subgoal**，online-learned physics 把每个 subgoal 的**反事实后果**渲染出来供它选，一个**可证明安全的低层控制器**保证执行不撞。
> 贡献落在 **safety-critical、低数据、无完整 simulator** 这个 niche。

---

## 1. 为什么转向（把负结果讲成动机）

旧 claim："VLM 能从轨迹图像里读出物理、自己选安全动作" —— 被证伪：
- prompt 里混入了 clearance / safe-unsafe label / score，VLM 其实在读安全 oracle，不是做视觉物理推理（泄漏）。
- 去泄漏后：PointHazard 100%→~90%，Reacher 要逐级加显式量才行，LunarLander 只有 ~20%。

这恰好是领域共识：**没人让 VLM 做低层控制**（SayCan / VoxPoser / VLM-MPC / VLMPC / SIMPACT 全是"VLM 管高层语义、physics/MPC 管低层安全"）。
所以负结果不是污点，是 **Path B 论文的第一张图**：先实证"VLM 当低层安全策略不可靠" → 把它上移成高层规划器。

---

## 2. 必须戳破的陷阱（决定成败）

`safe_expert.py` 的 A\*+PD 在**完全可观测的几何 toy**（纯 PointHazard）上几乎 100% 安全到达 → **在这种环境里 VLM 是多余的**，经典规划碾压。

⟹ Path B 成败不在"把 VLM 上移"，而在**把它放到有比较优势的轴上**：高层决策**无法被几何 cost function 写出来**时。三类 VLM-shaped 的轴：
1. **语义/语言约束**："走用户偏好侧 / 避开看起来像 X 的区域 / 先去拿那个再走" —— A\* 写不出这种 cost，VLM commonsense 能。✅ 主攻
2. **接触/操作里 cost 难手写**（PointPush）：先绕到箱子哪侧推、要不要重新站位、哪条走廊对箱子+agent 都安全。✅ 接触主环境
3. **部分可观测**（不喂特权坐标，VLM 从图读场景）：❌ 容易被检测器比下去，**不在这条上证明自己**。

> 核心纪律：第 1 月的实验**故意**证明"几何已知时 VLM 多余"——这是诚实性护城河，也直接回应导师"为什么不用经典方法"。

---

## 3. 现成资产（离 Path B 比想象近）

| 资产 | 文件 | Path B 角色 |
|---|---|---|
| 可证明安全的低层控制器 | `safe_expert.py`（A\*+PD） | **直接当 Path B 低层，别人要花两月搭的部分你已有** |
| 接触环境 + 专家 | `env_pointpushhazard.py` / `pointpush_expert.py` | 接触主环境的低层 |
| PIVOT 核心 | `pivot_vlm.py` | 候选生成 / VLM 选择 |
| 在线 learned physics | `shared_autonomy_*learned_physics*` | 给 subgoal 做"后果预览"渲染（代码复用，结果重跑） |
| VLA 对照 | `direct_vla_*` | 论文 baseline |

---

## 4. 半年路线（求稳版）

| 月 | 目标 | 产出 |
|---|---|---|
| **第 1 月** | **信心实验**：PointHazard 零泄漏三方对照 —— `VLM 直接选低层动作(去泄漏)` vs `VLM 选 subgoal→SafeExpert 执行` vs `纯 SafeExpert`。**故意**让 VLM 在几何 toy 上不占优 → 实证"几何已知时 VLM 多余"。 | 一张对照表 + 转向的实证理由。新建 `subgoal_pivot_hazard.py`。 |
| **第 2–3 月** | **造出 VLM 真正有优势的任务**：给 PointHazard/PointPush 加语义/语言约束（偏好侧、禁区语义、子任务顺序），A\* 写不出 cost 的那种。VLM 高层+SafeExpert 应**显著赢过**纯经典规划。 | 论文**主结果**。 |
| **第 3–4 月** | **PointPushHazard 做成接触主环境**：candidate 从 force 升成 `(push pose, contact side, direction, duration)`；online physics 渲染 box 轨迹+clearance；VLM 选策略，低层执行。（对应 MOKA/VoxPoser） | 接触物理主实验。 |
| **第 5 月** | **macro-action PIVOT 救 LunarLander**：candidate=动作序列，receding horizon，只执行第一段。把 20% 烂结果翻成 terminal-constraint 正面案例。（对应 VLMPC/Traj-VLMPC） | 第三个环境族。 |
| **第 6 月** | **极简真实桌面 pushing demo**：顶视相机 + ArUco + 在线 2D dynamics。现实可行性的最后一击。 | sim-to-real 收尾。 |

环境组织成**三个物理难点族**（不是堆 10 个无关 env）：惯性安全导航 / 接触推物 / 精密着陆。

---

## 5. 和相关工作的差异点（守住，别滑回"选 force"）

Path B 很挤（SayCan/VoxPoser/VLM-MPC 都在这条线）。锋利差异：
> **online-learned local physics + 把高层选择的反事实后果渲染给 VLM + uncertainty**，面向 safety-critical、低数据、无完整 simulator 的闭环控制。

- vs SIMPACT：它建完整 simulator；我们在线学轻量 local dynamics。
- vs VLMPC：它用重的 video prediction；我们渲染 compact trajectory。
- vs 经典规划：cost 能手写时它赢；我们专攻 cost 写不出（语义/接触）的轴。

---

## 6. Keep / Drop

**Keep**：泄漏审计 + 信息分级表、PointHazard 当干净诊断、Reacher/Lander 揭示 raw-trajectory 上限、learned physics 当"算物理量/做后果预览"的工具。
**Drop / Reframe**：① "轨迹可视化单独就给强物理推理"的强 claim；② 任何把 clearance/score 藏进 prompt 又当 VLM 推理解读的结果；③ 旧 100% 重述为 assisted-safety 上界。

---

## 7. 立即下一步

1. （可选）修掉 `env_pointhazard.make_env` 里坏的 `from hazard.hazard_renderer import`。
2. **新建 `subgoal_pivot_hazard.py`** —— 第 1 月对照实验主脚本：
   - 复用 `PointHazardEnv` + `SafeExpert`；
   - 薄封装："VLM 选离散 subgoal/绕行侧 → SafeExpert 跟踪"；
   - prompt **严格不含** clearance / safe label / score（防泄漏）；
   - matched-seed，输出 success / hazard / timeout / min-clearance / VLM calls 对照表。
</content>
