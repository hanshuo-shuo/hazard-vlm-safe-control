# hazard-vlm-safe-control

Research prototype for auditing semantic safety in VLM-guided robot control.
The current system uses a VLM for high-level semantic/routing decisions and an
exact-model, safety-oriented sampling MPC (empirical) controller for low-level execution.

> 🧭 **文档从哪读起 → [`docs/README.md`](docs/README.md)（索引，先看这个）。**
> 当前真相源：[`docs/RESEARCH_REVIEW_COMMENTS.md`](docs/RESEARCH_REVIEW_COMMENTS.md)
> （必须修改项）+ [`docs/ICLR_PLAN.md`](docs/ICLR_PLAN.md)（覆盖旧计划的新路线）。
> 现有 semantic 结果因场景 sampler 与对照混淆待重跑，不应作为论文数字。
> 代码逐文件地图见 [`STRUCTURE.md`](STRUCTURE.md)。

## Current research question

> Which part of a modular VLM-control pipeline deserves credit for apparent
> closed-loop safety: visual recognition, task/norm interpretation, spatial
> grounding, routing, privileged cues, or low-level execution?

## Key files

| File | Role |
|---|---|
| `env_pointhazard.py` | Point-mass env: random hazards + random goal, exact numpy dynamics |
| `hazard_renderer.py` | PIL top-down renderer |
| `mpc_expert.py` | **CEM-MPC** safety-oriented sampling MPC (empirical) using exact discrete dynamics; safety is reported from rollout metrics |
| `safe_expert.py` | A\*+PD safe controller (older/weaker baseline low-level) |
| `pivot_vlm.py` | PIVOT visual-prompting core (candidate generation / annotation / VLM select) |
| `subgoal_pivot_hazard.py` | **Month-1 experiment**: matched-seed 3-way comparison (see below) |
| `direct_vla_*.py` | end-to-end VLA baselines |
| `env_pointpushhazard.py`, `pointpush_*` | contact / pushing environment (later months) |

`legacy/` holds archived earlier variants; `docs/` holds the plan, failure-mode
analysis, experiment notes, and literature review.

## Claim boundary

| 当前支持的 claim | 当前不支持的 claim |
|---|---|
| CEM-MPC 是基于 exact discrete dynamics 的 safety-oriented sampling MPC（empirical）。 | 形式化安全定理、递归可行性或可证明的安全性质。 |
| 可报告 sampled rollouts 的 hazard、semantic violation、success 和 clearance 等 empirical metrics。 | 仅凭采样结果就宣称所有执行轨迹都不会碰撞。 |
| VLM 负责高层语义/路由选择，低层控制器负责执行；两层贡献可按 protocol 分开测量。 | 人机协作、人类 operator intent 或 user-study 结论（当前没有这些实验）。 |
| 当前仓库是用于 safety accounting 的研究原型，历史结果按 registry 状态管理。 | oracle 对齐、常识理解或超越 classical planning 的 headline claim。 |

## Quick start

```bash
pip install numpy pillow torch openai

# Offline smoke test (no API key; geometric stand-in pilot)
python subgoal_pivot_hazard.py --pilot_mode heuristic --episodes 20

# Real run against an OpenRouter VLM
OPENROUTER_API_KEY=sk-... python subgoal_pivot_hazard.py \
    --pilot_mode vlm --model google/gemini-3-flash-preview --episodes 30 \
    --out outputs/month1_confidence.json
```

The script compares three policies on matched seeds:

- `mpc` / `safe_expert` — pure low-level controller (no VLM)
- `subgoal` — VLM picks a high-level subgoal → low-level controller executes it
- `direct` — VLM picks the low-level force directly (de-leaked PIVOT)

and reports success / hazard / timeout / min-clearance / VLM-calls.

> **Anti-leakage:** the VLM prompts contain only the rendered image + a generic
> task description — no per-candidate clearance, safe/unsafe label, or score.

## Notes

- Model weights, gifs, run logs, and data are **not** committed (reproducible
  artifacts); see `.gitignore`.
- `.env` (OpenRouter key) is git-ignored.
