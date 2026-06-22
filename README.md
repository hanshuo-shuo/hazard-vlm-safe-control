# hazard-vlm-safe-control

Safety-critical, low-data shared autonomy on a 2D point-mass **PointHazard** task:
a VLM makes **high-level** (semantic / routing) decisions while a **provably-safe
low-level controller** guarantees collision-free execution.

This is the **Path B** line of the project — see [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)
for the thesis plan and [`STRUCTURE.md`](STRUCTURE.md) for a full file-by-file map.

## Core idea

> The VLM picks a semantic/strategic **subgoal**; an online-learned local physics
> model renders the **counterfactual consequence** of each choice for it to select;
> a safe low-level controller guarantees execution never hits a hazard.

The Month-1 "confidence experiment" deliberately shows that on a **pure-geometry**
task a classical controller is already near-perfect, so a VLM low-level is redundant
(and unreliable) — motivating moving the VLM **up** to semantic decisions a geometric
cost function can't express.

## Key files

| File | Role |
|---|---|
| `env_pointhazard.py` | Point-mass env: random hazards + random goal, exact numpy dynamics |
| `hazard_renderer.py` | PIL top-down renderer |
| `mpc_expert.py` | **CEM-MPC** safe low-level controller (rolls the exact dynamics, rejects colliding rollouts) |
| `safe_expert.py` | A\*+PD safe controller (older/weaker baseline low-level) |
| `pivot_vlm.py` | PIVOT visual-prompting core (candidate generation / annotation / VLM select) |
| `subgoal_pivot_hazard.py` | **Month-1 experiment**: matched-seed 3-way comparison (see below) |
| `direct_vla_*.py` | end-to-end VLA baselines |
| `env_pointpushhazard.py`, `pointpush_*` | contact / pushing environment (later months) |

`legacy/` holds archived earlier variants; `docs/` holds the plan, failure-mode
analysis, experiment notes, and literature review.

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
