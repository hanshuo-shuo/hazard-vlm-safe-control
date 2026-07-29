# Interface-Conditioned Safety for VLM Robot Control

This project studies a simple question: **can two interfaces that mean the same
thing make a vision-language robot behave differently?**

Start with the illustrated, plain-English
[`RESEARCH_STORY.md`](RESEARCH_STORY.md). It explains the research change, the
current evidence, the limitations, and the literature-supported next steps.

## Current result

The latest closed-loop scout used two VLMs, two native environments, five scene
seeds, and five kinds of equivalent interface pairs.

| Checkpoint | Matched-pair consistency |
|---|---:|
| Parse | 1.00 |
| Meaning after conversion | 0.70 |
| Grounded region | 0.35 |
| Planner action | 0.75 |
| Native action | 0.55 |
| Native trajectory | 0.55 |

All 120 responses parsed, but 9 of 20 matched interface pairs changed the exact
native action and trajectory. This passed the small continuation gate, but the
evidence remains **`PILOT_ONLY / NOT PAPER RESULT`**.

![Equivalent interfaces can diverge before native execution.](docs/assets/interface_contract_scout/consistency-results.png)

## Research boundary

This repository can support claims about observed closed-loop task success,
collision, semantic violation, clearance, Safe Task Completion, and matched
interface consistency.

It does not establish formal safety, human-intent alignment, a stable model
ranking, or superiority over a fair modern perception-and-planning baseline.
No semantic-safety result is currently `VALIDATED`. Check
[`docs/RESULTS_REGISTRY.md`](docs/RESULTS_REGISTRY.md) before citing a number.

## Repository map

| Path | Purpose |
|---|---|
| `RESEARCH_STORY.md` | Advisor-facing research narrative and next steps |
| `docs/` | Current protocols, result status, and reproducibility records |
| `envs/` | Minimal-permission adapters for native environments |
| `evaluation/` | Interface contracts, replay, metrics, and five-stage audit |
| `scripts/` | Reproducible experiment, analysis, and figure builders |
| `tests/` | Offline acceptance and regression tests |
| `results/` | Current machine-readable experiment artifacts |
| `legacy/` | Superseded narratives, implementations, and historical assets |
| `outputs/` | Historical artifacts kept at their registered provenance paths |

See [`STRUCTURE.md`](STRUCTURE.md) for the active/legacy boundary. Documentation
starts at [`docs/README.md`](docs/README.md).

## Main implementation path

```text
public image + robot capability
  -> equivalent interface contract
  -> parsed meaning + grounded region
  -> fixed planner and native controller
  -> detached collision / semantic / STC evaluation
```

The current implementation is centered on:

- `evaluation/interface_contracts.py` for interface normalization;
- `evaluation/interface_execution.py` for grounding-to-controller execution;
- `evaluation/scout_replay.py` for provider-free native replay;
- `evaluation/five_stage_audit.py` for failure localization;
- `envs/point_hazard_adapter.py` and `envs/safety_gym_goal_adapter.py` for the
  two native environments.

## Setup and verification

Core tests require Python, NumPy, Pillow, and pytest. Native Safety-Gymnasium is
optional and uses `requirements-safety-gym.txt`.

```bash
python -m pytest -q \
  tests/test_condition_contract.py \
  tests/test_interface_contract_protocol_v1.py \
  tests/test_interface_execution_bridge_v2.py \
  tests/test_interface_contract_scout.py
```

Rebuild the advisor figures with:

```bash
python scripts/build_research_story_figures.py
python scripts/build_interface_contract_scout_report_figures.py
```

The formal paid provider release is blocked. The checked-in scout is complete
and should be replayed from cache; no new provider call is needed to reproduce
its native analysis.
