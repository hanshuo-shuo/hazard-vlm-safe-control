# Interface-conditioned embodied safety — ICLR recovery plan

Date: 2026-07-26

Status: **promising exploratory direction; ICLR main result not yet recovered**

## Provider-free implementation status

The new preregistered protocol is
[`INTERFACE_CONTRACT_PROTOCOL.md`](INTERFACE_CONTRACT_PROTOCOL.md). Its checked-in
manifest remains `BLOCKED` and fixes paid seeds `20–24`, 480 new logical calls
per model, three attempts per request, and 1,440 attempts per model. The
provider-free dry-run artifact is
`results/interface_contract_provider_free_dry_run/`: 80 explicitly assigned
blocks, 1,440 logical request rows, zero provider calls/attempts, strict twins,
and native PointHazard plus native Safety-Gymnasium RGB/dynamics gates.

This is `INFRA`, not a paper result and not paid-run authorization. Gemini 2.5
Flash-Lite and Qwen3-VL-30B are ineligible for further paid calls under the new
global ceiling because their historical paid-seed union is `0–4` plus `20–24`.
Cached replay remains allowed and provider-free.

## Decision

The marker-only story remains terminated. The norm-applicability failure story
is also rejected: the earlier model outputs were semantically coherent, while
the evaluator mapped an ambiguous `applicable/not_applicable` field in the
opposite direction from the models' explicit reason and route fields.

The replacement research question is:

> **Are embodied safety evaluations measuring model capability, or artifacts
> of the interface contract between perception, reasoning, and control?**

Working title:

> **The Contract Is the Benchmark: Interface-Conditioned Safety in Embodied
> VLM Evaluation**

The paper cannot be “prompt sensitivity in robots.” Prompt/format/order
sensitivity is already established in language-model evaluation. The required
delta is a causal, closed-loop account of how semantically equivalent interface
contracts change physical decisions, executed safety, and model ranking in
modular embodied systems.

## Five-seed discovery result

The exploratory audit freezes:

- scene seeds `20–24` for every API model;
- three valid models: Gemini 2.5 Flash-Lite, Qwen3-VL-30B-A3B-Instruct, and
  Mistral Small 3.2 24B;
- two tasks: semantic route decision and candidate waypoint selection;
- two renderer/environment contracts: PointHazard and a Safety-Gym headless
  contract render;
- eight semantic/output contracts, three candidate-interface contracts, and
  five offline planner mappings.

The exact combined result is
[`../results/interface_contract_combined_analysis/RESULTS.json`](../results/interface_contract_combined_analysis/RESULTS.json).
It is `PILOT_ONLY`, not a formal paper result.

| Intervention | Same decision / total | Consistency |
|---|---:|---:|
| JSON field order | 23 / 30 | 0.767 |
| structured vs free-text | 21 / 30 | 0.700 |
| positive/negative constraint label | 27 / 30 | 0.900 |
| positive/negative compatibility label | 28 / 30 | 0.933 |
| constraint vs compatibility wording | 29 / 30 | 0.967 |
| explicit vs ambiguous applicability | 29 / 30 | 0.967 |
| marker-ID permutation | 23 / 30 | 0.767 |
| candidate-order permutation | 18 / 30 | 0.600 |

For the same ambiguous-applicability responses, plausible downstream mappings
produce executed aggregate STC from `0.50` to `0.80`; PointHazard uses the
repository fixed CEM-MPC and the second arm uses deterministic Safety-Gym
headless dynamics. The constraint interpretation reaches `0.70`. Gemini and Qwen
reverse relative rank under some interface conditions. Mistral is stable on
all matched semantic contracts and 0.90-consistent on marker/order
perturbations.

The preregistered mainline gate still fails because semantic-contract action
sensitivity is strong in the Safety-Gym render but does not replicate below the
0.80 threshold in PointHazard. Candidate-order sensitivity does appear in both
renderers, but reviving a marker-only paper from that result is forbidden.

## What the result does and does not show

Allowed pilot claims:

1. Equivalent serialization and output-mode contracts can alter actions for
   some model/environment pairs.
2. Marker identities and candidate ordering can alter selected physical
   waypoints even when aggregate selected-safe rate looks similar.
3. A downstream consumer can turn one model response into materially different
   safety/task outcomes through a plausible semantic mapping choice.
4. A single-score model ranking can be unstable across interface conditions.
5. Interface sensitivity is heterogeneous: a robust model is an important
   control, not a contradiction.
6. GPT-5-mini's invalid arm is an API/request compatibility failure, not a model
   capability score. It remains preserved, with 8/110 parsed rows and 87 empty
   contents.

Forbidden claims:

- universal VLM interface fragility;
- a validated cross-environment semantic-contract effect;
- native Safety-Gym replication;
- a native two-environment closed-loop benchmark result (the Safety-Gym arm is
  still a headless dynamics contract);
- formal safety or a paper-ready leaderboard;
- statistical independence of the 30 crossed model/environment/seed pairs.

## Metrics for the paper

The benchmark should report a contract envelope, never one privileged score.

### 1. Interface Equivalence Consistency (IEC)

For semantically equivalent contracts `c` and `c'`, IEC is the matched rate at
which the same physical action is selected. Report action IEC separately from
parse compliance so format failures do not masquerade as reasoning failures.

Primary pairs:

- marker ID permutation;
- candidate ordering;
- positive vs negative label polarity;
- constraint vs compatibility wording;
- field ordering;
- free-text vs structured schema.

### 2. Contract-Induced Safety Range (CISR)

For fixed model-visible evidence and model output, report `max(STC)-min(STC)`
across allowed planner mappings. Decompose the range into semantic violation,
task failure, false-conservative detour, and `unknown` handling.

### 3. Ranking Stability Envelope (RSE)

For every model pair, report whether relative order changes anywhere inside the
contract envelope. Include rank intervals and ties rather than selecting the
best wrapper for each model.

### 4. Five-stage attribution

Preserve recognition, applicability/compatibility, grounding, action proposal,
and enforcement as separate variables. A correct action with a contradictory
semantic field is an interface conflict; a correct semantic field executed by
the wrong mapping is an enforcement-contract failure.

## Formal experiment under the five-seed API ceiling

The ceiling applies to distinct RNG/model sampling seeds per API model. It must
not be evaded by silently relabeling seeds. Statistical coverage instead comes
from preregistered scenario families and matched interventions.

### Models

- at least five VLMs from at least three providers;
- at least two open-weight models;
- provider-returned model ID and provider revision frozen per row;
- a model/request compatibility smoke gate before formal calls;
- maximum five distinct RNG seeds per model.

### Environments and tasks

At least two **native** environments, with different control structure:

1. navigation: PointHazard plus native Safety-Gymnasium RGB/dynamics;
2. contact/manipulation: PointPushHazard, ManiSkill, or LIBERO-Safety with a
   genuinely task-coupled hazard.

Each environment must support both:

- a semantic route/action judgment;
- an executed candidate/subgoal decision through the same controller.

The current headless Safety-Gym render is a protocol unit test, not one of the
two formal environments.

### Scenario families

Use at least eight preregistered families per native environment, crossed with
five seeds:

- clear visible incompatible terrain;
- weak/occluded incompatible terrain;
- compatible look-alike;
- capability reversal twin;
- direct-path intersection;
- near-tangent path;
- multiple candidate detours;
- irrelevant terrain distractor.

Scenario family, not API repetition, is the principal statistical unit. Keep
family generation, image bytes, capability card, and physical geometry matched
across every interface intervention.

### Efficient intervention design

Do not run a full factorial. Use a balanced incomplete block with one-factor
matched pairs plus two prespecified interactions:

- `label semantics × planner mapping`;
- `perceptual ambiguity × output contract`.

Every scene receives a canonical contract and a balanced subset of
interventions. Planner mappings are evaluated offline from cached model outputs
and therefore consume no new API seeds.

### Statistical analysis

- cluster bootstrap by scenario family, retaining all five seed replicas in a
  cluster;
- mixed-effects logistic model with contract, model, environment, ambiguity,
  and their prespecified interactions;
- exact paired counts and intervals for every IEC pair;
- parse-compliant and all-call estimands side by side;
- model ranking under every contract plus worst/best envelope;
- no post-hoc wrapper selection.

## Go/no-go for an ICLR main result

Proceed to a formal paper only if all are true:

1. At least one non-marker semantic/output effect replicates in two native
   environments and two models with IEC below 0.80.
2. An executed controller shows CISR at least 0.15 without treating
   `unknown` as implicit traverse.
3. A rank reversal or rank-correlation drop survives parse-compliant analysis.
4. The finding is not driven only by one provider, one renderer, or API schema
   rejection.
5. Stage-wise attribution identifies whether the effect enters at perception,
   semantic normalization, action proposal, or enforcement.

Kill or redirect to workshop/technical report if:

- effects remain marker/order-only;
- semantic effects remain headless-render-only;
- native execution absorbs the planner-mapping range;
- only parse compliance changes while normalized actions remain stable;
- a robust-envelope evaluation preserves every model ranking.

## Novelty boundary and related work

The general observation that benchmark rankings change with option order or
answer extraction is not new: Alzahrani et al.,
[ACL 2024](https://aclanthology.org/2024.acl-long.744/), report leaderboard
rank changes under small benchmark perturbations. Output-format bias is studied
systematically by Long et al.,
[NAACL 2025](https://aclanthology.org/2025.naacl-long.15/). Hua et al.,
[EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1006/), further show that
some apparent prompt sensitivity comes from rigid evaluation rather than model
semantics. This project must therefore separate parser, normalized semantics,
physical action, and closed-loop outcome.

Embodied Agent Interface
([arXiv:2410.07166](https://arxiv.org/abs/2410.07166)) already provides a
generalized interface and stage-level embodied decision metrics. EmbodiedBench
([ICML 2025](https://proceedings.mlr.press/v267/yang25f.html)) already supplies
multi-environment capability evaluation. The defensible delta here is narrower:

> **matched causal interventions on the contract between embodied modules,
> with exact information provenance and executed safety/ranking envelopes.**

Without native closed-loop propagation and rank-envelope analysis, the work is
better positioned as a workshop paper or technical report.
