# Failure Mode Analysis: What Physical Information Does a VLM Need?

Date: 2026-06-09

This note replaces the earlier optimistic story about learned-physics PIVOT.
The old result is not clean enough to support a publishable claim, because the
prompt included extra physical safety information. The physics module was not
only used to draw future trajectories. It also exposed quantities such as
clearance, safety labels, or candidate scores. That means the VLM may have been
using an explicit safety oracle instead of inferring safety from the rendered
trajectory.

The new question is more honest:

> What form of physical information does a VLM actually need in order to act
> safely?

The current evidence suggests that raw trajectory visualization is often not
enough. VLMs improve when the trajectory is paired with more explicit physical
abstractions, such as safety labels, velocity, or progress-to-goal.

---

## 1. Main Finding

The earlier high success rate was partly caused by information leakage. The VLM
was not just seeing candidate paths. It was also being told how safe the
candidates were, either through explicit clearance values, safe/unsafe labels,
or score-like information.

When this extra information is removed, the system becomes weaker. This means
the original claim:

> A VLM can use visualized physics trajectories to choose safe actions.

is too strong.

A more accurate claim is:

> A VLM does not reliably extract safety-critical physical quantities from raw
> trajectory images. It may need explicit physical abstractions, or safety should
> be handled by a separate low-level controller.

---

## 2. Point Hazard Failure Mode

In the original PointHazard experiments, the prompt sometimes contained
candidate clearance or safety score information. This made the task much easier.
The VLM could choose the candidate with the best safety/progress tradeoff
without really computing hazard distance from the image.

After removing the exact clearance information, performance became worse. In a
preliminary check, a run that looked close to 100% success dropped to around
90%. This number should be treated as an informal diagnostic result, not as a
final benchmark.

This is still not terrible, but the interpretation changes. The result no
longer proves that the VLM learned physics from the trajectory drawing. It only
shows that:

- trajectory images may help a little;
- explicit safety information helps much more;
- the VLM often cannot reliably infer minimum hazard distance by itself.

An example VLM-style reasoning looked like:

```text
Candidate 4 provides the best balance of moving toward the green goal while
maintaining a safe distance from the red hazards. Candidates 1, 2, 3, and 8 move
the agent too close to or directly into the hazard immediately to the
right/top-right, while candidates 5, 6, and 7 move away from the goal.
```

This kind of answer sounds reasonable, but it is hard to know whether the VLM
computed the safety from the image or copied the safety structure from the
prompt. When a safe/unsafe label is provided, the task becomes much closer to
selecting from a learned safety classifier than doing visual physics reasoning.

### Interpretation

The PointHazard result suggests a hierarchy of physical information:

1. **Raw candidate arrows**: weak. The VLM sees possible actions but not their
   consequences.
2. **Predicted trajectories only**: better, but still weak. The VLM sees future
   motion but may not compute safety margins accurately.
3. **Trajectories plus clearance or safe/unsafe labels**: much stronger, but
   this moves safety reasoning out of the VLM and into the physics/safety
   module.
4. **Trajectories plus explicit candidate score**: strongest, but close to
   giving the answer.

The clean research question is therefore not whether a VLM can use physics at
all. The better question is which abstraction level is needed before the VLM can
make a good decision.

---

## 3. Lander and Reacher Ablation

The same pattern appears in Lander/Reacher-style tasks.

### Reacher-style Results

Different prompt information produced very different behavior:

| Prompt information | Success | Crash | Out of bounds | Timeout |
|---|---:|---:|---:|---:|
| Predicted trajectory only | 0.0 | 0.2 | 0.8 | 0.0 |
| Predicted trajectory + predicted velocity | 0.4 | 0.0 | 0.4 | 0.2 |
| Predicted trajectory + predicted velocity + progress to goal | 0.8 | 0.0 | 0.2 | 0.0 |

This is a useful result. It suggests that the VLM does not get enough
information from the path shape alone. It becomes more useful when the prompt
contains the physical quantities that matter for the task:

- where the state will go;
- how fast the system will be moving;
- whether the action makes progress toward the goal.

### Lander Result

On LunarLander-style tasks, even richer physics information is not enough. In a
preliminary check, a prompt with trajectory, predicted velocity, and
progress-to-goal only reached about 20% success.

This makes sense. Lander is not just a navigation task. It is a terminal
constraint task. A good policy must control:

- final position;
- vertical velocity;
- horizontal velocity;
- angle;
- angular velocity;
- fuel or action smoothness;
- touchdown timing.

The VLM is being forced to act like a low-level controller. That may be the wrong
role for it.

---

## 4. Updated Hypothesis

The current hypothesis is:

> VLMs are not reliable low-level safety controllers from raw visual rollouts.
> They need either explicit physical abstractions or a separate low-level
> controller that handles safety and dynamics.

This gives two possible research paths.

---

## 5. Path A: What Physical Information Does a VLM Need?

This path keeps the VLM inside the action-selection loop, but studies the input
representation carefully.

The main research question is:

> Which physical abstractions make VLM action selection safe and useful?

Instead of claiming that trajectory images are enough, this path treats the
prompt as an interface between a physics module and a VLM. The experiment is to
compare different levels of physical abstraction.

### Candidate Prompt Levels

| Level | Information shown to VLM | What it tests |
|---|---|---|
| L0 | Current image + action arrows | Can the VLM choose from action directions? |
| L1 | Predicted trajectory only | Can the VLM infer future safety from path shape? |
| L2 | Trajectory + predicted final state | Does final position/velocity help? |
| L3 | Trajectory + velocity metrics | Does the VLM need explicit speed information? |
| L4 | Trajectory + progress-to-goal | Does explicit task progress help? |
| L5 | Trajectory + safe/unsafe label | How much does a learned safety classifier help? |
| L6 | Trajectory + full candidate score/ranking | Upper bound; close to oracle assistance |

This would turn the failed result into a clean ablation paper:

> VLM control improves only when raw rollouts are converted into task-relevant
> physical abstractions.

### Possible Contributions

1. A taxonomy of physical information for VLM control.
2. A clean leakage-free prompt audit.
3. A set of ablations showing where VLMs fail:
   - hazard distance;
   - velocity;
   - terminal constraints;
   - contact/pushing.
4. A design rule:
   - use VLMs for semantic comparison and high-level choice;
   - expose computed physical quantities when exact safety matters.

### Good Environments for This Path

This path does not require many complex robotics environments at first. It needs
small environments that isolate different physical quantities:

- PointHazard: hazard distance and inertia.
- Reacher: progress and overshoot.
- LunarLander: velocity and terminal constraints.
- PointPushHazard: contact, object motion, and safety of both agent and object.

These are enough for a first clean story because each one tests a different
physical abstraction.

### Risk

The risk is that the conclusion may be mostly negative:

> VLMs need too much explicit information to be useful as low-level policies.

That is still scientifically useful, but it may be harder to publish as a main
robotics paper unless the experiments are very clean and the analysis is strong.

---

## 6. Path B: VLM as High-Level Planner, Physics as Low-Level Controller

This path changes the role of the VLM.

Instead of asking:

> Can the VLM pick the safest low-level action?

ask:

> Can the VLM choose high-level intent, subgoals, or strategy while a physics
> controller handles low-level safety?

This may be a better fit for what VLMs are good at. VLMs are strong at:

- interpreting scenes;
- choosing goals;
- selecting strategies;
- reasoning about task structure;
- explaining failures;
- proposing subgoals.

They are weaker at:

- precise hazard clearance;
- velocity control;
- terminal landing constraints;
- continuous feedback control;
- exact collision avoidance.

### Proposed System

```text
VLM:
  choose high-level mode, subgoal, or strategy

Physics / MPC / learned controller:
  evaluate candidates
  enforce safety
  control velocity
  execute low-level actions

Feedback:
  report failures or new scene changes back to VLM
```

For example:

### PointHazard

The VLM should not choose a force direction every step. It could choose:

- go around the hazard on the left;
- go around the hazard on the right;
- aim for an intermediate waypoint;
- slow down before entering a narrow gap.

Then a low-level MPC or safety controller chooses the actual action.

### PointPushHazard

The VLM can choose:

- which side of the box to approach;
- whether to reposition before pushing;
- which corridor is safer;
- whether the box path or the agent path is the main risk.

The low-level controller handles push contact and collision avoidance.

### LunarLander

The VLM can choose high-level flight phases:

- approach pad;
- reduce horizontal velocity;
- reduce vertical velocity;
- correct angle;
- hover;
- final descent.

The low-level controller handles thrust timing and terminal constraints.

### Why This Path May Be Stronger

This path avoids forcing the VLM to do the thing it is bad at. It also matches
many robotics systems:

- high-level planner for task intent;
- low-level controller for dynamics and safety;
- learned model or MPC for physical prediction.

The publishable story could be:

> VLMs should not be used as low-level safety policies. They are more useful as
> high-level strategy selectors when paired with explicit physics-based
> controllers.

---

## 7. What to Keep and What to Drop

### Keep

- The prompt audit and leakage analysis.
- The clean ablations about different physical information levels.
- The PointHazard environment as a controlled diagnostic task.
- The Lander/Reacher results, because they reveal the limits of raw trajectory
  prompting.
- The idea of learned physics, but use it as a tool to compute physical
  quantities or support MPC, not as proof that the VLM understands physics.

### Drop or Reframe

- Drop the claim that trajectory visualization alone gives strong VLM physical
  reasoning.
- Drop any result where clearance, safety score, or ranking was hidden inside
  the prompt but interpreted as VLM reasoning.
- Reframe the previous 100% success result as an assisted-safety upper bound,
  not as a clean VLM physics result.
- Reframe safety labels as a learned safety module, not as VLM inference.

---

## 8. Short-Term Plan

### Step 1: Make a Prompt Audit Table

For every old experiment, label what information was available:

| Experiment | Trajectory | Velocity | Progress | Clearance | Safe label | Score/rank | Clean? |
|---|---|---|---|---|---|---|---|
| Arrows only | no | no | no | no | no | no | yes |
| Trajectory only | yes | no | no | no | no | no | yes |
| Trajectory + velocity | yes | yes | no | no | no | no | yes |
| Trajectory + progress | yes | yes | yes | no | no | no | mostly |
| Trajectory + clearance | yes | maybe | maybe | yes | no | no | no for VLM-only claim |
| Trajectory + safe label | yes | maybe | maybe | no | yes | no | no for VLM-only claim |
| Trajectory + score | yes | maybe | yes | yes | yes | yes | no |

### Step 2: Run Small Matched Ablations

Do not chase 10 environments yet. Run small controlled experiments first:

- same seeds;
- same VLM;
- same candidate set;
- same number of episodes;
- only change prompt information.

The goal is not to get the best success rate. The goal is to measure what
information changes behavior.

### Step 3: Decide Which Path Has a Stronger Signal

If L1-L4 prompts improve gradually, Path A is promising:

> VLMs need structured physical abstractions.

If even L4 is weak and L5/L6 dominate, Path B is probably better:

> VLMs should stay high-level; safety belongs to a controller.

---

## 9. Current Best Direction

Right now, Path B looks more realistic for a strong robotics direction.

Path A is still worth doing as a diagnostic study, but it may become mostly a
negative result. Path B gives a cleaner system design:

```text
VLM = high-level task and strategy reasoning
Physics model = prediction and physical metrics
Safety/MPC controller = low-level action choice
```

The thesis direction could be:

> How should foundation models be connected to physical controllers for safe
> embodied decision making?

This is broader and safer than:

> Can VLMs directly choose safe low-level actions from rollouts?

The second question may have a negative answer. The first question can still
lead to a useful thesis.

---

## 10. One-Sentence Summary

The old result was helped by safety-information leakage. The new result is more
important: raw physical rollouts are not enough for safe VLM control. A VLM
either needs explicit physical abstractions, or it should act as a high-level
planner while a physics/MPC module handles safety-critical control.
