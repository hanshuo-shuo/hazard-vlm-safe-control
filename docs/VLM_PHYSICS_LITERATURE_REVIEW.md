# VLM + Physics / PIVOT 文献调研与下一步路线

日期：2026-06-04

这份笔记的目标不是做一个泛泛的 robotics foundation model survey，而是回答一个更具体的问题：

> 现有 VLM / VLA / VLM+physics 工作是怎么让视觉语言模型解决物理控制环境的？它们用了什么环境？哪些方法可以和当前的 learned-physics PIVOT 项目融合，帮助把故事推向 CoRL 风格的论文？

当前本地项目的核心现象已经很清楚：

- 直接让 VLM/PIVOT 选动作，在惯性和安全约束下会犯错。
- 只告诉 VLM 哪些动作安全，会更安全但保守，容易 timeout。
- 只给历史反馈，不给当前候选动作的未来后果，会更激进但不安全。
- 给 VLM 看每个候选动作的短期物理后果，效果明显更好。
- 用小的在线 dynamics model 学这些后果，再画到图里，可以替代 oracle rollout 的一部分作用。

所以最自然的论文位置是：

> **Counterfactual physics prompting for VLM control**：VLM 不缺目标语义，缺的是 action-conditioned dynamics。我们不把 VLM 直接改造成一个大 VLA，而是在 VLM 外面接一个轻量、在线学习的 physics predictor，把候选动作会导致的未来状态显式渲染出来，让 VLM 做可解释的选择。

---

## 1. 文献版图：大家到底怎么把 VLM 接到控制里

可以把相关工作分成五条线。

| 方向 | 代表工作 | 核心做法 | 对你的启发 |
|---|---|---|---|
| Visual prompting / action selection | PIVOT, MOKA, RoboPoint | 把连续动作、关键点或 affordance 画在图上，让 VLM 选 | 你的方法可以说是从“画动作”升级到“画动作后果” |
| LLM/VLM planner + affordance / value map | SayCan, Inner Monologue, Code as Policies, VoxPoser, PG-VLM | 大模型做高层语义规划，低层由 affordance/value function/motion planner 执行 | 可以借它们的模块化叙事：VLM 管语义选择，物理模块管可执行性 |
| VLM + MPC / world model | VLMPC, Traj-VLMPC, SIMPACT, PointWorld | 候选动作先通过预测模型/模拟器 rollout，再让 cost 或 VLM 评估 | 这是和你最贴近的方向；你的优势是轻量、在线、直接视觉化给 VLM |
| End-to-end VLA | RT-2, OpenVLA, Octo, pi0 | 把视觉、语言、动作统一训练成一个策略模型 | 你的 direct VLA baseline 可以说明端到端路线数据和 tuning 压力大 |
| Benchmark / environment | Safety Gymnasium, ManiSkill3, RLBench, LIBERO, Meta-World, VLMbench, Language Table, RoboDesk | 给安全、接触、多任务、语言控制提供环境族 | 你需要从 toy-looking env 扩到“物理难点族”，不只是多堆任务 |

---

## 2. Visual Prompting 线：从“选动作”到“选物理后果”

### 2.1 PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs

链接：[arXiv](https://arxiv.org/abs/2402.07872)，[项目页](https://pivot-prompt.github.io/)

**它的方法，用白话讲**

PIVOT 的出发点是：VLM 输出文字，但机器人控制需要连续坐标、连续动作或轨迹。PIVOT 不让 VLM 直接回归一个连续数值，而是把很多候选答案画在图片上，比如候选移动方向、候选抓取点、候选轨迹。然后问 VLM：“这些编号里哪个最好？”

它每轮做四步：

1. 在当前图像上采样一批候选动作/点/轨迹。
2. 把候选项用编号、箭头或点标在图上。
3. 让 VLM 选最好的一个或几个。
4. 根据 VLM 选中的候选项重新拟合一个更集中的分布，再采样更精细的候选项。

所以 PIVOT 的本质是：把连续控制问题改写成迭代的 VQA 选择题。

**它用了什么环境/任务**

PIVOT 主要验证在：

- real-world mobile manipulator navigation；
- real-world manipulation from images；
- simulation instruction following；
- RefCOCO 等空间定位任务。

项目页明确说它覆盖 real-world robotic navigation、real-world manipulation、simulation instruction following 和 localization 类空间推理。

**它解决问题的方式**

PIVOT 不是学 physics，也不是学 policy。它依赖 VLM 的视觉空间理解：如果候选动作画得足够清楚，VLM 能通过常识和图像理解选一个合理动作。

**局限**

PIVOT 画的是“动作候选”本身。对于没有强 dynamics 的任务，比如点击、抓取位置、简单导航，这可能够用。但对你的 PointHazard、PointPush、LunarLander 这种环境，动作本身不是关键，动作导致的速度变化、碰撞、未来 clearance、落地速度才是关键。VLM 看见一个箭头，不等于知道它在惯性系统里会把 agent 带到哪里。

**怎么和你的方法融合**

你的方法可以直接定位成 PIVOT 的 physics-aware extension：

> PIVOT asks the VLM to choose among visualized actions. We ask the VLM to choose among visualized action-conditioned futures.

这句话很关键。它把你的贡献从“小改 PIVOT prompt”变成“补上 PIVOT 在物理控制中的缺口”。

建议在论文里做一张对照图：

- Vanilla PIVOT：只画候选力箭头。
- Safe-filter PIVOT：画箭头并标 unsafe/safe。
- History PIVOT：给过去动作反馈。
- Physics-PIVOT：画每个候选动作 rollout 后的轨迹、最终状态、最小 clearance、速度。

你的 ablation 已经基本支持这个故事。

---

### 2.2 MOKA: Open-World Robotic Manipulation through Mark-Based Visual Prompting

链接：[arXiv](https://arxiv.org/abs/2403.03174)

**它的方法，用白话讲**

MOKA 也是 visual prompting。它不是让 VLM 直接输出低层控制，而是在图片上放 mark/keypoint，让 VLM 选择哪个点是任务相关的 affordance，比如哪里可以抓、哪里可以推、工具应该接触哪里。

它的核心是用点表示 affordance：

- 先把候选关键点标在图片上；
- 问 VLM 哪些点适合当前语言任务；
- 再把这些点转成机器人的 motion target；
- 后面可以通过 in-context learning 和 policy distillation 利用机器人经验提高效果。

**它用了什么环境/任务**

论文摘要提到 table-top manipulation，包括：

- tool use；
- deformable body manipulation；
- object rearrangement。

这些任务比纯导航更接近真实 manipulation，但它主要处理的是 affordance / keypoint grounding，不是短期 dynamics prediction。

**对你的启发**

MOKA 说明 VLM 对“选择哪个空间点”比“直接输出连续动作”更擅长。这和你的 PIVOT 方向一致。

可以融合的点：

- 在 PointPushHazard 或真实桌面 pushing 里，不只画动作 rollout，还画“理想接触点”或“push pose”。
- 对接触任务，候选可以从 force direction 扩展成 `(contact point, push direction, duration)`。
- VLM 先选接触 affordance，再由 physics predictor 评估几个候选 push 的后果。

这能把你的 PointPush 从“点 agent 推 box”讲成更机器人化的 contact-rich affordance + dynamics problem。

---

### 2.3 RoboPoint: A VLM for Spatial Affordance Prediction for Robotics

链接：[arXiv](https://arxiv.org/abs/2406.10721)，[项目页](https://robo-point.github.io/)

**它的方法，用白话讲**

RoboPoint 发现通用 VLM 很难用语言精确描述机器人动作点，所以他们直接训练一个 VLM 来预测图像中的 spatial affordance keypoint。它用自动合成数据生成 pipeline 来 instruction-tune VLM，不需要真实机器人数据或人类 demonstration。

它做的是：

1. 给图像和语言任务；
2. 让模型输出任务相关的关键点；
3. 用关键点驱动 navigation、manipulation 或 AR assistance。

**它用了什么环境/任务**

论文摘要说它支持 navigation、manipulation、AR assistance，并和 GPT-4o 以及 PIVOT 等 visual prompting 方法比较。

**对你的启发**

RoboPoint 强调“VLM 需要 robotics-specific spatial supervision”。你的方法可以强调另一种补法：

- RoboPoint 给 VLM 补 spatial affordance；
- PG-VLM 给 VLM 补 object physical properties；
- 你的方法给 VLM 补 **action-conditioned dynamics**。

这三者可以放在 related work 里形成一条清楚逻辑：

> 有些工作让 VLM 更懂哪里能动，有些工作让 VLM 更懂物体属性，我们让 VLM 在决策时看到动作会造成什么后果。

---

## 3. Planner + Affordance 线：VLM 管语义，物理模块管可执行性

### 3.1 SayCan: Do As I Can, Not As I Say

链接：[arXiv](https://arxiv.org/abs/2204.01691)

**它的方法，用白话讲**

SayCan 的核心思想很朴素：大语言模型知道“应该做什么”，但不知道机器人“能不能做”。所以它把每个高层 skill 的选择分成两个分数：

- LLM 给的语义分数：这个 skill 对完成语言任务有多合理？
- affordance/value function 给的可执行性分数：当前状态下机器人做这个 skill 成功概率多高？

最后选二者乘起来最高的 skill。

比如用户说“清理桌上的饮料”，LLM 可能知道要“拿海绵、擦桌子、丢垃圾”，但机器人当前如果够不到海绵，affordance 分数会压低这个动作。

**它用了什么环境/任务**

SayCan 主要是 Google 真实机器人环境，移动机械臂在厨房/桌面场景中执行长程任务。它使用预训练 skill 和 value function，而不是让 LLM 直接控制马达。

**和你的关系**

SayCan 是“语义合理性 × 可执行性”的经典模块化叙事。你的方法可以借这个框架，但把可执行性从离散 skill value function 换成候选 trajectory 的物理后果：

- SayCan：`score(skill) = language_relevance(skill) * affordance_success(skill, state)`
- 你的方法：`choose(action) = VLM(image with predicted trajectory, task, safety cues)`
- 你的 teacher：`score = goal_progress + clearance_weight * min_clearance`

你可以说：SayCan 用 learned affordance 判断 skill 是否可执行；我们用 online learned dynamics 把每个低层候选动作的可执行后果画出来，让 VLM 直接比较。

---

### 3.2 Inner Monologue: Embodied Reasoning through Planning with Language Models

链接：[arXiv](https://arxiv.org/abs/2207.05608)

**它的方法，用白话讲**

Inner Monologue 的核心是让语言模型在执行过程中不断读环境反馈，比如：

- 物体识别结果；
- 当前步骤是否成功；
- 场景描述；
- 人类反馈。

这些反馈都变成文字，放回 LLM prompt 里，让 LLM 形成“内心独白”，不断重规划。

**它用了什么环境/任务**

摘要里说它验证在三个 domain：

- simulated tabletop rearrangement；
- real tabletop rearrangement；
- real-world kitchen mobile manipulation。

**和你的关系**

你已经做过 history-only / self-iterating PIVOT，结果是：历史反馈让 agent 更激进，但不安全。这个正好可以和 Inner Monologue 做区分：

- Inner Monologue 说明 closed-loop feedback 对长程语义规划有帮助；
- 你的 ablation 说明，在低层物理安全控制里，历史反馈不能替代当前反事实 rollout；
- 也就是说，语言反馈是 necessary but not sufficient，尤其当动作后果由速度、接触、惯性决定时。

可融合点：

- 保留你当前的 real-step feedback，但不要把它当主要 physics grounding；
- 把它作为辅助 memory：告诉 VLM 上一步真实 rollout 和预测 rollout 的误差；
- 用于校准 physics model 或触发 conservative mode。

---

### 3.3 Code as Policies

链接：[arXiv](https://arxiv.org/abs/2209.07753)，[项目页](https://code-as-policies.github.io/)

**它的方法，用白话讲**

Code as Policies 让 LLM 直接写 robot policy code。这个 code 可以调用 perception API、控制 primitive、几何函数、循环和条件判断。它不是直接输出动作，而是输出一个可执行程序。

**它用了什么环境/任务**

项目页强调它在多个真实机器人平台上展示，包括 tabletop manipulation、trajectory-based control、pick-and-place 等。

**对你的启发**

它告诉我们一个重要原则：大模型适合组合已有模块，而不是从零学低层动力学。

你的方法也可以写成模块组合：

```text
VLM = visual chooser / semantic reasoner
Physics model = local counterfactual predictor
Renderer = interface that turns physics into visual prompt
Teacher = interpretable physical score
Controller = receding-horizon execution
```

如果之后想拓展，可以让 LLM/VLM 生成 candidate generator 的代码或参数，比如：

- 在 LunarLander 里生成“brake-left、brake-right、hover、tilt-correct”这类 macro-action；
- 在 PointPush 里生成“先绕到箱子后方，再推”的 subgoal；
- 在不同环境里选择不同 rollout horizon 和 safety weights。

但当前论文主线不建议变成 Code-as-Policies，因为那会稀释你的 physics-prompting claim。

---

### 3.4 VoxPoser: Composable 3D Value Maps for Robotic Manipulation

链接：[arXiv](https://arxiv.org/abs/2307.05973)，[项目页](https://voxposer.github.io/)

**它的方法，用白话讲**

VoxPoser 是非常值得你仔细看的 CoRL 2023 oral。它的思想是：LLM/VLM 不直接输出动作，而是生成 3D value maps。

流程大概是：

1. 输入 RGB-D 观察和语言任务。
2. LLM 写代码，调用 VLM 来识别目标、障碍、约束。
3. 这些信息变成 3D voxel grid 上的 affordance map 和 constraint map。
4. Motion planner / MPC 在这些 value maps 上找轨迹。
5. 因为 value maps 可以随着新视觉反馈重新计算，所以能 closed-loop replanning。

简单说：它把语言目标变成 3D 空间里的“哪里好、哪里危险、哪里约束强”。

**它用了什么环境/任务**

VoxPoser 在 simulated 和 real-robot environments 都做了大量 everyday manipulation 任务。项目页和摘要强调 open-set language instructions、open-set objects，以及真实桌面操作。

**它解决环境的方式**

VoxPoser 不是学一个端到端策略，而是让 foundation models 产出一个可被传统 planner 使用的中间表示：3D value map。

**和你的关系**

VoxPoser 很适合用来帮你讲“中间表示”的价值：

- VoxPoser 的中间表示：空间 value map / constraint map。
- 你的中间表示：候选动作的 counterfactual trajectory image。

你们都在避免让 VLM 直接做低层控制，都在把问题转成 VLM 更擅长的“看图判断”或“语义选择”。

可融合点：

- 对 PointHazard：除了 trajectory rollout，可以渲染一个 learned risk/progress heatmap，让 VLM 看到空间价值。
- 对 PointPush：可以渲染 push-pose value map：agent 应该站在箱子哪一侧，哪些位置会导致安全推箱。
- 对真实机械臂：用 RGB-D + segmentation 构造 2D/3D affordance map，再用你的 physics rollout 检查候选 push/grasp 的后果。

论文里可以把你和 VoxPoser 的区别写清楚：

> VoxPoser grounds language into spatial affordance and constraint maps for motion planning. Our method grounds candidate actions into predicted physical consequences for VLM selection.

---

### 3.5 Physically Grounded VLMs / PG-VLM

链接：[arXiv](https://arxiv.org/abs/2309.02561)，[项目页](https://iliad.stanford.edu/pg-vlm/)

**它的方法，用白话讲**

PG-VLM 关注的是 VLM 不懂物体物理属性，比如：

- fragile / 不易碎；
- heavy / light；
- liquid / solid；
- soft / hard；
- transparent / opaque；
- material / shape。

他们做了 PhysObjects 数据集，包含大量 household object 的物理概念标注，然后 fine-tune VLM，让它更懂这些物体属性。再把这个 physically grounded VLM 接到 LLM planner 里，帮助机器人规划。

**它用了什么环境/任务**

它主要是 robotic manipulation planning，包含真实机器人展示，任务要求机器人根据物体物理属性做选择。

**和你的关系**

PG-VLM 解决的是 object-centric physical knowledge。你的方法解决的是 action-conditioned physical dynamics。

二者可以互补：

- PG-VLM：这个物体能不能压？会不会碎？轻不轻？
- 你的 physics PIVOT：如果我这样推/走/落，会发生什么？

如果你之后上真实机械臂，PG-VLM 类方法可以帮你处理物体属性先验；你的 learned physics 处理局部动作后果。

---

## 4. VLM + MPC / World Model 线：和你最接近的相关工作

### 4.1 VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation

链接：[arXiv](https://arxiv.org/abs/2407.09829)，[GitHub](https://github.com/PPjmchen/vlmpc)

**它的方法，用白话讲**

VLMPC 是 RSS 2024，很接近你的方向。它把 VLM 接进 MPC：

1. 当前图像 + 目标图像或语言指令输入 VLM。
2. VLM 不直接给完整动作，而是给一个粗粒度动作方向，比如末端执行器应该往 x/y/z 哪个方向动、夹爪开合、旋转方向。
3. 这些 VLM 输出变成 action sampling distribution 的均值。
4. 从这个分布采样很多 7D end-effector action sequences。
5. 用 action-conditioned video prediction model 预测每个 action sequence 的未来图像。
6. 用 hierarchical cost function 评估未来图像：一部分是 pixel-level goal matching，一部分是 VLM-assisted knowledge-level cost。
7. 选最好的 action sequence，执行第一步，然后下一步重新规划。

这就是典型的 “look before you leap”：先想象未来，再执行。

**它用了什么环境/任务**

从论文和官方 repo：

- 官方 repo 提供 Language-Table 环境实现。
- 论文 simulation 部分：
  - RoboDesk：7 个任务；
  - Language Table：50 个 simulated environments / tasks。
- real-world experiments：
  - grasp towel；
  - put banana；
  - turn on lamp；
  - wipe water。
- 视觉预测模型 DMVFN-Act：
  - 先用 Open X-Embodiment 中的 Berkeley Autolab UR5、Columbia PushT、ASU TableTop Manipulation 三个子集预训练；
  - 再用目标环境中 20 episodes robot execution 做适配。

**它解决环境的方式**

VLMPC 解决的是 manipulation planning。它既用 VLM 提供 open-set 语义/视觉引导，又用 video prediction 提供 foresight。最后的选择不是让 VLM 自己凭感觉选，而是通过 cost function 评估 predicted future。

**它和你的方法的相同点**

- 都是 candidate-based。
- 都有预测未来。
- 都是 receding horizon。
- 都反对 VLM 直接输出低层连续控制。
- 都强调 VLM 自身缺少动态预测能力，需要外部 dynamics/world model。

**关键区别**

| 点 | VLMPC | 你的方法 |
|---|---|---|
| dynamics 表示 | action-conditioned future video frames | low-dimensional local physics rollout, rendered into prompt |
| VLM 角色 | action sampler + knowledge-level evaluator | visual chooser over physically annotated candidates |
| 成本 | video prediction 模型较重 | 小 online dynamics model |
| 学习方式 | 预训练 + 少量适配 | 在线从真实 transition 学 |
| 可解释性 | 未来图像 + cost | 每个候选 trajectory、clearance、progress 可直接看 |
| 适合任务 | 视觉 manipulation | inertial safety/control + simple contact |

**怎么融合**

这是最值得借的结构：

- 把你的 controller 正式写成 **MPC-style receding horizon PIVOT**。
- 对 LunarLander，不要只 rollout 单个 action 3 step，要采样 action sequence / macro-action，然后每次只执行第一步。
- 把 VLMPC 的 hierarchical cost 借过来：
  - low-level physics cost：goal progress、clearance、terminal velocity、tilt、fuel；
  - VLM-level visual preference：从渲染图判断路径是否绕开危险、是否合理。
- 对 PointPush，借它的 action sequence sampling，而不是单步 force candidates。

你可以在 related work 里说：

> VLMPC predicts future images for manipulation MPC; our method predicts compact local physical rollouts online and exposes them directly as counterfactual visual prompts to the VLM.

---

### 4.2 Traj-VLMPC: Vision-Language Model Predictive Control for Manipulation Planning and Trajectory Generation

链接：[arXiv](https://arxiv.org/abs/2504.05225)

**它的方法，用白话讲**

Traj-VLMPC 是 VLMPC 的后续。它发现 video prediction 很重，所以把“预测未来视频”换成“预测未来 motion trajectory”。这对你特别重要，因为你的方法本来就是画轨迹而不是生成完整图像。

大致流程：

1. VLM 根据目标图像/语言和当前观察生成候选 action sequences。
2. 不再对每个 sequence 生成未来图像，而是生成运动轨迹。
3. 用 VLM-based hierarchical cost function 评估这些候选轨迹。
4. 选择最优序列，执行并继续 MPC。

**它用了什么环境/任务**

它沿着 VLMPC 的方向做 manipulation planning，论文摘要强调 public benchmarks 和 real-world robotic manipulation tasks。VLMPC repo 里也把 VLMPC 和 Traj-VLMPC 放在同一个代码库。

**对你的启发**

Traj-VLMPC 是你的天然邻居，因为它也说“轨迹预测比视频预测更轻，更适合长 horizon / real time”。

你的区别可以写成：

- Traj-VLMPC 仍然是在 manipulation planning 框架里生成/评估 trajectory；
- 你的方法更强调 VLM prompt interface：把每个 candidate 的短期物理后果画给 VLM；
- 你的 dynamics model 是在线学的，目标是低数据、可解释、安全控制。

强建议你在 LunarLander 里直接采用 Traj-VLMPC 的思路：候选单位从 action 变成 action sequence / trajectory。

---

### 4.3 SIMPACT: Simulation-Enabled Action Planning using VLMs

链接：[arXiv](https://arxiv.org/abs/2512.05955)，[项目页](https://simpact-bot.github.io/)

**它的方法，用白话讲**

SIMPACT 是非常近的 VLM+physics 工作。它的核心观点和你几乎一样：

> VLM 从静态互联网图文数据训练来，缺少 causal interaction 和 action-conditioned change，所以很难做需要物理理解的精细 manipulation。

它的解决方案是 test-time simulation-in-the-loop：

1. 从单张 RGB-D 观察构建一个物理模拟器。
2. VLM 先根据任务提出 action sequence。
3. 把 action sequence 放进模拟器 rollout。
4. 把 rollout 结果给 VLM 看。
5. VLM 根据模拟结果优化 action sequence。
6. 迭代直到找到成功计划，再在真实机器人上执行。

**它用了什么环境/任务**

论文摘要说它在 real-world rigid-body and deformable manipulation tasks 上评估。搜索到的论文片段提到任务跨 rigid bodies 和 deformable materials，例如 cartons、bowls、boxes 等；CVPR 版本摘要说是五个真实物理推理任务，另有 PDF 片段提到七个任务版本。保守写法：它聚焦多个真实 rigid/deformable manipulation tasks。

**它解决环境的方式**

SIMPACT 不训练 VLM，也不把 VLM 改成 VLA。它在测试时构造物理仿真，让 VLM 通过模拟 rollout 获得物理 grounding。

**它和你的方法的关系**

这是你最应该引用、也最需要区分的工作。

相同点：

- 都认为 VLM 缺 action-conditioned physical dynamics。
- 都把 rollout 结果放进 VLM 决策循环。
- 都不是端到端 VLA。
- 都追求 test-time / online physical grounding。

区别：

| 点 | SIMPACT | 你的方法 |
|---|---|---|
| 物理来源 | 从 RGB-D 建完整/近似 simulator | 从真实 transition 在线学 local dynamics |
| 任务 | 真实 manipulation，刚体/软体 | 低维安全导航、接触推物、可扩展到 lander/robot pushing |
| VLM 角色 | sample / optimize / evaluate action sequence | choose among rendered candidate rollouts |
| 工程成本 | 需要 RGB-D、重建、物理参数、模拟器 | 只需状态/图像和在线 dynamics model |
| 强项 | 真实复杂物体交互 | 轻量、可解释、适合安全控制和低数据 |

**怎么融合**

SIMPACT 给你两个很好的升级方向：

1. **Iterative action optimization**：不要只让 VLM 一次性选候选动作。可以让 VLM 看一批 rollout 后，提出下一批更好的 candidates。也就是 PIVOT refinement 从“动作空间局部缩小”升级成“根据物理 rollout 改候选策略”。
2. **真实机器人 tabletop pushing**：你不需要一开始做复杂软体。可以做 SIMPACT-lite：顶视 RGB-D/RGB + ArUco tracking + local learned 2D dynamics，把真实桌面 pushing 变成你的 PointPushHazard 现实版。

论文 positioning：

> SIMPACT builds a test-time simulator from RGB-D to ground VLM planning. We instead learn a lightweight local dynamics model online and render counterfactual rollouts, targeting low-data, safety-critical closed-loop control where full simulation is unavailable or unnecessary.

---

### 4.4 VLM-MPC for Autonomous Driving

链接：[arXiv](https://arxiv.org/abs/2408.04821)

**它的方法，用白话讲**

这篇不是 manipulation，而是自动驾驶。它把 VLM 和 MPC 分成上下两层：

- 上层 VLM 看前视图像、ego 状态、交通环境和记忆，输出驾驶参数，比如目标速度、跟车距离。
- 下层 MPC 根据这些参数做实时控制，并考虑车辆动力学约束。

**它用了什么环境/任务**

它在 nuScenes 数据相关环境里验证，关注不同驾驶场景，比如夜间、雨天、路口。评估安全指标如 Post Encroachment Time、平滑性等。

**对你的启发**

这篇可以帮助你讲“VLM 不应该直接发低层控制”的控制架构：

- VLM 适合高层参数/偏好/候选选择；
- MPC/physics module 适合实时动力学约束；
- 安全控制需要 lower-level dynamics controller。

LunarLander 也可以借这个分层：

- VLM 选高层模式：approach / brake / hover / lateral correction / final descent；
- 低层 controller 或 macro-action 执行对应 thrust pattern；
- learned physics rollout 评估 terminal constraints。

---

### 4.5 PointWorld / 3D World Models

链接：[项目页](https://point-world.github.io/)

**它的方法，用白话讲**

PointWorld 是 CVPR 2026 的 3D world model。它从 RGB-D 和机器人动作预测 full-scene 3D point flow，也就是预测动作会让 3D 世界中的点怎么动。然后把这个世界模型放进 MPC 里做 manipulation。

**它用了什么环境/任务**

项目页说它训练数据来自 real + simulated manipulation，约 2M trajectories / 500 hours，覆盖 Franka 单臂和双臂 humanoid。真实任务包括 rigid-body pushing、deformable/articulated object manipulation、tool use。

**对你的启发**

PointWorld 是重型路线：大规模 3D world model + MPC。你现在不需要做这个，但可以借它的概念语言：

- 他们预测 full-scene 3D point flow；
- 你预测 compact state flow / local trajectory；
- 本质都是 action-conditioned future prediction。

这可以帮你把方法放进 broader world model 语境里。

---

## 5. End-to-End VLA 线：为什么你的 direct VLA baseline 很重要

### 5.1 RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

链接：[arXiv](https://arxiv.org/abs/2307.15818)

**它的方法，用白话讲**

RT-2 把 VLM 改造成 VLA。核心做法是把机器人动作表示成 token，和自然语言 token 放在同一个模型输出空间里。训练时同时喂互联网视觉语言任务和机器人轨迹数据，让模型既保留 web-scale 语义知识，又学会把图像/语言映射到机器人动作。

**它用了什么环境/任务**

RT-2 主要是 Google 真实机器人 manipulation，大量真实 trial，强调 novel objects、unseen commands、multi-stage semantic reasoning。论文摘要提到 6k evaluation trials。

**和你的关系**

RT-2 是端到端 VLA 的源头之一。你的论文里可以说：

- RT-2 展示了 web VLM knowledge 可以转移到机器人控制；
- 但这类方法需要大规模 robot trajectory data 和复杂 co-finetuning；
- 你的路线是低数据、模块化：不把所有 physics 都塞进 VLM 参数，而是在 test-time / online 显式外置。

---

### 5.2 OpenVLA

链接：[arXiv](https://arxiv.org/abs/2406.09246)，[项目页](https://openvla.github.io/)

**它的方法，用白话讲**

OpenVLA 是开源 7B VLA。它用 Llama 2 + DINOv2/SigLIP 视觉特征，在 970k real-world robot demonstrations 上训练。它强调两个点：

- 开源通用机器人策略；
- 可以用 LoRA/量化在消费级 GPU 上 fine-tune。

**它用了什么环境/任务**

论文摘要说它在 29 tasks、多 robot embodiments 上超过 RT-2-X，并能 fine-tune 到新任务、多物体语言 grounding 环境。

**和你的关系**

你本地 same-backbone direct VLA 结果很有价值：它不是拿 OpenVLA 这种大数据预训练模型来压你，而是用相同 Qwen backbone 做受控对照，说明直接回归动作在你的惯性安全环境里很难。

你可以这样写：

> Large VLAs are powerful when trained on hundreds of thousands of robot demonstrations, but our controlled same-backbone VLA baseline shows that simply attaching an action head to a VLM does not reliably learn safety-critical dynamics from modest data.

---

### 5.3 Octo

链接：[arXiv](https://arxiv.org/abs/2405.12213)，[项目页](https://octo-models.github.io/)

**它的方法，用白话讲**

Octo 是 open-source generalist robot policy。它用 Open X-Embodiment 的 800k trajectories 训练一个 transformer policy，可以接受语言命令或 goal image，并能在新 robot setup 上快速 fine-tune。

**它用了什么环境/任务**

论文摘要说在 9 robotic platforms 上做实验，强调不同 sensor/action spaces、不同机器人平台和 fine-tuning。

**和你的关系**

Octo 的故事是“通过大规模跨 embodiment 数据学通用策略”。你的故事不是要比它更 general，而是说：

- 如果目标是低数据、小模型、安全物理控制，显式 physics prompting 是另一条路线；
- 你可以把 Octo/OpenVLA 作为大规模 VLA family 的代表 baseline，而不是当前阶段必须复现的对象。

---

### 5.4 pi0: A Vision-Language-Action Flow Model for General Robot Control

链接：[arXiv](https://arxiv.org/abs/2410.24164)

**它的方法，用白话讲**

pi0 是 Physical Intelligence 的 VLA。它在预训练 VLM backbone 上加 flow matching action expert，用大规模多机器人数据训练连续动作生成。flow matching 的好处是能建模多模态、连续、长 chunk 的动作分布，比简单 regression head 表达力更强。

**它用了什么环境/任务**

论文摘要说它覆盖 single-arm、dual-arm、mobile manipulators，多种灵巧任务，比如 laundry folding、table cleaning、assembling boxes。2026 v4 摘要说它发表于 RSS 2025。

**和你的关系**

你的 direct VLA 里 regression head 和 flow head 的对照正好对应这里：

- flow action head 理论上更强；
- 但你实验里 head-only flow 崩，flow + LoRA 才起来；
- 说明 action decoder 和 fine-tuning choices 对 direct VLA 影响很大。

这可以强化你的 claim：

> For modest-scale VLM adaptation, the direct VLA route is sensitive to action decoder and fine-tuning choices, while physics-PIVOT keeps the VLM decision discrete and externalizes dynamics into an interpretable module.

---

## 6. Benchmark / Environment：别人用什么环境，你应该补什么

### 6.1 Safety Gymnasium

链接：[NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html)

**环境特点**

Safety Gymnasium 是 SafeRL benchmark，覆盖 single-agent 和 multi-agent safety-critical tasks，支持 vector input 和 vision-only input。它配套 SafePO，包含多种 SafeRL 算法。

**为什么适合你**

你的 PointHazard / PointPushHazard 很像 Safety Gymnasium 风格：

- reward + cost/safety constraint；
- hazard collision；
- navigation/pushing；
- safety-success tradeoff。

建议：

- 不一定立刻接入完整 Safety Gymnasium，但可以对齐它的指标：success、cost/hazard、timeout、return、constraint violation、min clearance。
- 如果要扩环境，优先选 Safety Gymnasium 里和你接近的 PointGoal / PointButton / PointPush / hazard 类任务。

---

### 6.2 ManiSkill3

链接：[arXiv](https://arxiv.org/abs/2410.00425)，[项目页](https://maniskill.ai/)

**环境特点**

ManiSkill3 是 GPU parallelized robotics simulator，面向 contact-rich manipulation。论文摘要强调：

- simulation + rendering 高速；
- 支持 pointcloud/voxel visual input；
- 12 个 domain；
- mobile manipulation、drawing、humanoids、dexterous manipulation；
- 提供 demonstration frames 和多种 RL / imitation baselines。

**为什么适合你**

如果你想从 2D toy 走向 CoRL，ManiSkill3 是很现实的下一步。它有接触、视觉、机器人 embodiment，但不用一开始买硬件。

建议任务：

- PushCube / PushT-like：最贴合你的 PointPush。
- PickCube：简单 manipulation，测试 VLM 是否能用 rollout 选择 approach/grasp。
- StackCube：需要更长 horizon。
- PegInsertion / PlugCharger：测试 precision。
- Drawer/OpenCabinet：测试 contact + constraints。

你的方法可以先只做 low-dimensional state + rendered top-down/third-person prompt，不必一开始处理 full RGB-D world model。

---

### 6.3 RLBench

链接：[arXiv](https://arxiv.org/abs/1909.12271)

**环境特点**

RLBench 是 CoppeliaSim/PyRep 上的机器人学习 benchmark，有 100 个手工设计任务，从 simple reaching、door opening 到 open oven and place tray 这类多阶段任务。它提供视觉、深度、segmentation、proprioception 和 motion-planned demonstrations。

**为什么适合你**

如果你要做“10+ environment / task”的论文，RLBench 很方便，因为任务多、语言指令多、视觉输入标准。但它可能工程量比 Safety Gymnasium / ManiSkill 更大。

建议：

- 先不要全量 RLBench；
- 选 5 个和物理后果有关的任务做 proof：reach target、push button、open drawer、slide block、insert/stack。

---

### 6.4 LIBERO

链接：[arXiv](https://arxiv.org/abs/2306.03310)

**环境特点**

LIBERO 是 lifelong robot learning benchmark，包含 4 个 task suites，总共 130 个任务，并提供 human teleoperated demonstrations。它关注 declarative/procedural knowledge transfer、task ordering、pretraining effects。

**为什么适合你**

LIBERO 更适合 VLA / imitation learning / lifelong learning，而不是你当前的 online physics-PIVOT 主线。但它适合做 direct VLA baseline 或少量 manipulation generalization。

建议：

- 如果你想加 VLA baseline，可以选 LIBERO 的小 subset；
- 如果主线是 physics prompting，LIBERO 不是第一优先级。

---

### 6.5 Meta-World

链接：[arXiv](https://arxiv.org/abs/1910.10897)

**环境特点**

Meta-World 是多任务/meta-RL benchmark，50 个 Sawyer manipulation tasks。

**为什么适合你**

Meta-World 任务多、轻量、标准，但语言/VLM 接口不是它的核心。它更适合做连续控制泛化，而不是 VLM prompt 方法。

建议：

- 可以用于补任务数量；
- 但不要把主要故事放在 Meta-World 上，除非你愿意做渲染和语言/视觉 prompt 包装。

---

### 6.6 VLMbench

链接：[arXiv](https://arxiv.org/abs/2206.08522)，[NeurIPS 页面](https://proceedings.neurips.cc/paper_files/paper/2022/hash/04543a88eae2683133c1acbef5a6bf77-Abstract-Datasets_and_Benchmarks.html)

**环境特点**

VLMbench 是 vision-and-language manipulation benchmark，用 AMSolver 自动生成机器人 demonstrations 和语言指令，强调 compositional language、object shapes/appearances、action types、motion constraints。

**为什么适合你**

VLMbench 很适合“VLM 相关”这个关键词，但它更偏语言组合泛化和 manipulation demonstration，而不是 physics safety。可以作为后续方向，不是最短路径。

---

### 6.7 Language Table / RoboDesk

VLMPC 用了这两个环境：

- RoboDesk：7 个 manipulation tasks，比如 buttons、slides、drawers、blocks 等；
- Language Table：50 个 simulated environments/tasks，典型任务如 move object to area。

这两个环境和 VLMPC 绑定很紧。如果你要直接对比 VLMPC，Language Table 是最自然的。但如果你的目标是安全/物理控制，Safety Gymnasium + ManiSkill 可能更贴。

---

## 7. 针对你当前环境的具体建议

### 7.1 PointHazard：保留为最干净的 mechanism test

它的作用不是炫复杂，而是证明机制：

- VLM/PIVOT 不懂惯性；
- safety filter 会保守；
- history feedback 不够；
- counterfactual rollout 是关键。

建议把 PointHazard 做成 paper 的第一组 controlled experiment，指标统一：

- success rate；
- hazard hit rate；
- timeout rate；
- return；
- final distance；
- min clearance；
- VLM calls per episode；
- parse success；
- physics prediction loss；
- predicted vs real progress correlation；
- predicted vs real clearance correlation。

### 7.2 PointPushHazard：升级成主环境之一

PointPush 比 PointHazard 更像机器人，因为它有接触 dynamics。建议重点做：

- agent hazard vs box hazard 分开统计；
- push-pose error；
- box-goal progress；
- candidate contact quality；
- 是否先绕到箱子背后；
- learned physics 在接触前后误差是否不同。

可以借 MOKA / RoboPoint / VoxPoser 的 affordance 思路：

- 画 ideal push pose；
- 画候选 contact/push direction；
- 画每个 push 的 box trajectory；
- 让 VLM 选择“站位 + 推力”而不只是 force arrow。

### 7.3 LunarReach / LunarLander：需要 macro-action PIVOT

LunarLander 卡住是非常合理的。它难在 terminal constraints：

- 落地速度不能太快；
- angle 不能太偏；
- horizontal velocity 要控制；
- fuel 不能乱用；
- 不是“离目标近”就成功。

当前 single-step action candidate 不够。应该改成：

```text
candidate = short action sequence / macro-action
examples:
  hover 8 steps
  brake vertical 12 steps
  tilt left + main thrust 8 steps
  cancel horizontal velocity 10 steps
  final gentle descent 15 steps
```

给 VLM 看的不是单步箭头，而是每个候选 macro 的 predicted descent curve：

- final x/y；
- final vx/vy；
- final angle/angular velocity；
- touchdown speed estimate；
- crash risk；
- fuel estimate。

这和 VLMPC / Traj-VLMPC 完全对齐：candidate action sequences + trajectory prediction + receding-horizon execution。

### 7.4 真实机械臂：可以做，但建议作为最后一击

不要现在就把真实机械臂当救命稻草。更好的路线：

1. 先把 simulation story 做完整；
2. 再做一个非常简单但漂亮的 real tabletop pushing；
3. 用它证明 learned local physics + visual rollout 可以从真实交互里在线适应。

推荐真实实验：

- 顶视相机；
- 桌面 puck/box；
- ArUco 或 AprilTag 做 pose tracking；
- 红色危险区贴纸；
- 目标区域；
- robot/pusher 推 box 到目标，避开危险区。

这基本就是 PointPushHazard-real，不需要一开始解决任意物体抓取。

---

## 8. 论文故事建议

### 8.1 最推荐标题

候选标题：

- Physics-PIVOT: Counterfactual Dynamics Visualization for Safe VLM Control
- Counterfactual Physics Prompting for Vision-Language Robotic Control
- Look Before You Act: Online Learned Dynamics Prompts for VLM Control

### 8.2 一句话 claim

> VLMs can choose useful actions when candidate futures are made visible; the bottleneck in physics-sensitive control is not action representation alone, but missing action-conditioned dynamics.

中文：

> VLM 做物理控制时，真正缺的不是“能不能输出动作”，而是“看不见动作后果”。把候选动作的反事实物理后果显式画出来，可以让 VLM 更安全、更可解释地做控制。

### 8.3 贡献点

可以写成三点：

1. **诊断**：通过 safe-filter、history-only、vanilla PIVOT、oracle/learned rollout 的 ablation，证明 VLM 控制失败主要来自缺少当前 action-conditioned dynamics，而不是只缺安全提示或历史反馈。
2. **方法**：提出 learned-physics PIVOT：在线学习小 dynamics model，把候选动作/动作序列的短期 rollout 渲染为 counterfactual visual prompt，让 VLM 在物理后果之间选择。
3. **验证**：在安全导航、接触推物、精密着陆/到达等任务族上，和 vanilla PIVOT、direct VLA、oracle rollout、teacher/MPC 等基线比较，展示更高 success、更低 hazard、更少 tuning。

### 8.4 主图建议

一张图讲完整故事：

```text
Current image/state
    -> candidate generator
    -> online learned physics rollout
    -> render counterfactual futures
    -> VLM chooses candidate
    -> execute one step/chunk
    -> real transition updates physics model
```

旁边放四个 prompt view：

- action arrows only；
- safe/unsafe labels；
- history feedback；
- predicted trajectories。

---

## 9. 实验矩阵建议

### 9.1 必做 baselines

| Baseline | 作用 |
|---|---|
| Vanilla PIVOT | 证明只画动作不够 |
| Safe-filter PIVOT | 证明只知道安全/不安全会保守 |
| History-only PIVOT | 证明历史反馈不能替代 counterfactual |
| Oracle rollout PIVOT | upper bound |
| Learned-physics PIVOT | 主方法 |
| Learned-physics PIVOT + LoRA/teacher | 主方法增强 |
| Direct VLA regression | same-backbone 端到端对照 |
| Direct VLA flow | 对照 pi0-style action head，但小数据下不稳 |
| Scripted expert / MPC teacher | sanity upper bound |

### 9.2 环境路线

短期最合理的 10+ task 不是 10 个完全无关环境，而是 3 个环境族，每个有多个变体：

#### A. Safe inertial navigation family

- PointHazard random hazards；
- narrow passage；
- moving hazards；
- larger mass / lower drag；
- delayed action / actuator lag；
- wind/disturbance；
- multi-goal waypoint；
- docking with speed constraint。

#### B. Contact pushing family

- PointPushHazard；
- heavier box；
- slippery box；
- narrow corridor push；
- hazard near box path；
- moving obstacle；
- two boxes / distractor box；
- real tabletop push。

#### C. Precision landing / terminal-constraint family

- LunarReach；
- LunarLander standard；
- LunarLander with wind；
- docking with velocity threshold；
- hover-and-land；
- target pad with hazard zones。

这样比随便堆 10 个 gym env 更有故事：每个环境族都测试一个物理难点。

### 9.3 评价指标

除了 success，要重点报告：

- safety violation / hazard hit；
- timeout；
- min clearance；
- final distance；
- terminal speed / terminal angle；
- number of VLM calls；
- parse/fallback rate；
- physics prediction error；
- predicted-real correlation；
- teacher agreement；
- data efficiency；
- latency。

CoRL reviewer 会关心：是不是只是 prompt trick？所以需要用这些指标证明 physics predictor 的质量和控制表现相关。

---

## 10. 最重要的融合路线

如果只选三件事做，我建议：

### 路线 1：把方法正式升级为 Trajectory/Macro-action PIVOT

对应 VLMPC / Traj-VLMPC。

现在你是：

```text
candidate = one force direction
rollout = repeat same action for H steps
```

下一版：

```text
candidate = short action sequence / macro-action
rollout = execute candidate sequence in learned dynamics
VLM sees: trajectory + key physical metrics
execute: first action or first chunk
```

这会直接解决 LunarLander 和 precision control 的瓶颈。

### 路线 2：给 PointPush 加 affordance + dynamics

对应 MOKA / RoboPoint / VoxPoser。

不要只让 VLM 选 force，改成让候选包含：

```text
push pose
contact side
push direction
duration
```

渲染：

- agent path；
- box path；
- contact point；
- predicted final box position；
- hazard clearance。

这样 PointPushHazard 会变成一个真正的 robot manipulation task，而不是看起来像 2D toy。

### 路线 3：加 uncertainty-aware physics prompting

对应 safe control / world model uncertainty。

用 ensemble dynamics model 或 dropout，给每个 candidate 估计：

- mean rollout；
- uncertainty band；
- worst-case clearance；
- disagreement score。

然后：

- 图上用虚线/透明带画不确定性；
- teacher score 惩罚 high uncertainty；
- VLM prompt 明确说不确定性高的候选更危险。

这会让 safety story 更强。

---

## 11. 推荐引用清单

### 核心必引

- [PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs](https://arxiv.org/abs/2402.07872)
- [VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation](https://arxiv.org/abs/2407.09829)
- [Vision-Language Model Predictive Control for Manipulation Planning and Trajectory Generation](https://arxiv.org/abs/2504.05225)
- [SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models](https://arxiv.org/abs/2512.05955)
- [VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models](https://arxiv.org/abs/2307.05973)
- [Physically Grounded Vision-Language Models for Robotic Manipulation](https://arxiv.org/abs/2309.02561)

### VLA 对照必引

- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)
- [pi0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)

### Visual affordance / grounding 相关

- [MOKA: Open-World Robotic Manipulation through Mark-Based Visual Prompting](https://arxiv.org/abs/2403.03174)
- [RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics](https://arxiv.org/abs/2406.10721)
- [SayCan: Do As I Can, Not As I Say](https://arxiv.org/abs/2204.01691)
- [Inner Monologue: Embodied Reasoning through Planning with Language Models](https://arxiv.org/abs/2207.05608)
- [Code as Policies](https://arxiv.org/abs/2209.07753)

### 环境 / benchmark

- [Safety Gymnasium](https://papers.nips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html)
- [ManiSkill3](https://arxiv.org/abs/2410.00425)
- [RLBench](https://arxiv.org/abs/1909.12271)
- [LIBERO](https://arxiv.org/abs/2306.03310)
- [Meta-World](https://arxiv.org/abs/1910.10897)
- [VLMbench](https://arxiv.org/abs/2206.08522)

---

## 12. 结论：你应该怎么讲这个故事

最强故事不是：

> 我做了一个 VLM + PIVOT 的小扩展。

而是：

> VLM-based control 在物理任务里失败，是因为 VLM 的输入没有包含 action-conditioned futures。我们提出一种轻量的 counterfactual physics prompting 方法：在线学习局部动力学，把每个候选动作或动作序列的未来物理后果画出来，让 VLM 在这些后果之间做选择。相比端到端 VLA，它更省数据、更可解释；相比传统 PIVOT，它能处理惯性、安全和接触；相比完整 simulation-in-the-loop，它更轻量、更适合在线适应。

如果按 CoRL 风格推进，下一步应该优先做：

1. **macro-action / trajectory PIVOT**，解决 LunarLander 这类 terminal constraint；
2. **PointPushHazard 完整化**，加 affordance/contact candidate，让它成为接触物理主实验；
3. **环境族扩展**，围绕 inertial safety、contact pushing、precision landing 三类物理难点组织 10+ tasks；
4. **uncertainty-aware learned physics**，强化 safety claim；
5. **一个简单真实桌面 pushing demo**，作为现实可行性的最后一击。

这样讲出来的论文不是“prompt 工程”，而是一个清楚的 embodied control 论点：**VLM 需要看见动作的物理后果，才能安全地行动。**
