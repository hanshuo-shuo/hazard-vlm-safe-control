# C³-Safe：本轮实验与论文推进

日期：2026-09-12。结果状态：**PILOT_ONLY / DEVELOPMENT_DIAGNOSTIC**。

本轮把“远端分支看起来已经有很高分数”推进到“知道这些分数依赖什么，能修复一个具体问题，并写出对应论文草稿”。
没有训练 SAC actor，没有把 simulator 标签写成 VLM 标签，也没有将结果升格为正式 VALIDATED。

## 已完成的工作

1. 恢复 Quest 成熟分支的准确源码和三个小型关系模型，验证全部 **15 个 protected hash**。
2. 下载并验证 **17 份历史逐场景数组**，重跑原始 KEEP/DISCARD 比较器。
3. 先冻结新的配对实验，再运行 **3 个视觉模型种子**；增加公式、无 RGB 空间先验、角色元数据等对照。
4. 第一颗种子暴露问题后，另行冻结修复诊断，再运行 **3 个相同种子的模型**。
5. 保存并核验 **6 份视觉 checkpoint、93 个 manifest 引用文件**，重算所有 arm 的 regret 和 false-safe 计数。
6. 在本机 CPU 重新加载六份 checkpoint，各重放一对 RGB；二值场与原 GPU 输出全部一致。
7. 写成英文研究草稿，包含方法、误差界、实验协议、结果、负面发现、局限和 8 篇已核对的参考文献，并生成三张图和 PDF。

论文入口：[`../paper/compositional_visual_risk/manuscript.md`](../paper/compositional_visual_risk/manuscript.md)。
数值真相源：[`../research/composition_audit_20260912/SUMMARY.json`](../research/composition_audit_20260912/SUMMARY.json)。
复现说明：[`../research/composition_audit_20260912/README.md`](../research/composition_audit_20260912/README.md)。

## 这次发现的具体难点

### 1. 模型原来能把“位置”当成“属性”

旧 decision generator 把水固定放在左侧附近，把 fragile 放在右侧附近，只做小幅随机扰动。
对颜色和纹理做 OOD，仍然保留了这一关系。原训练方法的新三种子模型在普通 appearance probe 上
平均 mIoU 达到 **0.945350**，很像已经解决了视觉任务。

这次把同一场景的两种真实属性交换后重新渲染：水可以出现在右边、fragile 可以出现在左边。
轨迹、足迹、卡片、背景渲染 seed 都不变；候选顺序随机打乱，但在一对图中使用相同打乱方式。
RGB 和 ground truth 一起改变，不是只交换评测标签。

结果是交换后的 mIoU **三颗种子均为 0**。两张图中需要改变最优动作的卡片，一对内同时选对的比例也均为 **0%**。
可视化中，原模型在交换后仍把左边预测成水、右边预测成 fragile。这支持位置依赖的判断。

注意：这是本轮捕获的最终训练 recipe 的新实验。历史最优 visual checkpoint 没有保存下来，
不能断言旧 AR03/AR05 模型在同一测试上也恰好为 0。

### 2. 已经做了一个可以复查的修复

没有增加网络、没有加 CF loss。仅把训练集 800 张中的 400 张、验证集 100 张中的 50 张，
交换真实属性并重绘。图像数量和 geometry identity 不变，模型种子、训练代码和 24 epoch 上限不变。
验证集分布也改变了，所以这项实验考察的是训练与验证数据的共同修复，不能声称只隔离了训练增强。

以下使用同一预测属性场 + 已知代价公式。每个条件有 **200 对场景、400 个需要改变决策的卡片对**。
这 400 个是 scene 内的 A/D 两张有效卡片，不是 400 个独立 scene。

| 条件 | 原模型：原图 mIoU | 原模型：交换图 mIoU | 修复：原图 mIoU | 修复：交换图 mIoU | 原模型配对正确率 | 修复配对正确率 |
|---|---:|---:|---:|---:|---:|---:|
| IID | 0.9556 | 0.0000 | 0.9441 | 0.9497 | 0.00% | 87.17% |
| Appearance families | 0.9454 | 0.0000 | 0.8327 | 0.9172 | 0.00% | 85.33% |
| Joint families | 0.9379 | 0.0000 | 0.9193 | 0.9080 | 0.08% | 90.33% |

Appearance 下，修复后三种子分别为 **87.75%、81.75%、86.50%**。
先对这三颗种子平均，再以 scene pair 为单位 bootstrap，95% 区间为 **[81.25%, 89.08%]**。
这个区间仅包含 scene 抽样不确定性；不能说它完整覆盖了新训练种子的波动。

原图/交换图各占一半时，appearance regret 从 **0.016907175** 降为 **0.001480094**，
下降约 **91.25%**；配对下降量的条件 95% 区间为 **[0.014747943, 0.016114673]**。

修复并非所有指标都改善：原图分割从 0.9454 降到 0.8327；balanced scene-macro false-safe
从 **2.3093% 升到 2.9798%**。下一步需要同时处理布局泛化与风险校准，不能只报配对正确率。

修复是在第一颗原模型结果出来后选择的，属于 **adaptive diagnostic**。
这批 probe 已经被使用，未来不应再次称为 untouched holdout。

### 3. 学习关系函数的增益目前没有被证明

旧协议的真实代价是公开的：

`g(e, κ, q) = 1 - (1 - (1-κ)*e_water) * (1 - q*e_fragile)`。

这次让 `factorized_no_cf`、`factorized_full` 和 `predicted_formula` 使用完全相同的预测暴露。
在原模型和修复模型的 appearance 主测试上，三者的 regret **逐场景相同**；主要差值及 bootstrap
区间都是 **0、[0, 0]**。CF 关系头也没有带来这一指标的改善。

这不等于证明所有学习关系模型永远无用。数值预测仍有很小的差异，其他未知相互作用也可能需要学习。
但在这个已知公式的任务中，不能把现在的提升归功于学习关系头或 CF loss。收益来自修复视觉训练分布，
公式已经足以完成当前的组合规则。

### 4. 旧 candidate generator 还需要变强

旧高分辨率 audit 要求每张卡固定偏好某个候选角色：A 偏好 fragile-heavy，C 偏好 balanced，D 偏好 water-heavy。
因此“卡片 + generator 角色”是一个过于容易的诊断基准。在原 appearance 图上，角色元数据的训练均值查表
regret 为 **0.000665**，接近视觉公式模型的 **0.000660**；交换后查表 regret 升到 **0.034416**。

这里使用角色元数据的 arm 是明确标记的 privileged diagnostic。
实际视觉模型只接收 RGB，关系头接收暴露和卡片；不能把这一发现误报成已确认的模型输入泄漏。

真实无 RGB、但允许读取动作足迹的空间均值场对照，原图 regret 是 **0.011207**，交换后是 **0.027526**。
它优于原图 cards-only 的 0.026359，但明显弱于原图视觉模型，也没有解决交换测试。

## 历史分支成绩现在应如何理解

重跑未修改的原比较器后：

| 历史三种子 confirm | Appearance mIoU | Regret | 原规则结论 |
|---|---:|---:|---|
| baseline-confirm-v2 | 0.1739 | 0.007460 | 比较基准 |
| ar03-calibrated-polarity-confirm | 0.9334 | 0.000691 | DISCARD：irrelevant-swap guard 未过 |
| ar05-runtime-margin-confirm | 0.9365 | 0.000592 | KEEP |
| ar10-cosine-schedule-confirm | 0.9409 | 0.000600 | DISCARD：irrelevant-swap guard 未过 |

所以，旧开发循环里最值得保留的是符合原规则的 AR05；不能只按 IoU 挑 AR10，也不能仅看 AR03 的明显提升。
这些 KEEP/DISCARD 只是开发选择。原分支持续使用同一开发分布，且没有导出对应视觉权重，不能直接构成正式论文证据。

本轮所用 `train.py` 哈希以 `838f49158b7c` 开头，是服务器留下的 AR13 channel-permutation recipe。
没有把它命名为 AR03 或 AR05 复现，也没有合并不存在于本地 Git 对象库中的旧 commit。

## 对论文主张的具体影响

现在已写出的草稿题目是 **Compositional Visual Risk Under Property Interventions**。
它先围绕可检验的事实组织正文：空间属性与动作暴露分解、匹配的视觉干预、同感知公式对照，以及数据修复。
这是当前证据的写法，不是对最终投稿主题或主线协议的永久改名。

可以写进研究草稿的内容：

- 对某个明确的 procedural generator，高普通分割成绩掩盖了 property/location shortcut。
- 配对视觉干预可暴露这一缺陷；训练与验证中平衡属性位置，能显著缓解它。
- 相同暴露输入下，公式在主要决策指标上匹配当前关系网络；不支持额外 CF loss 的必要性。
- 已知公式的 exposure-error 到 cost-error 的界，以及最优选择 regret 的 2δ 界；这些是代数性质，不是机器人安全定理。

现在不能写成已完成的主张：VLM 蒸馏成功、counterfactual loss 是核心创新、SAC 安全策略有效、
真实机器人安全、独立 joint-OOD 正式验证通过，或全面优于 VLM-SAFE/PROCO。

真实 teacher 的 126 次回答仍然是 **0 个属性正标注**。本轮没有用这些空标签训练六个学生，也没有继续盲目扩充调用量。
三通道/q95 的本地主线和两通道/mean 的成熟远端协议仍然分开统计。

## 下一步应按什么顺序做

1. **把 generator 从“左右两种排列”升级为属性、位置、形状和动作独立变化。**
   每个配对的有效性从实际低分辨率暴露和卡片真值算出，不能从 `candidate_id` 名字推断。
   在生成和评分前冻结新的 families、seed、主要指标、失败条件；保留公式和无 RGB 对照。
2. **做确认性比较和风险校准。**
   比较原 recipe、位置平衡 recipe；保持训练预算和监督权限一致。提前规定阈值，报告 false-safe、
   risk–coverage、配对正确率和原布局退化。加入至少五个新模型种子，避免把现有三种子区间解释过度。
3. **重新资格审查 teacher 输入与标签。**
   先做小而平衡的可辨识性样本和人工参考标注，区分渲染是否表达属性、模型是否看懂、坐标是否可靠。
   确认正标签质量与未知区域覆盖后再扩大蒸馏数据。这里真正需要人工参与的是标签审核，不是再找硬盘或重复授权 SSH。
4. **组件证据成立后再接闭环。**
   把 learned field 与 actor/critics 接通，分开 semantic/native cost，加入 success/return/latency。
   当前静态候选选择成绩不等于已经运行过安全策略。

## 运行、验证与资源

- Quest 根目录：`/projects/p33100/siosio/hazard_composition_audit_20260912`。
- Slurm arrays：`6161267`（原模型）和 `6161459`（修复）；六个任务全部 `COMPLETED / 0:0`。
- 总计约 292 GPU 秒的作业占用；PyTorch 2.5.1+cu124，A100 40/80 GB。没有据异构硬件比较速度优劣。
- 本地完整结果：`results/c3_composition_audit_20260912/`，约 285 MB；远端也保留。
- GPU 预测与 CPU 重放的最大像素差为 0.001224 以下；12 张重放图的二值场一致率均为 100%。
  跨设备浮点卷积并非逐位一致，论文指标始终来自冻结的 GPU 预测；验证没有替换原数组。
- 初次 CPU 重放用 0.001 的最大差门限，有一个模型超出到 0.001223；随后改为报告实际数值差和二值场一致性，
  不把门限调整当成科学实验成功。目标重构最大差小于 4.1e-8，所有 arm 的 regret 和 false-safe 计数独立核算通过。
- 新调用 VLM / 付费 API：**0**。旧 131 次本地模型生成仍保留原计数。

本轮没有需要用户补充的机器或存储信息。剩余工作是研究有效性与方法验证，不是服务器接入。
