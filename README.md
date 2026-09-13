# Process-Supervised RL for Mathematical Reasoning


![过程监督 RL 的目标与问题定义](docs/target.png)


## 已完成工作

- **数据管线**：GSM8K 规范化、最终答案抽取、推理步骤切分与 debug subset 构造。
- **奖励建模**：final reward、规则过程 reward、anti-hacking penalty、候选 reranking 与 Python verifier。
- **过程监督数据**：多候选生成、LLM judge、preference 数据构造、轻量 PRM 训练与诊断。
- **训练路径**：LoRA SFT、Skywork PRM 接入、warmup adapter merge、fresh LoRA gated GRPO。
- **工程质量**：数据、reward、PRM、训练入口和 CLI 均有单元测试；大模型资产与完整数据不提交 Git。


## 路线演进 核心


**1. 用规则代替 reward model 打分第一版，具体有下面四个标准。**

$$
\hat R_{\mathrm{rule}}=\mathrm{norm}\!\left(\frac{1}{T}\sum_t
  [r_{\mathrm{valid}}+r_{\mathrm{consistent}}+r_{\mathrm{progress}}-0.5r_{\mathrm{hack}}]\right).
$$

其中 `validity` 看数字和运算，`consistency` 查变量赋值冲突，`progress` 看是否引入新的数值/推导，`anti-hacking` 查重复、空泛 filler、提示词或代码污染。

问题：第一，换行只是文本格式：一行可能把两步推理也可能两行是一步推理过程，所以它不是可标注的语义步骤。第二，步骤和步骤之间有关系 ，规则会奖励它容易数到的表面特征，而非真正的推理质量。

在 100 题 × 4 候选的同一批轨迹上做 reranking。`final-only` 与 `final + rule-process` 的 top-1 都是 **93%**，但出现了

$$
\mathrm{corr}(\mathrm{num\_steps},\ \mathrm{process\_reward})=-0.8478.
$$

也就是说，过程分与推理长度强烈负相关，明显偏爱短轨迹；50/100 道题虽换了选择，却没有带来准确率增益。个人认为很可能是规则设置问题，例如长步骤在后续的新变量出现次数少。

**2. 考虑到推理链条有很强的上下文联系，先让 LLM 从全局视角评一次，类似 LLM judge。** 改为将同一题的 4 条完整候选连同题目和参考答案交给强 LLM judge。它一次看到整条推理，能比较跳步、前后矛盾、无关展开和最终结论之间的关系；在 100 道题上形成 **324 个** `chosen/rejected` preference pairs。
目的是用这些偏好数据训练一个打分的小模型，**但后续实验效果明显不理想**。

实际诊断也验证了这一点：轻量模型在这 324 个训练 pair 上的 accuracy 达到 **0.9722**，但在候选选择评估中，它与原始 LLM judge 选中结果的一致率只有约 **0.71**。
所以后续直接接入已经在大规模过程监督数据上预训练的 Skywork PRM，

**3. 接入预训练 Skywork PRM（1.5B），但仍只评整条 CoT。** 为避免用 324 个 pair 从零训练过程模型，后续改用预训练的 Skywork PRM。当前用法是把“题目 + 完整 CoT”整体输入模型，得到一个 `raw_prm_score`；GRPO 对每条轨迹只接收这一个标量过程分。不做语义步骤切分，也不对单独某一步打分或更新。随后在 MATH L3/4 warmup 后进行 GRPO。

关于这里其实我们的就不算严格意义的一个PRM了相当于给最后的奖励得分做了不同维度的一个加权，现在我们的目标就变成了跑通后训练的基本流程尽量找一些前言论文的方法做些消融实验

训练阶段没有只跑一种 GRPO 配置，而是参考 DAPO 等工作中关于 clipping、loss 与采样的思路，在固定模型、warmup、reward 和评测设置下做了三组训练策略探索：Clip-Higher GRPO、DAPO loss、以及 Dynamic Sampling GRPO。MATH500 前 100 题的单 seed quick check 分别为 `39% / 39% / 49%`；它们用于判断哪种训练策略值得继续，而不是宣称复现了 DAPO/GSPO，或证明过程奖励本身带来因果提升。

这里最大的教训是：轻量 LoRA 很适合 SFT warmup，因为 SFT 主要做格式、风格和领域冷启动；但 GRPO 需要在采样噪声和 reward 信号下真正改变策略。早期 small-rank、约 200-step 的 RL 只能算 smoke test，容量和训练强度都不足。因此先将 SFT adapter merge 到基座，再挂 fresh `r=512`、`alpha=1024` 的 RL LoRA，并将训练拉到 1000 steps；步数太少也会削弱 RL 更新的效果。

最终答案仍是不可绕过的验证门；过程总分只在答对候选之间影响 reward。这是当前项目对“既要利用过程信号、又要抑制 reward hacking”的实际折中，而不是声称已经解决严格的逐步标注问题。

相关阶段记录：[step1：工程与规则原型](docs/history/README_process_supervised_rl_step1.md) · [step2：reranking 与规则偏差](docs/history/README_process_supervised_rl_step2.md) · [step3：LLM judge 与轻量偏好模型](docs/history/README_process_supervised_rl_step3.md) · [step4：Skywork PRM 与 gated GRPO](docs/history/README_process_supervised_rl_step4.md)

## 探索性结果

![MATH500 exploratory training-strategy ablation](docs/figures/math500_exploratory_ablation.svg)

三组均使用相同的模型、MATH L3/4 warmup、gated PRM reward、MATH500 first-100、greedy decoding 和 seed 42；Dynamic Sampling GRPO 的准确率为 49/100。完整指标与来源见 [结果表](docs/evidence/math500_exploratory_ablation.csv)。

这是 single-seed 的训练策略探索，**不是正式 benchmark，也不能证明过程奖励单独带来提升**：目前尚未归档同设定的 `final-only GRPO` 控制组。

训练日志也保留了 Dynamic-Sampling run 的 loss、reward 和最终正确率；由于其在 step 500 后以 weights-only 方式恢复，图中将两段训练分开呈现。

![Dynamic-Sampling GRPO training dynamics](docs/figures/exp4_dynamic_sampling_training_dynamics.svg)

早期 GSM8K reranking 诊断则覆盖 100 题、400 条候选：final-only 与 final + process 的 top-1 准确率均为 93%，后者改变了 50 个选择且没有 `1→0` 退化；同时发现过程分与步骤数存在较强负相关，因而没有直接将其当作训练结论。[查看原始报告](docs/evidence/gsm8k_reranking_100_report.md)

## 快速开始

```bash
# 轻量工程验证
pytest -q

# 准备 GSM8K
python scripts/prepare_gsm8k.py \
  --input data/raw/train.jsonl \
  --output data/processed/gsm8k_train.jsonl \
  --split train

# 准备 MATH L3/4 训练子集
python scripts/prepare_math_l34.py \
  --output data/processed/math_l34_train_3000_seed42.jsonl \
  --limit 3000 \
  --seed 42
```

完整 SFT、PRM、GRPO 命令与模型加载要求见 [复现流程](docs/repro/current_pipeline.md)。评测 GRPO 时必须加载：

```text
base + merged warmup SFT adapter + r512 GRPO adapter
```

## 项目结构

```text
src/psrl/     data、reward、PRM、GRPO 与评测核心逻辑
scripts/      数据准备、候选生成、judge、PRM、SFT、GRPO、benchmark 入口
configs/      数据、reward、训练与评测配置
tests/        单元测试与 CLI smoke tests
docs/         设计、阶段记录与复现说明
```

更多背景与实验决策：
[设计文档](docs/superpowers/specs/2026-04-19-process-supervised-rl-design.md) ·
[阶段记录](docs/history/README_process_supervised_rl_step4.md) ·
[复现流程](docs/repro/current_pipeline.md)

## Next

- 完成 final-only GRPO vs. gated PRM-GRPO 的同设定、多 seed 对照；
- 在 MATH500 full 上评测，并归档曲线、逐题预测和可复用结果表。
