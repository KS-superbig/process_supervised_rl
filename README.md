# GSM8K 过程监督强化学习实验

这个仓库是一个面向 `GSM8K` 的过程监督推理实验仓库，目标是把核心流程和核心训练代码讲清楚，而不是保留所有探索期的细枝末节。

当前重点不是盲目继续扩大 GRPO，而是先固定一个强 SFT anchor，诊断哪些 reward source 真正提供新增信息，再把有效 reward 接入 GRPO。

## 当前状态

- 当前阶段：第一阶段，`step3` 之后
- 当前数据集：`GSM8K`
- 当前 policy anchor：`SFT v3 spacefix`
- 当前 RL baseline：`GRPO v3 cfg3`
- 当前工程重点：Python verifier step reward、DAPO Clip-Higher GRPO、reward source 诊断
- 当前工作流：代码保存在本地/GitHub；数据、模型产物、长时间训练放在远端机器

固定 `256` 条 GSM8K test 子集上的主要结果：

| 模型 | 准确率 |
| --- | ---: |
| Base 7B | `150/256 = 0.5859` |
| old PRM-filtered SFT | `166/256 = 0.6484` |
| clean chat SFT v2 | `175/256 = 0.6836` |
| SFT v3 spacefix | `187/256 = 0.7305` |
| GRPO v3 cfg3 | `190/256 = 0.7422` |

当前 GRPO 只有小幅提升，说明旧 PRM reward 在 PRM-filtered SFT 之后的边际信息有限。下一步重点是验证 Python verifier 这种 step-level reward，或者更强的 PPM/RM，是否能提供更好的 RL reward signal。

## 核心流程

新合作者建议先看当前复现流程：

- [docs/repro/current_pipeline.md](docs/repro/current_pipeline.md)

主线流程：

1. 准备 GSM8K JSONL 数据。
2. 为每道题生成多条候选推理轨迹。
3. 用 final correctness 和可选 LLM judge 对候选排序/打标。
4. 构造 PRM/RM 诊断用 preference 数据。
5. 从过滤后的候选轨迹训练干净的 SFT LoRA。
6. 从 SFT adapter 初始化并运行 GRPO。
7. 在高权重接入训练前，先离线诊断 reward source 是否真的有效。

## 关键代码路径

| 模块 | 路径 |
| --- | --- |
| GSM8K 预处理 | `scripts/prepare_gsm8k.py`, `src/psrl/data/` |
| 候选生成 | `scripts/generate_candidates.py`, `src/psrl/candidates.py` |
| LLM judge preference | `scripts/judge_candidates_with_llm.py`, `scripts/build_prm_dataset.py` |
| SFT 数据过滤 | `scripts/build_prm_filtered_sft_data.py`, `src/psrl/sft_data.py` |
| SFT LoRA 训练 | `scripts/train_sft_lora_chat.py` |
| GRPO 训练 | `scripts/train_grpo_smoke.py`, `src/psrl/rl/` |
| DAPO Clip-Higher adapter | `src/psrl/rl/dapo_grpo_trainer.py` |
| GRPO reward 组合 | `src/psrl/rl/grpo_rewards.py` |
| Python verifier reward | `src/psrl/reward/python_verifier.py`, `scripts/diagnose_python_verifier.py` |
| 外部 Qwen PRM 诊断 | `scripts/evaluate_qwen_prm_benchmark.py` |
| MLP PRM v2 | `scripts/train_prm_v2.py`, `src/psrl/prm_v2.py` |

## 当前 reward 路线

推荐顺序：

1. 在固定 SFT/GRPO benchmark 输出上离线诊断 Python verifier reward。
2. 如果 verifier 在 correct/wrong 之间有明显 gap，并且能在同题多候选里区分好坏，再用保守权重接入 GRPO。
3. 如果这条路线 work，再训练更强的 process preference model，优先考虑直接使用 PRM800K 这类 step label 数据，避免重复造 step label。
4. 外部 RM/PRM 先作为离线诊断工具。只有当它和 final-answer improvement 明显一致时，才考虑接入训练。

注意：不要在没有诊断边际信息的情况下，让同一个 PRM 同时承担 SFT 主过滤器和 RL 主 reward。这样 SFT 阶段可能已经吸收了大部分 PRM 偏好，导致 RL 阶段提升很小。

## GRPO Clip-Higher

训练脚本已经暴露 DAPO 风格的非对称 clipping：

```bash
python scripts/train_grpo_smoke.py \
  --epsilon-low 0.2 \
  --epsilon-high 0.28
```

兼容旧的对称 clip 写法：

```bash
python scripts/train_grpo_smoke.py --epsilon 0.2
```

真正的 TRL loss 实现在远端 Python 环境的安装包里；但本仓库已经拥有自己的 adapter 层：`src/psrl/rl/dapo_grpo_trainer.py`。后续如果要 override GRPO loss、clip、token-level mask 或日志指标，优先在这个文件里接管，不要直接改远端 `site-packages`。

## 历史文档

探索期长文档已经归档到 `docs/history/`：

- [step1 bootstrap](docs/history/README_process_supervised_rl_step1.md)
- [step2 process reward baseline](docs/history/README_process_supervised_rl_step2.md)
- [step3 PRM/SFT/GRPO diagnostics](docs/history/README_process_supervised_rl_step3.md)

这些文档用于保留实验历史。新合作者应优先阅读当前 `README.md` 和 [docs/repro/current_pipeline.md](docs/repro/current_pipeline.md)，不要从历史长文开始。

## 仓库卫生规则

- Git 中保留：代码、配置、测试、小型 Markdown summary。
- Git 中不保留：大数据、模型 checkpoint、LoRA adapter、完整 JSONL 日志。
- 远端机器用于 GPU 训练和 benchmark 执行。
- 本地/GitHub 是代码源，远端只同步代码并执行实验。
