# 当前复现流程

这个文档是当前 `GSM8K` 过程监督 RL 主线的短版复现说明。它只覆盖当前仍然重要的流程，不试图复述所有历史实验。

## 1. 数据准备

预期的处理后数据文件：

```text
data/processed/gsm8k_train.jsonl
data/processed/gsm8k_test.jsonl
```

从原始 GSM8K JSONL 构建：

```bash
python scripts/prepare_gsm8k.py \
  --input data/raw/train.jsonl \
  --output data/processed/gsm8k_train.jsonl \
  --split train

python scripts/prepare_gsm8k.py \
  --input data/raw/test.jsonl \
  --output data/processed/gsm8k_test.jsonl \
  --split test
```

## 2. 候选轨迹生成

为每道题生成多条候选推理轨迹：

```bash
python scripts/generate_candidates.py \
  --input data/processed/gsm8k_train.jsonl \
  --output logs/candidates/gsm8k_train_candidates.jsonl \
  --num-candidates 4
```

当前阶段，候选文件和完整日志应放在远端机器，不提交到 Git。

## 3. preference 构造

使用 final-answer correctness 和可选 LLM judge 排名构造 preference 数据：

```bash
python scripts/judge_candidates_with_llm.py \
  --input logs/candidates/gsm8k_train_candidates.jsonl \
  --output logs/llm_judge/gsm8k_train_candidates_judged.jsonl

python scripts/build_prm_dataset.py \
  --candidates logs/candidates/gsm8k_train_candidates.jsonl \
  --judgements logs/llm_judge/gsm8k_train_candidates_judged.jsonl \
  --output data/prm/gsm8k_train_prm_preferences.jsonl
```

## 4. PRM v2 诊断

当前仓库内的 PRM baseline 是 MLP pairwise model：

```bash
python scripts/train_prm_v2.py \
  --preferences data/prm/gsm8k_train_prm_preferences.jsonl \
  --candidates logs/candidates/gsm8k_train_candidates.jsonl \
  --output-dir logs/prm_v2/run
```

PRM v2 主要用于候选过滤、reranking 和 reward smoke test。不要在没有离线相关性诊断的情况下，把它当成强 RL reward。

## 5. SFT

先构建过滤后的 SFT 数据，再训练 chat LoRA adapter：

```bash
python scripts/build_prm_filtered_sft_data.py \
  --scored-candidates logs/prm_v2/run/scored_candidates.jsonl \
  --output data/sft/prm_filtered_sft.jsonl \
  --stats-json data/sft/prm_filtered_sft_stats.json \
  --strict-format

python scripts/train_sft_lora_chat.py \
  --train-jsonl data/sft/prm_filtered_sft.jsonl \
  --model-name /root/autodl-tmp/models/deepseek-math-7b-base \
  --output-dir logs/sft/prm_filtered_lora
```

当前最佳 SFT anchor 是 `SFT v3 spacefix` adapter。关键修复是：对会吞普通 ASCII 空格的 DeepSeek tokenizer，在 assistant target 中保留可训练的 space marker，避免模型学出无空格输出。

## 6. GRPO

从 SFT adapter 初始化并运行 GRPO：

```bash
python scripts/train_grpo_smoke.py \
  --train-jsonl data/processed/gsm8k_train.jsonl \
  --model-name /root/autodl-tmp/models/deepseek-math-7b-base \
  --sft-adapter logs/sft/prm_filtered_lora_1000x4_v3_strict_chat_spacefix/final \
  --prm-dir logs/prm_v2/run \
  --output-dir logs/rl/grpo_run \
  --epsilon-low 0.2 \
  --epsilon-high 0.28
```

`--epsilon-low` 和 `--epsilon-high` 暴露 DAPO Clip-Higher。传入旧参数 `--epsilon` 时，会退化为对称 clipping。

项目内 GRPO 扩展入口：

```text
src/psrl/rl/dapo_grpo_trainer.py
```

reward 组合逻辑：

```text
src/psrl/rl/grpo_rewards.py
```

## 7. Python verifier reward

先做离线诊断，再决定是否接入训练：

```bash
python scripts/diagnose_python_verifier.py \
  logs/benchmark_256/sft_v3_predictions.jsonl \
  logs/python_verifier/sft_v3_summary.json \
  --scored-output logs/python_verifier/sft_v3_details.jsonl
```

有效信号应当满足两点：

- final correct 和 final wrong 输出之间有明显 verifier reward gap。
- 在同一道题的 mixed candidates 中，verifier 能帮助区分好坏轨迹。

## 8. 外部 PRM/RM 诊断

Qwen Math PRM 这类外部 reward model 应先离线评估：

```bash
python scripts/evaluate_qwen_prm_benchmark.py \
  --model-path /root/autodl-tmp/models/Qwen2.5-Math-PRM-7B \
  --input sft_v3=logs/benchmark_256/sft_v3_predictions.jsonl \
  --output-dir logs/qwen_prm_eval
```

如果外部 reward 信号偏弱，或者和 final-answer improvement 打架，就只把它当作小权重辅助项或诊断工具。

## 9. Git 中应该保留什么

保留：

```text
configs/
scripts/
src/
tests/
docs/
小型 Markdown summaries
```

不要保留：

```text
完整 candidate JSONL
完整 benchmark JSONL
model checkpoint
LoRA adapter
大型生成日志
远端专用数据集
```
