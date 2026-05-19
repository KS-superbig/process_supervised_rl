# Current Pipeline

This document is the short reproducibility guide for the active GSM8K process-supervised RL pipeline.

## 1. Data

Expected processed files:

```text
data/processed/gsm8k_train.jsonl
data/processed/gsm8k_test.jsonl
```

Build them from raw GSM8K JSONL files with:

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

## 2. Candidate Generation

Generate multiple trajectories per question:

```bash
python scripts/generate_candidates.py \
  --input data/processed/gsm8k_train.jsonl \
  --output logs/candidates/gsm8k_train_candidates.jsonl \
  --num-candidates 4
```

For the current stage, candidate files and full logs should live on the remote machine and should not be committed to Git.

## 3. Preference Construction

Use final-answer correctness and optional LLM judge ranking to construct preferences:

```bash
python scripts/judge_candidates_with_llm.py \
  --input logs/candidates/gsm8k_train_candidates.jsonl \
  --output logs/llm_judge/gsm8k_train_candidates_judged.jsonl

python scripts/build_prm_dataset.py \
  --candidates logs/candidates/gsm8k_train_candidates.jsonl \
  --judgements logs/llm_judge/gsm8k_train_candidates_judged.jsonl \
  --output data/prm/gsm8k_train_prm_preferences.jsonl
```

## 4. PRM v2 Diagnostics

The current in-repo PRM baseline is the MLP pairwise model:

```bash
python scripts/train_prm_v2.py \
  --preferences data/prm/gsm8k_train_prm_preferences.jsonl \
  --candidates logs/candidates/gsm8k_train_candidates.jsonl \
  --output-dir logs/prm_v2/run
```

Use PRM v2 mainly for candidate filtering, reranking, and reward smoke tests. Do not assume it is a strong standalone RL reward without offline correlation checks.

## 5. SFT

Build filtered SFT rows, then train the chat LoRA adapter:

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

The current best SFT anchor is the `SFT v3 spacefix` adapter. The important fix is preserving assistant target spaces for DeepSeek-style tokenizers that otherwise drop normal ASCII spaces in labels.

## 6. GRPO

Run GRPO from the SFT adapter:

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

`--epsilon-low` and `--epsilon-high` expose DAPO Clip-Higher. Passing legacy `--epsilon` keeps symmetric clipping.

Project-owned GRPO extension point:

```text
src/psrl/rl/dapo_grpo_trainer.py
```

Reward composition:

```text
src/psrl/rl/grpo_rewards.py
```

## 7. Python Verifier Reward

Diagnose before training:

```bash
python scripts/diagnose_python_verifier.py \
  logs/benchmark_256/sft_v3_predictions.jsonl \
  logs/python_verifier/sft_v3_summary.json \
  --scored-output logs/python_verifier/sft_v3_details.jsonl
```

Useful signal should show a meaningful gap between final-correct and final-wrong outputs, and should help rank mixed candidates within the same question.

## 8. External PRM/RM Diagnostics

External reward models such as Qwen Math PRM should be evaluated offline first:

```bash
python scripts/evaluate_qwen_prm_benchmark.py \
  --model-path /root/autodl-tmp/models/Qwen2.5-Math-PRM-7B \
  --input sft_v3=logs/benchmark_256/sft_v3_predictions.jsonl \
  --output-dir logs/qwen_prm_eval
```

If the signal is weak or fights final-answer improvements, use it only as a small auxiliary term or keep it as a diagnostic.

## 9. What To Keep In Git

Keep:

```text
configs/
scripts/
src/
tests/
docs/
small Markdown summaries
```

Do not keep:

```text
full candidate JSONL files
full benchmark JSONL files
model checkpoints
LoRA adapters
large generated logs
remote-only datasets
```
