# 当前复现流程

这份文档保留当前仍然重要的短版复现说明。历史探索、废弃路线和阶段性判断已经归档到 `docs/history/`。

最终项目主线是：

```text
deepseek-math-7b-instruct
  + SFT v3 / MATH L3-4 warmup adapter
  -> 将 warmup adapter merge 到模型权重
  + fresh r512 LoRA adapter
  -> 使用 Skywork PRM gated reward 做 GRPO
```

早期 GSM8K 规则 reward、LLM judge、PRM 和 SFT 工作仍然是重要的项目历史和工程底座，但最终模型路线已经收敛到 MATH L3/4 warmup + gated PRM GRPO。

## 1. 数据准备

GSM8K 预处理脚本仍可用于早期实验复现：

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

最终 GRPO 路线使用 MATH L3/4 数据：

```bash
python scripts/prepare_math_l34.py \
  --output data/processed/math_l34_train_3000_seed42.jsonl \
  --limit 3000 \
  --seed 42
```

原始数据和处理后的数据都不提交到 Git。

## 2. 候选轨迹、LLM Judge 与 PRM 诊断

早期过程监督路线会为每道题生成多条候选轨迹，用强 LLM 做过程质量评审，再把结果转成 preference pairs：

```bash
python scripts/generate_candidates.py \
  --input data/processed/gsm8k_train.jsonl \
  --output logs/candidates/gsm8k_train_candidates.jsonl \
  --num-candidates 4

python scripts/judge_candidates_with_llm.py \
  --input logs/candidates/gsm8k_train_candidates.jsonl \
  --output logs/llm_judge/gsm8k_train_candidates_judged.jsonl

python scripts/build_prm_dataset.py \
  --candidates logs/candidates/gsm8k_train_candidates.jsonl \
  --judgements logs/llm_judge/gsm8k_train_candidates_judged.jsonl \
  --output data/prm/gsm8k_train_prm_preferences.jsonl
```

仓库中保留了轻量 PRM baseline，用于 reranking 和 reward signal 诊断：

```bash
python scripts/train_prm_v2.py \
  --preferences data/prm/gsm8k_train_prm_preferences.jsonl \
  --candidates logs/candidates/gsm8k_train_candidates.jsonl \
  --output-dir logs/prm_v2/run
```

这些部分主要用于复现项目推理过程和工程路径，不是最终 GRPO 使用的 reward source。

## 3. 最终 MATH GRPO 路线

最终训练路线会先加载 warmup SFT adapter，将其 merge 到 base model，再挂 fresh r512 LoRA adapter，并使用 Skywork PRM gated reward 做 GRPO：

```bash
python scripts/train_grpo_smoke.py \
  --train-jsonl data/processed/math_l34_train_3000_seed42.jsonl \
  --model-name /root/autodl-tmp/models/deepseek-math-7b-instruct \
  --sft-adapter /root/autodl-tmp/psrl_outputs/sft_v3_math_l34_3000_e0p4_lr1e5_len2048/final \
  --prm-dir /root/autodl-tmp/models/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
  --prm-backend skywork \
  --output-dir /root/autodl-tmp/psrl_outputs/grpo_math_l34_r512_gated_len768_1000_s42 \
  --limit 3000 \
  --num-generations 4 \
  --generation-batch-size 4 \
  --reward-mode gated_prm \
  --wrong-final-reward 0.0 \
  --final-weight 1.0 \
  --prm-weight 0.2 \
  --max-prompt-length 512 \
  --max-completion-length 768 \
  --learning-rate 1e-6 \
  --beta 0.04 \
  --loss-type grpo \
  --epsilon-low 0.2 \
  --epsilon-high 0.2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing \
  --max-steps 1000 \
  --save-steps 500 \
  --save-total-limit 1 \
  --save-only-model \
  --delete-saved-ref-adapter \
  --merge-sft-adapter \
  --new-lora-r 512 \
  --new-lora-alpha 1024 \
  --new-lora-dropout 0.05 \
  --new-lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --seed 42 \
  --device cuda
```

评测时最重要的加载规则是：

```text
base + warmup SFT adapter merge + r512 GRPO adapter
```

不要评测成 `base + r512 GRPO adapter`。那样会漏掉 warmup adapter，测到的是错误模型。

## 4. 评测说明

方向性 quick check 使用 MATH500 前 40 题：

```text
SFT warmup: 16/40 = 40%
GRPO r512 gated: 22/40 = 55%
```

这不是正式 benchmark 结论。正式评测建议使用：

```yaml
dataset: HuggingFaceH4/MATH-500
subset: first 100 examples or full 500
decode: greedy
max_gen_toks: 768
verifier: math_verify
```

## 5. Git 归档策略

保留在 Git 中：

```text
configs/
scripts/
src/
tests/
docs/
小型 Markdown 结果摘要
adapter checksum
```

不保留在 Git 中：

```text
完整 candidate JSONL
完整 benchmark JSONL
原始或处理后数据集
model checkpoint
LoRA adapter 二进制文件
大型 reward-debug 日志
远端机器专用输出
```

warmup adapter 的 checksum 保存在：

```text
artifacts/sft_v3_math_l34_3000_e0p4_lr1e5_len2048.tar.gz.sha256
```

adapter 压缩包本身被 `.gitignore` 排除。如果需要共享 adapter，建议通过外部 artifact 渠道发布，并在这里保留 checksum。
