# Remote asset manifest

Downloaded on 2026-09-11 from the project training instance. Large model assets remain local-only and are excluded from Git.

| Asset | Local location | Integrity / size |
| --- | --- | --- |
| GSM8K processed train split | `data/processed/gsm8k_train.jsonl` | 7,473 JSONL records |
| GSM8K processed test split | `data/processed/gsm8k_test.jsonl` | 1,319 JSONL records |
| MATH L3/4 SFT warmup LoRA | `artifacts/local/sft_v3_math_l34_3000_e0p4_lr1e5_len2048_final/` | `adapter_model.safetensors` SHA256: `b8115fa1c0e3d8e602441151c6201d752bab0337930991be14e8f734b1af267c` |
| Early candidate-selection diagnostic | `docs/evidence/gsm8k_reranking_100_report.md` | 100 samples / 400 candidates |

The remote instance did not contain GRPO outputs, MATH500 predictions, TensorBoard/W&B curves, or a strict final-only GRPO control run.
