# Process-Supervised RL on GSM8K

This repository is a compact experimental pipeline for process-supervised reasoning on `GSM8K`.

The current goal is not to keep scaling GRPO blindly. The current goal is to keep a strong SFT anchor, diagnose which reward sources add new signal, and then use the useful reward signal in GRPO.

## Current Status

- Stage: phase 1, after `step3`
- Dataset: `GSM8K`
- Current policy anchor: `SFT v3 spacefix`
- Current RL baseline: `GRPO v3 cfg3`
- Current engineering focus: Python verifier step reward, DAPO Clip-Higher GRPO, and reward-source diagnostics
- Execution workflow: code is committed locally/GitHub; data, model artifacts, and long runs live on the remote machine

Key benchmark snapshot on the fixed 256-sample GSM8K test subset:

| Model | Accuracy |
| --- | ---: |
| Base 7B | `150/256 = 0.5859` |
| Old PRM-filtered SFT | `166/256 = 0.6484` |
| Clean chat SFT v2 | `175/256 = 0.6836` |
| SFT v3 spacefix | `187/256 = 0.7305` |
| GRPO v3 cfg3 | `190/256 = 0.7422` |

The small GRPO gain suggests that the previous PRM reward source has limited marginal information after PRM-filtered SFT. The next experiments focus on whether a step-level Python verifier or a stronger PPM/RM provides a better RL reward signal.

## Core Pipeline

Read the current reproducible pipeline first:

- [docs/repro/current_pipeline.md](docs/repro/current_pipeline.md)

Main flow:

1. Prepare GSM8K JSONL files.
2. Generate multiple candidate trajectories per question.
3. Judge/rank candidates with final correctness and optional LLM judge.
4. Build preference data for PRM/RM diagnostics.
5. Train clean SFT LoRA from filtered trajectories.
6. Run GRPO from the SFT adapter.
7. Diagnose reward sources before giving them high training weight.

## Important Code Paths

| Area | Path |
| --- | --- |
| GSM8K preprocessing | `scripts/prepare_gsm8k.py`, `src/psrl/data/` |
| Candidate generation | `scripts/generate_candidates.py`, `src/psrl/candidates.py` |
| LLM judge preferences | `scripts/judge_candidates_with_llm.py`, `scripts/build_prm_dataset.py` |
| SFT data filtering | `scripts/build_prm_filtered_sft_data.py`, `src/psrl/sft_data.py` |
| SFT LoRA training | `scripts/train_sft_lora_chat.py` |
| GRPO training | `scripts/train_grpo_smoke.py`, `src/psrl/rl/` |
| DAPO Clip-Higher adapter | `src/psrl/rl/dapo_grpo_trainer.py` |
| GRPO reward composition | `src/psrl/rl/grpo_rewards.py` |
| Python verifier reward | `src/psrl/reward/python_verifier.py`, `scripts/diagnose_python_verifier.py` |
| External Qwen PRM diagnostic | `scripts/evaluate_qwen_prm_benchmark.py` |
| MLP PRM v2 | `scripts/train_prm_v2.py`, `src/psrl/prm_v2.py` |

## Current Reward Direction

The current recommended sequence is:

1. Diagnose the Python verifier reward offline on fixed SFT/GRPO benchmark outputs.
2. If the verifier creates a clear correct-vs-wrong gap and useful within-question ranking, connect it to GRPO with conservative weight.
3. If this works, train a stronger process preference model, likely using PRM800K-style step labels instead of spending extra effort recreating step labels from scratch.
4. Keep external RM/PRM models as offline diagnostics unless they prove strong alignment with final-answer improvements.

Avoid using the same PRM as both the main SFT filter and the main RL reward without checking marginal signal. That design can make RL gains look small because SFT has already absorbed the PRM preference.

## GRPO Clip-Higher

The project now exposes DAPO-style non-symmetric clipping through the training script:

```bash
python scripts/train_grpo_smoke.py \
  --epsilon-low 0.2 \
  --epsilon-high 0.28
```

Backward compatibility is preserved:

```bash
python scripts/train_grpo_smoke.py --epsilon 0.2
```

The actual installed TRL loss implementation lives in the remote Python environment, but this repository owns the adapter layer in `src/psrl/rl/dapo_grpo_trainer.py`. Future GRPO loss overrides should go there instead of editing `site-packages`.

## Historical Notes

Long-form step documents are archived under `docs/history/`:

- [step1 bootstrap](docs/history/README_process_supervised_rl_step1.md)
- [step2 process reward baseline](docs/history/README_process_supervised_rl_step2.md)
- [step3 PRM/SFT/GRPO diagnostics](docs/history/README_process_supervised_rl_step3.md)

These files preserve the experiment history. New contributors should start from this README and `docs/repro/current_pipeline.md`, not from the archived notes.

## Repository Hygiene

- Keep code, configs, tests, and small Markdown summaries in Git.
- Keep large data, model checkpoints, adapter outputs, and full JSONL logs outside Git.
- Use the remote machine for GPU training and benchmark execution.
- Use local/GitHub as the source of truth for code.
