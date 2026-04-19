# Process-Supervised RL Minimal System Design

## Goal

Build a professional but lightweight research codebase for a first-stage experiment on process-supervised reasoning.

The first milestone is not full online RL. The first milestone is a reproducible minimal closed loop that can compare:

- `final-only reward`
- `final + process reward`

The project should support local development, GitHub-based collaboration, and remote GPU execution on either a lab server or a rented cloud machine.

## Decisions Already Fixed

- Task domain: mathematical reasoning
- Initial dataset: `GSM8K`
- Initial model size: small open model in the `1.5B-7B` range
- Reward design: rule-based process reward
- Training path: `SFT + rerank / sample filtering / reward-weighted training`
- Infrastructure assumption: lab server first, cloud as backup

## Non-Goals For Phase 1

- No full-scale PPO/GRPO pipeline in the first milestone
- No very large model training
- No end-to-end learned reward model in the first milestone
- No giant multi-dataset ingestion system before the first baseline works

## Success Criteria

Phase 1 is considered successful if the team can:

1. Load and normalize `GSM8K`
2. Generate or ingest step-by-step solutions
3. Split solutions into steps deterministically
4. Compute final-answer reward
5. Compute rule-based process reward
6. Aggregate rewards at sample level
7. Run a small training or reward-weighted fine-tuning experiment
8. Produce a comparison report between `final-only` and `final+process`

## Recommended Hardware

### Minimum Practical Remote Setup

- `1 x 24GB` GPU
- `64GB` RAM
- `16-32` CPU cores
- `500GB-1TB` disk

This is enough for:

- local-scale experimentation with `1.5B-7B` models
- LoRA or QLoRA style fine-tuning
- debug-sized and moderate-sized processed GSM8K workflows

### More Comfortable Setup

- `2-4` GPUs with `40GB-80GB` VRAM each
- `128GB+` RAM
- `1TB+` disk

This is not required for phase 1, but it makes repeated ablations much easier.

## Infrastructure Model

### Local Machine Responsibilities

- write code
- edit configs
- run unit tests
- run tiny debug subsets
- inspect a few processed samples
- prepare Git commits and PRs

### GitHub Responsibilities

- host code
- host configs
- host experiment notes
- host small example data only

### Remote Server Responsibilities

- store full datasets
- cache processed artifacts
- build training environment
- run training jobs
- save checkpoints and logs

### Data Placement Rule

Do not keep the full training dataset on the local machine by default.

Recommended split:

- local: tiny debug subset only
- remote: full raw data, processed data, training outputs
- optional object storage: checkpoints and long-term backup

## Data Strategy

### Dataset Choice

Start with `GSM8K` only.

Reasons:

- clean and widely used
- fast to parse and debug
- enough to validate the first process-reward pipeline
- lower engineering risk than starting with more complex process datasets

### Unified Sample Schema

All processed samples should follow one schema:

```json
{
  "id": "gsm8k-train-000001",
  "source": "gsm8k",
  "split": "train",
  "question": "...",
  "answer_final": "...",
  "answer_final_normalized": "...",
  "solution_raw": "...",
  "steps": ["...", "..."],
  "metadata": {}
}
```

### Data Cleaning Scope

Cleaning does not need to be complex in phase 1. It needs to be deterministic and reproducible.

Required cleaning steps:

1. remove empty or malformed rows
2. normalize question and answer fields
3. parse final answer into a normalized comparable form
4. split raw solution into steps
5. filter obviously broken or extreme-length samples
6. assign fixed `train/dev/test` split
7. export a tiny debug subset

### Where Cleaning Runs

Two-stage cleaning is recommended:

1. local dry-run on a tiny sample
2. remote full run on the complete dataset

This means:

- the cleaning code lives in the repository
- the same script can run both locally and remotely
- local execution validates logic
- remote execution handles the full dataset size

So the answer to "do we need a cleaning script on the remote machine" is:

- yes, but not a remote-only script
- write one reusable repo script, then execute it locally on small samples and remotely on the full dataset

## Reward Design

### Final Reward

The final reward should compare predicted final answer and gold final answer after normalization.

Example signals:

- exact match for integers
- normalized comparison for fractions and decimals
- optional tolerance for numeric equivalence

### Process Reward V0

Phase-1 process reward should be rule-based and modular.

Initial components:

1. `step_validity`
   Checks whether a step appears mathematically or logically plausible by simple heuristics.

2. `step_consistency`
   Checks for obvious contradiction with previous steps.

3. `progress_contribution`
   Rewards steps that appear to move toward a result instead of repeating setup.

4. `anti_hacking_penalty`
   Penalizes repetition, filler language, and low-information verbosity.

### Aggregation

The sample-level reward should support:

- `final-only`
- `final + process`

Initial aggregation can be simple weighted summation:

```text
R = alpha * R_final + beta * sum(step_rewards)
```

Weights should be configuration-driven instead of hard-coded.

## Model Recommendation

### Default Recommendation

Choose a math-capable small open model that is easy to fine-tune and easy to run.

Recommended order:

1. `Qwen2.5-Math-1.5B` or `Qwen2.5-Math-7B`
2. `DeepSeekMath-7B`
3. another strong `7B` reasoning model if local availability is better

### Recommendation Rationale

For phase 1, the most important thing is stable experimentation, not benchmark prestige.

`Qwen2.5-Math` is a strong default if you want:

- easier tooling compatibility
- smaller variants for faster debugging
- smooth scaling from `1.5B` to `7B`

`DeepSeekMath-7B` is also a very reasonable choice if you want stronger math orientation from the start.

### Final Recommendation

Use this staged policy:

- local/debug baseline: `Qwen2.5-Math-1.5B`
- remote main phase-1 run: `DeepSeekMath-7B` or `Qwen2.5-Math-7B`

This gives fast local iteration and a credible remote baseline.

## Project Structure

The initial repository should be organized like this:

```text
.
├── README.md
├── configs/
│   ├── data/
│   ├── reward/
│   ├── train/
│   └── eval/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── debug/
├── docs/
│   └── superpowers/
│       └── specs/
├── experiments/
├── logs/
├── scripts/
│   ├── prepare_gsm8k.py
│   ├── build_debug_subset.py
│   ├── score_samples.py
│   └── run_sft.py
└── src/
    ├── data/
    ├── reward/
    ├── train/
    ├── eval/
    └── utils/
```

## Implementation Order

Implementation should proceed in this order:

1. create project structure
2. implement shared sample schema and config loading
3. implement GSM8K ingestion and normalization
4. implement step splitter
5. implement final-answer evaluator
6. implement rule-based process reward
7. implement sample-level reward aggregation
8. implement a simple training entrypoint
9. implement evaluation and reporting scripts

## Collaboration Model

Recommended collaboration flow:

1. everyone works from GitHub branches
2. local machines are for development only
3. remote jobs are launched from versioned code
4. experiment notes are committed in a structured format
5. large data, model weights, and logs are not committed to GitHub

## Risks

### Risk 1

Rule-based process reward may be noisy.

Mitigation:

- keep the first reward version simple
- inspect samples manually
- store intermediate scores for debugging

### Risk 2

Step splitting may introduce errors that contaminate reward.

Mitigation:

- make the splitter deterministic
- save both raw solution and split steps
- audit a random small batch visually

### Risk 3

Trying full RL too early may stall progress.

Mitigation:

- keep phase 1 on reward-weighted supervised training or reranking
- upgrade to stronger optimization only after the baseline works

## Immediate Next Step

After this design is approved, the next step is to write an implementation plan and then scaffold the project structure plus the first data pipeline files.
