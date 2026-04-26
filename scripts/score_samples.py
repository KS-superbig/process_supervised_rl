#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score sample trajectories with final + process reward.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL samples path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL with reward details.")
    parser.add_argument(
        "--reward-config",
        type=Path,
        default=Path("configs/reward/process_reward_v0.yaml"),
        help="Reward config YAML path.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows for quick debug runs.")
    return parser


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from psrl.config import load_yaml_config
    from psrl.reward.aggregator import combine_rewards
    from psrl.reward.final_reward import compute_final_reward
    from psrl.reward.process_reward_v0 import score_process_steps

    cfg = load_yaml_config(args.reward_config)
    component_weights = cfg.get("components", {})
    final_w = float(cfg.get("final_reward_weight", 1.0))
    process_w = float(cfg.get("process_reward_weight", 0.0))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    final_scores: list[float] = []
    process_scores: list[float] = []
    total_scores: list[float] = []

    with args.output.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(iter_jsonl(args.input)):
            if args.limit is not None and idx >= args.limit:
                break

            gold_final = row.get("answer_final_normalized", "")
            predicted_final = row.get("candidate_final", row.get("answer_final_normalized", ""))
            predicted_steps = row.get("candidate_steps", row.get("steps", []))

            final_reward = compute_final_reward(gold_final, predicted_final)
            process_result = score_process_steps(predicted_steps, component_weights)
            sample_reward = combine_rewards(
                final_reward=final_reward,
                process_reward=process_result.score,
                final_reward_weight=final_w,
                process_reward_weight=process_w,
            )

            flags = []
            if process_result.component_means["anti_hacking_penalty"] > 0.6:
                flags.append("high_hacking_penalty")
            if process_result.component_means["progress_contribution"] < 0.2 and len(predicted_steps) >= 4:
                flags.append("low_progress_signal")

            payload = {
                "id": row.get("id", f"sample-{idx + 1:06d}"),
                "question": row.get("question", ""),
                "num_steps": len(predicted_steps),
                "final_reward": sample_reward.final_reward,
                "process_reward": sample_reward.process_reward,
                "total_reward": sample_reward.total_reward,
                "component_means": process_result.component_means,
                "flags": flags,
            }
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")

            final_scores.append(sample_reward.final_reward)
            process_scores.append(sample_reward.process_reward)
            total_scores.append(sample_reward.total_reward)

    total = len(total_scores)
    if total == 0:
        print("No rows scored.")
        return

    print(f"Scored {total} samples -> {args.output}")
    print(
        "Summary: "
        f"final_mean={statistics.mean(final_scores):.4f}, "
        f"process_mean={statistics.mean(process_scores):.4f}, "
        f"total_mean={statistics.mean(total_scores):.4f}"
    )


if __name__ == "__main__":
    main()
