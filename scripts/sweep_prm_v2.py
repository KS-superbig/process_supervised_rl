#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psrl.candidates import iter_jsonl, write_jsonl
from psrl.prm import build_prm_selection_report
from psrl.prm_protocol import build_eval_protocol_report
from psrl.prm_v2 import save_mlp_prm, score_candidates_with_mlp_prm, train_mlp_preference_prm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep PRM v2 hyperparameters with protocol-based early stop.")
    parser.add_argument("--preferences", type=Path, required=True, help="Preference JSONL path.")
    parser.add_argument("--candidates", type=Path, required=True, help="Candidate JSONL path.")
    parser.add_argument("--judgements", type=Path, required=True, help="Judge JSONL path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Sweep output directory.")
    parser.add_argument("--devices", default="cuda", help="Device to train on, e.g. cpu or cuda.")
    parser.add_argument("--max-features-grid", default="8000,12000", help="Comma-separated max_features values.")
    parser.add_argument("--hidden-dim-grid", default="256,384,512", help="Comma-separated hidden_dim values.")
    parser.add_argument("--epochs-grid", default="8,12,16", help="Comma-separated epochs values.")
    parser.add_argument("--batch-size-grid", default="64", help="Comma-separated batch_size values.")
    parser.add_argument("--learning-rate-grid", default="0.001,0.0015,0.002", help="Comma-separated learning_rate values.")
    parser.add_argument("--weight-decay-grid", default="0.00005,0.0001", help="Comma-separated weight_decay values.")
    parser.add_argument("--target-judge-agree-rate", type=float, default=0.74, help="Early-stop target for judge_agree_prm_rate.")
    parser.add_argument("--target-final-accuracy", type=float, default=0.93, help="Early-stop target for final_plus_prm_accuracy.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--shuffle-grid", action="store_true", help="Shuffle trial order.")
    parser.add_argument("--max-trials", type=int, default=0, help="Optional hard limit on number of trials.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    prefs = list(iter_jsonl(args.preferences))
    candidates = list(iter_jsonl(args.candidates))
    judgements = list(iter_jsonl(args.judgements))

    grid = _build_grid(args)
    if args.shuffle_grid:
        random.Random(args.seed).shuffle(grid)
    if args.max_trials > 0:
        grid = grid[: args.max_trials]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = args.output_dir / "trials.jsonl"
    best_path = args.output_dir / "best.json"

    best = None
    best_scored = None
    stopped = False
    with trials_path.open("w", encoding="utf-8") as trial_handle:
        for idx, cfg in enumerate(grid, start=1):
            trial_dir = args.output_dir / f"trial_{idx:03d}"
            prm, metrics = train_mlp_preference_prm(
                prefs,
                max_features=cfg["max_features"],
                hidden_dim=cfg["hidden_dim"],
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                learning_rate=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
                seed=args.seed + idx,
                device=args.devices,
            )
            scored = score_candidates_with_mlp_prm(candidates, prm, device=args.devices)
            report = build_prm_selection_report(scored)
            protocol = build_eval_protocol_report(scored, judgements)

            model_path, meta_path = save_mlp_prm(prm, trial_dir, metrics)
            scored_path = trial_dir / "scored_candidates.jsonl"
            protocol_json = trial_dir / "protocol_summary.json"
            protocol_md = trial_dir / "protocol_summary.md"
            report_md = trial_dir / "final_only_vs_final_plus_prm_selection_report.md"
            metrics_json = trial_dir / "metrics.json"
            write_jsonl(scored, scored_path)
            protocol_json.write_text(json.dumps(protocol.summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            protocol_md.write_text(protocol.markdown, encoding="utf-8")
            report_md.write_text(report.markdown, encoding="utf-8")
            metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            row = {
                "trial_index": idx,
                "config": cfg,
                "metrics": metrics,
                "selection_summary": report.summary,
                "protocol_summary": protocol.summary,
                "paths": {
                    "model_path": str(model_path),
                    "meta_path": str(meta_path),
                    "scored_path": str(scored_path),
                    "protocol_json": str(protocol_json),
                },
            }
            trial_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            trial_handle.flush()

            score_key = (
                float(protocol.summary["judge_agree_prm_rate"]),
                float(report.summary["final_plus_prm_accuracy"]),
                -int(protocol.summary["changed_count"]),
            )
            if best is None or score_key > best["score_key"]:
                best = {"score_key": score_key, "row": row}
                best_scored = scored

            print(
                f"trial {idx}/{len(grid)} "
                f"agree_prm={protocol.summary['judge_agree_prm_rate']:.4f} "
                f"final_acc={report.summary['final_plus_prm_accuracy']:.4f} "
                f"changed={protocol.summary['changed_count']}",
                flush=True,
            )

            if (
                float(protocol.summary["judge_agree_prm_rate"]) >= args.target_judge_agree_rate
                and float(report.summary["final_plus_prm_accuracy"]) >= args.target_final_accuracy
            ):
                stopped = True
                print("Early stop reached target metrics.", flush=True)
                break

    if best is None:
        raise RuntimeError("No trials executed.")

    best_payload = {
        "stopped_early": stopped,
        "num_trials_executed": (best["row"]["trial_index"] if stopped else len(grid)),
        "best": best["row"],
    }
    best_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    best_dir = args.output_dir / "best_export"
    best_dir.mkdir(parents=True, exist_ok=True)
    if best_scored is not None:
        write_jsonl(best_scored, best_dir / "scored_candidates.jsonl")

    print(f"Wrote sweep trials -> {trials_path}")
    print(f"Wrote best summary -> {best_path}")
    print(json.dumps(best_payload, ensure_ascii=False, indent=2))


def _build_grid(args: argparse.Namespace) -> list[dict]:
    max_features_grid = [int(v) for v in args.max_features_grid.split(",") if v.strip()]
    hidden_dim_grid = [int(v) for v in args.hidden_dim_grid.split(",") if v.strip()]
    epochs_grid = [int(v) for v in args.epochs_grid.split(",") if v.strip()]
    batch_size_grid = [int(v) for v in args.batch_size_grid.split(",") if v.strip()]
    learning_rate_grid = [float(v) for v in args.learning_rate_grid.split(",") if v.strip()]
    weight_decay_grid = [float(v) for v in args.weight_decay_grid.split(",") if v.strip()]

    grid = []
    for max_features in max_features_grid:
        for hidden_dim in hidden_dim_grid:
            for epochs in epochs_grid:
                for batch_size in batch_size_grid:
                    for learning_rate in learning_rate_grid:
                        for weight_decay in weight_decay_grid:
                            grid.append(
                                {
                                    "max_features": max_features,
                                    "hidden_dim": hidden_dim,
                                    "epochs": epochs,
                                    "batch_size": batch_size,
                                    "learning_rate": learning_rate,
                                    "weight_decay": weight_decay,
                                }
                            )
    return grid


if __name__ == "__main__":
    main()
