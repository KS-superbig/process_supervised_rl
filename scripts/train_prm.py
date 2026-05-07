#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psrl.candidates import iter_jsonl, write_jsonl
from psrl.prm import build_prm_selection_report, score_candidates_with_prm, train_linear_preference_prm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a trajectory-level preference PRM from chosen/rejected rows.")
    parser.add_argument("--preferences", type=Path, required=True, help="PRM preference JSONL path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for model, metrics, and optional reports.")
    parser.add_argument("--candidates", type=Path, help="Optional candidate JSONL path to score with the trained PRM.")
    parser.add_argument("--scored-output", type=Path, help="Optional scored candidate JSONL path.")
    parser.add_argument("--report-output", type=Path, help="Optional final-only vs final+PRM markdown report path.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of pairwise training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Pairwise logistic regression learning rate.")
    parser.add_argument("--l2", type=float, default=0.0001, help="L2 regularization strength.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    preferences = list(iter_jsonl(args.preferences))
    model, metrics = train_linear_preference_prm(
        preferences,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.json"
    metrics_path = args.output_dir / "metrics.json"
    model.save(model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Trained PRM on {metrics['num_pairs']} preference pairs")
    print(f"train_pair_accuracy={metrics['train_pair_accuracy']:.4f}")
    print(f"final_loss={metrics['final_loss']:.6f}")
    print(f"Wrote model -> {model_path}")
    print(f"Wrote metrics -> {metrics_path}")

    if args.candidates:
        scored_output = args.scored_output or (args.output_dir / "scored_candidates.jsonl")
        report_output = args.report_output or (args.output_dir / "final_only_vs_final_plus_prm_selection_report.md")
        candidates = list(iter_jsonl(args.candidates))
        scored = score_candidates_with_prm(candidates, model)
        write_jsonl(scored, scored_output)
        report = build_prm_selection_report(scored)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(report.markdown, encoding="utf-8")
        print(f"Scored {len(scored)} candidates -> {scored_output}")
        print(f"Wrote selection report -> {report_output}")
        print(report.markdown)


if __name__ == "__main__":
    main()
