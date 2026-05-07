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
from psrl.prm import build_prm_selection_report
from psrl.prm_v2 import save_mlp_prm, score_candidates_with_mlp_prm, train_mlp_preference_prm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PRM v2 (tokenizer + lightweight MLP) from preference pairs.")
    parser.add_argument("--preferences", type=Path, required=True, help="Preference JSONL path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for model and metrics.")
    parser.add_argument("--candidates", type=Path, required=True, help="Candidate JSONL path for reranking eval.")
    parser.add_argument("--scored-output", type=Path, help="Optional scored candidate JSONL output path.")
    parser.add_argument("--report-output", type=Path, help="Optional markdown report output path.")
    parser.add_argument("--max-features", type=int, default=8000, help="Maximum tokenizer vocabulary size.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="MLP hidden dimension.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    preferences = list(iter_jsonl(args.preferences))
    candidates = list(iter_jsonl(args.candidates))

    prm, metrics = train_mlp_preference_prm(
        preferences,
        max_features=args.max_features,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = save_mlp_prm(prm, args.output_dir, metrics)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    scored_output = args.scored_output or (args.output_dir / "scored_candidates.jsonl")
    report_output = args.report_output or (args.output_dir / "final_only_vs_final_plus_prm_selection_report.md")
    scored = score_candidates_with_mlp_prm(candidates, prm, device=args.device)
    write_jsonl(scored, scored_output)
    report = build_prm_selection_report(scored)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(report.markdown, encoding="utf-8")

    print(f"Trained PRM v2 on {metrics['num_pairs']} preference pairs")
    print(f"train_pair_accuracy={metrics['train_pair_accuracy']:.4f}")
    print(f"final_loss={metrics['final_loss']:.6f}")
    print(f"Wrote model weights -> {model_path}")
    print(f"Wrote model meta -> {meta_path}")
    print(f"Wrote metrics -> {metrics_path}")
    print(f"Scored {len(scored)} candidates -> {scored_output}")
    print(f"Wrote selection report -> {report_output}")
    print(report.markdown)


if __name__ == "__main__":
    main()
