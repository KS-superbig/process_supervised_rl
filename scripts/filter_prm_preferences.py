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
from psrl.prm_protocol import filter_preference_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Denoise and cap PRM preference rows before training.")
    parser.add_argument("--preferences", type=Path, required=True, help="Input preference JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Filtered preference JSONL output path.")
    parser.add_argument("--stats-json", type=Path, required=True, help="Filtering stats JSON output path.")
    parser.add_argument("--judgements", type=Path, help="Optional LLM judgement JSONL path (for score-gap filtering).")
    parser.add_argument("--min-text-chars", type=int, default=20, help="Minimum text length per chosen/rejected side.")
    parser.add_argument("--min-score-gap", type=float, default=0.0, help="Minimum (chosen_score - rejected_score).")
    parser.add_argument("--max-pairs-per-sample", type=int, default=4, help="Maximum kept pairs per sample.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    preferences = list(iter_jsonl(args.preferences))
    judgements = list(iter_jsonl(args.judgements)) if args.judgements else []
    filtered, stats = filter_preference_rows(
        preferences,
        judgements=judgements,
        min_text_chars=args.min_text_chars,
        min_score_gap=args.min_score_gap,
        max_pairs_per_sample=args.max_pairs_per_sample,
    )

    kept = write_jsonl(filtered, args.output)
    stats["written_rows"] = kept
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote filtered preference rows -> {args.output}")
    print(f"Wrote filtering stats -> {args.stats_json}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
