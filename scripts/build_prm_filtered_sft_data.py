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
from psrl.sft_data import select_prm_filtered_sft_rows, summarize_sft_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build clean chat-message SFT data from PRM-scored candidates.")
    parser.add_argument("--scored-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--min-prm-score", type=float, default=None)
    parser.add_argument("--min-text-chars", type=int, default=40)
    parser.add_argument("--min-space-ratio", type=float, default=0.04)
    parser.add_argument("--allow-final-wrong", action="store_true")
    parser.add_argument("--allow-run-on", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidates = list(iter_jsonl(args.scored_candidates))
    rows, stats = select_prm_filtered_sft_rows(
        candidates,
        require_final_correct=not args.allow_final_wrong,
        min_prm_score=args.min_prm_score,
        min_text_chars=args.min_text_chars,
        min_space_ratio=args.min_space_ratio,
        reject_run_on=not args.allow_run_on,
    )
    written = write_jsonl(rows, args.output)
    stats["written_rows"] = written
    stats["quality_summary"] = summarize_sft_rows(rows)

    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote clean SFT rows -> {args.output}")
    print(f"Wrote stats -> {args.stats_json}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
