#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psrl.candidates import iter_jsonl, write_jsonl
from psrl.prm_dataset import build_preference_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build PRM preference data from candidate rows and LLM judgements.")
    parser.add_argument("--candidates", type=Path, required=True, help="Candidate JSONL path.")
    parser.add_argument("--judgements", type=Path, required=True, help="LLM judgement JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="PRM preference JSONL output path.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    candidates = list(iter_jsonl(args.candidates))
    judgements = list(iter_jsonl(args.judgements))
    rows = build_preference_rows(candidates, judgements)
    count = write_jsonl(rows, args.output)
    print(f"Wrote {count} PRM preference rows -> {args.output}")


if __name__ == "__main__":
    main()
