#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psrl.data.gsm8k import write_reasoning_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare GSM8K JSONL into unified sample schema.")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw GSM8K JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Path to processed JSONL.")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"], help="Dataset split name.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    count = write_reasoning_samples(args.input, args.output, args.split)
    print(f"Wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
