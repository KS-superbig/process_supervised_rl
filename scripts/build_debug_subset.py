#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a small debug subset from processed JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Processed JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Debug subset JSONL path.")
    parser.add_argument("--limit", type=int, default=100, help="Number of rows to keep.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src):
            if idx >= args.limit:
                break
            dst.write(line)

    print(f"Wrote debug subset to {args.output}")


if __name__ == "__main__":
    main()
