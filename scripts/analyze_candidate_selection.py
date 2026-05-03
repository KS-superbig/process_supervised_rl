#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psrl.config import load_yaml_config
from psrl.eval.candidate_selection import (
    build_selection_report,
    read_jsonl,
    score_candidate_rows,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score candidates and compare final-only vs final+process top1.")
    parser.add_argument("--input", type=Path, required=True, help="Candidate JSONL path.")
    parser.add_argument("--scored-output", type=Path, required=True, help="Scored candidate JSONL path.")
    parser.add_argument("--report-output", type=Path, required=True, help="Markdown report path.")
    parser.add_argument(
        "--reward-config",
        type=Path,
        default=Path("configs/reward/process_reward_v0.yaml"),
        help="Reward config YAML path.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    reward_cfg = load_yaml_config(args.reward_config)
    rows = read_jsonl(args.input)
    scored = score_candidate_rows(rows, reward_cfg)
    write_jsonl(scored, args.scored_output)

    report = build_selection_report(scored)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(report.markdown, encoding="utf-8")

    print(f"Scored {len(scored)} candidates -> {args.scored_output}")
    print(f"Wrote selection report -> {args.report_output}")
    print(report.markdown)


if __name__ == "__main__":
    main()
