#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psrl.candidates import iter_jsonl
from psrl.prm_protocol import build_eval_protocol_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed PRM evaluation protocol and export summary.")
    parser.add_argument("--scored", type=Path, required=True, help="Scored candidate JSONL path.")
    parser.add_argument("--judgements", type=Path, required=True, help="Original LLM judgement JSONL path.")
    parser.add_argument("--audit", type=Path, help="Optional changed-case deepseek audit JSONL path.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Output summary JSON path.")
    parser.add_argument("--summary-md", type=Path, required=True, help="Output summary markdown path.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    scored = list(iter_jsonl(args.scored))
    judgements = list(iter_jsonl(args.judgements))
    audit_rows = list(iter_jsonl(args.audit)) if args.audit else []

    result = build_eval_protocol_report(scored, judgements, audit_rows=audit_rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(result.markdown, encoding="utf-8")

    print(f"Wrote protocol summary JSON -> {args.summary_json}")
    print(f"Wrote protocol summary markdown -> {args.summary_md}")
    print(result.markdown)


if __name__ == "__main__":
    main()
