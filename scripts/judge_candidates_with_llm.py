#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psrl.candidates import iter_jsonl
from psrl.llm_judge import (
    build_judge_prompt,
    call_deepseek_judge,
    group_candidates_by_sample,
    parse_judge_response,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge candidate reasoning traces with an LLM API.")
    parser.add_argument("--input", type=Path, required=True, help="Candidate JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="LLM judgement JSONL path.")
    parser.add_argument("--provider", choices=["deepseek"], default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to judge.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=1, help="Retries when API returns invalid JSON.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip sample_ids already in output.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.provider != "deepseek":
        raise SystemExit(f"Unsupported provider: {args.provider}")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY environment variable.")

    rows = list(iter_jsonl(args.input))
    grouped = group_candidates_by_sample(rows)
    sample_ids = list(grouped)
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]

    completed = set()
    if args.output.exists() and not args.no_resume:
        completed = {row["sample_id"] for row in iter_jsonl(args.output)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for ordinal, sample_id in enumerate(sample_ids, start=1):
            if sample_id in completed:
                print(f"Skipping {sample_id}: already judged", flush=True)
                continue

            candidates = grouped[sample_id]
            first = candidates[0]
            prompt = build_judge_prompt(
                sample_id=sample_id,
                question=first.get("question", ""),
                gold_final=first.get("gold_final", first.get("answer_final_normalized", "")),
                candidates=candidates,
            )
            last_parse_error = None
            for parse_attempt in range(args.parse_retries + 1):
                raw_response, usage = call_deepseek_judge(
                    api_key=api_key,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                try:
                    judgement = parse_judge_response(
                        raw_response,
                        sample_id=sample_id,
                        expected_candidate_ids=[row["candidate_id"] for row in candidates],
                    )
                    break
                except ValueError as exc:
                    last_parse_error = exc
                    if parse_attempt >= args.parse_retries:
                        raise
                    print(f"Retrying {sample_id}: invalid judge JSON ({exc})", flush=True)
            else:
                raise RuntimeError(f"Failed to parse judgement for {sample_id}: {last_parse_error}")
            judgement["provider"] = args.provider
            judgement["model"] = args.model
            judgement["usage"] = usage
            handle.write(json.dumps(judgement, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            print(f"Judged {ordinal}/{len(sample_ids)}: {sample_id}", flush=True)
            if args.sleep:
                time.sleep(args.sleep)

    print(f"Wrote {written} new judgements -> {args.output}")


if __name__ == "__main__":
    main()
