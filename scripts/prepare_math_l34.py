#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psrl.candidates import build_prompt, write_jsonl


DEFAULT_DATASET = "hendrycks/competition_math"
DEFAULT_LEVELS = (3, 4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare MATH competition level 3/4 rows for lightweight SFT and GRPO training."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id or local dataset path.")
    parser.add_argument("--split", default="train", help="Dataset split to load.")
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="MATH difficulty levels to keep, e.g. --levels 3 4.",
    )
    parser.add_argument("--limit", type=int, help="Optional maximum number of kept rows.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed used when --shuffle is set.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle kept rows before applying --limit.")
    parser.add_argument(
        "--sft-output",
        type=Path,
        default=Path("data/sft/math_l34_sft.jsonl"),
        help="Output chat-message JSONL for scripts/train_sft_lora_chat.py.",
    )
    parser.add_argument(
        "--grpo-output",
        type=Path,
        default=Path("data/processed/math_l34_train.jsonl"),
        help="Output reasoning JSONL consumable by scripts/train_grpo_smoke.py.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/processed/math_l34_manifest.json"),
        help="Output manifest with dataset/filter metadata.",
    )
    return parser


def load_dataset_rows(dataset: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(dataset, split=split)]


def parse_level(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value)
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def extract_boxed_answer(solution: str) -> str:
    marker = r"\boxed{"
    start = solution.rfind(marker)
    if start < 0:
        return ""
    index = start + len(marker)
    depth = 1
    chars: list[str] = []
    while index < len(solution):
        char = solution[index]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
            chars.append(char)
        else:
            chars.append(char)
        index += 1
    return normalize_math_answer("".join(chars))


def normalize_math_answer(answer: str) -> str:
    text = str(answer).strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def answer_from_solution(solution: str) -> str:
    boxed = extract_boxed_answer(solution)
    if boxed:
        return boxed
    lines = [line.strip() for line in str(solution).splitlines() if line.strip()]
    return normalize_math_answer(lines[-1]) if lines else ""


def filter_math_rows(
    rows: Iterable[dict[str, Any]],
    *,
    levels: set[int],
) -> list[dict[str, Any]]:
    kept = []
    for idx, row in enumerate(rows, start=1):
        level = parse_level(row.get("level"))
        if level not in levels:
            continue
        problem = str(row.get("problem", row.get("question", ""))).strip()
        solution = str(row.get("solution", row.get("answer", ""))).strip()
        if not problem or not solution:
            continue
        answer = answer_from_solution(solution)
        kept.append(
            {
                "sample_id": f"math-train-l{level}-{idx:06d}",
                "source": "competition_math",
                "split": "train",
                "question": problem,
                "answer_final": answer,
                "answer_final_normalized": answer,
                "solution_raw": solution,
                "steps": [line.strip() for line in solution.splitlines() if line.strip()],
                "metadata": {
                    "level": level,
                    "type": row.get("type"),
                },
            }
        )
    return kept


def to_sft_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    sft_rows = []
    for row in rows:
        sft_rows.append(
            {
                "messages": [
                    {"role": "user", "content": build_prompt(row["question"])},
                    {"role": "assistant", "content": row["solution_raw"]},
                ],
                "metadata": {
                    "sample_id": row["sample_id"],
                    **row.get("metadata", {}),
                },
            }
        )
    return sft_rows


def write_manifest(path: Path, *, args: argparse.Namespace, total_rows: int, kept_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "levels": args.levels,
                "shuffle": args.shuffle,
                "seed": args.seed,
                "limit": args.limit,
                "total_rows": total_rows,
                "kept_rows": kept_rows,
                "sft_output": str(args.sft_output),
                "grpo_output": str(args.grpo_output),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    import random

    args = build_parser().parse_args()
    raw_rows = load_dataset_rows(args.dataset, args.split)
    rows = filter_math_rows(raw_rows, levels=set(args.levels))
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    grpo_count = write_jsonl(rows, args.grpo_output)
    sft_count = write_jsonl(to_sft_rows(rows), args.sft_output)
    write_manifest(args.manifest_output, args=args, total_rows=len(raw_rows), kept_rows=len(rows))
    print(f"Wrote {grpo_count} GRPO rows -> {args.grpo_output}")
    print(f"Wrote {sft_count} SFT rows -> {args.sft_output}")
    print(f"Wrote manifest -> {args.manifest_output}")


if __name__ == "__main__":
    main()
