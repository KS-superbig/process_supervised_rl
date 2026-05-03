from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from psrl.data.normalize import extract_solution_text, normalize_final_answer
from psrl.data.step_splitter import split_solution_steps


def build_prompt(question: str) -> str:
    return (
        f"{question.strip()}\n"
        "Please reason step by step, and put your final answer within \\boxed{}. "
        "Put each reasoning step on its own line."
    )


def clean_candidate_text(text: str) -> str:
    replacements = {
        "Ġ": " ",
        "Ċ": "\n",
        "▁": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def extract_candidate_final(candidate_text: str) -> str:
    normalized = normalize_final_answer(candidate_text)
    if "####" in candidate_text and normalized:
        return normalized

    answer_match = re.search(
        r"(?:answer|final answer)\s*(?:is|=|:)?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        candidate_text,
        flags=re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).replace(",", "")

    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", candidate_text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def build_candidate_row(sample: dict, candidate_index: int, candidate_text: str) -> dict:
    candidate_text = clean_candidate_text(candidate_text)
    sample_id = sample.get("id", sample.get("sample_id", "sample-unknown"))
    gold_final = sample.get("answer_final_normalized", sample.get("gold_final", ""))
    solution_text = extract_solution_text(candidate_text)
    candidate_final = extract_candidate_final(candidate_text)
    return {
        "sample_id": sample_id,
        "candidate_id": f"{sample_id}-cand-{candidate_index:02d}",
        "candidate_index": candidate_index,
        "question": sample.get("question", ""),
        "gold_final": gold_final,
        "answer_final_normalized": gold_final,
        "candidate_text": candidate_text,
        "candidate_final": candidate_final,
        "candidate_steps": split_solution_steps(solution_text),
    }


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
