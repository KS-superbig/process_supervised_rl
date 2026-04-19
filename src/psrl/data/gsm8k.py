import json
from pathlib import Path
from typing import Iterable

from psrl.data.normalize import extract_solution_text, normalize_final_answer
from psrl.data.schema import ReasoningSample
from psrl.data.step_splitter import split_solution_steps


def build_reasoning_sample(sample_id: str, split: str, row: dict) -> ReasoningSample:
    raw_answer = row["answer"]
    solution_raw = extract_solution_text(raw_answer)
    return ReasoningSample(
        sample_id=sample_id,
        source="gsm8k",
        split=split,
        question=row["question"].strip(),
        answer_final=raw_answer.strip(),
        answer_final_normalized=normalize_final_answer(raw_answer),
        solution_raw=solution_raw,
        steps=split_solution_steps(solution_raw),
        metadata={},
    )


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_reasoning_samples(input_path: Path, output_path: Path, split: str) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(iter_jsonl(input_path), start=1):
            sample = build_reasoning_sample(f"gsm8k-{split}-{idx:06d}", split, row)
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
