from __future__ import annotations

from collections import defaultdict
import re
import statistics
from typing import Iterable


_GLUED_NAME_VERB = re.compile(
    r"\b[A-Z][a-z]{2,}(?:earns|earned|starts|started|has|does|costs|gets|"
    r"buys|bought|sells|sold|makes|made|needs|needed|wants|wanted|reads|"
    r"read|writes|wrote)\b"
)
_LOWER_TO_UPPER = re.compile(r"[a-z][A-Z]")


def has_run_on_artifact(text: str) -> bool:
    """Detect obvious tokenizer/spacing artifacts in generated reasoning text."""
    normalized = str(text)
    return bool(_GLUED_NAME_VERB.search(normalized) or _LOWER_TO_UPPER.search(normalized))


def whitespace_ratio(text: str) -> float:
    text = str(text)
    if not text:
        return 0.0
    return text.count(" ") / len(text)


def is_usable_sft_candidate(
    row: dict,
    *,
    require_final_correct: bool = True,
    min_text_chars: int = 40,
    min_space_ratio: float = 0.04,
    reject_run_on: bool = True,
) -> tuple[bool, str]:
    if require_final_correct and float(row.get("final_reward", 0.0)) <= 0.0:
        return False, "final_wrong"
    text = str(row.get("candidate_text", "")).strip()
    if len(text) < min_text_chars:
        return False, "short_text"
    if whitespace_ratio(text) < min_space_ratio:
        return False, "low_space_ratio"
    if reject_run_on and has_run_on_artifact(text):
        return False, "run_on"
    return True, "ok"


def build_sft_row(row: dict) -> dict:
    question = str(row.get("question", "")).strip()
    answer = str(row.get("candidate_text", "")).strip()
    return {
        "sample_id": row.get("sample_id", ""),
        "candidate_id": row.get("candidate_id", ""),
        "final_reward": float(row.get("final_reward", 0.0)),
        "prm_score": float(row.get("prm_score", 0.0)),
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


def select_prm_filtered_sft_rows(
    candidate_rows: Iterable[dict],
    *,
    require_final_correct: bool = True,
    min_prm_score: float | None = None,
    min_text_chars: int = 40,
    min_space_ratio: float = 0.04,
    reject_run_on: bool = True,
) -> tuple[list[dict], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        grouped[str(row.get("sample_id", ""))].append(row)

    selected: list[dict] = []
    stats = {
        "samples": len(grouped),
        "selected_rows": 0,
        "dropped_no_usable_candidate": 0,
        "rejected_final_wrong": 0,
        "rejected_short_text": 0,
        "rejected_low_space_ratio": 0,
        "rejected_run_on": 0,
        "rejected_low_prm_score": 0,
    }

    for sample_id in sorted(grouped):
        ranked = sorted(
            grouped[sample_id],
            key=lambda row: (
                float(row.get("final_reward", 0.0)),
                float(row.get("prm_score", 0.0)),
                -int(row.get("candidate_index", 0)),
            ),
            reverse=True,
        )
        picked = None
        for row in ranked:
            if min_prm_score is not None and float(row.get("prm_score", 0.0)) < min_prm_score:
                stats["rejected_low_prm_score"] += 1
                continue
            ok, reason = is_usable_sft_candidate(
                row,
                require_final_correct=require_final_correct,
                min_text_chars=min_text_chars,
                min_space_ratio=min_space_ratio,
                reject_run_on=reject_run_on,
            )
            if ok:
                picked = row
                break
            stats[f"rejected_{reason}"] += 1
        if picked is None:
            stats["dropped_no_usable_candidate"] += 1
            continue
        selected.append(build_sft_row(picked))

    stats["selected_rows"] = len(selected)
    return selected, stats


def summarize_sft_rows(rows: Iterable[dict]) -> dict:
    assistant_texts = []
    for row in rows:
        messages = row.get("messages", [])
        for message in messages:
            if message.get("role") == "assistant":
                assistant_texts.append(str(message.get("content", "")))
                break

    chars = [len(text) for text in assistant_texts]
    spaces = [text.count(" ") for text in assistant_texts]
    return {
        "rows": len(assistant_texts),
        "run_on_rows": sum(1 for text in assistant_texts if has_run_on_artifact(text)),
        "low_space_ratio_rows": sum(1 for text in assistant_texts if whitespace_ratio(text) < 0.04),
        "assistant_mean_chars": statistics.mean(chars) if chars else 0.0,
        "assistant_mean_spaces": statistics.mean(spaces) if spaces else 0.0,
        "assistant_mean_space_ratio": statistics.mean(
            [whitespace_ratio(text) for text in assistant_texts]
        )
        if assistant_texts
        else 0.0,
    }
