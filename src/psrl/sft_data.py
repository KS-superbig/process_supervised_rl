from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import re
import statistics
from typing import Iterable


_GLUED_NAME_VERB = re.compile(
    r"\b[A-Z][a-z]{2,}(?:earns|earned|starts|started|has|does|costs|gets|"
    r"buys|bought|sells|sold|makes|made|needs|needed|wants|wanted|reads|"
    r"read|writes|wrote)\b"
)
_LOWER_TO_UPPER = re.compile(r"[a-z][A-Z]")
_ALPHA_TOKEN = re.compile(r"[A-Za-z]+")
_PUNCT = re.compile(r"[.!?,;:]")
_PUNCT_MISSING_SPACE = re.compile(r"[.!?,;:](?=[A-Za-z])")
_TOKENIZER_ARTIFACT = re.compile(r"[ĠĊ▁]")
_MISSING_SPACE_AFTER_PUNCT = re.compile(r"([.!?,;:])(?=([A-Za-z]))")
_SPACE_AROUND_NEWLINES = re.compile(r"[ \t]*\n[ \t]*")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class FormatQualityMetrics:
    chars: int
    spaces: int
    space_ratio: float
    alpha_token_count: int
    avg_alpha_token_len: float
    max_alpha_token_len: int
    long_alpha_token_count: int
    punctuation_count: int
    punctuation_missing_space_count: int
    punctuation_missing_space_ratio: float
    lower_to_upper_count: int
    tokenizer_artifact_count: int


def has_run_on_artifact(text: str) -> bool:
    """Detect obvious tokenizer/spacing artifacts in generated reasoning text."""
    normalized = str(text)
    metrics = format_quality_metrics(normalized)
    return bool(
        _GLUED_NAME_VERB.search(normalized)
        or metrics.lower_to_upper_count
        or metrics.tokenizer_artifact_count
        or metrics.max_alpha_token_len >= 36
        or metrics.punctuation_missing_space_ratio >= 0.45
    )


def whitespace_ratio(text: str) -> float:
    text = str(text)
    if not text:
        return 0.0
    return text.count(" ") / len(text)


def normalize_candidate_format(text: str) -> str:
    """Apply only deterministic spacing repairs that do not guess word boundaries."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = _TOKENIZER_ARTIFACT.sub(" ", text)
    text = _MISSING_SPACE_AFTER_PUNCT.sub(r"\1 ", text)
    text = _SPACE_AROUND_NEWLINES.sub("\n", text)
    lines = [_MULTI_SPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def format_quality_metrics(text: str) -> FormatQualityMetrics:
    text = str(text)
    alpha_tokens = _ALPHA_TOKEN.findall(text)
    alpha_lens = [len(token) for token in alpha_tokens]
    punct_count = len(_PUNCT.findall(text))
    punct_missing = len(_PUNCT_MISSING_SPACE.findall(text))
    return FormatQualityMetrics(
        chars=len(text),
        spaces=text.count(" "),
        space_ratio=whitespace_ratio(text),
        alpha_token_count=len(alpha_tokens),
        avg_alpha_token_len=statistics.mean(alpha_lens) if alpha_lens else 0.0,
        max_alpha_token_len=max(alpha_lens) if alpha_lens else 0,
        long_alpha_token_count=sum(1 for length in alpha_lens if length >= 24),
        punctuation_count=punct_count,
        punctuation_missing_space_count=punct_missing,
        punctuation_missing_space_ratio=punct_missing / punct_count if punct_count else 0.0,
        lower_to_upper_count=len(_LOWER_TO_UPPER.findall(text)),
        tokenizer_artifact_count=len(_TOKENIZER_ARTIFACT.findall(text)),
    )


def is_format_healthy(
    text: str,
    *,
    min_space_ratio: float = 0.12,
    max_avg_alpha_token_len: float = 10.0,
    max_alpha_token_len: int = 28,
    max_long_alpha_tokens: int = 0,
    max_punctuation_missing_space_ratio: float = 0.20,
    reject_run_on: bool = True,
) -> tuple[bool, str, FormatQualityMetrics]:
    metrics = format_quality_metrics(text)
    if metrics.space_ratio < min_space_ratio:
        return False, "low_space_ratio", metrics
    if metrics.avg_alpha_token_len > max_avg_alpha_token_len:
        return False, "high_avg_alpha_token_len", metrics
    if metrics.max_alpha_token_len > max_alpha_token_len:
        return False, "long_alpha_token", metrics
    if metrics.long_alpha_token_count > max_long_alpha_tokens:
        return False, "long_alpha_token", metrics
    if metrics.punctuation_missing_space_ratio > max_punctuation_missing_space_ratio:
        return False, "punctuation_spacing", metrics
    if metrics.tokenizer_artifact_count:
        return False, "tokenizer_artifact", metrics
    if reject_run_on and has_run_on_artifact(text):
        return False, "run_on", metrics
    return True, "ok", metrics


def is_usable_sft_candidate(
    row: dict,
    *,
    require_final_correct: bool = True,
    min_text_chars: int = 40,
    min_space_ratio: float = 0.04,
    reject_run_on: bool = True,
    strict_format: bool = False,
) -> tuple[bool, str]:
    if require_final_correct and float(row.get("final_reward", 0.0)) <= 0.0:
        return False, "final_wrong"
    text = normalize_candidate_format(row.get("candidate_text", ""))
    if len(text) < min_text_chars:
        return False, "short_text"
    if strict_format:
        ok, reason, _metrics = is_format_healthy(
            text,
            min_space_ratio=max(min_space_ratio, 0.12),
            reject_run_on=reject_run_on,
        )
        if not ok:
            if reason == "low_space_ratio":
                return False, reason
            return False, "bad_format"
    elif whitespace_ratio(text) < min_space_ratio:
        return False, "low_space_ratio"
    if not strict_format and reject_run_on and has_run_on_artifact(text):
        return False, "run_on"
    return True, "ok"


def build_sft_row(row: dict) -> dict:
    question = str(row.get("question", "")).strip()
    answer = normalize_candidate_format(row.get("candidate_text", ""))
    return {
        "sample_id": row.get("sample_id", ""),
        "candidate_id": row.get("candidate_id", ""),
        "final_reward": float(row.get("final_reward", 0.0)),
        "prm_score": float(row.get("prm_score", 0.0)),
        "format_quality": asdict(format_quality_metrics(answer)),
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
    strict_format: bool = False,
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
        "rejected_bad_format": 0,
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
                strict_format=strict_format,
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
    metrics = [format_quality_metrics(text) for text in assistant_texts]
    return {
        "rows": len(assistant_texts),
        "run_on_rows": sum(1 for text in assistant_texts if has_run_on_artifact(text)),
        "low_space_ratio_rows": sum(1 for text in assistant_texts if whitespace_ratio(text) < 0.12),
        "bad_format_rows": sum(1 for text in assistant_texts if not is_format_healthy(text)[0]),
        "assistant_mean_chars": statistics.mean(chars) if chars else 0.0,
        "assistant_mean_spaces": statistics.mean(spaces) if spaces else 0.0,
        "assistant_mean_space_ratio": statistics.mean(
            [whitespace_ratio(text) for text in assistant_texts]
        )
        if assistant_texts
        else 0.0,
        "assistant_mean_avg_alpha_token_len": statistics.mean(
            [item.avg_alpha_token_len for item in metrics]
        )
        if metrics
        else 0.0,
        "assistant_max_alpha_token_len": max([item.max_alpha_token_len for item in metrics], default=0),
        "assistant_mean_punctuation_missing_space_ratio": statistics.mean(
            [item.punctuation_missing_space_ratio for item in metrics]
        )
        if metrics
        else 0.0,
    }
