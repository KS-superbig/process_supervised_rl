from __future__ import annotations

from dataclasses import dataclass
import re

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "we",
}

_FILLER_PATTERNS = (
    "it is clear",
    "obviously",
    "let us",
    "now we can",
    "as we know",
)

_CONCLUSION_CUES = ("therefore", "thus", "hence", "so", "answer", "final")


@dataclass
class StepRewardDetail:
    step_idx: int
    step_text: str
    components: dict[str, float]
    weighted_score: float


@dataclass
class ProcessRewardResult:
    score: float
    component_means: dict[str, float]
    step_details: list[StepRewardDetail]


def score_process_steps(steps: list[str], component_weights: dict[str, float]) -> ProcessRewardResult:
    if not steps:
        return ProcessRewardResult(
            score=0.0,
            component_means={
                "step_validity": 0.0,
                "step_consistency": 0.0,
                "progress_contribution": 0.0,
                "anti_hacking_penalty": 0.0,
            },
            step_details=[],
        )

    seen_tokens: set[str] = set()
    assignment_history: dict[str, str] = {}
    previous_step = ""

    details: list[StepRewardDetail] = []
    component_sums = {
        "step_validity": 0.0,
        "step_consistency": 0.0,
        "progress_contribution": 0.0,
        "anti_hacking_penalty": 0.0,
    }

    normalizer = sum(abs(float(v)) for v in component_weights.values()) or 1.0

    for idx, step in enumerate(steps):
        validity = _score_step_validity(step)
        consistency = _score_step_consistency(step, assignment_history)
        progress = _score_progress_contribution(step, seen_tokens)
        hacking = _score_anti_hacking_penalty(step, previous_step)

        components = {
            "step_validity": validity,
            "step_consistency": consistency,
            "progress_contribution": progress,
            "anti_hacking_penalty": hacking,
        }

        weighted = 0.0
        for key, value in components.items():
            weighted += component_weights.get(key, 0.0) * value
            component_sums[key] += value

        details.append(
            StepRewardDetail(
                step_idx=idx,
                step_text=step,
                components=components,
                weighted_score=weighted / normalizer,
            )
        )

        _update_assignments(step, assignment_history)
        seen_tokens.update(_tokens(step))
        previous_step = step

    count = float(len(steps))
    component_means = {key: value / count for key, value in component_sums.items()}
    score = sum(detail.weighted_score for detail in details) / count
    return ProcessRewardResult(score=score, component_means=component_means, step_details=details)


def _score_step_validity(step: str) -> float:
    text = step.strip().lower()
    score = 0.0
    if any(char.isdigit() for char in text):
        score += 0.4
    if any(op in text for op in ("=", "+", "-", "*", "/", "%", "^")):
        score += 0.4
    if len(text) >= 12:
        score += 0.2
    if "??" in text or "idk" in text:
        score -= 0.6
    return _clamp(score, 0.0, 1.0)


def _score_step_consistency(step: str, assignment_history: dict[str, str]) -> float:
    assignments = _extract_assignments(step)
    if not assignments:
        return 0.8

    for var, value in assignments.items():
        old = assignment_history.get(var)
        if old is not None and old != value:
            return 0.0
    return 1.0


def _score_progress_contribution(step: str, seen_tokens: set[str]) -> float:
    tokens = _tokens(step)
    if not tokens:
        return 0.0

    novel = len([tok for tok in tokens if tok not in seen_tokens])
    novelty_ratio = novel / len(tokens)
    score = novelty_ratio
    if any(cue in step.lower() for cue in _CONCLUSION_CUES):
        score += 0.2
    return _clamp(score, 0.0, 1.0)


def _score_anti_hacking_penalty(step: str, previous_step: str) -> float:
    penalty = 0.0
    lowered = step.lower()

    if previous_step:
        similarity = _jaccard_similarity(_tokens(step), _tokens(previous_step))
        if similarity > 0.85:
            penalty += 1.0

    for pattern in _FILLER_PATTERNS:
        if pattern in lowered:
            penalty += 0.3

    if len(step) > 220 and sum(ch.isdigit() for ch in step) < 2:
        penalty += 0.5

    return _clamp(penalty, 0.0, 1.0)


def _extract_assignments(step: str) -> dict[str, str]:
    matches = re.findall(r"\b([a-zA-Z])\s*=\s*([-+]?\d+(?:\.\d+)?)", step)
    return {var.lower(): value for var, value in matches}


def _update_assignments(step: str, assignment_history: dict[str, str]) -> None:
    assignment_history.update(_extract_assignments(step))


def _tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {tok for tok in toks if tok not in _STOPWORDS}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
