from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def build_preference_rows(
    candidates: Iterable[dict],
    judgements: Iterable[dict],
    *,
    min_score_delta: float = 0.05,
) -> list[dict]:
    candidates_by_id = {row["candidate_id"]: row for row in candidates}
    candidates_by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in candidates_by_id.values():
        candidates_by_sample[row["sample_id"]].append(row)

    rows = []
    for judgement in judgements:
        sample_id = judgement["sample_id"]
        preferences = judgement.get("pairwise_preferences") or _ranking_to_adjacent_preferences(
            judgement.get("ranking", []),
            judgement.get("scores", {}),
            min_score_delta,
        )
        for pref in preferences:
            chosen_id = pref["chosen"]
            rejected_id = pref["rejected"]
            if chosen_id not in candidates_by_id:
                raise ValueError(f"unknown chosen candidate id: {chosen_id}")
            if rejected_id not in candidates_by_id:
                raise ValueError(f"unknown rejected candidate id: {rejected_id}")

            chosen = candidates_by_id[chosen_id]
            rejected = candidates_by_id[rejected_id]
            if chosen["sample_id"] != sample_id or rejected["sample_id"] != sample_id:
                raise ValueError("preference candidates must belong to the judgement sample")

            rows.append(
                {
                    "sample_id": sample_id,
                    "question": chosen.get("question", rejected.get("question", "")),
                    "chosen_candidate_id": chosen_id,
                    "chosen_text": chosen.get("candidate_text", ""),
                    "rejected_candidate_id": rejected_id,
                    "rejected_text": rejected.get("candidate_text", ""),
                    "judge_reason": str(pref.get("reason", "")).strip(),
                }
            )
    return rows


def _ranking_to_adjacent_preferences(ranking: list[str], scores: dict[str, float], min_score_delta: float) -> list[dict]:
    preferences = []
    for chosen, rejected in zip(ranking, ranking[1:]):
        if scores:
            chosen_score = float(scores.get(chosen, 0.0))
            rejected_score = float(scores.get(rejected, 0.0))
            if chosen_score - rejected_score < min_score_delta:
                continue
        preferences.append(
            {
                "chosen": chosen,
                "rejected": rejected,
                "reason": "Derived from adjacent LLM judge ranking.",
            }
        )
    return preferences
