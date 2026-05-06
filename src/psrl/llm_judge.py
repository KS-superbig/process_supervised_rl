from __future__ import annotations

from collections import defaultdict
import json
import time
from typing import Iterable
from urllib import error, request


JUDGE_SYSTEM_PROMPT = """You are a strict but fair math reasoning judge.
Evaluate candidate solutions for process quality, not only final answer.
Prefer correct, faithful, complete, and coherent reasoning.
Penalize unrelated continuations, code/prompt artifacts, contradictions, unsupported jumps, and thin explanations.
Do not reward a solution merely because it is short.
Return only valid JSON."""


def group_candidates_by_sample(rows: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    return {
        sample_id: sorted(sample_rows, key=lambda row: int(row.get("candidate_index", 0)))
        for sample_id, sample_rows in sorted(grouped.items())
    }


def build_judge_prompt(sample_id: str, question: str, gold_final: str, candidates: list[dict]) -> str:
    candidate_blocks = []
    for candidate in candidates:
        candidate_blocks.append(
            "\n".join(
                [
                    f"Candidate ID: {candidate['candidate_id']}",
                    f"Candidate final answer: {candidate.get('candidate_final', '')}",
                    "Candidate reasoning:",
                    str(candidate.get("candidate_text", "")).strip(),
                ]
            )
        )

    return "\n\n".join(
        [
            "Judge the reasoning quality of multiple candidate solutions for the same math problem.",
            f"Sample ID: {sample_id}",
            f"Question: {question.strip()}",
            f"Gold final answer: {gold_final}",
            "Scoring priorities:",
            "1. Correct final answer matters, but process quality also matters.",
            "2. Among correct answers, prefer clear, faithful, complete, non-contradictory reasoning.",
            "3. Penalize unrelated continuations, prompt/code artifacts, malformed text, hidden contradictions, and unsupported jumps.",
            "4. Do not prefer a candidate merely because it is shorter.",
            "5. If all candidates are wrong, rank the one with the most useful partial reasoning highest.",
            "Candidates:",
            "\n\n---\n\n".join(candidate_blocks),
            "Return only valid JSON with this exact shape:",
            json.dumps(
                {
                    "best_candidate_id": "<candidate id>",
                    "ranking": ["<candidate id>", "<candidate id>"],
                    "scores": {"<candidate id>": 0.0},
                    "pairwise_preferences": [
                        {
                            "chosen": "<candidate id>",
                            "rejected": "<candidate id>",
                            "reason": "short reason",
                        }
                    ],
                    "notes": "short notes",
                },
                ensure_ascii=False,
                indent=2,
            ),
        ]
    )


def parse_judge_response(text: str, sample_id: str, expected_candidate_ids: list[str]) -> dict:
    payload = json.loads(_strip_json_fence(text))
    if not isinstance(payload, dict):
        raise ValueError("judge response must be a JSON object")

    expected = set(expected_candidate_ids)
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError("ranking must be a list")
    unknown_ranking_ids = set(ranking) - expected
    if unknown_ranking_ids:
        raise ValueError(f"unknown candidate id in ranking: {sorted(unknown_ranking_ids)[0]}")
    if set(ranking) != expected:
        raise ValueError("ranking must contain exactly the expected candidate ids")

    best_candidate_id = payload.get("best_candidate_id")
    if best_candidate_id not in expected:
        raise ValueError(f"unknown candidate id in best_candidate_id: {best_candidate_id}")

    scores = payload.get("scores")
    if not isinstance(scores, dict) or set(scores) != expected:
        raise ValueError("scores must contain exactly the expected candidate ids")
    normalized_scores = {}
    for candidate_id, score in scores.items():
        if candidate_id not in expected:
            raise ValueError(f"unknown candidate id in scores: {candidate_id}")
        value = float(score)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"score for {candidate_id} must be between 0 and 1")
        normalized_scores[candidate_id] = value

    preferences = payload.get("pairwise_preferences", [])
    if not isinstance(preferences, list):
        raise ValueError("pairwise_preferences must be a list")
    normalized_preferences = []
    for pref in preferences:
        chosen = pref.get("chosen")
        rejected = pref.get("rejected")
        if chosen not in expected:
            raise ValueError(f"unknown candidate id in pairwise preference: {chosen}")
        if rejected not in expected:
            raise ValueError(f"unknown candidate id in pairwise preference: {rejected}")
        if chosen == rejected:
            raise ValueError("pairwise preference chosen and rejected must differ")
        normalized_preferences.append(
            {
                "chosen": chosen,
                "rejected": rejected,
                "reason": str(pref.get("reason", "")).strip(),
            }
        )

    return {
        "sample_id": sample_id,
        "best_candidate_id": best_candidate_id,
        "ranking": ranking,
        "scores": normalized_scores,
        "pairwise_preferences": normalized_preferences,
        "notes": str(payload.get("notes", "")).strip(),
    }


def call_deepseek_judge(
    api_key: str,
    model: str,
    prompt: str,
    *,
    base_url: str = "https://api.deepseek.com",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout: int = 120,
    max_retries: int = 2,
) -> tuple[str, dict]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = base_url.rstrip("/") + "/chat/completions"

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                response_payload = json.loads(resp.read().decode("utf-8"))
            content = response_payload["choices"][0]["message"]["content"]
            return content, response_payload.get("usage", {})
        except (error.HTTPError, error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"DeepSeek judge request failed after {max_retries + 1} attempts: {last_error}")


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped
