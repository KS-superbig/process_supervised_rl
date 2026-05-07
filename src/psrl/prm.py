from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
from typing import Iterable

from psrl.reward.final_reward import compute_final_reward


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class PRMSelectionReport:
    summary: dict[str, float | int]
    markdown: str
    changed_cases: list[dict]


@dataclass
class LinearPreferencePRM:
    weights: dict[str, float]
    bias: float = 0.0

    def score(self, question: str, candidate_text: str) -> float:
        features = featurize_solution(question, candidate_text)
        return self.bias + sum(self.weights.get(name, 0.0) * value for name, value in features.items())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "linear_bow_preference_prm",
            "bias": self.bias,
            "weights": dict(sorted(self.weights.items())),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinearPreferencePRM":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_type") != "linear_bow_preference_prm":
            raise ValueError(f"unsupported PRM model type: {payload.get('model_type')}")
        return cls(weights={str(k): float(v) for k, v in payload.get("weights", {}).items()}, bias=float(payload.get("bias", 0.0)))


def train_linear_preference_prm(
    preference_rows: Iterable[dict],
    *,
    epochs: int = 80,
    learning_rate: float = 0.05,
    l2: float = 0.0001,
) -> tuple[LinearPreferencePRM, dict[str, float | int]]:
    rows = list(preference_rows)
    if not rows:
        raise ValueError("preference_rows must not be empty")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")

    pair_features = []
    for row in rows:
        question = row.get("question", "")
        chosen = featurize_solution(question, row.get("chosen_text", ""))
        rejected = featurize_solution(question, row.get("rejected_text", ""))
        pair_features.append(_diff_features(chosen, rejected))

    weights: dict[str, float] = {}
    losses = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for diff in pair_features:
            margin = sum(weights.get(name, 0.0) * value for name, value in diff.items())
            epoch_loss += _softplus(-margin)
            grad_scale = -_sigmoid(-margin)
            for name, value in diff.items():
                current = weights.get(name, 0.0)
                grad = grad_scale * value + l2 * current
                updated = current - learning_rate * grad
                if abs(updated) > 1e-12:
                    weights[name] = updated
                elif name in weights:
                    del weights[name]
        losses.append(epoch_loss / len(pair_features))

    model = LinearPreferencePRM(weights=weights)
    metrics = {
        "num_pairs": len(pair_features),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "train_pair_accuracy": _pair_accuracy(model, rows),
        "weight_count": len(weights),
    }
    return model, metrics


def score_candidates_with_prm(candidate_rows: Iterable[dict], model: LinearPreferencePRM) -> list[dict]:
    scored = []
    for idx, row in enumerate(candidate_rows, start=1):
        sample_id = row.get("sample_id", row.get("id", f"sample-{idx:06d}"))
        candidate_id = row.get("candidate_id", f"{sample_id}-cand-{idx:02d}")
        candidate_index = int(row.get("candidate_index", idx))
        gold_final = row.get("gold_final", row.get("answer_final_normalized", ""))
        candidate_final = row.get("candidate_final", "")
        final_reward = compute_final_reward(gold_final, candidate_final)
        prm_score = model.score(row.get("question", ""), row.get("candidate_text", ""))
        scored.append(
            {
                **row,
                "id": candidate_id,
                "sample_id": sample_id,
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "gold_final": gold_final,
                "candidate_final": candidate_final,
                "final_reward": final_reward,
                "prm_score": prm_score,
            }
        )
    return scored


def build_prm_selection_report(scored_rows: list[dict], max_changed_cases: int = 20) -> PRMSelectionReport:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in scored_rows:
        grouped[row["sample_id"]].append(row)

    final_only_selected = []
    final_plus_prm_selected = []
    changed_cases = []

    for sample_id in sorted(grouped):
        rows = sorted(grouped[sample_id], key=lambda row: int(row.get("candidate_index", 0)))
        final_only = max(rows, key=lambda row: (float(row["final_reward"]), -int(row["candidate_index"])))
        final_plus_prm = max(rows, key=lambda row: (float(row["final_reward"]), float(row["prm_score"]), -int(row["candidate_index"])))
        final_only_selected.append(final_only)
        final_plus_prm_selected.append(final_plus_prm)

        if final_only["candidate_id"] != final_plus_prm["candidate_id"]:
            changed_cases.append(
                {
                    "sample_id": sample_id,
                    "question": final_plus_prm.get("question", ""),
                    "final_only_candidate_id": final_only["candidate_id"],
                    "final_plus_prm_candidate_id": final_plus_prm["candidate_id"],
                    "final_only_final_reward": final_only["final_reward"],
                    "final_plus_prm_final_reward": final_plus_prm["final_reward"],
                    "final_only_prm_score": final_only["prm_score"],
                    "final_plus_prm_score": final_plus_prm["prm_score"],
                }
            )

    summary = {
        "num_samples": len(grouped),
        "num_candidates": len(scored_rows),
        "final_only_accuracy": _mean([row["final_reward"] for row in final_only_selected]),
        "final_plus_prm_accuracy": _mean([row["final_reward"] for row in final_plus_prm_selected]),
        "final_only_prm_score_mean": _mean([row["prm_score"] for row in final_only_selected]),
        "final_plus_prm_score_mean": _mean([row["prm_score"] for row in final_plus_prm_selected]),
        "changed_selection_count": len(changed_cases),
        "changed_selection_rate": len(changed_cases) / len(grouped) if grouped else 0.0,
    }
    return PRMSelectionReport(
        summary=summary,
        markdown=_render_prm_markdown(summary, scored_rows, changed_cases[:max_changed_cases]),
        changed_cases=changed_cases,
    )


def featurize_solution(question: str, candidate_text: str) -> dict[str, float]:
    del question
    tokens = _tokenize(candidate_text)
    counts = Counter(tokens)
    length = max(1.0, math.sqrt(sum(counts.values())))
    features = {f"tok={token}": count / length for token, count in counts.items()}
    features["bias:length_log"] = math.log1p(len(candidate_text)) / 10.0
    features["bias:line_count"] = min(candidate_text.count("\n") + 1, 20) / 20.0
    features["bias:number_count"] = min(len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", candidate_text)), 30) / 30.0
    return features


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _diff_features(chosen: dict[str, float], rejected: dict[str, float]) -> dict[str, float]:
    names = set(chosen) | set(rejected)
    return {name: chosen.get(name, 0.0) - rejected.get(name, 0.0) for name in names}


def _pair_accuracy(model: LinearPreferencePRM, rows: list[dict]) -> float:
    correct = 0
    for row in rows:
        question = row.get("question", "")
        if model.score(question, row.get("chosen_text", "")) > model.score(question, row.get("rejected_text", "")):
            correct += 1
    return correct / len(rows) if rows else 0.0


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def _softplus(value: float) -> float:
    if value > 30:
        return value
    if value < -30:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def _render_prm_markdown(summary: dict, scored_rows: list[dict], changed_cases: list[dict]) -> str:
    prm_scores = [row["prm_score"] for row in scored_rows]
    lines = [
        "# final-only vs final+PRM candidate selection report",
        "",
        "## Summary",
        f"- samples: {summary['num_samples']}",
        f"- candidates: {summary['num_candidates']}",
        f"- final_only_accuracy: {summary['final_only_accuracy']:.4f}",
        f"- final_plus_prm_accuracy: {summary['final_plus_prm_accuracy']:.4f}",
        f"- final_only_selected_prm_score_mean: {summary['final_only_prm_score_mean']:.4f}",
        f"- final_plus_prm_selected_prm_score_mean: {summary['final_plus_prm_score_mean']:.4f}",
        f"- changed_selection_count: {summary['changed_selection_count']}",
        f"- changed_selection_rate: {summary['changed_selection_rate']:.4f}",
        f"- all_candidate_prm_score_mean: {_mean(prm_scores):.4f}",
        "",
        "## Changed Top1 Cases",
    ]
    if changed_cases:
        for case in changed_cases:
            lines.append(
                "- "
                f"{case['sample_id']}: "
                f"{case['final_only_candidate_id']} -> {case['final_plus_prm_candidate_id']}; "
                f"final={case['final_only_final_reward']:.1f}->{case['final_plus_prm_final_reward']:.1f}; "
                f"prm={case['final_only_prm_score']:.4f}->{case['final_plus_prm_score']:.4f}"
            )
    else:
        lines.append("- no changed top1 cases")
    return "\n".join(lines) + "\n"
