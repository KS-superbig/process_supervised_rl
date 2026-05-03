from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from typing import Iterable

from psrl.reward.aggregator import combine_rewards
from psrl.reward.final_reward import compute_final_reward
from psrl.reward.process_reward_v0 import score_process_steps


@dataclass
class SelectionReport:
    summary: dict[str, float | int]
    markdown: str
    changed_cases: list[dict]


def score_candidate_rows(rows: Iterable[dict], reward_config: dict) -> list[dict]:
    component_weights = reward_config.get("components", {})
    final_w = float(reward_config.get("final_reward_weight", 1.0))
    process_w = float(reward_config.get("process_reward_weight", 0.0))
    scored = []

    for idx, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id", row.get("id", f"sample-{idx:06d}"))
        candidate_id = row.get("candidate_id", f"{sample_id}-cand-{idx:02d}")
        candidate_index = int(row.get("candidate_index", idx))
        gold_final = row.get("gold_final", row.get("answer_final_normalized", ""))
        predicted_final = row.get("candidate_final", "")
        predicted_steps = row.get("candidate_steps", [])

        final_reward = compute_final_reward(gold_final, predicted_final)
        process_result = score_process_steps(predicted_steps, component_weights)
        sample_reward = combine_rewards(
            final_reward=final_reward,
            process_reward=process_result.score,
            final_reward_weight=final_w,
            process_reward_weight=process_w,
        )

        flags = []
        if process_result.component_means["anti_hacking_penalty"] > 0.6:
            flags.append("high_hacking_penalty")
        if process_result.component_means["progress_contribution"] < 0.2 and len(predicted_steps) >= 4:
            flags.append("low_progress_signal")

        scored.append(
            {
                "id": candidate_id,
                "sample_id": sample_id,
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "question": row.get("question", ""),
                "gold_final": gold_final,
                "candidate_final": predicted_final,
                "candidate_text": row.get("candidate_text", ""),
                "num_steps": len(predicted_steps),
                "final_reward": sample_reward.final_reward,
                "process_reward": sample_reward.process_reward,
                "total_reward": sample_reward.total_reward,
                "component_means": process_result.component_means,
                "flags": flags,
            }
        )

    return scored


def build_selection_report(scored_rows: list[dict], max_changed_cases: int = 20) -> SelectionReport:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in scored_rows:
        grouped[row["sample_id"]].append(row)

    final_only_selected = []
    final_plus_selected = []
    changed_cases = []

    for sample_id in sorted(grouped):
        rows = sorted(grouped[sample_id], key=lambda row: int(row.get("candidate_index", 0)))
        final_only = max(rows, key=lambda row: (float(row["final_reward"]), -int(row["candidate_index"])))
        final_plus = max(rows, key=lambda row: (float(row["total_reward"]), -int(row["candidate_index"])))
        final_only_selected.append(final_only)
        final_plus_selected.append(final_plus)

        if final_only["candidate_id"] != final_plus["candidate_id"]:
            changed_cases.append(
                {
                    "sample_id": sample_id,
                    "question": final_plus.get("question", ""),
                    "final_only_candidate_id": final_only["candidate_id"],
                    "final_plus_candidate_id": final_plus["candidate_id"],
                    "final_only_final_reward": final_only["final_reward"],
                    "final_plus_final_reward": final_plus["final_reward"],
                    "final_only_process_reward": final_only["process_reward"],
                    "final_plus_process_reward": final_plus["process_reward"],
                    "final_only_total_reward": final_only["total_reward"],
                    "final_plus_total_reward": final_plus["total_reward"],
                }
            )

    summary = {
        "num_samples": len(grouped),
        "num_candidates": len(scored_rows),
        "final_only_accuracy": _mean([row["final_reward"] for row in final_only_selected]),
        "final_plus_process_accuracy": _mean([row["final_reward"] for row in final_plus_selected]),
        "final_only_process_reward_mean": _mean([row["process_reward"] for row in final_only_selected]),
        "final_plus_process_reward_mean": _mean([row["process_reward"] for row in final_plus_selected]),
        "changed_selection_count": len(changed_cases),
        "changed_selection_rate": len(changed_cases) / len(grouped) if grouped else 0.0,
    }
    markdown = _render_markdown(summary, scored_rows, changed_cases[:max_changed_cases])
    return SelectionReport(summary=summary, markdown=markdown, changed_cases=changed_cases)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _render_markdown(summary: dict, scored_rows: list[dict], changed_cases: list[dict]) -> str:
    process_scores = [row["process_reward"] for row in scored_rows]
    num_steps = [row["num_steps"] for row in scored_rows]
    flags = Counter(flag for row in scored_rows for flag in row.get("flags", []))

    lines = [
        "# final-only vs final+process candidate selection report",
        "",
        "## Summary",
        f"- samples: {summary['num_samples']}",
        f"- candidates: {summary['num_candidates']}",
        f"- final_only_accuracy: {summary['final_only_accuracy']:.4f}",
        f"- final_plus_process_accuracy: {summary['final_plus_process_accuracy']:.4f}",
        f"- final_only_selected_process_reward_mean: {summary['final_only_process_reward_mean']:.4f}",
        f"- final_plus_selected_process_reward_mean: {summary['final_plus_process_reward_mean']:.4f}",
        f"- changed_selection_count: {summary['changed_selection_count']}",
        f"- changed_selection_rate: {summary['changed_selection_rate']:.4f}",
        f"- all_candidate_process_reward_mean: {_mean(process_scores):.4f}",
        f"- corr(num_steps, process_reward): {_corr(num_steps, process_scores):.4f}",
        "",
        "## Flags",
    ]

    if flags:
        for flag, count in sorted(flags.items()):
            lines.append(f"- {flag}: {count}")
    else:
        lines.append("- no flags")

    lines.extend(["", "## Changed Top1 Cases"])
    if changed_cases:
        for case in changed_cases:
            lines.append(
                "- "
                f"{case['sample_id']}: "
                f"{case['final_only_candidate_id']} -> {case['final_plus_candidate_id']}; "
                f"final={case['final_only_final_reward']:.1f}->{case['final_plus_final_reward']:.1f}; "
                f"process={case['final_only_process_reward']:.4f}->{case['final_plus_process_reward']:.4f}"
            )
    else:
        lines.append("- no changed top1 cases")

    lines.extend(
        [
            "",
            "## First-pass Reading Guide",
            "- If final_plus_process_accuracy is lower than final_only_accuracy, process reward is hurting selection.",
            "- If accuracy is tied and selected process reward is higher, process reward is adding useful ranking signal.",
            "- Changed cases should be manually inspected before moving to training.",
        ]
    )
    return "\n".join(lines) + "\n"


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (vx * vy) ** 0.5
