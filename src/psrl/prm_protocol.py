from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import re
from typing import Iterable


@dataclass
class PRMEvalProtocolResult:
    summary: dict[str, float | int | dict]
    markdown: str


def build_eval_protocol_report(
    scored_rows: Iterable[dict],
    judgements: Iterable[dict],
    *,
    audit_rows: Iterable[dict] | None = None,
) -> PRMEvalProtocolResult:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in scored_rows:
        grouped[str(row["sample_id"])].append(row)

    judgement_by_sample = {str(row["sample_id"]): row for row in judgements}
    audit_by_sample = {str(row["sample_id"]): row for row in (audit_rows or [])}

    changed = 0
    changed_breakdown: Counter[str] = Counter()
    judge_agree_final_only = 0
    judge_agree_prm = 0
    judge_agree_neither = 0
    audit_agree_final_only = 0
    audit_agree_prm = 0

    for sample_id in sorted(grouped):
        rows = sorted(grouped[sample_id], key=lambda r: int(r.get("candidate_index", 0)))
        final_only = max(rows, key=lambda r: (float(r["final_reward"]), -int(r["candidate_index"])))
        final_prm = max(rows, key=lambda r: (float(r["final_reward"]), float(r["prm_score"]), -int(r["candidate_index"])))
        judge_best = str(judgement_by_sample[sample_id]["best_candidate_id"]) if sample_id in judgement_by_sample else ""

        if judge_best == final_only["candidate_id"]:
            judge_agree_final_only += 1
        if judge_best == final_prm["candidate_id"]:
            judge_agree_prm += 1
        if judge_best and judge_best not in {final_only["candidate_id"], final_prm["candidate_id"]}:
            judge_agree_neither += 1

        if final_only["candidate_id"] != final_prm["candidate_id"]:
            changed += 1
            changed_breakdown[f"{int(final_only['final_reward'])}->{int(final_prm['final_reward'])}"] += 1
            audit = audit_by_sample.get(sample_id)
            if audit:
                if bool(audit.get("agrees_with_final_only")):
                    audit_agree_final_only += 1
                if bool(audit.get("agrees_with_prm")):
                    audit_agree_prm += 1

    num_samples = len(grouped)
    summary = {
        "num_samples": num_samples,
        "changed_count": changed,
        "changed_rate": (changed / num_samples) if num_samples else 0.0,
        "changed_breakdown": dict(sorted(changed_breakdown.items())),
        "judge_agree_final_only": judge_agree_final_only,
        "judge_agree_prm": judge_agree_prm,
        "judge_agree_neither": judge_agree_neither,
        "judge_agree_final_only_rate": (judge_agree_final_only / num_samples) if num_samples else 0.0,
        "judge_agree_prm_rate": (judge_agree_prm / num_samples) if num_samples else 0.0,
        "audit_changed_samples": len(audit_by_sample),
        "audit_agree_final_only": audit_agree_final_only,
        "audit_agree_prm": audit_agree_prm,
        "audit_agree_final_only_rate": (audit_agree_final_only / len(audit_by_sample)) if audit_by_sample else 0.0,
        "audit_agree_prm_rate": (audit_agree_prm / len(audit_by_sample)) if audit_by_sample else 0.0,
    }
    markdown = _render_protocol_markdown(summary)
    return PRMEvalProtocolResult(summary=summary, markdown=markdown)


def filter_preference_rows(
    preferences: Iterable[dict],
    *,
    judgements: Iterable[dict] | None = None,
    min_text_chars: int = 20,
    min_score_gap: float = 0.0,
    max_pairs_per_sample: int = 4,
) -> tuple[list[dict], dict[str, int | float]]:
    judgement_by_sample = {str(row["sample_id"]): row for row in (judgements or [])}
    dedup_seen = set()
    kept: dict[str, list[dict]] = defaultdict(list)
    stats = Counter()

    for row in preferences:
        stats["input_rows"] += 1
        sample_id = str(row["sample_id"])
        chosen = str(row.get("chosen_candidate_id", ""))
        rejected = str(row.get("rejected_candidate_id", ""))
        chosen_text = str(row.get("chosen_text", ""))
        rejected_text = str(row.get("rejected_text", ""))

        key = (sample_id, chosen, rejected)
        if key in dedup_seen:
            stats["drop_duplicate"] += 1
            continue
        dedup_seen.add(key)

        if chosen == rejected:
            stats["drop_same_candidate"] += 1
            continue
        if _normalize_text(chosen_text) == _normalize_text(rejected_text):
            stats["drop_same_text"] += 1
            continue
        if len(chosen_text.strip()) < min_text_chars or len(rejected_text.strip()) < min_text_chars:
            stats["drop_short_text"] += 1
            continue

        if min_score_gap > 0 and sample_id in judgement_by_sample:
            judge = judgement_by_sample[sample_id]
            scores = judge.get("scores", {}) if isinstance(judge.get("scores"), dict) else {}
            gap = float(scores.get(chosen, 0.0)) - float(scores.get(rejected, 0.0))
            if gap < min_score_gap:
                stats["drop_low_score_gap"] += 1
                continue

        kept[sample_id].append(row)

    output = []
    for sample_id in sorted(kept):
        rows = sorted(
            kept[sample_id],
            key=lambda row: (
                -_judge_score_gap(sample_id, row, judgement_by_sample),
                len(str(row.get("judge_reason", ""))),
            ),
        )
        output.extend(rows[:max_pairs_per_sample])
        if len(rows) > max_pairs_per_sample:
            stats["drop_over_cap"] += len(rows) - max_pairs_per_sample

    stats["output_rows"] = len(output)
    stats["output_samples"] = len({str(row["sample_id"]) for row in output})
    return output, dict(stats)


def _judge_score_gap(sample_id: str, row: dict, judgement_by_sample: dict[str, dict]) -> float:
    judge = judgement_by_sample.get(sample_id)
    if not judge:
        return 0.0
    scores = judge.get("scores", {}) if isinstance(judge.get("scores"), dict) else {}
    chosen = str(row.get("chosen_candidate_id", ""))
    rejected = str(row.get("rejected_candidate_id", ""))
    return float(scores.get(chosen, 0.0)) - float(scores.get(rejected, 0.0))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _render_protocol_markdown(summary: dict[str, int | float | dict]) -> str:
    lines = [
        "# PRM Evaluation Protocol",
        "",
        "## Core Metrics",
        f"- num_samples: {summary['num_samples']}",
        f"- changed_count: {summary['changed_count']}",
        f"- changed_rate: {float(summary['changed_rate']):.4f}",
        f"- changed_breakdown: {json.dumps(summary['changed_breakdown'], ensure_ascii=False)}",
        f"- judge_agree_final_only: {summary['judge_agree_final_only']}",
        f"- judge_agree_prm: {summary['judge_agree_prm']}",
        f"- judge_agree_neither: {summary['judge_agree_neither']}",
        f"- judge_agree_final_only_rate: {float(summary['judge_agree_final_only_rate']):.4f}",
        f"- judge_agree_prm_rate: {float(summary['judge_agree_prm_rate']):.4f}",
        "",
        "## Changed-Case Audit",
        f"- audit_changed_samples: {summary['audit_changed_samples']}",
        f"- audit_agree_final_only: {summary['audit_agree_final_only']}",
        f"- audit_agree_prm: {summary['audit_agree_prm']}",
        f"- audit_agree_final_only_rate: {float(summary['audit_agree_final_only_rate']):.4f}",
        f"- audit_agree_prm_rate: {float(summary['audit_agree_prm_rate']):.4f}",
    ]
    return "\n".join(lines) + "\n"
