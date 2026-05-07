#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psrl.candidates import iter_jsonl
from psrl.llm_judge import build_judge_prompt, call_deepseek_judge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay v2 new changed-top1 samples and optionally run DeepSeek pairwise audit.")
    parser.add_argument("--baseline-scored", type=Path, required=True, help="Baseline scored JSONL path.")
    parser.add_argument("--new-scored", type=Path, required=True, help="New scored JSONL path.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output replay sample JSONL path.")
    parser.add_argument("--output-md", type=Path, required=True, help="Output markdown summary path.")
    parser.add_argument("--sample-size", type=int, default=20, help="Max sampled rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--run-audit", action="store_true", help="Run DeepSeek pairwise audit for sampled rows.")
    parser.add_argument("--audit-model", default="deepseek-v4-flash", help="DeepSeek model for pairwise audit.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    baseline_rows = list(iter_jsonl(args.baseline_scored))
    new_rows = list(iter_jsonl(args.new_scored))
    baseline_sel, final_only_sel = _select_by_sample(baseline_rows)
    new_sel, _ = _select_by_sample(new_rows)

    new_changed = []
    for sample_id in sorted(new_sel):
        fo = final_only_sel[sample_id]
        base = baseline_sel[sample_id]
        new = new_sel[sample_id]
        if new["candidate_id"] != fo["candidate_id"] and new["candidate_id"] != base["candidate_id"]:
            new_changed.append(_build_replay_row(sample_id, fo, base, new))

    random.Random(args.seed).shuffle(new_changed)
    sampled = new_changed[: args.sample_size]

    if args.run_audit:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("Missing DEEPSEEK_API_KEY for --run-audit")
        for row in sampled:
            audit = _run_pairwise_audit(row, api_key=api_key, model=args.audit_model)
            row["audit"] = audit

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_lines = [
        "# V2 New Changed-Case Replay",
        "",
        f"- total_new_changed: {len(new_changed)}",
        f"- sampled_rows: {len(sampled)}",
        f"- baseline_scored: {args.baseline_scored}",
        f"- new_scored: {args.new_scored}",
        "",
        "## Sampled Cases",
    ]
    for row in sampled:
        line = (
            f"- {row['sample_id']}: final_only={row['final_only_candidate_id']} "
            f"baseline={row['baseline_candidate_id']} new={row['new_candidate_id']} "
            f"final={row['final_only_final_reward']:.1f}->{row['new_final_reward']:.1f}"
        )
        if "audit" in row:
            line += f" audit_best={row['audit'].get('best_candidate_id', '')}"
        summary_lines.append(line)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Wrote replay rows -> {args.output_jsonl}")
    print(f"Wrote replay summary -> {args.output_md}")
    print("\n".join(summary_lines[:12]))


def _select_by_sample(rows: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    selected = {}
    final_only = {}
    for sample_id, sample_rows in grouped.items():
        sample_rows = sorted(sample_rows, key=lambda r: int(r.get("candidate_index", 0)))
        fo = max(sample_rows, key=lambda r: (float(r["final_reward"]), -int(r["candidate_index"])))
        new = max(sample_rows, key=lambda r: (float(r["final_reward"]), float(r["prm_score"]), -int(r["candidate_index"])))
        selected[sample_id] = new
        final_only[sample_id] = fo
    return selected, final_only


def _build_replay_row(sample_id: str, final_only: dict, baseline: dict, new: dict) -> dict:
    return {
        "sample_id": sample_id,
        "question": new.get("question", ""),
        "gold_final": new.get("gold_final", ""),
        "final_only_candidate_id": final_only["candidate_id"],
        "baseline_candidate_id": baseline["candidate_id"],
        "new_candidate_id": new["candidate_id"],
        "final_only_final_reward": float(final_only["final_reward"]),
        "new_final_reward": float(new["final_reward"]),
        "final_only_candidate_text": final_only.get("candidate_text", ""),
        "new_candidate_text": new.get("candidate_text", ""),
    }


def _run_pairwise_audit(row: dict, *, api_key: str, model: str) -> dict:
    candidates = [
        {
            "candidate_id": row["final_only_candidate_id"],
            "candidate_index": 1,
            "candidate_final": "",
            "candidate_text": row.get("final_only_candidate_text", ""),
        },
        {
            "candidate_id": row["new_candidate_id"],
            "candidate_index": 2,
            "candidate_final": "",
            "candidate_text": row.get("new_candidate_text", ""),
        },
    ]
    prompt = build_judge_prompt(
        sample_id=row["sample_id"],
        question=row.get("question", ""),
        gold_final=row.get("gold_final", ""),
        candidates=candidates,
    )
    raw, usage = call_deepseek_judge(
        api_key=api_key,
        model=model,
        prompt=prompt,
        max_tokens=2048,
        temperature=0.0,
    )
    parsed = _parse_lenient(raw, expected_ids=[c["candidate_id"] for c in candidates])
    parsed["usage"] = usage
    return parsed


def _parse_lenient(raw_text: str, *, expected_ids: list[str]) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    payload = json.loads(text)
    ranking = payload.get("ranking") if isinstance(payload.get("ranking"), list) else []
    ranking = [cid for cid in ranking if cid in expected_ids]
    for cid in expected_ids:
        if cid not in ranking:
            ranking.append(cid)
    best = payload.get("best_candidate_id")
    if best not in expected_ids:
        best = ranking[0]
    return {
        "best_candidate_id": best,
        "ranking": ranking,
        "notes": str(payload.get("notes", "")),
    }


if __name__ == "__main__":
    main()
