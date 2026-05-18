#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psrl.reward.python_verifier import python_verifier_reward


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Python verifier reward on benchmark generations.")
    parser.add_argument("input", type=Path, help="Generation JSONL path.")
    parser.add_argument("output", type=Path, help="Summary JSON output path.")
    parser.add_argument("--alpha-step", type=float, default=0.3)
    parser.add_argument("--beta-pass-rate", type=float, default=0.2)
    parser.add_argument("--gamma-no-eq-penalty", type=float, default=0.0)
    parser.add_argument("--scored-output", type=Path, help="Optional per-row scored JSONL output path.")
    return parser


def diagnose(
    generations_jsonl: Path,
    output_path: Path,
    *,
    alpha_step: float = 0.3,
    beta_pass_rate: float = 0.2,
    gamma_no_eq_penalty: float = 0.0,
    scored_output: Path | None = None,
) -> dict[str, Any]:
    correct_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []

    with generations_jsonl.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completion = _get_completion(row)
            final_correct = _get_final_correct(row)
            reward = python_verifier_reward(
                completion,
                final_correct,
                alpha_step=alpha_step,
                beta_pass_rate=beta_pass_rate,
                gamma_no_eq_penalty=gamma_no_eq_penalty,
            )
            scored = {
                "index": row.get("index", idx),
                "sample_id": row.get("sample_id"),
                "is_correct": final_correct,
                **reward,
            }
            scored_rows.append(scored)
            if final_correct:
                correct_rows.append(scored)
            else:
                wrong_rows.append(scored)

    summary = {
        "n": len(scored_rows),
        "n_correct": len(correct_rows),
        "n_wrong": len(wrong_rows),
        "accuracy": _mean([1.0 if row["is_correct"] else 0.0 for row in scored_rows]),
        "step_mean_gap": _mean_key(correct_rows, "r_step_mean") - _mean_key(wrong_rows, "r_step_mean"),
        "pass_rate_gap": _mean_key(correct_rows, "r_pass_rate") - _mean_key(wrong_rows, "r_pass_rate"),
        "reward_gap": _mean_key(correct_rows, "reward") - _mean_key(wrong_rows, "reward"),
        "correct_avg_step_mean": _mean_key(correct_rows, "r_step_mean"),
        "wrong_avg_step_mean": _mean_key(wrong_rows, "r_step_mean"),
        "correct_avg_pass_rate": _mean_key(correct_rows, "r_pass_rate"),
        "wrong_avg_pass_rate": _mean_key(wrong_rows, "r_pass_rate"),
        "correct_no_eq_rate": _no_eq_rate(correct_rows),
        "wrong_no_eq_rate": _no_eq_rate(wrong_rows),
        "avg_steps": _mean_key(scored_rows, "n_steps"),
        "avg_steps_with_eq": _mean_key(scored_rows, "n_steps_with_eq"),
        "config": {
            "alpha_step": alpha_step,
            "beta_pass_rate": beta_pass_rate,
            "gamma_no_eq_penalty": gamma_no_eq_penalty,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if scored_output is not None:
        _write_jsonl(scored_rows, scored_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def _get_completion(row: dict[str, Any]) -> str:
    for key in ("completion", "generated_text", "candidate_text", "answer"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise KeyError("generation row must contain completion, generated_text, candidate_text, or answer")


def _get_final_correct(row: dict[str, Any]) -> bool:
    for key in ("final_correct", "is_correct", "correct"):
        if key in row:
            value = row[key]
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes"}
            return bool(value)
    raise KeyError("generation row must contain final_correct, is_correct, or correct")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _mean_key(rows: list[dict[str, Any]], key: str) -> float:
    return _mean([float(row[key]) for row in rows])


def _no_eq_rate(rows: list[dict[str, Any]]) -> float:
    return _mean([1.0 if int(row["n_steps_with_eq"]) == 0 else 0.0 for row in rows])


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    diagnose(
        args.input,
        args.output,
        alpha_step=args.alpha_step,
        beta_pass_rate=args.beta_pass_rate,
        gamma_no_eq_penalty=args.gamma_no_eq_penalty,
        scored_output=args.scored_output,
    )


if __name__ == "__main__":
    main()
