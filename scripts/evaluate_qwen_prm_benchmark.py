#!/usr/bin/env python3
"""Score benchmark generations with Qwen2.5-Math-PRM."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


def split_steps(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    return parts or [text.strip()]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"steps": 0, "mean": None, "min": None, "last": None}
    return {
        "steps": len(values),
        "mean": float(mean(values)),
        "min": float(min(values)),
        "last": float(values[-1]),
    }


def finite_values(rows: list[dict], key: str) -> list[float]:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return [float(v) for v in vals if math.isfinite(float(v))]


def group_stats(rows: list[dict], key: str) -> dict:
    correct = [r for r in rows if r.get("is_correct")]
    wrong = [r for r in rows if not r.get("is_correct")]
    cvals = finite_values(correct, key)
    wvals = finite_values(wrong, key)
    return {
        "correct_n": len(correct),
        "wrong_n": len(wrong),
        "correct_mean": mean(cvals) if cvals else None,
        "wrong_mean": mean(wvals) if wvals else None,
        "gap_correct_minus_wrong": (mean(cvals) - mean(wvals)) if cvals and wvals else None,
    }


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = config.eos_token_id
    config.use_cache = False
    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    return tokenizer, model


def score_one(tokenizer, model, step_sep_id: int, question: str, answer: str) -> list[float]:
    steps = split_steps(answer)
    messages = [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
    ]
    conversation = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    input_ids = tokenizer.encode(
        conversation,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
    probs = F.softmax(logits, dim=-1)
    token_mask = input_ids == step_sep_id
    return probs[0, token_mask[0], 1].float().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", action="append", required=True, help="name=path.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(args.model_path)
    step_sep_id = tokenizer.encode("<extra_0>")[0]

    summary = {}
    for spec in args.input:
        if "=" not in spec:
            raise ValueError(f"--input must be name=path, got {spec}")
        name, path_str = spec.split("=", 1)
        rows = []
        input_path = Path(path_str)
        output_path = out_dir / f"{name}.qwen_prm_scores.jsonl"
        with input_path.open() as f, output_path.open("w") as out:
            for idx, line in enumerate(f):
                if args.limit is not None and idx >= args.limit:
                    break
                row = json.loads(line)
                scores = score_one(
                    tokenizer,
                    model,
                    step_sep_id,
                    row["question"],
                    row["generated_text"],
                )
                score_summary = summarize(scores)
                scored = {
                    "index": row.get("index", idx),
                    "sample_id": row.get("sample_id"),
                    "is_correct": bool(row.get("is_correct")),
                    "pred_final": row.get("pred_final"),
                    "gold_final": row.get("gold_final"),
                    "generated_tokens": row.get("generated_tokens"),
                    "truncated": row.get("truncated"),
                    "qwen_prm_steps": score_summary["steps"],
                    "qwen_prm_mean": score_summary["mean"],
                    "qwen_prm_min": score_summary["min"],
                    "qwen_prm_last": score_summary["last"],
                    "qwen_prm_scores": scores,
                }
                rows.append(scored)
                out.write(json.dumps(scored, ensure_ascii=False) + "\n")
                if (idx + 1) % 16 == 0:
                    print(f"{name} progress {idx + 1}", flush=True)

        summary[name] = {
            "n": len(rows),
            "accuracy": mean([1.0 if r["is_correct"] else 0.0 for r in rows]) if rows else None,
            "mean_score_stats": group_stats(rows, "qwen_prm_mean"),
            "min_score_stats": group_stats(rows, "qwen_prm_min"),
            "last_score_stats": group_stats(rows, "qwen_prm_last"),
            "scores_path": str(output_path),
        }
        print(json.dumps({name: summary[name]}, ensure_ascii=False, indent=2), flush=True)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done -> {summary_path}")


if __name__ == "__main__":
    main()
