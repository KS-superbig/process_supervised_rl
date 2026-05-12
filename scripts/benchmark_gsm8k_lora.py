#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psrl.candidates import build_prompt, clean_candidate_text, extract_candidate_final, iter_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate base and LoRA adapters on a GSM8K JSONL subset.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/gsm8k_test.jsonl"))
    parser.add_argument("--base-model", required=True, help="Base model path or Hugging Face id.")
    parser.add_argument(
        "--model-spec",
        action="append",
        required=True,
        help="Model to evaluate. Use 'base' or 'name=/path/to/adapter'. Repeat for multiple models.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--progress-every", type=int, default=16)
    return parser


def parse_model_spec(spec: str) -> tuple[str, Path | None]:
    if spec == "base":
        return "base", None
    if "=" not in spec:
        raise ValueError(f"Invalid model spec {spec!r}. Expected 'base' or 'name=/adapter/path'.")
    name, adapter = spec.split("=", 1)
    name = name.strip()
    adapter = adapter.strip()
    if not name or not adapter:
        raise ValueError(f"Invalid model spec {spec!r}. Expected non-empty name and adapter path.")
    return name, Path(adapter)


def answer_is_correct(generated_text: str, gold_final: str) -> tuple[bool, str]:
    pred = extract_candidate_final(generated_text)
    return pred == str(gold_final).strip(), pred


def build_summary(name: str, predictions_path: Path, elapsed_sec: float) -> dict:
    rows = list(iter_jsonl(predictions_path))
    total = len(rows)
    correct = sum(1 for row in rows if row.get("is_correct"))
    token_counts = [int(row.get("generated_tokens", 0)) for row in rows]
    truncated = [bool(row.get("truncated")) for row in rows]
    space_counts = [str(row.get("generated_text", "")).count(" ") for row in rows]
    return {
        "name": name,
        "correct": correct,
        "total": total,
        "acc": correct / total if total else 0.0,
        "avg_generated_tokens": sum(token_counts) / total if total else 0.0,
        "truncated_rate": sum(truncated) / total if total else 0.0,
        "mean_spaces": sum(space_counts) / total if total else 0.0,
        "elapsed_sec": round(elapsed_sec, 2),
        "predictions_path": str(predictions_path),
    }


def load_model(base_model: str, adapter_path: Path | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    try:
        model.generation_config = GenerationConfig.from_pretrained(base_model)
    except OSError:
        pass
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model


def encode_prompt(tokenizer, question: str):
    prompt = build_prompt(question)
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if hasattr(encoded, "keys") and "input_ids" in encoded:
            return {key: encoded[key] for key in encoded.keys()}
        return {"input_ids": encoded}
    return tokenizer(prompt, return_tensors="pt")


def evaluate_model(
    *,
    name: str,
    base_model: str,
    adapter_path: Path | None,
    samples: list[dict],
    output_dir: Path,
    max_new_tokens: int,
    progress_every: int,
) -> dict:
    import torch

    tokenizer, model = load_model(base_model, adapter_path)
    predictions_path = output_dir / f"{name}.jsonl"
    rows = []
    started = time.time()
    with torch.inference_mode():
        for idx, sample in enumerate(samples, start=1):
            inputs = encode_prompt(tokenizer, sample["question"])
            if hasattr(model, "device"):
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][prompt_len:]
            generated_text = clean_candidate_text(tokenizer.decode(generated_ids, skip_special_tokens=True))
            gold_final = sample.get("answer_final_normalized", sample.get("gold_final", ""))
            is_correct, pred_final = answer_is_correct(generated_text, gold_final)
            rows.append(
                {
                    "index": idx - 1,
                    "sample_id": sample.get("id", sample.get("sample_id", f"sample-{idx}")),
                    "question": sample.get("question", ""),
                    "gold_final": gold_final,
                    "pred_final": pred_final,
                    "is_correct": is_correct,
                    "generated_tokens": int(generated_ids.numel()),
                    "truncated": int(generated_ids.numel()) >= max_new_tokens,
                    "generated_text": generated_text,
                }
            )
            if progress_every and idx % progress_every == 0:
                print(f"{name} progress {idx}/{len(samples)}", flush=True)

    write_jsonl(rows, predictions_path)
    summary = build_summary(name, predictions_path, time.time() - started)
    print(summary, flush=True)
    return summary


def write_report(summaries: list[dict], path: Path) -> None:
    lines = [
        "# GSM8K Benchmark",
        "",
        "| model | accuracy | correct/total | avg_generated_tokens | truncated_rate | mean_spaces | elapsed_sec |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {name} | {acc:.4f} | {correct}/{total} | {avg_generated_tokens:.2f} | "
            "{truncated_rate:.4f} | {mean_spaces:.2f} | {elapsed_sec:.2f} |".format(**item)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = list(iter_jsonl(args.input))
    if args.limit is not None:
        samples = samples[: args.limit]

    summaries = []
    for spec in args.model_spec:
        name, adapter_path = parse_model_spec(spec)
        summaries.append(
            evaluate_model(
                name=name,
                base_model=args.base_model,
                adapter_path=adapter_path,
                samples=samples,
                output_dir=args.output_dir,
                max_new_tokens=args.max_new_tokens,
                progress_every=args.progress_every,
            )
        )
    write_jsonl(summaries, args.output_dir / "results.jsonl")
    write_report(summaries, args.output_dir / "report.md")
    print(f"done -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
