#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psrl.candidates import build_candidate_row, build_prompt, clean_candidate_text, iter_jsonl, write_jsonl
from psrl.config import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate model candidate solutions for GSM8K reranking.")
    parser.add_argument("--input", type=Path, default=Path("data/debug/gsm8k_train_debug.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("logs/candidates/gsm8k_train_debug_candidates.jsonl"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train/sft_baseline.yaml"))
    parser.add_argument("--model-name", default=None, help="Model name/path. Defaults to train config model_name.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train_cfg = load_yaml_config(args.train_config)
    model_name = args.model_name or train_cfg.get("model_name")
    if not model_name:
        raise SystemExit("No model name supplied and configs/train/sft_baseline.yaml has no model_name.")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    except ImportError as exc:
        raise SystemExit(
            "Missing generation dependency. Install torch and transformers on the remote GPU environment."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
    except OSError as exc:
        raise SystemExit(
            f"Cannot load model {model_name!r}. Provide a valid local model path or an accessible Hugging Face repo id."
        ) from exc
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_name)
    except OSError:
        pass
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    rows = []
    samples = list(iter_jsonl(args.input))
    if args.limit is not None:
        samples = samples[: args.limit]

    for sample_idx, sample in enumerate(samples, start=1):
        prompt = build_prompt(sample["question"])
        if hasattr(tokenizer, "apply_chat_template"):
            chat_inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            if hasattr(chat_inputs, "keys") and "input_ids" in chat_inputs:
                inputs = {key: chat_inputs[key] for key in chat_inputs.keys()}
            else:
                inputs = {"input_ids": chat_inputs}
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_candidates,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[-1]
        for idx, output_ids in enumerate(outputs, start=1):
            candidate_text = clean_candidate_text(tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True))
            rows.append(build_candidate_row(sample, idx, candidate_text))
        print(f"Generated candidates for {sample_idx}/{len(samples)}: {sample.get('id', sample_idx)}", flush=True)

    count = write_jsonl(rows, args.output)
    print(f"Generated {count} candidates -> {args.output}")


if __name__ == "__main__":
    main()
