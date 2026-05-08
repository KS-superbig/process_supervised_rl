#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psrl.candidates import build_prompt, iter_jsonl
from psrl.prm_v2 import load_mlp_prm
from psrl.rl.grpo_rewards import RewardConfig, build_reward_breakdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a conservative GRPO smoke run with final + PRM rewards.")
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/processed/gsm8k_train.jsonl"))
    parser.add_argument("--model-name", required=True, help="Base model path or Hugging Face id.")
    parser.add_argument("--sft-adapter", type=Path, required=True, help="LoRA-SFT adapter used as policy init.")
    parser.add_argument("--prm-dir", type=Path, required=True, help="PRM v2 directory containing model.pt and meta.json.")
    parser.add_argument("--prm-calibration", type=Path, help="Scored candidate JSONL used to z-score PRM rewards.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512, help="Number of GSM8K train rows for smoke run.")
    parser.add_argument("--final-weight", type=float, default=1.0)
    parser.add_argument("--prm-weight", type=float, default=0.2)
    parser.add_argument("--prm-clip", type=float, default=3.0)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.04, help="KL coefficient passed to TRL GRPOConfig.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="Device for the lightweight PRM scorer.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing GRPO dependency. Install torch, transformers, peft, datasets, and trl on the remote GPU environment."
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_train_rows(args.train_jsonl, limit=args.limit)
    dataset = Dataset.from_list(rows)
    prm = load_mlp_prm(args.prm_dir, device=args.device)
    prm_mean, prm_std = _load_prm_calibration(args.prm_calibration)
    reward_config = RewardConfig(
        final_weight=args.final_weight,
        prm_weight=args.prm_weight,
        prm_mean=prm_mean,
        prm_std=prm_std,
        prm_clip=args.prm_clip,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(args.sft_adapter))
    model.config.use_cache = False
    tracked_param_names = _select_tracked_lora_param_names(model, limit=5)
    pre_snapshots = {name: _capture_param_snapshot(model, name) for name in tracked_param_names}
    for name in tracked_param_names:
        snap = pre_snapshots[name]
        print(f"[adapter-diag] pre-train {name}: sha256={snap['sha256']} norm={snap['norm']:.8f}", flush=True)

    def reward_func(prompts, completions, gold_final, question, **_kwargs):
        rewards = []
        for raw_question, completion, gold in zip(question, completions, gold_final):
            completion_text = _completion_to_text(completion)
            raw_prm_score = prm_score(prm, raw_question, completion_text, device=args.device)
            breakdown = build_reward_breakdown(
                gold_final=gold,
                completion_text=completion_text,
                raw_prm_score=raw_prm_score,
                config=reward_config,
            )
            rewards.append(breakdown.total_reward)
        return rewards

    training_args = GRPOConfig(**_build_grpo_config_kwargs(GRPOConfig, args, bf16=torch.cuda.is_available()))
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        train_dataset=dataset,
    )
    grad_stats = _collect_grad_stats(model)
    print(
        "[adapter-diag] grad-stats "
        f"trainable_params={grad_stats['trainable_params']} total_params={grad_stats['total_params']} "
        f"trainable_lora_params={grad_stats['trainable_lora_params']} total_lora_params={grad_stats['total_lora_params']}",
        flush=True,
    )
    trainer.create_optimizer()
    optimizer_stats = _collect_optimizer_stats(model, trainer.optimizer)
    print(
        "[adapter-diag] optimizer-stats "
        f"optimizer_params={optimizer_stats['optimizer_params']} "
        f"optimizer_lora_params={optimizer_stats['optimizer_lora_params']} "
        f"trainable_lora_params={optimizer_stats['trainable_lora_params']}",
        flush=True,
    )
    _write_run_manifest(args, reward_config, len(rows))
    trainer.train()
    post_train_snapshots = {name: _capture_param_snapshot(model, name) for name in tracked_param_names}
    for name in tracked_param_names:
        snap = post_train_snapshots[name]
        print(f"[adapter-diag] post-train {name}: sha256={snap['sha256']} norm={snap['norm']:.8f}", flush=True)
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))
    saved_snapshots = {name: _capture_saved_adapter_snapshot(args.output_dir / "final", name) for name in tracked_param_names}
    for name in tracked_param_names:
        snap = saved_snapshots[name]
        print(f"[adapter-diag] saved-adapter {name}: sha256={snap['sha256']} norm={snap['norm']:.8f}", flush=True)
    _write_adapter_diagnostics(
        args.output_dir,
        tracked_param_names=tracked_param_names,
        pre_snapshots=pre_snapshots,
        post_train_snapshots=post_train_snapshots,
        saved_snapshots=saved_snapshots,
        grad_stats=grad_stats,
        optimizer_stats=optimizer_stats,
    )


def prm_score(prm, question: str, completion: str, *, device: str) -> float:
    scored = prm.model(
        _tensor_for_prm(prm, question=question, completion=completion, device=device)
    ).squeeze(-1)
    return float(scored.detach().cpu().item())


def _tensor_for_prm(prm, *, question: str, completion: str, device: str):
    import torch

    return torch.tensor([prm.vectorize(question, completion)], dtype=torch.float32, device=device)


def _load_train_rows(path: Path, *, limit: int | None) -> list[dict]:
    rows = []
    for row in iter_jsonl(path):
        question = str(row.get("question", "")).strip()
        gold_final = str(row.get("answer_final_normalized", row.get("gold_final", ""))).strip()
        if question and gold_final:
            rows.append({"prompt": build_prompt(question), "question": question, "gold_final": gold_final})
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError(f"No usable GSM8K rows loaded from {path}")
    return rows


def _completion_to_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(completion)


def _build_grpo_config_kwargs(grpo_config_cls, args: argparse.Namespace, *, bf16: bool) -> dict:
    candidate_kwargs = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "beta": args.beta,
        "seed": args.seed,
        "bf16": bf16,
        "report_to": [],
        "logging_steps": 1,
        "save_steps": max(args.max_steps, 1),
    }
    supported = set(inspect.signature(grpo_config_cls.__init__).parameters)
    return {key: value for key, value in candidate_kwargs.items() if key in supported}


def _select_tracked_lora_param_names(model, *, limit: int = 5) -> list[str]:
    names = []
    for name, _ in model.named_parameters():
        if "lora_" in name and name.endswith("weight"):
            names.append(name)
    if not names:
        raise RuntimeError("No LoRA weight parameter found in model; cannot run adapter diagnostics.")
    return names[:limit]


def _capture_param_snapshot(model, param_name: str) -> dict[str, float | str]:
    param = dict(model.named_parameters()).get(param_name)
    if param is None:
        raise RuntimeError(f"Missing tracked parameter in model: {param_name}")
    tensor = param.detach().float().cpu()
    payload = tensor.numpy().tobytes()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "norm": float(tensor.norm().item()),
        "numel": int(tensor.numel()),
    }


def _capture_saved_adapter_snapshot(final_dir: Path, param_name: str) -> dict[str, float | str]:
    from safetensors.torch import load_file

    state_dict = load_file(str(final_dir / "adapter_model.safetensors"), device="cpu")
    tensor = None
    for key in _saved_param_key_candidates(param_name):
        tensor = state_dict.get(key)
        if tensor is not None:
            break
    if tensor is None:
        raise RuntimeError(f"Tracked parameter not present in saved adapter: {param_name}")
    t = tensor.detach().float().cpu()
    payload = t.numpy().tobytes()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "norm": float(t.norm().item()),
        "numel": int(t.numel()),
    }


def _saved_param_key_candidates(param_name: str) -> list[str]:
    candidates = [param_name]
    if ".default." in param_name:
        candidates.append(param_name.replace(".default.", "."))
    return candidates


def _collect_grad_stats(model) -> dict[str, int]:
    total_params = 0
    trainable_params = 0
    total_lora_params = 0
    trainable_lora_params = 0
    for name, param in model.named_parameters():
        numel = int(param.numel())
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
        if "lora_" in name:
            total_lora_params += numel
            if param.requires_grad:
                trainable_lora_params += numel
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_lora_params": total_lora_params,
        "trainable_lora_params": trainable_lora_params,
    }


def _collect_optimizer_stats(model, optimizer) -> dict[str, int]:
    id_to_name = {id(param): name for name, param in model.named_parameters()}
    id_to_numel = {id(param): int(param.numel()) for _, param in model.named_parameters()}
    optimizer_param_ids = set()
    optimizer_params = 0
    optimizer_lora_params = 0
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            pid = id(param)
            if pid in optimizer_param_ids:
                continue
            optimizer_param_ids.add(pid)
            optimizer_params += id_to_numel.get(pid, int(param.numel()))
            name = id_to_name.get(pid, "")
            if "lora_" in name:
                optimizer_lora_params += id_to_numel.get(pid, int(param.numel()))
    grad_stats = _collect_grad_stats(model)
    return {
        "optimizer_params": optimizer_params,
        "optimizer_lora_params": optimizer_lora_params,
        "trainable_lora_params": grad_stats["trainable_lora_params"],
    }


def _write_adapter_diagnostics(
    output_dir: Path,
    *,
    tracked_param_names: list[str],
    pre_snapshots: dict[str, dict[str, float | str]],
    post_train_snapshots: dict[str, dict[str, float | str]],
    saved_snapshots: dict[str, dict[str, float | str]],
    grad_stats: dict[str, int],
    optimizer_stats: dict[str, int],
) -> None:
    per_param = {}
    for name in tracked_param_names:
        pre = pre_snapshots[name]
        post = post_train_snapshots[name]
        saved = saved_snapshots[name]
        per_param[name] = {
            "pre_train": pre,
            "post_train": post,
            "saved_final_adapter": saved,
            "changed_in_memory_after_train": pre["sha256"] != post["sha256"],
            "saved_matches_post_train_memory": saved["sha256"] == post["sha256"],
        }
    any_changed = any(item["changed_in_memory_after_train"] for item in per_param.values())
    all_saved_match = all(item["saved_matches_post_train_memory"] for item in per_param.values())
    diagnostics = {
        "tracked_param_names": tracked_param_names,
        "grad_stats": grad_stats,
        "optimizer_stats": optimizer_stats,
        "per_param": per_param,
        "any_changed_in_memory_after_train": any_changed,
        "all_saved_match_post_train_memory": all_saved_match,
    }
    (output_dir / "adapter_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_prm_calibration(path: Path | None) -> tuple[float, float]:
    if path is None:
        return 0.0, 1.0
    scores = [float(row["prm_score"]) for row in iter_jsonl(path) if "prm_score" in row]
    if len(scores) < 2:
        return (scores[0], 1.0) if scores else (0.0, 1.0)
    return statistics.fmean(scores), statistics.pstdev(scores) or 1.0


def _write_run_manifest(args: argparse.Namespace, config: RewardConfig, num_rows: int) -> None:
    manifest = {
        "train_jsonl": str(args.train_jsonl),
        "model_name": args.model_name,
        "sft_adapter": str(args.sft_adapter),
        "prm_dir": str(args.prm_dir),
        "prm_calibration": str(args.prm_calibration) if args.prm_calibration else None,
        "num_rows": num_rows,
        "reward": {
            "version": "final_plus_prm_zscore_v1",
            "final_weight": config.final_weight,
            "prm_weight": config.prm_weight,
            "prm_mean": config.prm_mean,
            "prm_std": config.prm_std,
            "prm_clip": config.prm_clip,
        },
        "training": {
            "max_steps": args.max_steps,
            "num_generations": args.num_generations,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "seed": args.seed,
        },
        "diagnostics": {
            "adapter_diagnostics_path": str(args.output_dir / "adapter_diagnostics.json"),
        },
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
