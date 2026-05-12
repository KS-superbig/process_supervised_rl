#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA SFT from chat-message JSONL with assistant-only loss.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            if not _valid_messages(messages):
                continue
            rows.append({"messages": messages})
    if len(rows) < 64:
        raise RuntimeError(f"Not enough rows for SFT: {len(rows)}")
    return rows


def _valid_messages(messages) -> bool:
    if not isinstance(messages, list) or len(messages) != 2:
        return False
    return messages[0].get("role") == "user" and messages[1].get("role") == "assistant"


def mask_user_tokens(input_ids: list[int], assistant_start: int) -> list[int]:
    assistant_start = max(0, min(assistant_start, len(input_ids)))
    return [-100] * assistant_start + input_ids[assistant_start:]


def format_example(tokenizer, messages: list[dict], max_length: int) -> dict:
    user_messages = [messages[0]]
    full_messages = _messages_with_trainable_space_markers(tokenizer, messages)
    full_text = _apply_chat_template(tokenizer, full_messages, add_generation_prompt=False)
    prefix_text = _apply_chat_template(tokenizer, user_messages, add_generation_prompt=True)

    full_ids = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)["input_ids"]
    prefix_ids = tokenizer(prefix_text, truncation=True, max_length=max_length, padding=False)["input_ids"]
    assistant_start = min(len(prefix_ids), len(full_ids))
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": mask_user_tokens(full_ids, assistant_start),
    }


def _messages_with_trainable_space_markers(tokenizer, messages: list[dict]) -> list[dict]:
    if not _tokenizer_drops_ascii_spaces(tokenizer):
        return messages
    marked = []
    for message in messages:
        if message.get("role") != "assistant":
            marked.append(message)
            continue
        marked.append({**message, "content": _mark_spaces(str(message.get("content", "")))})
    return marked


def _tokenizer_drops_ascii_spaces(tokenizer) -> bool:
    if getattr(tokenizer, "_psrl_drops_ascii_spaces", None) is not None:
        return bool(tokenizer._psrl_drops_ascii_spaces)
    marker_id = None
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        marker_id = tokenizer.convert_tokens_to_ids("Ġ")
    if marker_id is None:
        tokenizer._psrl_drops_ascii_spaces = False
        return False
    encoded = tokenizer("A B", truncation=True, max_length=16, padding=False)["input_ids"]
    decoded = tokenizer.decode(encoded, skip_special_tokens=False)
    tokenizer._psrl_drops_ascii_spaces = " " not in decoded and "Ġ" not in decoded
    return bool(tokenizer._psrl_drops_ascii_spaces)


def _mark_spaces(text: str) -> str:
    return str(text).replace(" ", "Ġ")


def _apply_chat_template(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    if add_generation_prompt:
        return f"User: {messages[0]['content'].strip()}\n\nAssistant:"
    return "\n\n".join(f"{m['role'].title()}: {m['content'].strip()}" for m in messages)


class AssistantOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
            batch["labels"].append(feature["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.train_jsonl)
    split = max(1, min(args.eval_size, len(rows) // 10))
    eval_rows = rows[:split]
    train_rows = rows[split:]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    def tokenize_row(row: dict) -> dict:
        return format_example(tokenizer, row["messages"], args.max_length)

    train_ds = Dataset.from_list(train_rows).map(tokenize_row, remove_columns=["messages"])
    eval_ds = Dataset.from_list(eval_rows).map(tokenize_row, remove_columns=["messages"])

    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=AssistantOnlyCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
