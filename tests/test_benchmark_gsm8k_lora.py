import importlib.util
import json
from pathlib import Path


def _load_script_module(name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_benchmark_parser_accepts_multiple_model_specs():
    module = _load_script_module("benchmark_gsm8k_lora.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--input",
            "data/processed/gsm8k_test.jsonl",
            "--base-model",
            "models/base",
            "--model-spec",
            "base",
            "--model-spec",
            "sft=logs/sft/final",
            "--output-dir",
            "logs/benchmark",
            "--limit",
            "16",
        ]
    )

    assert args.input == Path("data/processed/gsm8k_test.jsonl")
    assert args.base_model == "models/base"
    assert args.model_spec == ["base", "sft=logs/sft/final"]
    assert args.output_dir == Path("logs/benchmark")
    assert args.limit == 16


def test_parse_model_spec_supports_base_and_adapter():
    module = _load_script_module("benchmark_gsm8k_lora.py")

    assert module.parse_model_spec("base") == ("base", None)
    assert module.parse_model_spec("clean=logs/sft/clean/final") == ("clean", Path("logs/sft/clean/final"))


def test_build_summary_counts_accuracy_and_spacing(tmp_path):
    module = _load_script_module("benchmark_gsm8k_lora.py")
    path = tmp_path / "predictions.jsonl"
    rows = [
        {"is_correct": True, "generated_text": "one two three", "generated_tokens": 3, "truncated": False},
        {"is_correct": False, "generated_text": "onetwothree", "generated_tokens": 2, "truncated": True},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = module.build_summary("clean", path, elapsed_sec=12.34)

    assert summary["name"] == "clean"
    assert summary["correct"] == 1
    assert summary["total"] == 2
    assert summary["acc"] == 0.5
    assert summary["avg_generated_tokens"] == 2.5
    assert summary["truncated_rate"] == 0.5
    assert summary["mean_spaces"] == 1.0
