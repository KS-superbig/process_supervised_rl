import importlib.util
from pathlib import Path


def _load_script_module(name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_train_sft_lora_chat_parser():
    module = _load_script_module("train_sft_lora_chat.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--train-jsonl",
            "data/sft/train.jsonl",
            "--model-name",
            "models/base",
            "--output-dir",
            "logs/sft/chat",
            "--max-length",
            "768",
            "--epochs",
            "1.5",
        ]
    )

    assert args.train_jsonl == Path("data/sft/train.jsonl")
    assert args.model_name == "models/base"
    assert args.output_dir == Path("logs/sft/chat")
    assert args.max_length == 768
    assert args.epochs == 1.5


def test_mask_user_tokens_keeps_only_assistant_labels():
    module = _load_script_module("train_sft_lora_chat.py")
    input_ids = [11, 12, 13, 14, 15]
    labels = module.mask_user_tokens(input_ids, assistant_start=3)

    assert labels == [-100, -100, -100, 14, 15]
