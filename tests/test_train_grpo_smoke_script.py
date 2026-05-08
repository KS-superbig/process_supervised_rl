import importlib.util
from pathlib import Path
import json


def _load_script_module(name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_train_grpo_smoke_parser_defaults_to_rl_smoke_paths():
    module = _load_script_module("train_grpo_smoke.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-name",
            "/models/base",
            "--sft-adapter",
            "logs/sft/policy/final",
            "--prm-dir",
            "logs/prm_v2/best/trial_001",
            "--output-dir",
            "logs/rl/smoke",
        ]
    )

    assert args.train_jsonl == Path("data/processed/gsm8k_train.jsonl")
    assert args.model_name == "/models/base"
    assert args.sft_adapter == Path("logs/sft/policy/final")
    assert args.prm_dir == Path("logs/prm_v2/best/trial_001")
    assert args.output_dir == Path("logs/rl/smoke")
    assert args.limit == 512
    assert args.final_weight == 1.0
    assert args.prm_weight == 0.2


def test_train_grpo_smoke_parser_accepts_reward_and_generation_controls():
    module = _load_script_module("train_grpo_smoke.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-name",
            "/models/base",
            "--sft-adapter",
            "logs/sft/policy/final",
            "--prm-dir",
            "logs/prm_v2/best/trial_001",
            "--output-dir",
            "logs/rl/smoke",
            "--limit",
            "16",
            "--num-generations",
            "4",
            "--max-prompt-length",
            "256",
            "--max-completion-length",
            "128",
            "--prm-weight",
            "0.35",
            "--prm-clip",
            "2.5",
        ]
    )

    assert args.limit == 16
    assert args.num_generations == 4
    assert args.max_prompt_length == 256
    assert args.max_completion_length == 128
    assert args.prm_weight == 0.35
    assert args.prm_clip == 2.5


def test_load_train_rows_keeps_raw_question_for_prm_reward(tmp_path):
    module = _load_script_module("train_grpo_smoke.py")
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        '{"question": "What is 2 + 2?", "answer_final_normalized": "4"}\n',
        encoding="utf-8",
    )

    rows = module._load_train_rows(train_path, limit=1)

    assert rows == [
        {
            "prompt": "What is 2 + 2?\nPlease reason step by step, and put your final answer within \\boxed{}. Put each reasoning step on its own line.",
            "question": "What is 2 + 2?",
            "gold_final": "4",
        }
    ]


def test_completion_to_text_handles_chat_completion_shape():
    module = _load_script_module("train_grpo_smoke.py")

    text = module._completion_to_text([{"role": "assistant", "content": "Step 1\n#### 4"}])

    assert text == "Step 1\n#### 4"


def test_grpo_config_kwargs_filters_unsupported_trl_parameters():
    module = _load_script_module("train_grpo_smoke.py")

    class FakeGRPOConfig:
        def __init__(self, output_dir=None, max_completion_length=None, beta=0.0):
            pass

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model-name",
            "/models/base",
            "--sft-adapter",
            "logs/sft/policy/final",
            "--prm-dir",
            "logs/prm_v2/best/trial_001",
            "--output-dir",
            "logs/rl/smoke",
            "--max-prompt-length",
            "256",
            "--max-completion-length",
            "128",
            "--beta",
            "0.04",
        ]
    )

    kwargs = module._build_grpo_config_kwargs(FakeGRPOConfig, args, bf16=False)

    assert kwargs == {
        "output_dir": "logs/rl/smoke",
        "max_completion_length": 128,
        "beta": 0.04,
    }


def test_select_tracked_lora_param_name_picks_lora_weight():
    module = _load_script_module("train_grpo_smoke.py")

    class FakeModel:
        def named_parameters(self):
            return [
                ("base_model.model.layers.0.attn.q_proj.weight", object()),
                ("base_model.model.layers.0.attn.q_proj.lora_A.weight", object()),
            ]

    name = module._select_tracked_lora_param_name(FakeModel())
    assert name == "base_model.model.layers.0.attn.q_proj.lora_A.weight"


def test_write_adapter_diagnostics_sets_change_flags(tmp_path):
    module = _load_script_module("train_grpo_smoke.py")
    pre = {"sha256": "aaa", "norm": 1.0, "numel": 2}
    post = {"sha256": "bbb", "norm": 1.2, "numel": 2}
    saved = {"sha256": "bbb", "norm": 1.2, "numel": 2}

    module._write_adapter_diagnostics(
        tmp_path,
        tracked_param_name="x.lora_A.weight",
        pre_snapshot=pre,
        post_train_snapshot=post,
        saved_snapshot=saved,
    )

    payload = json.loads((tmp_path / "adapter_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["tracked_param_name"] == "x.lora_A.weight"
    assert payload["changed_in_memory_after_train"] is True
    assert payload["saved_matches_post_train_memory"] is True
