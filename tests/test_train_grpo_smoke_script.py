import importlib.util
from pathlib import Path
import json

import pytest

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
    assert args.prm_backend == "auto"
    assert args.output_dir == Path("logs/rl/smoke")
    assert args.limit == 512
    assert args.num_generations == 8
    assert args.generation_batch_size is None
    assert args.dynamic_sampling is False
    assert args.max_num_gen_batches == 4
    assert args.loss_type == "dapo"
    assert args.final_weight == 1.0
    assert args.prm_weight == 0.2
    assert args.python_verifier_weight == 0.0
    assert args.length_penalty_weight == 0.0
    assert args.length_penalty_start == 0
    assert args.length_penalty_max == 0
    assert module._resolve_clip_epsilons(args) == (0.2, 0.28)


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
            "--prm-backend",
            "skywork",
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
            "--python-verifier-weight",
            "0.4",
            "--python-verifier-alpha-step",
            "0.25",
            "--python-verifier-beta-pass-rate",
            "0.15",
            "--length-penalty-weight",
            "1.0",
            "--length-penalty-start",
            "96",
            "--length-penalty-max",
            "128",
            "--epsilon-low",
            "0.18",
            "--epsilon-high",
            "0.31",
            "--dynamic-sampling",
            "--max-num-gen-batches",
            "6",
            "--generation-batch-size",
            "32",
            "--loss-type",
            "bnpo",
        ]
    )

    assert args.limit == 16
    assert args.prm_backend == "skywork"
    assert args.num_generations == 4
    assert args.max_prompt_length == 256
    assert args.max_completion_length == 128
    assert args.prm_weight == 0.35
    assert args.prm_clip == 2.5
    assert args.python_verifier_weight == 0.4
    assert args.python_verifier_alpha_step == 0.25
    assert args.python_verifier_beta_pass_rate == 0.15
    assert args.length_penalty_weight == 1.0
    assert args.length_penalty_start == 96
    assert args.length_penalty_max == 128
    assert args.dynamic_sampling is True
    assert args.max_num_gen_batches == 6
    assert args.generation_batch_size == 32
    assert args.loss_type == "bnpo"
    assert module._resolve_clip_epsilons(args) == (0.18, 0.31)


def test_train_grpo_smoke_parser_keeps_legacy_symmetric_epsilon():
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
            "--epsilon",
            "0.17",
        ]
    )

    assert module._resolve_clip_epsilons(args) == (0.17, 0.17)


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


def test_completion_token_length_prefers_trl_completion_ids_over_retokenizing():
    module = _load_script_module("train_grpo_smoke.py")

    class ExplodingTokenizer:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("tokenizer fallback should not be used")

    length = module._completion_token_length(
        "decoded text",
        completion_ids=[11, 12, 13],
        tokenizer=ExplodingTokenizer(),
    )

    assert length == 3


def test_completion_token_length_falls_back_to_tokenizer_when_ids_are_missing():
    module = _load_script_module("train_grpo_smoke.py")

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            assert text == "decoded text"
            assert add_special_tokens is False
            return {"input_ids": [1, 2, 3, 4]}

    length = module._completion_token_length(
        "decoded text",
        completion_ids=None,
        tokenizer=FakeTokenizer(),
    )

    assert length == 4


def test_resolve_prm_backend_auto_prefers_mlp_when_v2_files_exist(tmp_path):
    module = _load_script_module("train_grpo_smoke.py")
    (tmp_path / "model.pt").write_bytes(b"weights")
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")

    assert module._resolve_prm_backend(tmp_path, backend="auto") == "mlp"


def test_resolve_prm_backend_auto_uses_skywork_for_hf_model_dir(tmp_path):
    module = _load_script_module("train_grpo_smoke.py")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    assert module._resolve_prm_backend(tmp_path, backend="auto") == "skywork"


def test_completion_step_prefixes_builds_incremental_prefixes():
    module = _load_script_module("train_grpo_smoke.py")

    prefixes = module._completion_step_prefixes("step one\n\nstep two\n")

    assert prefixes == ["step one", "step one\nstep two"]


def test_prepare_skywork_input_marks_each_newline_step_reward():
    module = _load_script_module("train_grpo_smoke.py")

    class FakeTokenizer:
        bos_token = "<s>"

        def encode(self, text):
            return [ord(char) for char in text]

    input_ids, reward_flags = module._prepare_skywork_input(
        "Q?",
        "step one\nstep two",
        tokenizer=FakeTokenizer(),
    )

    assert len(input_ids) == len(reward_flags)
    assert sum(reward_flags) == 2
    assert [input_ids[index] for index, flag in enumerate(reward_flags) if flag] == [ord("\n"), ord("\n")]


def test_extract_reward_scalar_reads_token_level_last_non_pad_score():
    import torch

    module = _load_script_module("train_grpo_smoke.py")

    class Output:
        scores = torch.tensor([[0.1, 0.2, 0.3, 9.9]])

    score = module._extract_reward_scalar(Output(), torch.tensor([[1, 1, 1, 0]]))

    assert score == pytest.approx(0.3)


def test_grpo_config_kwargs_filters_unsupported_trl_parameters():
    module = _load_script_module("train_grpo_smoke.py")

    class FakeGRPOConfig:
        def __init__(
            self,
            output_dir=None,
            max_completion_length=None,
            beta=0.0,
            loss_type="dapo",
            epsilon=0.2,
            epsilon_high=None,
            generation_batch_size=None,
            reward_weights=None,
        ):
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
            "--loss-type",
            "dapo",
            "--generation-batch-size",
            "32",
            "--epsilon-low",
            "0.19",
            "--epsilon-high",
            "0.29",
        ]
    )

    kwargs = module._build_grpo_config_kwargs(FakeGRPOConfig, args, bf16=False, reward_weights=[1.0, 0.0])

    assert kwargs == {
        "output_dir": "logs/rl/smoke",
        "max_completion_length": 128,
        "beta": 0.04,
        "loss_type": "dapo",
        "epsilon": 0.19,
        "epsilon_high": 0.29,
        "generation_batch_size": 32,
        "reward_weights": [1.0, 0.0],
    }


def test_final_correctness_recorder_returns_binary_final_rewards():
    module = _load_script_module("train_grpo_smoke.py")
    recorder = module.FinalCorrectnessRecorder()

    rewards = recorder(
        prompts=["p1", "p2"],
        completions=["We get \\boxed{4}.", "We get \\boxed{5}."],
        gold_final=["4", "4"],
    )

    assert rewards == [1.0, 0.0]
    assert recorder.last_values == [1.0, 0.0]


def test_select_tracked_lora_param_names_picks_lora_weights():
    module = _load_script_module("train_grpo_smoke.py")

    class FakeModel:
        def named_parameters(self):
            return [
                ("base_model.model.layers.0.attn.q_proj.weight", object()),
                ("base_model.model.layers.0.attn.q_proj.lora_A.weight", object()),
                ("base_model.model.layers.0.attn.q_proj.lora_B.weight", object()),
            ]

    names = module._select_tracked_lora_param_names(FakeModel(), limit=2)
    assert names == [
        "base_model.model.layers.0.attn.q_proj.lora_A.weight",
        "base_model.model.layers.0.attn.q_proj.lora_B.weight",
    ]


def test_write_adapter_diagnostics_sets_change_flags(tmp_path):
    module = _load_script_module("train_grpo_smoke.py")
    pre = {"x.lora_A.weight": {"sha256": "aaa", "norm": 1.0, "numel": 2}}
    post = {"x.lora_A.weight": {"sha256": "bbb", "norm": 1.2, "numel": 2}}
    saved = {"x.lora_A.weight": {"sha256": "bbb", "norm": 1.2, "numel": 2}}

    module._write_adapter_diagnostics(
        tmp_path,
        tracked_param_names=["x.lora_A.weight"],
        pre_snapshots=pre,
        post_train_snapshots=post,
        saved_snapshots=saved,
        grad_stats={"total_params": 10, "trainable_params": 2, "total_lora_params": 2, "trainable_lora_params": 2},
        optimizer_stats={"optimizer_params": 2, "optimizer_lora_params": 2, "trainable_lora_params": 2},
    )

    payload = json.loads((tmp_path / "adapter_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["tracked_param_names"] == ["x.lora_A.weight"]
    assert payload["any_changed_in_memory_after_train"] is True
    assert payload["all_saved_match_post_train_memory"] is True


def test_saved_param_key_candidates_adds_non_default_fallback():
    module = _load_script_module("train_grpo_smoke.py")

    keys = module._saved_param_key_candidates("a.b.lora_A.default.weight")

    assert keys == ["a.b.lora_A.default.weight", "a.b.lora_A.weight"]
