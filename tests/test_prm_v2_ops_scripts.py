import importlib.util
from pathlib import Path


def _load_script_module(name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sweep_prm_v2_parser():
    module = _load_script_module("sweep_prm_v2.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--preferences",
            "data/prm/prefs.jsonl",
            "--candidates",
            "logs/candidates/candidates.jsonl",
            "--judgements",
            "logs/llm_judge/judged.jsonl",
            "--output-dir",
            "logs/prm/sweep_v2",
            "--target-judge-agree-rate",
            "0.74",
            "--target-final-accuracy",
            "0.93",
            "--max-trials",
            "12",
        ]
    )
    assert args.preferences == Path("data/prm/prefs.jsonl")
    assert args.output_dir == Path("logs/prm/sweep_v2")
    assert abs(args.target_judge_agree_rate - 0.74) < 1e-9
    assert args.max_trials == 12


def test_replay_changed_cases_parser():
    module = _load_script_module("replay_changed_cases.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--baseline-scored",
            "logs/prm/baseline/scored.jsonl",
            "--new-scored",
            "logs/prm/v2/scored.jsonl",
            "--output-jsonl",
            "logs/prm/v2/replay.jsonl",
            "--output-md",
            "logs/prm/v2/replay.md",
            "--sample-size",
            "16",
            "--seed",
            "7",
        ]
    )
    assert args.baseline_scored == Path("logs/prm/baseline/scored.jsonl")
    assert args.new_scored == Path("logs/prm/v2/scored.jsonl")
    assert args.sample_size == 16
