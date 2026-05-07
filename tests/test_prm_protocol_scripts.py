import importlib.util
from pathlib import Path


def _load_script_module(name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_filter_prm_preferences_parser():
    module = _load_script_module("filter_prm_preferences.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--preferences",
            "data/prm/prefs.jsonl",
            "--output",
            "data/prm/prefs_filtered.jsonl",
            "--stats-json",
            "logs/prm/filter_stats.json",
            "--min-text-chars",
            "30",
            "--min-score-gap",
            "0.1",
            "--max-pairs-per-sample",
            "3",
        ]
    )
    assert args.preferences == Path("data/prm/prefs.jsonl")
    assert args.min_text_chars == 30
    assert abs(args.min_score_gap - 0.1) < 1e-9
    assert args.max_pairs_per_sample == 3


def test_evaluate_prm_protocol_parser():
    module = _load_script_module("evaluate_prm_protocol.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--scored",
            "logs/prm/scored.jsonl",
            "--judgements",
            "logs/llm_judge/judged.jsonl",
            "--summary-json",
            "logs/prm/eval.json",
            "--summary-md",
            "logs/prm/eval.md",
        ]
    )
    assert args.scored == Path("logs/prm/scored.jsonl")
    assert args.judgements == Path("logs/llm_judge/judged.jsonl")
    assert args.summary_json == Path("logs/prm/eval.json")
