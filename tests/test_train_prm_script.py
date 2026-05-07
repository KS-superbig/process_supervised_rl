import importlib.util
from pathlib import Path


def test_train_prm_parser_exposes_smoke_training_arguments():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_prm.py"
    spec = importlib.util.spec_from_file_location("train_prm", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--preferences",
            "data/prm/preferences.jsonl",
            "--output-dir",
            "logs/prm/smoke",
            "--candidates",
            "logs/candidates/candidates.jsonl",
        ]
    )

    assert args.preferences == Path("data/prm/preferences.jsonl")
    assert args.output_dir == Path("logs/prm/smoke")
    assert args.candidates == Path("logs/candidates/candidates.jsonl")
