import importlib.util
from pathlib import Path


def test_train_prm_v2_parser_arguments():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_prm_v2.py"
    spec = importlib.util.spec_from_file_location("train_prm_v2", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--preferences",
            "data/prm/prefs.jsonl",
            "--output-dir",
            "logs/prm/v2",
            "--candidates",
            "logs/candidates/candidates.jsonl",
            "--max-features",
            "4096",
            "--hidden-dim",
            "128",
            "--epochs",
            "4",
            "--batch-size",
            "32",
            "--learning-rate",
            "0.001",
            "--weight-decay",
            "0.0005",
            "--seed",
            "7",
            "--device",
            "cpu",
        ]
    )

    assert args.preferences == Path("data/prm/prefs.jsonl")
    assert args.output_dir == Path("logs/prm/v2")
    assert args.max_features == 4096
    assert args.hidden_dim == 128
    assert args.epochs == 4
