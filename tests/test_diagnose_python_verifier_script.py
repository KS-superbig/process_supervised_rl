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


def test_diagnose_accepts_existing_benchmark_field_names(tmp_path):
    module = _load_script_module("diagnose_python_verifier.py")
    input_path = tmp_path / "generations.jsonl"
    output_path = tmp_path / "summary.json"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"completion": "2 + 2 = 4", "correct": 1}),
                json.dumps({"completion": "2 + 2 = 5", "correct": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = module.diagnose(input_path, output_path)

    assert summary["n_correct"] == 1
    assert summary["n_wrong"] == 1
    assert summary["step_mean_gap"] == 1.0
    assert summary["pass_rate_gap"] == 1.0
    assert json.loads(output_path.read_text(encoding="utf-8")) == summary
