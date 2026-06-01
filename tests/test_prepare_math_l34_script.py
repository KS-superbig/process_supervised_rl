import importlib.util
from pathlib import Path


def _load_script_module(name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_math_l34_parser_defaults():
    module = _load_script_module("prepare_math_l34.py")
    args = module.build_parser().parse_args([])

    assert args.dataset == "hendrycks/competition_math"
    assert args.split == "train"
    assert args.levels == [3, 4]
    assert args.sft_output == Path("data/sft/math_l34_sft.jsonl")
    assert args.grpo_output == Path("data/processed/math_l34_train.jsonl")


def test_extract_boxed_answer_handles_nested_braces():
    module = _load_script_module("prepare_math_l34.py")

    assert module.extract_boxed_answer(r"Thus \boxed{\frac{1}{2}}.") == r"\frac{1}{2}"


def test_filter_math_rows_keeps_levels_three_and_four():
    module = _load_script_module("prepare_math_l34.py")
    rows = [
        {"problem": "p1", "solution": r"sol \boxed{7}", "level": "Level 2", "type": "Algebra"},
        {"problem": "p2", "solution": r"sol \boxed{x+1}", "level": "Level 3", "type": "Algebra"},
        {"problem": "p3", "solution": "last line answer", "level": "Level 4", "type": "Geometry"},
    ]

    kept = module.filter_math_rows(rows, levels={3, 4})

    assert [row["question"] for row in kept] == ["p2", "p3"]
    assert kept[0]["answer_final_normalized"] == "x+1"
    assert kept[1]["answer_final_normalized"] == "last line answer"
    assert kept[0]["metadata"]["level"] == 3
    assert kept[1]["metadata"]["type"] == "Geometry"


def test_to_sft_rows_uses_chat_messages():
    module = _load_script_module("prepare_math_l34.py")
    rows = [
        {
            "sample_id": "math-train-l3-000001",
            "question": "What is 1+1?",
            "solution_raw": r"We get \boxed{2}.",
            "metadata": {"level": 3},
        }
    ]

    sft_rows = module.to_sft_rows(rows)

    assert sft_rows[0]["messages"][0]["role"] == "user"
    assert "Please reason step by step" in sft_rows[0]["messages"][0]["content"]
    assert sft_rows[0]["messages"][1] == {"role": "assistant", "content": r"We get \boxed{2}."}
    assert sft_rows[0]["metadata"]["sample_id"] == "math-train-l3-000001"
