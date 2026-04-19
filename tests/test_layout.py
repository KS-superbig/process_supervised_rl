from pathlib import Path


def test_expected_project_directories_exist():
    for rel_path in [
        "configs",
        "data/raw",
        "data/interim",
        "data/processed",
        "data/debug",
        "experiments",
        "logs",
    ]:
        assert Path(rel_path).exists(), rel_path
