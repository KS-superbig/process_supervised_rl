from pathlib import Path

from psrl.config import load_yaml_config


def test_load_yaml_config_reads_mapping():
    config = load_yaml_config(Path("configs/data/gsm8k.yaml"))
    assert config["dataset_name"] == "gsm8k"
    assert config["debug_limit"] == 100
