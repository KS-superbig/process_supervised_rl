import sys
import types

import pytest

from psrl.rl.dapo_grpo_trainer import get_grpo_trainer_classes


def test_get_grpo_trainer_classes_returns_project_owned_subclass(monkeypatch):
    class FakeGRPOConfig:
        def __init__(self, epsilon=0.2, epsilon_high=None):
            pass

    class FakeGRPOTrainer:
        def marker(self):
            return "epsilon_low epsilon_high"

    fake_trl = types.SimpleNamespace(GRPOConfig=FakeGRPOConfig, GRPOTrainer=FakeGRPOTrainer)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    config_cls, trainer_cls = get_grpo_trainer_classes()

    assert config_cls is FakeGRPOConfig
    assert issubclass(trainer_cls, FakeGRPOTrainer)
    assert trainer_cls.__name__ == "PSRLDAPOGRPOTrainer"


def test_get_grpo_trainer_classes_rejects_trl_without_clip_higher(monkeypatch):
    class FakeGRPOConfig:
        def __init__(self, epsilon=0.2):
            pass

    class FakeGRPOTrainer:
        pass

    fake_trl = types.SimpleNamespace(GRPOConfig=FakeGRPOConfig, GRPOTrainer=FakeGRPOTrainer)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    with pytest.raises(RuntimeError, match="epsilon_high"):
        get_grpo_trainer_classes()
