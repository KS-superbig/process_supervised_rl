import sys
import types

import pytest

from psrl.rl.dapo_grpo_trainer import (
    _set_num_items_in_batch,
    build_dynamic_sampling_group_mask,
    expand_group_mask,
    get_grpo_trainer_classes,
)


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


def test_build_dynamic_sampling_group_mask_keeps_only_mixed_correctness_groups():
    group_mask, stats = build_dynamic_sampling_group_mask(
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
        num_generations=4,
    )

    assert group_mask == [False, True, False]
    assert stats.total_groups == 3
    assert stats.kept_groups == 1
    assert stats.all_wrong_groups == 1
    assert stats.all_correct_groups == 1


def test_expand_group_mask_repeats_each_group_for_generation_rows():
    assert expand_group_mask([False, True, False], num_generations=3) == [
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_set_num_items_in_batch_uses_gathered_completion_token_count():
    class FakeScalar:
        def __init__(self, value):
            self.value = value

        def sum(self):
            return self.value

        def __eq__(self, other):
            return self.value == other

    class FakeMask:
        def __init__(self, count):
            self.count = count

        def sum(self):
            return FakeScalar(self.count)

    class FakeAccelerator:
        def gather(self, token_count):
            return FakeScalar(token_count.value * 2)

    batch = {"completion_mask": FakeMask(7)}

    _set_num_items_in_batch(batch, accelerator=FakeAccelerator())

    assert batch["num_items_in_batch"] == 14
