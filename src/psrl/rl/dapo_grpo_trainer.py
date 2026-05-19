from __future__ import annotations

import inspect
from typing import Any


class PSRLDAPOGRPOTrainerMixin:
    """Project-owned extension point for GRPO/DAPO algorithm changes.

    The installed TRL version used by this project already implements DAPO-style
    Clip-Higher through `GRPOConfig.epsilon` (low) and `epsilon_high` (high).
    Keeping this class in the repository gives us a stable place to override
    loss details later without editing site-packages on the remote machine.
    """


def get_grpo_trainer_classes() -> tuple[type[Any], type[Any]]:
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing TRL GRPO dependency. Install a TRL version with GRPOTrainer and GRPOConfig."
        ) from exc

    _validate_clip_higher_support(GRPOConfig, GRPOTrainer)

    class PSRLDAPOGRPOTrainer(PSRLDAPOGRPOTrainerMixin, GRPOTrainer):
        pass

    return GRPOConfig, PSRLDAPOGRPOTrainer


def _validate_clip_higher_support(grpo_config_cls: type[Any], grpo_trainer_cls: type[Any]) -> None:
    config_params = set(inspect.signature(grpo_config_cls.__init__).parameters)
    missing = {"epsilon", "epsilon_high"} - config_params
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise RuntimeError(f"Installed TRL GRPOConfig lacks Clip-Higher parameter(s): {missing_text}")

    trainer_source = _safe_getsource(grpo_trainer_cls)
    required_markers = ("epsilon_low", "epsilon_high")
    if trainer_source and not all(marker in trainer_source for marker in required_markers):
        raise RuntimeError("Installed TRL GRPOTrainer does not expose Clip-Higher epsilon bounds.")


def _safe_getsource(obj: Any) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return ""
