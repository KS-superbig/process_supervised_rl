from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any


@dataclass(frozen=True)
class DynamicSamplingStats:
    total_groups: int
    kept_groups: int
    all_wrong_groups: int
    all_correct_groups: int
    partial_wrong_groups: int


class PSRLDAPOGRPOTrainerMixin:
    """Project-owned extension point for GRPO/DAPO algorithm changes.

    The installed TRL version used by this project already implements DAPO-style
    Clip-Higher through `GRPOConfig.epsilon` (low) and `epsilon_high` (high).
    Keeping this class in the repository gives us a stable place to override
    loss details later without editing site-packages on the remote machine.
    """

    dynamic_sampling_enabled: bool = False
    dynamic_sampling_max_gen_batches: int = 4
    dynamic_sampling_min_correct: int = 1
    dynamic_sampling_max_correct: int | None = None
    dynamic_sampling_correctness_source: Any = None

    def _prepare_inputs(self, inputs):
        if not self.dynamic_sampling_enabled:
            return super()._prepare_inputs(inputs)

        from trl.trainer.utils import (
            shuffle_sequence_dict,
            split_pixel_values_by_grid,
            split_tensor_dict,
            unsplit_pixel_values_by_grid,
        )

        mode = "train" if self.model.training else "eval"
        if mode != "train":
            return super()._prepare_inputs(inputs)

        generate_every = self.args.steps_per_generation * self.num_iterations
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            sampled_batch = self._generate_dynamic_sampling_batch(inputs)
            sampled_batch = split_pixel_values_by_grid(sampled_batch)
            sampled_batch = shuffle_sequence_dict(sampled_batch)
            generation_batches = split_tensor_dict(sampled_batch, self.args.steps_per_generation)
            self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]

        return self._buffered_inputs[self._step % self.args.steps_per_generation]

    def _generate_dynamic_sampling_batch(self, generation_batch):
        accepted_batches = []
        last_batch = None
        target_groups = None
        stats_totals = {
            "attempts": 0,
            "total_groups": 0,
            "kept_groups": 0,
            "all_wrong_groups": 0,
            "all_correct_groups": 0,
            "partial_wrong_groups": 0,
        }

        max_batches = max(1, int(self.dynamic_sampling_max_gen_batches))
        for _ in range(max_batches):
            generated = self._generate_and_score_completions(generation_batch)
            last_batch = generated
            num_generations = int(self.num_generations)
            rows = _batch_row_count(generated)
            if rows % num_generations != 0:
                raise RuntimeError(
                    f"Dynamic sampling expected rows to be divisible by num_generations: {rows} vs {num_generations}"
                )

            if target_groups is None:
                target_groups = rows // num_generations

            correctness = self._consume_dynamic_sampling_correctness(rows)
            group_mask, stats = build_dynamic_sampling_group_mask(
                correctness,
                num_generations=num_generations,
                min_correct=self.dynamic_sampling_min_correct,
                max_correct=self.dynamic_sampling_max_correct,
            )
            stats_totals["attempts"] += 1
            stats_totals["total_groups"] += stats.total_groups
            stats_totals["kept_groups"] += stats.kept_groups
            stats_totals["all_wrong_groups"] += stats.all_wrong_groups
            stats_totals["all_correct_groups"] += stats.all_correct_groups
            stats_totals["partial_wrong_groups"] += stats.partial_wrong_groups

            if stats.kept_groups:
                row_mask = expand_group_mask(group_mask, num_generations)
                accepted_batches.append(_filter_generation_batch(generated, row_mask, accelerator=self.accelerator))

            kept_so_far = sum(_batch_row_count(batch) // num_generations for batch in accepted_batches)
            if target_groups is not None and kept_so_far >= target_groups:
                break

        mode = "train" if self.model.training else "eval"
        self._log_dynamic_sampling_metrics(mode, stats_totals)

        if accepted_batches and target_groups is not None:
            merged = _concat_generation_batches(accepted_batches, accelerator=self.accelerator)
            return _truncate_generation_batch(merged, target_groups * int(self.num_generations), accelerator=self.accelerator)

        if last_batch is None:
            raise RuntimeError("Dynamic sampling did not produce any generation batch.")
        return last_batch

    def _consume_dynamic_sampling_correctness(self, expected_rows: int) -> list[float]:
        source = self.dynamic_sampling_correctness_source
        if source is None or not hasattr(source, "last_values"):
            raise RuntimeError("Dynamic sampling requires a correctness recorder with a last_values attribute.")

        values = list(source.last_values)
        if len(values) != expected_rows:
            raise RuntimeError(
                f"Dynamic sampling correctness length mismatch: got {len(values)}, expected {expected_rows}."
            )
        return [float(value) for value in values]

    def _log_dynamic_sampling_metrics(self, mode: str, totals: dict[str, int]) -> None:
        attempts = max(1, totals["attempts"])
        total_groups = max(1, totals["total_groups"])
        self._metrics[mode]["dynamic_sampling/num_gen_batches"].append(float(attempts))
        self._metrics[mode]["dynamic_sampling/keep_rate"].append(totals["kept_groups"] / total_groups)
        self._metrics[mode]["dynamic_sampling/all_wrong_rate"].append(totals["all_wrong_groups"] / total_groups)
        self._metrics[mode]["dynamic_sampling/all_correct_rate"].append(totals["all_correct_groups"] / total_groups)


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


def build_dynamic_sampling_group_mask(
    correctness: list[float],
    *,
    num_generations: int,
    min_correct: int = 1,
    max_correct: int | None = None,
) -> tuple[list[bool], DynamicSamplingStats]:
    if num_generations <= 0:
        raise ValueError("num_generations must be positive.")
    if len(correctness) % num_generations != 0:
        raise ValueError("correctness length must be divisible by num_generations.")

    max_correct = num_generations - 1 if max_correct is None else max_correct
    group_mask = []
    all_wrong = 0
    all_correct = 0
    partial_wrong = 0

    for start in range(0, len(correctness), num_generations):
        group = correctness[start : start + num_generations]
        correct_count = sum(1 for value in group if float(value) > 0.0)
        if correct_count == 0:
            all_wrong += 1
        elif correct_count == num_generations:
            all_correct += 1
        else:
            partial_wrong += 1
        group_mask.append(min_correct <= correct_count <= max_correct)

    kept = sum(1 for keep in group_mask if keep)
    stats = DynamicSamplingStats(
        total_groups=len(group_mask),
        kept_groups=kept,
        all_wrong_groups=all_wrong,
        all_correct_groups=all_correct,
        partial_wrong_groups=partial_wrong,
    )
    return group_mask, stats


def expand_group_mask(group_mask: list[bool], *, num_generations: int) -> list[bool]:
    if num_generations <= 0:
        raise ValueError("num_generations must be positive.")
    return [keep for keep in group_mask for _ in range(num_generations)]


def _batch_row_count(batch: dict[str, Any]) -> int:
    return int(batch["completion_ids"].shape[0])


def _filter_generation_batch(batch: dict[str, Any], row_mask: list[bool], *, accelerator: Any = None) -> dict[str, Any]:
    import torch

    mask = torch.tensor(row_mask, dtype=torch.bool, device=batch["completion_ids"].device)
    rows = len(row_mask)
    filtered: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "shape") and len(value.shape) > 0 and int(value.shape[0]) == rows:
            filtered[key] = value[mask]
        elif key != "num_items_in_batch":
            filtered[key] = value
    _set_num_items_in_batch(filtered, accelerator=accelerator)
    return filtered


def _concat_generation_batches(batches: list[dict[str, Any]], *, accelerator: Any = None) -> dict[str, Any]:
    import torch

    merged: dict[str, Any] = {}
    first = batches[0]
    for key, value in first.items():
        if key == "num_items_in_batch":
            continue
        if hasattr(value, "shape") and len(value.shape) > 0:
            merged[key] = torch.cat([batch[key] for batch in batches], dim=0)
        else:
            merged[key] = value
    _set_num_items_in_batch(merged, accelerator=accelerator)
    return merged


def _truncate_generation_batch(batch: dict[str, Any], max_rows: int, *, accelerator: Any = None) -> dict[str, Any]:
    truncated: dict[str, Any] = {}
    rows = _batch_row_count(batch)
    limit = min(rows, max_rows)
    for key, value in batch.items():
        if hasattr(value, "shape") and len(value.shape) > 0 and int(value.shape[0]) == rows:
            truncated[key] = value[:limit]
        elif key != "num_items_in_batch":
            truncated[key] = value
    _set_num_items_in_batch(truncated, accelerator=accelerator)
    return truncated


def _set_num_items_in_batch(batch: dict[str, Any], *, accelerator: Any = None) -> None:
    completion_mask = batch["completion_mask"]
    if "tool_mask" in batch:
        token_count = (completion_mask * batch["tool_mask"]).sum()
    else:
        token_count = completion_mask.sum()
    if accelerator is not None:
        token_count = accelerator.gather(token_count).sum()
    batch["num_items_in_batch"] = token_count
