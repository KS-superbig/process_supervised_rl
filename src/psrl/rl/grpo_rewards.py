from __future__ import annotations

from dataclasses import dataclass

from psrl.candidates import extract_candidate_final
from psrl.reward.final_reward import compute_final_reward
from psrl.reward.python_verifier import python_verifier_reward


@dataclass(frozen=True)
class RewardConfig:
    final_weight: float = 1.0
    prm_weight: float = 0.2
    prm_mean: float = 0.0
    prm_std: float = 1.0
    prm_clip: float = 3.0
    python_verifier_weight: float = 0.0
    python_verifier_alpha_step: float = 0.3
    python_verifier_beta_pass_rate: float = 0.2
    python_verifier_gamma_no_eq_penalty: float = 0.0
    length_penalty_weight: float = 0.0
    length_penalty_start: int = 0
    length_penalty_max: int = 0


@dataclass(frozen=True)
class RewardBreakdown:
    final_reward: float
    prm_reward: float
    python_verifier_reward: float
    length_penalty: float
    total_reward: float


def normalize_prm_score(score: float, *, mean: float, std: float, clip: float) -> float:
    if std <= 0:
        normalized = score - mean
    else:
        normalized = (score - mean) / std
    if clip > 0:
        normalized = max(-clip, min(clip, normalized))
    return float(normalized)


def compute_soft_length_penalty(length: int | None, *, start: int, max_length: int) -> float:
    if length is None or max_length <= 0:
        return 0.0
    if max_length <= start:
        raise ValueError("length_penalty_max must be greater than length_penalty_start.")
    if length <= start:
        return 0.0
    if length >= max_length:
        return -1.0
    return -float(length - start) / float(max_length - start)


def build_reward_breakdown(
    *,
    gold_final: str,
    completion_text: str,
    completion_token_length: int | None = None,
    raw_prm_score: float,
    config: RewardConfig,
) -> RewardBreakdown:
    candidate_final = extract_candidate_final(completion_text)
    final_reward = compute_final_reward(gold_final, candidate_final)
    prm_reward = normalize_prm_score(
        raw_prm_score,
        mean=config.prm_mean,
        std=config.prm_std,
        clip=config.prm_clip,
    )
    verifier = python_verifier_reward(
        completion_text,
        bool(final_reward),
        alpha_step=config.python_verifier_alpha_step,
        beta_pass_rate=config.python_verifier_beta_pass_rate,
        gamma_no_eq_penalty=config.python_verifier_gamma_no_eq_penalty,
    )
    verifier_reward = float(verifier["reward"] - verifier["r_final"])
    length_penalty = compute_soft_length_penalty(
        completion_token_length,
        start=config.length_penalty_start,
        max_length=config.length_penalty_max,
    )
    total_reward = (
        config.final_weight * final_reward
        + config.prm_weight * prm_reward
        + config.python_verifier_weight * verifier_reward
        + config.length_penalty_weight * length_penalty
    )
    return RewardBreakdown(
        final_reward=float(final_reward),
        prm_reward=float(prm_reward),
        python_verifier_reward=float(verifier_reward),
        length_penalty=float(length_penalty),
        total_reward=float(total_reward),
    )
