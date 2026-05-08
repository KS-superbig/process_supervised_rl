from __future__ import annotations

from dataclasses import dataclass

from psrl.candidates import extract_candidate_final
from psrl.reward.final_reward import compute_final_reward


@dataclass(frozen=True)
class RewardConfig:
    final_weight: float = 1.0
    prm_weight: float = 0.2
    prm_mean: float = 0.0
    prm_std: float = 1.0
    prm_clip: float = 3.0


@dataclass(frozen=True)
class RewardBreakdown:
    final_reward: float
    prm_reward: float
    total_reward: float


def normalize_prm_score(score: float, *, mean: float, std: float, clip: float) -> float:
    if std <= 0:
        normalized = score - mean
    else:
        normalized = (score - mean) / std
    if clip > 0:
        normalized = max(-clip, min(clip, normalized))
    return float(normalized)


def build_reward_breakdown(
    *,
    gold_final: str,
    completion_text: str,
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
    total_reward = config.final_weight * final_reward + config.prm_weight * prm_reward
    return RewardBreakdown(
        final_reward=float(final_reward),
        prm_reward=float(prm_reward),
        total_reward=float(total_reward),
    )
