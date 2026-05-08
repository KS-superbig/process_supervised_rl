from psrl.rl.grpo_rewards import (
    RewardConfig,
    build_reward_breakdown,
    normalize_prm_score,
)


def test_normalize_prm_score_uses_zscore_and_clip():
    assert normalize_prm_score(13.0, mean=10.0, std=2.0, clip=1.0) == 1.0
    assert normalize_prm_score(7.0, mean=10.0, std=2.0, clip=1.0) == -1.0


def test_build_reward_breakdown_combines_final_and_prm_rewards():
    config = RewardConfig(final_weight=1.0, prm_weight=0.25, prm_mean=10.0, prm_std=2.0, prm_clip=3.0)

    breakdown = build_reward_breakdown(
        gold_final="42",
        completion_text="Reasoning here. The final answer is 42.",
        raw_prm_score=12.0,
        config=config,
    )

    assert breakdown.final_reward == 1.0
    assert breakdown.prm_reward == 1.0
    assert breakdown.total_reward == 1.25


def test_build_reward_breakdown_penalizes_wrong_final_answer_but_keeps_prm_component():
    config = RewardConfig(final_weight=1.0, prm_weight=0.2, prm_mean=0.0, prm_std=1.0, prm_clip=3.0)

    breakdown = build_reward_breakdown(
        gold_final="42",
        completion_text="Reasoning here. The final answer is 41.",
        raw_prm_score=0.5,
        config=config,
    )

    assert breakdown.final_reward == 0.0
    assert breakdown.prm_reward == 0.5
    assert breakdown.total_reward == 0.1
