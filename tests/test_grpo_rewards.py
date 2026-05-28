from psrl.rl.grpo_rewards import (
    RewardConfig,
    build_reward_breakdown,
    compute_soft_length_penalty,
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


def test_build_reward_breakdown_can_include_python_verifier_component():
    config = RewardConfig(
        final_weight=1.0,
        prm_weight=0.0,
        python_verifier_weight=1.0,
        python_verifier_alpha_step=0.3,
        python_verifier_beta_pass_rate=0.2,
    )

    breakdown = build_reward_breakdown(
        gold_final="13",
        completion_text="16 - 3 = 13. The final answer is 13.",
        raw_prm_score=0.0,
        config=config,
    )

    assert breakdown.final_reward == 1.0
    assert breakdown.python_verifier_reward == 0.5
    assert breakdown.total_reward == 1.5


def test_compute_soft_length_penalty_is_linear_inside_penalty_window():
    assert compute_soft_length_penalty(255, start=256, max_length=384) == 0.0
    assert compute_soft_length_penalty(256, start=256, max_length=384) == 0.0
    assert compute_soft_length_penalty(320, start=256, max_length=384) == -0.5
    assert compute_soft_length_penalty(384, start=256, max_length=384) == -1.0
    assert compute_soft_length_penalty(512, start=256, max_length=384) == -1.0


def test_build_reward_breakdown_can_include_soft_length_penalty():
    config = RewardConfig(
        final_weight=1.0,
        prm_weight=0.0,
        python_verifier_weight=0.0,
        length_penalty_weight=1.0,
        length_penalty_start=256,
        length_penalty_max=384,
    )

    breakdown = build_reward_breakdown(
        gold_final="42",
        completion_text="The final answer is 42.",
        completion_token_length=320,
        raw_prm_score=0.0,
        config=config,
    )

    assert breakdown.final_reward == 1.0
    assert breakdown.length_penalty == -0.5
    assert breakdown.total_reward == 0.5
