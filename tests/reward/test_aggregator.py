from psrl.reward.aggregator import combine_rewards


def test_combine_rewards_uses_configured_weights():
    result = combine_rewards(
        final_reward=1.0,
        process_reward=0.5,
        final_reward_weight=1.0,
        process_reward_weight=0.3,
    )
    assert result.total_reward == 1.15
