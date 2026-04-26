from dataclasses import dataclass


@dataclass
class SampleRewardResult:
    final_reward: float
    process_reward: float
    total_reward: float


def combine_rewards(
    final_reward: float,
    process_reward: float,
    final_reward_weight: float,
    process_reward_weight: float,
) -> SampleRewardResult:
    total = final_reward_weight * final_reward + process_reward_weight * process_reward
    return SampleRewardResult(
        final_reward=final_reward,
        process_reward=process_reward,
        total_reward=total,
    )
