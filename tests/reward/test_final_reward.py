from psrl.reward.final_reward import compute_final_reward


def test_compute_final_reward_exact_match_after_normalization():
    assert compute_final_reward("#### 42", "42") == 1.0


def test_compute_final_reward_mismatch():
    assert compute_final_reward("#### 42", "41") == 0.0
