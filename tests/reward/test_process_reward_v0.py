from psrl.reward.process_reward_v0 import score_process_steps


def test_score_process_steps_penalizes_repetition():
    weights = {
        "step_validity": 1.0,
        "step_consistency": 1.0,
        "progress_contribution": 1.0,
        "anti_hacking_penalty": -0.5,
    }
    steps = [
        "x = 2 + 3 = 5",
        "x = 2 + 3 = 5",
    ]
    result = score_process_steps(steps, weights)
    assert result.component_means["anti_hacking_penalty"] > 0.4


def test_score_process_steps_detects_assignment_conflict():
    weights = {
        "step_validity": 1.0,
        "step_consistency": 1.0,
        "progress_contribution": 1.0,
        "anti_hacking_penalty": -0.5,
    }
    steps = [
        "x = 3",
        "x = 4",
    ]
    result = score_process_steps(steps, weights)
    assert result.step_details[1].components["step_consistency"] == 0.0


def test_score_process_steps_handles_empty_steps():
    weights = {
        "step_validity": 1.0,
        "step_consistency": 1.0,
        "progress_contribution": 1.0,
        "anti_hacking_penalty": -0.5,
    }
    result = score_process_steps([], weights)
    assert result.score == 0.0
    assert result.step_details == []
