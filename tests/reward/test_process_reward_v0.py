from psrl.reward.process_reward_v0 import score_process_steps


WEIGHTS = {
    "step_validity": 1.0,
    "step_consistency": 1.0,
    "progress_contribution": 1.0,
    "anti_hacking_penalty": -0.5,
}


def test_score_process_steps_penalizes_repetition():
    steps = [
        "x = 2 + 3 = 5",
        "x = 2 + 3 = 5",
    ]
    result = score_process_steps(steps, WEIGHTS)
    assert result.component_means["anti_hacking_penalty"] > 0.4


def test_score_process_steps_detects_assignment_conflict():
    steps = [
        "x = 3",
        "x = 4",
    ]
    result = score_process_steps(steps, WEIGHTS)
    assert result.step_details[1].components["step_consistency"] == 0.0


def test_score_process_steps_handles_empty_steps():
    result = score_process_steps([], WEIGHTS)
    assert result.score == 0.0
    assert result.step_details == []


def test_complete_multistep_trace_beats_compressed_run_on_trace():
    complete_steps = [
        "Each pie has 10 slices, so 3 pies have 3 * 10 = 30 slices.",
        "Manny, 24 classmates, and the teacher ate 24 + 1 + 1 = 26 slices.",
        "The number of slices left is 30 - 26 = 4.",
        "The answer is 4.",
    ]
    compressed_steps = [
        "There were 3 x 10 = 30 slices.If Manny, his classmates, and the teacher all had 1 slice each, then they consumed 24 + 1 + 1 = 26 slices.Therefore, there were 30 - 26 = 4 slices left.The answer is 4."
    ]

    complete = score_process_steps(complete_steps, WEIGHTS)
    compressed = score_process_steps(compressed_steps, WEIGHTS)

    assert complete.score >= compressed.score


def test_prompt_artifacts_are_penalized_as_reward_hacking():
    steps = [
        "At his main job, James earns $20/hour x 30 hours = $600 per week.",
        "User: Find the solution to the equation x^2 - 6x + 8 = 0.",
    ]

    result = score_process_steps(steps, WEIGHTS)

    assert result.component_means["anti_hacking_penalty"] > 0.2


def test_prompt_artifact_trace_scores_below_on_task_trace():
    on_task_steps = [
        "At his main job, James earns 20 * 30 = 600 dollars.",
        "His second job pays 20% less, so he earns 20 * 0.8 = 16 dollars per hour.",
        "He works half of 30 hours there, so he works 15 hours.",
        "His second-job pay is 16 * 15 = 240 dollars.",
        "His total weekly pay is 600 + 240 = 840 dollars.",
    ]
    artifact_steps = [
        "At his main job, James earns $20/hour x 30 hours = $600 per week.",
        "User: Find the solution to the equation x^2 - 6x + 8 = 0.",
        "The solutions are x = 2 or x = 4.",
    ]

    on_task = score_process_steps(on_task_steps, WEIGHTS)
    artifact = score_process_steps(artifact_steps, WEIGHTS)

    assert on_task.score > artifact.score
