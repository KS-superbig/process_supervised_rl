from psrl.reward.python_verifier import (
    python_verifier_reward,
    split_into_steps,
    verify_step,
)


def test_split_into_steps_prefers_explicit_markers():
    trace = "Step 1: Add 2 and 3 = 5.\nStep 2. Multiply 5 * 4 = 20."

    assert split_into_steps(trace) == [
        "Add 2 and 3 = 5.",
        "Multiply 5 * 4 = 20.",
    ]


def test_split_into_steps_uses_double_newlines_before_single_newlines():
    trace = "First paragraph\nstill first paragraph\n\nSecond paragraph"

    assert split_into_steps(trace) == [
        "First paragraph\nstill first paragraph",
        "Second paragraph",
    ]


def test_verify_step_accepts_gsm8k_money_and_x_multiplication():
    result = verify_step("The cost of 3 dozen donuts is 3x$68=$204.")

    assert result["has_equation"] is True
    assert result["all_correct"] is True
    assert result["checked"] == 1
    assert result["passed"] == 1


def test_verify_step_accepts_fraction_of_language():
    result = verify_step("Half the amount is 1/2 of 10 = 5.")

    assert result["has_equation"] is True
    assert result["all_correct"] is True


def test_verify_step_accepts_percent_of_language():
    result = verify_step("20% of 20 students is 20% of 20 = 4.")

    assert result["has_equation"] is True
    assert result["all_correct"] is True


def test_verify_step_marks_incorrect_arithmetic():
    result = verify_step("She has 16 - 3 = 12 eggs left.")

    assert result["has_equation"] is True
    assert result["all_correct"] is False
    assert result["checked"] == 1
    assert result["passed"] == 0


def test_verify_step_ignores_unsafe_or_non_arithmetic_text():
    result = verify_step("Use foo(1) = 1 and keep going.")

    assert result == {
        "has_equation": False,
        "all_correct": None,
        "checked": 0,
        "passed": 0,
        "details": [],
    }


def test_python_verifier_reward_aggregates_equation_steps_only():
    result = python_verifier_reward(
        "Step 1: 16 - 3 = 13.\nStep 2: Explain why this matters.\nStep 3: 13 - 4 = 8.",
        final_correct=True,
        alpha_step=0.3,
        beta_pass_rate=0.2,
    )

    assert result["r_final"] == 1.0
    assert result["r_step_mean"] == 0.5
    assert result["r_pass_rate"] == 0.5
    assert result["step_scores"] == [1.0, 0.0]
    assert result["n_steps"] == 3
    assert result["n_steps_with_eq"] == 2
    assert result["reward"] == 1.25
