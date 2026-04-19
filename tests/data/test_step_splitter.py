from psrl.data.step_splitter import split_solution_steps


def test_split_solution_steps_uses_newlines():
    raw = "Step 1: Add 2 and 3\nStep 2: Get 5"
    assert split_solution_steps(raw) == ["Step 1: Add 2 and 3", "Step 2: Get 5"]


def test_split_solution_steps_drops_empty_lines():
    raw = "First line\n\nSecond line"
    assert split_solution_steps(raw) == ["First line", "Second line"]
