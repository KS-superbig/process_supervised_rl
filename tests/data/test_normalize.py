from psrl.data.normalize import extract_solution_text, normalize_final_answer


def test_normalize_final_answer_extracts_hash_answer():
    raw = "The answer is 42 #### 42"
    assert normalize_final_answer(raw) == "42"


def test_normalize_final_answer_trims_whitespace():
    assert normalize_final_answer("  3/4 ") == "3/4"


def test_extract_solution_text_drops_final_marker():
    raw = "First compute 2 + 2. #### 4"
    assert extract_solution_text(raw) == "First compute 2 + 2."
