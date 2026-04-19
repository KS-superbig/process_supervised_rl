from psrl.data.gsm8k import build_reasoning_sample


def test_build_reasoning_sample_populates_fields():
    row = {"question": "2+2?", "answer": "We compute 2 + 2.\nTherefore the result is 4. #### 4"}
    sample = build_reasoning_sample("gsm8k-train-1", "train", row)
    assert sample.answer_final_normalized == "4"
    assert sample.solution_raw == "We compute 2 + 2.\nTherefore the result is 4."
    assert sample.steps == ["We compute 2 + 2.", "Therefore the result is 4."]
