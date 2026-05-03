from psrl.candidates import build_candidate_row, build_prompt, clean_candidate_text, extract_candidate_final


def test_extract_candidate_final_prefers_hash_marker():
    text = "Compute 2 + 2 = 4.\n#### 4"

    assert extract_candidate_final(text) == "4"


def test_extract_candidate_final_falls_back_to_last_number():
    text = "We add 3 and 5 to get 8. The answer is 8."

    assert extract_candidate_final(text) == "8"


def test_build_candidate_row_preserves_gold_and_steps():
    sample = {
        "id": "gsm8k-train-000001",
        "question": "What is 2 + 2?",
        "answer_final_normalized": "4",
    }

    row = build_candidate_row(sample, candidate_index=1, candidate_text="2 + 2 = 4.\n#### 4")

    assert row["sample_id"] == "gsm8k-train-000001"
    assert row["candidate_id"] == "gsm8k-train-000001-cand-01"
    assert row["gold_final"] == "4"
    assert row["answer_final_normalized"] == "4"
    assert row["candidate_final"] == "4"
    assert row["candidate_steps"] == ["2 + 2 = 4."]


def test_clean_candidate_text_decodes_tokenizer_space_artifacts():
    text = "Nataliasoldclipsto48ofherfriendsinApril.ĊInMay,sheĠsold half."

    assert clean_candidate_text(text) == "Nataliasoldclipsto48ofherfriendsinApril.\nInMay,she sold half."


def test_build_prompt_uses_deepseek_instruct_format():
    prompt = build_prompt("What is 2 + 2?")

    assert "Please reason step by step" in prompt
    assert "\\boxed{}" in prompt
    assert "Question:" not in prompt
