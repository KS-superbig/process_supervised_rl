import json
from pathlib import Path

from psrl.sft_data import (
    build_sft_row,
    format_quality_metrics,
    has_run_on_artifact,
    normalize_candidate_format,
    select_prm_filtered_sft_rows,
    summarize_sft_rows,
)


def test_has_run_on_artifact_flags_glued_name_and_verb():
    assert has_run_on_artifact("Wengearns $12 per hour.")
    assert has_run_on_artifact("Therefore,Wengearned $10.")
    assert not has_run_on_artifact("Weng earns $12 per hour.")


def test_build_sft_row_uses_chat_messages_not_flat_user_assistant_text():
    row = build_sft_row(
        {
            "sample_id": "s1",
            "candidate_id": "s1-c1",
            "question": "What is 2 + 2?",
            "candidate_text": "2 + 2 = 4. The answer is $\\boxed{4}$.",
            "final_reward": 1.0,
            "prm_score": 8.0,
        }
    )

    assert row["messages"] == [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 = 4. The answer is $\\boxed{4}$."},
    ]
    assert "User:" not in json.dumps(row["messages"])
    assert "Assistant:" not in json.dumps(row["messages"])


def test_select_prm_filtered_sft_rows_skips_run_on_and_falls_back_to_clean_candidate():
    candidates = [
        {
            "sample_id": "s1",
            "candidate_id": "s1-c1",
            "candidate_index": 1,
            "question": "What did Weng earn?",
            "candidate_text": "Wengearns $10. The answer is $\\boxed{10}$.",
            "final_reward": 1.0,
            "prm_score": 10.0,
        },
        {
            "sample_id": "s1",
            "candidate_id": "s1-c2",
            "candidate_index": 2,
            "question": "What did Weng earn?",
            "candidate_text": "Weng earns $10. The answer is $\\boxed{10}$.",
            "final_reward": 1.0,
            "prm_score": 8.0,
        },
    ]

    rows, stats = select_prm_filtered_sft_rows(candidates)

    assert [row["candidate_id"] for row in rows] == ["s1-c2"]
    assert stats["selected_rows"] == 1
    assert stats["rejected_run_on"] == 1


def test_format_quality_metrics_flags_missing_spaces_after_punctuation():
    text = "Janetlays16eggsperday.Sheeatsthreeforbreakfast.Theansweris$\\boxed{18}$."

    metrics = format_quality_metrics(text)

    assert metrics.space_ratio < 0.08
    assert metrics.max_alpha_token_len >= 24
    assert metrics.punctuation_missing_space_ratio > 0.5


def test_normalize_candidate_format_repairs_safe_spacing_without_guessing_words():
    text = "Weng earns $10.She spends $3,then saves $7.  The answer is $\\boxed{7}$."

    normalized = normalize_candidate_format(text)

    assert normalized == "Weng earns $10. She spends $3, then saves $7. The answer is $\\boxed{7}$."


def test_strict_sft_selection_rejects_fully_glued_correct_candidate():
    candidates = [
        {
            "sample_id": "s1",
            "candidate_id": "s1-c1",
            "candidate_index": 1,
            "question": "How much?",
            "candidate_text": "Janetlays16eggsperday.Sheeatsthreeforbreakfast.Theansweris$\\boxed{18}$.",
            "final_reward": 1.0,
            "prm_score": 20.0,
        },
        {
            "sample_id": "s1",
            "candidate_id": "s1-c2",
            "candidate_index": 2,
            "question": "How much?",
            "candidate_text": "Janet lays 16 eggs per day. She eats three for breakfast. The answer is $\\boxed{18}$.",
            "final_reward": 1.0,
            "prm_score": 12.0,
        },
    ]

    rows, stats = select_prm_filtered_sft_rows(candidates, strict_format=True)

    assert [row["candidate_id"] for row in rows] == ["s1-c2"]
    assert stats["rejected_low_space_ratio"] == 1
    assert rows[0]["messages"][1]["content"].startswith("Janet lays 16 eggs")


def test_summarize_sft_rows_counts_message_spacing_artifacts(tmp_path: Path):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "Clean answer with spaces."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "Wengearns money."},
            ]
        },
    ]

    summary = summarize_sft_rows(rows)

    assert summary["rows"] == 2
    assert summary["run_on_rows"] == 1
    assert summary["assistant_mean_chars"] > 0
