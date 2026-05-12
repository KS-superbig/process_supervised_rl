import json
from pathlib import Path

from psrl.sft_data import (
    build_sft_row,
    has_run_on_artifact,
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
