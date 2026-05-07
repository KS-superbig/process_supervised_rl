from psrl.prm_protocol import build_eval_protocol_report, filter_preference_rows


def test_build_eval_protocol_report_counts_agreement_and_changed_breakdown():
    scored = [
        {
            "sample_id": "s1",
            "candidate_id": "s1-c1",
            "candidate_index": 1,
            "final_reward": 1.0,
            "prm_score": 0.2,
        },
        {
            "sample_id": "s1",
            "candidate_id": "s1-c2",
            "candidate_index": 2,
            "final_reward": 1.0,
            "prm_score": 0.9,
        },
        {
            "sample_id": "s2",
            "candidate_id": "s2-c1",
            "candidate_index": 1,
            "final_reward": 0.0,
            "prm_score": 0.1,
        },
        {
            "sample_id": "s2",
            "candidate_id": "s2-c2",
            "candidate_index": 2,
            "final_reward": 0.0,
            "prm_score": 0.8,
        },
    ]
    judgements = [
        {"sample_id": "s1", "best_candidate_id": "s1-c2"},
        {"sample_id": "s2", "best_candidate_id": "s2-c1"},
    ]
    audit = [
        {"sample_id": "s1", "agrees_with_prm": True, "agrees_with_final_only": False},
        {"sample_id": "s2", "agrees_with_prm": False, "agrees_with_final_only": True},
    ]

    result = build_eval_protocol_report(scored, judgements, audit_rows=audit)

    assert result.summary["num_samples"] == 2
    assert result.summary["changed_count"] == 2
    assert result.summary["judge_agree_prm"] == 1
    assert result.summary["judge_agree_final_only"] == 1
    assert result.summary["changed_breakdown"] == {"0->0": 1, "1->1": 1}


def test_filter_preference_rows_dedup_short_and_gap_rules():
    prefs = [
        {
            "sample_id": "s1",
            "chosen_candidate_id": "s1-c1",
            "rejected_candidate_id": "s1-c2",
            "chosen_text": "A long enough chosen text.",
            "rejected_text": "A long enough rejected text.",
            "judge_reason": "x",
        },
        {
            "sample_id": "s1",
            "chosen_candidate_id": "s1-c1",
            "rejected_candidate_id": "s1-c2",
            "chosen_text": "A long enough chosen text.",
            "rejected_text": "A long enough rejected text.",
            "judge_reason": "dup",
        },
        {
            "sample_id": "s1",
            "chosen_candidate_id": "s1-c3",
            "rejected_candidate_id": "s1-c4",
            "chosen_text": "short",
            "rejected_text": "also short",
            "judge_reason": "short",
        },
        {
            "sample_id": "s2",
            "chosen_candidate_id": "s2-c1",
            "rejected_candidate_id": "s2-c2",
            "chosen_text": "Chosen text for second sample is long enough.",
            "rejected_text": "Rejected text for second sample is long enough.",
            "judge_reason": "ok",
        },
    ]
    judgements = [
        {
            "sample_id": "s1",
            "scores": {"s1-c1": 0.9, "s1-c2": 0.7, "s1-c3": 0.6, "s1-c4": 0.55},
        },
        {
            "sample_id": "s2",
            "scores": {"s2-c1": 0.65, "s2-c2": 0.64},
        },
    ]

    filtered, stats = filter_preference_rows(
        prefs,
        judgements=judgements,
        min_text_chars=20,
        min_score_gap=0.05,
        max_pairs_per_sample=4,
    )

    assert len(filtered) == 1
    assert filtered[0]["sample_id"] == "s1"
    assert stats["drop_duplicate"] == 1
    assert stats["drop_short_text"] == 1
    assert stats["drop_low_score_gap"] == 1
