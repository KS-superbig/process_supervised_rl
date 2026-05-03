from psrl.eval.candidate_selection import build_selection_report, score_candidate_rows


def test_score_candidate_rows_uses_gold_final_field():
    rows = [
        {
            "sample_id": "s1",
            "candidate_id": "s1-cand-01",
            "question": "2+2?",
            "gold_final": "4",
            "candidate_final": "4",
            "candidate_steps": ["2 + 2 = 4."],
        }
    ]
    cfg = {
        "final_reward_weight": 1.0,
        "process_reward_weight": 0.3,
        "components": {
            "step_validity": 1.0,
            "step_consistency": 1.0,
            "progress_contribution": 1.0,
            "anti_hacking_penalty": -0.5,
        },
    }

    scored = score_candidate_rows(rows, cfg)

    assert scored[0]["final_reward"] == 1.0
    assert scored[0]["sample_id"] == "s1"
    assert scored[0]["candidate_id"] == "s1-cand-01"
    assert scored[0]["total_reward"] > 1.0


def test_build_selection_report_compares_top1_strategies():
    scored_rows = [
        {
            "sample_id": "s1",
            "candidate_id": "s1-cand-01",
            "candidate_index": 1,
            "question": "2+2?",
            "candidate_text": "wrong",
            "final_reward": 0.0,
            "process_reward": 0.9,
            "total_reward": 0.27,
            "num_steps": 1,
            "flags": [],
        },
        {
            "sample_id": "s1",
            "candidate_id": "s1-cand-02",
            "candidate_index": 2,
            "question": "2+2?",
            "candidate_text": "right but terse",
            "final_reward": 1.0,
            "process_reward": 0.2,
            "total_reward": 1.06,
            "num_steps": 1,
            "flags": [],
        },
        {
            "sample_id": "s1",
            "candidate_id": "s1-cand-03",
            "candidate_index": 3,
            "question": "2+2?",
            "candidate_text": "right and clearer",
            "final_reward": 1.0,
            "process_reward": 0.8,
            "total_reward": 1.24,
            "num_steps": 2,
            "flags": [],
        },
    ]

    report = build_selection_report(scored_rows)

    assert report.summary["num_samples"] == 1
    assert report.summary["final_only_accuracy"] == 1.0
    assert report.summary["final_plus_process_accuracy"] == 1.0
    assert report.summary["changed_selection_count"] == 1
    assert "s1-cand-02 -> s1-cand-03" in report.markdown
