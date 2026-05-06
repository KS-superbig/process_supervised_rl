from psrl.prm_dataset import build_preference_rows


def test_build_preference_rows_joins_judge_preferences_to_candidate_text():
    candidates = [
        {
            "sample_id": "sample-1",
            "candidate_id": "sample-1-cand-01",
            "question": "What is 2 + 2?",
            "candidate_text": "2 + 2 = 4. The answer is 4.",
        },
        {
            "sample_id": "sample-1",
            "candidate_id": "sample-1-cand-02",
            "question": "What is 2 + 2?",
            "candidate_text": "2 + 2 = 5. The answer is 5.",
        },
    ]
    judgements = [
        {
            "sample_id": "sample-1",
            "pairwise_preferences": [
                {
                    "chosen": "sample-1-cand-01",
                    "rejected": "sample-1-cand-02",
                    "reason": "Candidate 1 is correct.",
                }
            ],
        }
    ]

    rows = build_preference_rows(candidates, judgements)

    assert rows == [
        {
            "sample_id": "sample-1",
            "question": "What is 2 + 2?",
            "chosen_candidate_id": "sample-1-cand-01",
            "chosen_text": "2 + 2 = 4. The answer is 4.",
            "rejected_candidate_id": "sample-1-cand-02",
            "rejected_text": "2 + 2 = 5. The answer is 5.",
            "judge_reason": "Candidate 1 is correct.",
        }
    ]


def test_build_preference_rows_falls_back_to_ranking_pairs():
    candidates = [
        {
            "sample_id": "sample-1",
            "candidate_id": "sample-1-cand-01",
            "question": "What is 2 + 2?",
            "candidate_text": "Correct.",
        },
        {
            "sample_id": "sample-1",
            "candidate_id": "sample-1-cand-02",
            "question": "What is 2 + 2?",
            "candidate_text": "Wrong.",
        },
        {
            "sample_id": "sample-1",
            "candidate_id": "sample-1-cand-03",
            "question": "What is 2 + 2?",
            "candidate_text": "Also wrong.",
        },
    ]
    judgements = [
        {
            "sample_id": "sample-1",
            "ranking": ["sample-1-cand-01", "sample-1-cand-02", "sample-1-cand-03"],
            "pairwise_preferences": [],
        }
    ]

    rows = build_preference_rows(candidates, judgements)

    assert [(row["chosen_candidate_id"], row["rejected_candidate_id"]) for row in rows] == [
        ("sample-1-cand-01", "sample-1-cand-02"),
        ("sample-1-cand-02", "sample-1-cand-03"),
    ]


def test_build_preference_rows_skips_score_ties_without_explicit_preferences():
    candidates = [
        {
            "sample_id": "sample-1",
            "candidate_id": "sample-1-cand-01",
            "question": "What is 2 + 2?",
            "candidate_text": "Correct.",
        },
        {
            "sample_id": "sample-1",
            "candidate_id": "sample-1-cand-02",
            "question": "What is 2 + 2?",
            "candidate_text": "Also correct.",
        },
    ]
    judgements = [
        {
            "sample_id": "sample-1",
            "ranking": ["sample-1-cand-01", "sample-1-cand-02"],
            "scores": {"sample-1-cand-01": 1.0, "sample-1-cand-02": 1.0},
            "pairwise_preferences": [],
        }
    ]

    assert build_preference_rows(candidates, judgements) == []
