from pathlib import Path

from psrl.prm import (
    LinearPreferencePRM,
    build_prm_selection_report,
    score_candidates_with_prm,
    train_linear_preference_prm,
)


def test_train_linear_preference_prm_scores_chosen_above_rejected():
    rows = [
        {
            "sample_id": "s1",
            "question": "What is 2 + 2?",
            "chosen_text": "2 plus 2 equals 4. The answer is 4.",
            "rejected_text": "2 plus 2 equals 5. The answer is 5.",
        },
        {
            "sample_id": "s2",
            "question": "What is 3 + 3?",
            "chosen_text": "3 plus 3 equals 6. The answer is 6.",
            "rejected_text": "3 plus 3 equals 7. The answer is 7.",
        },
    ]

    model, metrics = train_linear_preference_prm(rows, epochs=30, learning_rate=0.2, l2=0.0)

    assert metrics["num_pairs"] == 2
    assert metrics["train_pair_accuracy"] == 1.0
    for row in rows:
        assert model.score(row["question"], row["chosen_text"]) > model.score(row["question"], row["rejected_text"])


def test_linear_preference_prm_round_trips(tmp_path: Path):
    model = LinearPreferencePRM(weights={"tok=good": 1.5, "tok=bad": -1.0}, bias=0.25)
    path = tmp_path / "model.json"

    model.save(path)
    loaded = LinearPreferencePRM.load(path)

    assert loaded.score("question", "good") == model.score("question", "good")
    assert loaded.score("question", "bad") == model.score("question", "bad")


def test_score_candidates_with_prm_and_report_compares_prm_top1():
    model = LinearPreferencePRM(weights={"tok=clear": 1.0, "tok=wrong": -1.0})
    candidates = [
        {
            "sample_id": "s1",
            "candidate_id": "s1-cand-01",
            "candidate_index": 1,
            "question": "2+2?",
            "gold_final": "4",
            "candidate_final": "4",
            "candidate_text": "right but terse",
        },
        {
            "sample_id": "s1",
            "candidate_id": "s1-cand-02",
            "candidate_index": 2,
            "question": "2+2?",
            "gold_final": "4",
            "candidate_final": "4",
            "candidate_text": "right and clear",
        },
        {
            "sample_id": "s2",
            "candidate_id": "s2-cand-01",
            "candidate_index": 1,
            "question": "1+1?",
            "gold_final": "2",
            "candidate_final": "3",
            "candidate_text": "wrong",
        },
    ]

    scored = score_candidates_with_prm(candidates, model)
    report = build_prm_selection_report(scored)

    assert scored[1]["prm_score"] > scored[0]["prm_score"]
    assert report.summary["num_samples"] == 2
    assert report.summary["final_only_accuracy"] == 0.5
    assert report.summary["final_plus_prm_accuracy"] == 0.5
    assert report.summary["changed_selection_count"] == 1
    assert "s1-cand-01 -> s1-cand-02" in report.markdown
