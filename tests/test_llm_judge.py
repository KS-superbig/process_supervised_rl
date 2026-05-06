import json

from psrl.llm_judge import build_judge_prompt, parse_judge_response


def test_build_judge_prompt_contains_question_gold_and_all_candidates():
    candidates = [
        {
            "candidate_id": "sample-1-cand-01",
            "candidate_text": "2 + 2 = 4. The answer is \\boxed{4}.",
            "candidate_final": "4",
        },
        {
            "candidate_id": "sample-1-cand-02",
            "candidate_text": "2 + 2 = 5. The answer is \\boxed{5}.",
            "candidate_final": "5",
        },
    ]

    prompt = build_judge_prompt(
        sample_id="sample-1",
        question="What is 2 + 2?",
        gold_final="4",
        candidates=candidates,
    )

    assert "sample-1" in prompt
    assert "What is 2 + 2?" in prompt
    assert "Gold final answer: 4" in prompt
    assert "sample-1-cand-01" in prompt
    assert "sample-1-cand-02" in prompt
    assert "Return only valid JSON" in prompt


def test_parse_judge_response_accepts_json_fence_and_validates_ids():
    response = """```json
{
  "best_candidate_id": "sample-1-cand-01",
  "ranking": ["sample-1-cand-01", "sample-1-cand-02"],
  "scores": {
    "sample-1-cand-01": 0.91,
    "sample-1-cand-02": 0.20
  },
  "pairwise_preferences": [
    {
      "chosen": "sample-1-cand-01",
      "rejected": "sample-1-cand-02",
      "reason": "The first candidate is correct and complete."
    }
  ],
  "notes": "ok"
}
```"""

    parsed = parse_judge_response(
        response,
        sample_id="sample-1",
        expected_candidate_ids=["sample-1-cand-01", "sample-1-cand-02"],
    )

    assert parsed["sample_id"] == "sample-1"
    assert parsed["best_candidate_id"] == "sample-1-cand-01"
    assert parsed["scores"]["sample-1-cand-02"] == 0.20


def test_parse_judge_response_rejects_unknown_candidate_id():
    response = json.dumps(
        {
            "best_candidate_id": "sample-1-cand-03",
            "ranking": ["sample-1-cand-03", "sample-1-cand-01"],
            "scores": {"sample-1-cand-03": 0.9, "sample-1-cand-01": 0.1},
            "pairwise_preferences": [],
        }
    )

    try:
        parse_judge_response(
            response,
            sample_id="sample-1",
            expected_candidate_ids=["sample-1-cand-01", "sample-1-cand-02"],
        )
    except ValueError as exc:
        assert "unknown candidate id" in str(exc)
    else:
        raise AssertionError("Expected parse_judge_response to reject unknown candidate ids")
