from psrl.data.normalize import normalize_final_answer


def compute_final_reward(gold_final: str, predicted_final: str) -> float:
    """Return 1.0 for exact normalized match, otherwise 0.0."""
    gold = normalize_final_answer(gold_final)
    pred = normalize_final_answer(predicted_final)
    return 1.0 if gold and pred and gold == pred else 0.0
