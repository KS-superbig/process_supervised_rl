def normalize_final_answer(raw: str) -> str:
    text = raw.strip()
    if "####" in text:
        text = text.split("####")[-1].strip()
    return " ".join(text.split())


def extract_solution_text(raw: str) -> str:
    text = raw.strip()
    if "####" in text:
        text = text.split("####")[0].strip()
    return text
