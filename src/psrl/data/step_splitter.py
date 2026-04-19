def split_solution_steps(raw: str) -> list[str]:
    lines = [line.strip() for line in raw.splitlines()]
    return [line for line in lines if line]
