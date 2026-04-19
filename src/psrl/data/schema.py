from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ReasoningSample:
    sample_id: str
    source: str
    split: str
    question: str
    answer_final: str
    answer_final_normalized: str
    solution_raw: str
    steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["id"] = payload.pop("sample_id")
        return payload
