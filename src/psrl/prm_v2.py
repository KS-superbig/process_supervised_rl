from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Iterable

from psrl.reward.final_reward import compute_final_reward


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class MLPPreferencePRM:
    vocab: dict[str, int]
    model: object
    max_features: int

    def vectorize(self, question: str, candidate_text: str) -> list[float]:
        vec = [0.0] * self.max_features
        q_tokens = _tokenize(question)
        c_tokens = _tokenize(candidate_text)
        for token in q_tokens:
            key = f"q:{token}"
            idx = self.vocab.get(key)
            if idx is not None:
                vec[idx] += 1.0
        for token in c_tokens:
            key = f"c:{token}"
            idx = self.vocab.get(key)
            if idx is not None:
                vec[idx] += 1.0
        norm = sum(vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


def train_mlp_preference_prm(
    preference_rows: Iterable[dict],
    *,
    max_features: int = 8000,
    hidden_dim: int = 256,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[MLPPreferencePRM, dict[str, float | int]]:
    rows = list(preference_rows)
    if not rows:
        raise ValueError("preference_rows must not be empty")

    torch = _import_torch()
    _seed_everything(seed, torch)

    vocab = _build_vocab(rows, max_features=max_features)
    model = _build_model(torch, input_dim=max_features, hidden_dim=hidden_dim)
    model.to(device)

    chosen_matrix = torch.tensor(
        [_vectorize_with_vocab(vocab, max_features, row.get("question", ""), row.get("chosen_text", "")) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    rejected_matrix = torch.tensor(
        [_vectorize_with_vocab(vocab, max_features, row.get("question", ""), row.get("rejected_text", "")) for row in rows],
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses = []
    for _ in range(epochs):
        order = list(range(len(rows)))
        random.shuffle(order)
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            c = chosen_matrix[batch_idx]
            r = rejected_matrix[batch_idx]
            c_score = model(c).squeeze(-1)
            r_score = model(r).squeeze(-1)
            margin = c_score - r_score
            loss = torch.nn.functional.softplus(-margin).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(batch_idx)
        losses.append(epoch_loss / len(rows))

    with torch.no_grad():
        c_score = model(chosen_matrix).squeeze(-1)
        r_score = model(rejected_matrix).squeeze(-1)
        pair_acc = float((c_score > r_score).float().mean().cpu())

    prm = MLPPreferencePRM(vocab=vocab, model=model, max_features=max_features)
    metrics = {
        "num_pairs": len(rows),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_features": max_features,
        "hidden_dim": hidden_dim,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "train_pair_accuracy": pair_acc,
    }
    return prm, metrics


def save_mlp_prm(prm: MLPPreferencePRM, output_dir: Path, metrics: dict[str, float | int]) -> tuple[Path, Path]:
    torch = _import_torch()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    meta_path = output_dir / "meta.json"
    torch.save(prm.model.state_dict(), model_path)
    meta = {
        "model_type": "mlp_preference_prm_v2",
        "max_features": prm.max_features,
        "vocab": prm.vocab,
        "metrics": metrics,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return model_path, meta_path


def load_mlp_prm(output_dir: Path, *, device: str = "cpu") -> MLPPreferencePRM:
    torch = _import_torch()
    meta_path = output_dir / "meta.json"
    model_path = output_dir / "model.pt"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    max_features = int(meta["max_features"])
    metrics = meta.get("metrics", {})
    hidden_dim = int(metrics.get("hidden_dim", 256))
    model = _build_model(torch, input_dim=max_features, hidden_dim=hidden_dim)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return MLPPreferencePRM(vocab=meta["vocab"], model=model, max_features=max_features)


def score_candidates_with_mlp_prm(candidate_rows: Iterable[dict], prm: MLPPreferencePRM, *, device: str = "cpu") -> list[dict]:
    torch = _import_torch()
    model = prm.model
    model.eval()
    model.to(device)

    rows = list(candidate_rows)
    features = torch.tensor(
        [prm.vectorize(row.get("question", ""), row.get("candidate_text", "")) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        scores = model(features).squeeze(-1).detach().cpu().tolist()

    output = []
    for idx, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id", row.get("id", f"sample-{idx:06d}"))
        candidate_id = row.get("candidate_id", f"{sample_id}-cand-{idx:02d}")
        candidate_index = int(row.get("candidate_index", idx))
        gold_final = row.get("gold_final", row.get("answer_final_normalized", ""))
        candidate_final = row.get("candidate_final", "")
        output.append(
            {
                **row,
                "id": candidate_id,
                "sample_id": sample_id,
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "gold_final": gold_final,
                "candidate_final": candidate_final,
                "final_reward": compute_final_reward(gold_final, candidate_final),
                "prm_score": float(scores[idx - 1]),
            }
        )
    return output


def _build_vocab(rows: list[dict], *, max_features: int) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        for token in _tokenize(str(row.get("question", ""))):
            counts[f"q:{token}"] += 1
        for token in _tokenize(str(row.get("chosen_text", ""))):
            counts[f"c:{token}"] += 1
        for token in _tokenize(str(row.get("rejected_text", ""))):
            counts[f"c:{token}"] += 1
    most_common = counts.most_common(max_features)
    return {token: idx for idx, (token, _) in enumerate(most_common)}


def _vectorize_with_vocab(vocab: dict[str, int], max_features: int, question: str, candidate_text: str) -> list[float]:
    vec = [0.0] * max_features
    for token in _tokenize(question):
        idx = vocab.get(f"q:{token}")
        if idx is not None:
            vec[idx] += 1.0
    for token in _tokenize(candidate_text):
        idx = vocab.get(f"c:{token}")
        if idx is not None:
            vec[idx] += 1.0
    norm = sum(vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _build_model(torch: object, *, input_dim: int, hidden_dim: int) -> object:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    )


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _seed_everything(seed: int, torch: object) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _import_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for PRM v2 training. Please install torch in the runtime.") from exc
    return torch
