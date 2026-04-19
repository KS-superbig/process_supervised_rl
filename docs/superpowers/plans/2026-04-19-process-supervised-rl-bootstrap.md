# Process-Supervised RL Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the initial research-grade project scaffold for a process-supervised reasoning experiment that supports local development and remote execution.

**Architecture:** Keep the repository thin and modular. Put all reusable Python logic under `src/`, keep runnable entrypoints in `scripts/`, keep settings in `configs/`, and make data preparation deterministic so the same code can run locally on debug subsets and remotely on full data.

**Tech Stack:** Python 3.11+, pytest, YAML configs, JSONL data artifacts

---

## File Map

- Create: `README.md`
- Create: `.gitignore`
- Create: `configs/data/gsm8k.yaml`
- Create: `configs/reward/process_reward_v0.yaml`
- Create: `configs/train/sft_baseline.yaml`
- Create: `configs/eval/baseline_eval.yaml`
- Create: `scripts/prepare_gsm8k.py`
- Create: `scripts/build_debug_subset.py`
- Create: `src/psrl/__init__.py`
- Create: `src/psrl/config.py`
- Create: `src/psrl/data/__init__.py`
- Create: `src/psrl/data/schema.py`
- Create: `src/psrl/data/gsm8k.py`
- Create: `src/psrl/data/normalize.py`
- Create: `src/psrl/data/step_splitter.py`
- Create: `tests/test_config.py`
- Create: `tests/data/test_normalize.py`
- Create: `tests/data/test_step_splitter.py`

### Task 1: Scaffold Repository Layout

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `configs/.gitkeep`
- Create: `data/raw/.gitkeep`
- Create: `data/interim/.gitkeep`
- Create: `data/processed/.gitkeep`
- Create: `data/debug/.gitkeep`
- Create: `experiments/.gitkeep`
- Create: `logs/.gitkeep`

- [ ] **Step 1: Write the failing smoke test for repository layout**

```python
from pathlib import Path


def test_expected_project_directories_exist():
    for rel_path in [
        "configs",
        "data/raw",
        "data/interim",
        "data/processed",
        "data/debug",
        "experiments",
        "logs",
    ]:
        assert Path(rel_path).exists(), rel_path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layout.py -v`
Expected: FAIL because directories or the test file do not exist yet.

- [ ] **Step 3: Create the directory scaffold and minimal repo docs**

```text
README.md
.gitignore
configs/
data/raw/
data/interim/
data/processed/
data/debug/
experiments/
logs/
```

`README.md` should explain:
- local development only
- remote server runs data prep, inference, training, evaluation
- phase-1 scope is GSM8K + rule-based process reward

`.gitignore` should exclude:
- Python cache
- virtual environments
- processed data outputs
- checkpoints
- logs

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_layout.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md .gitignore configs data experiments logs tests/test_layout.py
git commit -m "feat: add initial project scaffold"
```

### Task 2: Add Config Loader and Base Config Files

**Files:**
- Create: `src/psrl/__init__.py`
- Create: `src/psrl/config.py`
- Create: `configs/data/gsm8k.yaml`
- Create: `configs/reward/process_reward_v0.yaml`
- Create: `configs/train/sft_baseline.yaml`
- Create: `configs/eval/baseline_eval.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing config test**

```python
from pathlib import Path

from psrl.config import load_yaml_config


def test_load_yaml_config_reads_mapping():
    config = load_yaml_config(Path("configs/data/gsm8k.yaml"))
    assert config["dataset_name"] == "gsm8k"
    assert config["debug_limit"] == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with import error or missing file error.

- [ ] **Step 3: Write minimal config loader and YAML files**

```python
from pathlib import Path
import yaml


def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}
```

Example `configs/data/gsm8k.yaml`:

```yaml
dataset_name: gsm8k
raw_path: data/raw/gsm8k
processed_train_path: data/processed/gsm8k_train.jsonl
processed_test_path: data/processed/gsm8k_test.jsonl
debug_limit: 100
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/psrl/__init__.py src/psrl/config.py configs tests/test_config.py
git commit -m "feat: add base config loader"
```

### Task 3: Define Sample Schema and Normalization Helpers

**Files:**
- Create: `src/psrl/data/__init__.py`
- Create: `src/psrl/data/schema.py`
- Create: `src/psrl/data/normalize.py`
- Test: `tests/data/test_normalize.py`

- [ ] **Step 1: Write the failing normalization tests**

```python
from psrl.data.normalize import normalize_final_answer


def test_normalize_final_answer_extracts_hash_answer():
    raw = "The answer is 42 #### 42"
    assert normalize_final_answer(raw) == "42"


def test_normalize_final_answer_trims_whitespace():
    assert normalize_final_answer("  3/4 ") == "3/4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_normalize.py -v`
Expected: FAIL because module does not exist.

- [ ] **Step 3: Write minimal schema and normalization code**

```python
from dataclasses import dataclass, field


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
    metadata: dict = field(default_factory=dict)
```

```python
def normalize_final_answer(raw: str) -> str:
    text = raw.strip()
    if "####" in text:
        text = text.split("####")[-1].strip()
    return " ".join(text.split())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_normalize.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/psrl/data tests/data/test_normalize.py
git commit -m "feat: add sample schema and answer normalization"
```

### Task 4: Implement Step Splitter

**Files:**
- Create: `src/psrl/data/step_splitter.py`
- Test: `tests/data/test_step_splitter.py`

- [ ] **Step 1: Write the failing step splitter tests**

```python
from psrl.data.step_splitter import split_solution_steps


def test_split_solution_steps_uses_newlines():
    raw = "Step 1: Add 2 and 3\nStep 2: Get 5"
    assert split_solution_steps(raw) == ["Step 1: Add 2 and 3", "Step 2: Get 5"]


def test_split_solution_steps_drops_empty_lines():
    raw = "First line\n\nSecond line"
    assert split_solution_steps(raw) == ["First line", "Second line"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_step_splitter.py -v`
Expected: FAIL because function does not exist.

- [ ] **Step 3: Write minimal splitter implementation**

```python
def split_solution_steps(raw: str) -> list[str]:
    lines = [line.strip() for line in raw.splitlines()]
    return [line for line in lines if line]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_step_splitter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/psrl/data/step_splitter.py tests/data/test_step_splitter.py
git commit -m "feat: add initial step splitter"
```

### Task 5: Implement GSM8K Preparation Script

**Files:**
- Create: `src/psrl/data/gsm8k.py`
- Create: `scripts/prepare_gsm8k.py`
- Create: `scripts/build_debug_subset.py`

- [ ] **Step 1: Write the failing ingestion test**

```python
from psrl.data.gsm8k import build_reasoning_sample


def test_build_reasoning_sample_populates_fields():
    row = {"question": "2+2?", "answer": "We compute #### 4"}
    sample = build_reasoning_sample("gsm8k-train-1", "train", row)
    assert sample.answer_final_normalized == "4"
    assert sample.steps == ["We compute #### 4"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_gsm8k.py -v`
Expected: FAIL because module or test file does not exist.

- [ ] **Step 3: Write minimal ingestion and CLI scripts**

`src/psrl/data/gsm8k.py` should expose:

```python
def build_reasoning_sample(sample_id: str, split: str, row: dict) -> ReasoningSample:
    ...
```

`scripts/prepare_gsm8k.py` should:
- load raw rows from a JSONL path
- normalize each row
- write processed JSONL

`scripts/build_debug_subset.py` should:
- read a processed JSONL
- write the first `N` rows to `data/debug/`

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_gsm8k.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/psrl/data/gsm8k.py scripts/prepare_gsm8k.py scripts/build_debug_subset.py tests/data/test_gsm8k.py
git commit -m "feat: add gsm8k preparation pipeline"
```

### Task 6: Verification Sweep

**Files:**
- Verify: `README.md`
- Verify: `configs/`
- Verify: `src/psrl/`
- Verify: `scripts/`
- Verify: `tests/`

- [ ] **Step 1: Run targeted tests**

Run: `pytest tests/test_layout.py tests/test_config.py tests/data -v`
Expected: all PASS

- [ ] **Step 2: Run a config smoke command**

Run: `python scripts/prepare_gsm8k.py --help`
Expected: usage text prints without traceback

- [ ] **Step 3: Review git diff**

Run: `git status --short`
Expected: only planned scaffold files are changed

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "chore: bootstrap phase-1 project structure"
```
