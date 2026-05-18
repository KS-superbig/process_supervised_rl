from __future__ import annotations

import ast
from dataclasses import dataclass
import operator
import re
from typing import Any

_STEP_MARKER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Step\s*\d+[\.:]\s*|\d+\.\s+)",
    re.IGNORECASE,
)
_EQUATION_PATTERN = re.compile(
    r"(?<![A-Za-z_])([\$\d,\.\s\+\-\*\/×÷xX\(\)%]*(?:\bof\b[\$\d,\.\s\+\-\*\/×÷xX\(\)%]*)?)\s*=\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@dataclass(frozen=True)
class EquationCheck:
    expr: str
    expected: float
    actual: float
    ok: bool

    def as_tuple(self) -> tuple[str, float, float, bool]:
        return (self.expr, self.expected, self.actual, self.ok)


def split_into_steps(trace: str) -> list[str]:
    """Split a generated reasoning trace into coarse reasoning steps."""
    text = trace.strip()
    if not text:
        return []

    if _STEP_MARKER_PATTERN.search(text):
        chunks = _STEP_MARKER_PATTERN.split(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    if "\n\n" in text:
        return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    return [line.strip() for line in text.splitlines() if len(line.strip()) > 5]


def normalize_expr(expr: str) -> str:
    """Normalize common GSM8K arithmetic notation into a Python AST expression."""
    normalized = expr.replace("$", "").replace(",", "")
    normalized = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", normalized)
    normalized = re.sub(r"\bof\b", "*", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("×", "*").replace("÷", "/")
    normalized = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d|\$)", "*", normalized)
    normalized = re.sub(r"(?<=\))\s*[xX]\s*(?=\d|\()", "*", normalized)
    return normalized.strip()


def safe_eval(expr: str) -> float | None:
    """Evaluate a basic arithmetic expression without exposing Python eval."""
    normalized = normalize_expr(expr)
    if not normalized or not re.fullmatch(r"[\d\.\s\+\-\*\/\(\)]+", normalized):
        return None
    try:
        node = ast.parse(normalized, mode="eval")
        value = _eval_ast(node)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
        return None
    return float(value)


def verify_step(step_text: str, tol: float = 1e-3) -> dict[str, Any]:
    """Check explicit arithmetic equations inside one reasoning step."""
    matches = _EQUATION_PATTERN.findall(step_text)
    if not matches:
        return _empty_result()

    details: list[EquationCheck] = []
    passed = 0
    for lhs, rhs in matches:
        actual = safe_eval(lhs)
        if actual is None:
            continue
        try:
            expected = float(rhs.replace(",", ""))
        except ValueError:
            continue
        ok = abs(actual - expected) <= tol
        details.append(EquationCheck(lhs.strip(), expected, actual, ok))
        if ok:
            passed += 1

    if not details:
        return _empty_result()

    return {
        "has_equation": True,
        "all_correct": passed == len(details),
        "checked": len(details),
        "passed": passed,
        "details": [detail.as_tuple() for detail in details],
    }


def python_verifier_reward(
    completion: str,
    final_correct: bool,
    *,
    alpha_step: float = 0.3,
    beta_pass_rate: float = 0.2,
    gamma_no_eq_penalty: float = 0.0,
) -> dict[str, Any]:
    """Compute a trajectory reward from final correctness plus arithmetic step checks."""
    steps = split_into_steps(completion)
    step_scores: list[float] = []
    n_with_eq = 0
    total_checked = 0
    total_passed = 0

    for step in steps:
        result = verify_step(step)
        if not result["has_equation"]:
            continue
        n_with_eq += 1
        total_checked += int(result["checked"])
        total_passed += int(result["passed"])
        step_scores.append(1.0 if result["all_correct"] else 0.0)

    r_final = 1.0 if final_correct else 0.0
    r_step_mean = sum(step_scores) / len(step_scores) if step_scores else 0.0
    r_pass_rate = total_passed / total_checked if total_checked else 0.0
    reward = (
        r_final
        + alpha_step * r_step_mean
        + beta_pass_rate * r_pass_rate
        - gamma_no_eq_penalty * (1.0 if n_with_eq == 0 else 0.0)
    )

    return {
        "reward": float(reward),
        "r_final": r_final,
        "r_step_mean": float(r_step_mean),
        "r_pass_rate": float(r_pass_rate),
        "step_scores": step_scores,
        "n_steps": len(steps),
        "n_steps_with_eq": n_with_eq,
        "total_checked": total_checked,
        "total_passed": total_passed,
    }


def _empty_result() -> dict[str, Any]:
    return {
        "has_equation": False,
        "all_correct": None,
        "checked": 0,
        "passed": 0,
        "details": [],
    }


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ValueError("non-numeric constant")
    if isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported binary operator")
        return float(op(_eval_ast(node.left), _eval_ast(node.right)))
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported unary operator")
        return float(op(_eval_ast(node.operand)))
    raise ValueError("unsupported expression")
