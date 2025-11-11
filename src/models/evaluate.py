"""Shared evaluation helpers for classifier baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass
class EvalResult:
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float


def build_result(model_name: str, y_true, y_pred) -> EvalResult:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return EvalResult(model_name, acc, precision, recall, f1)


def format_markdown_table(results: Iterable[EvalResult]) -> str:
    rows: List[str] = ["| Model | Accuracy | Precision | Recall | F1 |", "| --- | --- | --- | --- | --- |"]
    for res in results:
        rows.append(
            f"| {res.model} | {res.accuracy:.4f} | {res.precision:.4f} | {res.recall:.4f} | {res.f1:.4f} |"
        )
    return "\n".join(rows)
