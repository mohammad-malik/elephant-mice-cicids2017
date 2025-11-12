"""Shared evaluation helpers for classifier baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass
class EvalResult:
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass
class AggregateEvalResult:
    model: str
    accuracy_mean: float
    accuracy_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    f1_mean: float
    f1_std: float


def build_result(model_name: str, y_true, y_pred) -> EvalResult:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return EvalResult(model_name, acc, precision, recall, f1)


def aggregate_results(model_name: str, results: Iterable[EvalResult]) -> AggregateEvalResult:
    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    for res in results:
        metrics["accuracy"].append(res.accuracy)
        metrics["precision"].append(res.precision)
        metrics["recall"].append(res.recall)
        metrics["f1"].append(res.f1)

    def _mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        arr = np.array(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    acc_mean, acc_std = _mean_std(metrics["accuracy"])
    pre_mean, pre_std = _mean_std(metrics["precision"])
    rec_mean, rec_std = _mean_std(metrics["recall"])
    f1_mean, f1_std = _mean_std(metrics["f1"])

    return AggregateEvalResult(
        model=model_name,
        accuracy_mean=acc_mean,
        accuracy_std=acc_std,
        precision_mean=pre_mean,
        precision_std=pre_std,
        recall_mean=rec_mean,
        recall_std=rec_std,
        f1_mean=f1_mean,
        f1_std=f1_std,
    )


def format_markdown_table(results: Iterable[AggregateEvalResult]) -> str:
    rows: List[str] = ["| Model | Accuracy | Precision | Recall | F1 |", "| --- | --- | --- | --- | --- |"]

    def _fmt(mean: float, std: float) -> str:
        return f"{mean:.4f} ± {std:.4f}"

    for res in results:
        rows.append(
            "| {model} | {acc} | {pre} | {rec} | {f1} |".format(
                model=res.model,
                acc=_fmt(res.accuracy_mean, res.accuracy_std),
                pre=_fmt(res.precision_mean, res.precision_std),
                rec=_fmt(res.recall_mean, res.recall_std),
                f1=_fmt(res.f1_mean, res.f1_std),
            )
        )
    return "\n".join(rows)
