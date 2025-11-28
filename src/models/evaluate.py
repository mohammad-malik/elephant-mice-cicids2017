"""Shared evaluation helpers for classifier baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass
class EvalResult:
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None


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
    auc_roc_mean: Optional[float] = None
    auc_roc_std: Optional[float] = None
    auc_pr_mean: Optional[float] = None
    auc_pr_std: Optional[float] = None


def build_result(model_name: str, y_true, y_pred, y_score=None) -> EvalResult:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc_roc = None
    auc_pr = None
    if y_score is not None:
        y_true_arr = np.asarray(y_true)
        if np.unique(y_true_arr).size > 1:
            try:
                auc_roc = roc_auc_score(y_true, y_score)
                auc_pr = average_precision_score(y_true, y_score)
            except ValueError:
                pass
    return EvalResult(model_name, acc, precision, recall, f1, auc_roc, auc_pr)


def aggregate_results(model_name: str, results: Iterable[EvalResult]) -> AggregateEvalResult:
    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc_roc": [],
        "auc_pr": [],
    }
    for res in results:
        metrics["accuracy"].append(res.accuracy)
        metrics["precision"].append(res.precision)
        metrics["recall"].append(res.recall)
        metrics["f1"].append(res.f1)
        if res.auc_roc is not None:
            metrics["auc_roc"].append(res.auc_roc)
        if res.auc_pr is not None:
            metrics["auc_pr"].append(res.auc_pr)

    def _mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        arr = np.array(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    def _mean_std_optional(values: list[float]) -> tuple[Optional[float], Optional[float]]:
        if not values:
            return None, None
        arr = np.array(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    acc_mean, acc_std = _mean_std(metrics["accuracy"])
    pre_mean, pre_std = _mean_std(metrics["precision"])
    rec_mean, rec_std = _mean_std(metrics["recall"])
    f1_mean, f1_std = _mean_std(metrics["f1"])
    auc_roc_mean, auc_roc_std = _mean_std_optional(metrics["auc_roc"])
    auc_pr_mean, auc_pr_std = _mean_std_optional(metrics["auc_pr"])

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
        auc_roc_mean=auc_roc_mean,
        auc_roc_std=auc_roc_std,
        auc_pr_mean=auc_pr_mean,
        auc_pr_std=auc_pr_std,
    )


def format_markdown_table(results: Iterable[AggregateEvalResult]) -> str:
    rows: List[str] = [
        "| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    def _fmt(mean: float, std: float) -> str:
        return f"{mean:.4f} ± {std:.4f}"

    def _fmt_optional(mean: Optional[float], std: Optional[float]) -> str:
        if mean is None or std is None:
            return "N/A"
        return f"{mean:.4f} ± {std:.4f}"

    for res in results:
        rows.append(
            "| {model} | {acc} | {pre} | {rec} | {f1} | {auc_roc} | {auc_pr} |".format(
                model=res.model,
                acc=_fmt(res.accuracy_mean, res.accuracy_std),
                pre=_fmt(res.precision_mean, res.precision_std),
                rec=_fmt(res.recall_mean, res.recall_std),
                f1=_fmt(res.f1_mean, res.f1_std),
                auc_roc=_fmt_optional(res.auc_roc_mean, res.auc_roc_std),
                auc_pr=_fmt_optional(res.auc_pr_mean, res.auc_pr_std),
            )
        )
    return "\n".join(rows)
