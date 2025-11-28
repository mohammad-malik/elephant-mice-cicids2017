"""Train classical baselines on the paper-aligned experiment set."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from pandas.util import hash_pandas_object
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import CONFIG
from src.models.evaluate import aggregate_results, build_result, format_markdown_table
from src.utils.logging_utils import get_logger

LOGGER = get_logger("classical_baselines")
DATA_PATH = CONFIG.chebyshev_csv
GROUP_SOURCE_PATH = CONFIG.chebyshev_csv
PLOTS_DIR = Path("reports") / "plots"
MODEL_ARTIFACT_PATH = Path("artifacts") / "model.joblib"

GROUP_COLUMNS = ["src_ip", "dst_ip", "src_port", "dst_port"]


def _get_feature_and_target_cols(feature_set: str) -> tuple[list[str], str]:
    if feature_set == "baseline":
        return list(CONFIG.feature_columns), "target_traffic"
    if feature_set == "non_leaky":
        return list(CONFIG.feature_columns_non_leaky), "target_99pct"
    if feature_set == "early":
        return list(CONFIG.feature_columns_early), "target_99pct"
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _format_priors(series: pd.Series) -> dict[int, str]:
    return {int(k): f"{v * 100:.4f}%" for k, v in series.items()}


def _load_group_ids(groups_df: pd.DataFrame, expected_len: int) -> pd.Series:
    missing = [col for col in GROUP_COLUMNS if col not in groups_df.columns]
    if missing:
        raise ValueError(f"Missing group columns in dataset: {missing}")
    if len(groups_df) != expected_len:
        raise ValueError(
            f"Group source rows ({len(groups_df)}) do not match modeling rows ({expected_len})"
        )
    hashes = hash_pandas_object(groups_df, index=False).astype("uint64")
    return hashes


def load_data(feature_set: str = "baseline"):
    feature_cols, target_col = _get_feature_and_target_cols(feature_set)
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    missing = [col for col in [*feature_cols, target_col] if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for feature_set '{feature_set}': {missing}"
        )

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[target_col].astype(int)
    groups = _load_group_ids(df[GROUP_COLUMNS], len(df))

    priors = y.value_counts(normalize=True)
    LOGGER.info(
        "Feature set=%s | target=%s | dataset size=%d | positive fraction=%.4f%%",
        feature_set,
        target_col,
        len(df),
        100 * y.mean(),
    )
    LOGGER.info("Class priors: %s", _format_priors(priors))
    return X, y, groups, feature_cols, target_col


def get_models():
    return {
        "log_reg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True)),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=CONFIG.random_state,
        ),
        "lda": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LinearDiscriminantAnalysis()),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "gnb": GaussianNB(),
        "dt": DecisionTreeClassifier(random_state=CONFIG.random_state),
    }


def _get_prediction_scores(model, X_test) -> list[float] | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1].tolist()
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test).tolist()
    return None


def _plot_confusion_matrix(model_name: str, y_true: list[int], y_pred: list[int], suffix: str = "") -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Mice", "Elephant"],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f"confusion_matrix{suffix}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved confusion matrix to %s", out_path)


def _plot_roc_pr(model_name: str, y_true: list[int], y_score: list[float], feature_set: str) -> None:
    if not y_score:
        return
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax_roc, name=model_name)
    ax_roc.set_title(f"ROC Curve - {model_name} ({feature_set})")
    plt.tight_layout()
    roc_path = PLOTS_DIR / f"{feature_set}_roc.png"
    fig_roc.savefig(roc_path, dpi=300)
    plt.close(fig_roc)
    LOGGER.info("Saved ROC curve to %s", roc_path)

    fig_pr, ax_pr = plt.subplots(figsize=(5.5, 4.5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax_pr, name=model_name)
    ax_pr.set_title(f"Precision-Recall Curve - {model_name} ({feature_set})")
    plt.tight_layout()
    pr_path = PLOTS_DIR / f"{feature_set}_pr.png"
    fig_pr.savefig(pr_path, dpi=300)
    plt.close(fig_pr)
    LOGGER.info("Saved PR curve to %s", pr_path)


def run_baselines(feature_set: str = "baseline", export_model: bool = False) -> tuple[list, dict]:
    X, y, groups, feature_cols, target_col = load_data(feature_set=feature_set)
    models = get_models()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CONFIG.random_state)

    aggregated_results = []
    predictions: dict[str, tuple[list[int], list[int], list[float] | None]] = {}

    for name, estimator in models.items():
        LOGGER.info("Evaluating %s via 5-fold StratifiedGroupKFold", name)
        fold_results = []
        y_true_all: list[int] = []
        y_pred_all: list[int] = []
        y_score_all: list[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            model = clone(estimator)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = _get_prediction_scores(model, X_test)

            fold_results.append(build_result(name, y_test, y_pred, y_score))
            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())
            if y_score is not None:
                y_score_all.extend(y_score)
            LOGGER.info(
                "Fold %d (%s) | train=%d test=%d | elephants in test=%d",
                fold_idx,
                name,
                len(train_idx),
                len(test_idx),
                int(y_test.sum()),
            )

        aggregated = aggregate_results(name, fold_results)
        aggregated_results.append(aggregated)
        predictions[name] = (y_true_all, y_pred_all, y_score_all if y_score_all else None)

    aggregated_results.sort(key=lambda res: res.f1_mean, reverse=True)
    best = aggregated_results[0]
    best_true, best_pred, best_scores = predictions[best.model]
    suffix = f"_{feature_set}" if feature_set != "baseline" else ""
    _plot_confusion_matrix(best.model, best_true, best_pred, suffix=suffix)
    if best_scores:
        _plot_roc_pr(best.model, best_true, best_scores, feature_set)

    if export_model:
        LOGGER.info("Exporting best model '%s' to %s", best.model, MODEL_ARTIFACT_PATH)
        MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_estimator = clone(models[best.model])
        final_estimator.fit(X, y)
        joblib.dump(
            {
                "model": final_estimator,
                "features": feature_cols,
                "label": target_col,
                "feature_set": feature_set,
            },
            MODEL_ARTIFACT_PATH,
        )
        LOGGER.info("Persisted best model artifact.")

    print(f"# Baseline results - feature_set={feature_set} (5-fold stratified group CV)\n")
    print(format_markdown_table(aggregated_results))
    return aggregated_results, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical baselines on elephant/mice flows.")
    parser.add_argument(
        "--export-model",
        action="store_true",
        help="Fit the best cross-validated model on the full dataset and save to artifacts/model.joblib.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="baseline",
        choices=["baseline", "non_leaky", "early"],
        help="Which feature configuration to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_baselines(feature_set=args.feature_set, export_model=args.export_model)


if __name__ == "__main__":
    main()
