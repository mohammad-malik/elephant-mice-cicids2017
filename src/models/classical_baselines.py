"""Train classical baselines on the paper-aligned experiment set."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas.util import hash_pandas_object
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
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
DATA_PATH = CONFIG.paper_small_csv
GROUP_SOURCE_PATH = CONFIG.chebyshev_csv
PLOTS_DIR = Path("reports") / "plots"
CONFUSION_MATRIX_PATH = PLOTS_DIR / "confusion_matrix.png"

FEATURE_COLUMNS = list(CONFIG.feature_columns)
GROUP_COLUMNS = ["src_ip", "dst_ip", "src_port", "dst_port"]


def _format_priors(series: pd.Series) -> dict[int, str]:
    return {int(k): f"{v * 100:.4f}%" for k, v in series.items()}


def _load_group_ids(expected_len: int) -> pd.Series:
    groups_df = pd.read_csv(GROUP_SOURCE_PATH, usecols=GROUP_COLUMNS)
    if len(groups_df) != expected_len:
        raise ValueError(
            f"Group source rows ({len(groups_df)}) do not match modeling rows ({expected_len})"
        )
    hashes = hash_pandas_object(groups_df, index=False).astype("uint64")
    return hashes


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    missing = [col for col in [*FEATURE_COLUMNS, "target_traffic"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df["target_traffic"].astype(int)
    groups = _load_group_ids(len(df))

    priors = y.value_counts(normalize=True)
    LOGGER.info("Dataset size=%d | Elephant fraction=%.4f%%", len(df), 100 * y.mean())
    LOGGER.info("Class priors: %s", _format_priors(priors))
    return X, y, groups


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
                ("clf", SVC(kernel="rbf")),
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


def _plot_confusion_matrix(model_name: str, y_true: list[int], y_pred: list[int]) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Mice", "Elephant"],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved confusion matrix to %s", CONFUSION_MATRIX_PATH)


def run_baselines() -> None:
    X, y, groups = load_data()
    models = get_models()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CONFIG.random_state)

    aggregated_results = []
    predictions: dict[str, tuple[list[int], list[int]]] = {}

    for name, estimator in models.items():
        LOGGER.info("Evaluating %s via 5-fold StratifiedGroupKFold", name)
        fold_results = []
        y_true_all: list[int] = []
        y_pred_all: list[int] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            model = clone(estimator)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_results.append(build_result(name, y_test, y_pred))
            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())
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
        predictions[name] = (y_true_all, y_pred_all)

    aggregated_results.sort(key=lambda res: res.f1_mean, reverse=True)
    best = aggregated_results[0]
    best_true, best_pred = predictions[best.model]
    _plot_confusion_matrix(best.model, best_true, best_pred)

    print("# Baseline results (5-fold stratified group CV)\n")
    print(format_markdown_table(aggregated_results))
    return aggregated_results, predictions


if __name__ == "__main__":
    run_baselines()
