"""Train classical baselines on the paper-aligned experiment set."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import CONFIG
from src.models.evaluate import build_result, format_markdown_table
from src.utils.logging_utils import get_logger

LOGGER = get_logger("classical_baselines")
DATA_PATH = CONFIG.paper_small_csv

BASE_FEATURES = [
    "src_port",
    "dst_port",
    "bidirectional_first_seen_ms",
    "bidirectional_last_seen_ms",
    "bidirectional_bytes",
]
INCLUDE_SIZE_TRAFFIC = True


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    use_cols = BASE_FEATURES.copy()
    if INCLUDE_SIZE_TRAFFIC and "size_traffic" in df.columns:
        df = pd.get_dummies(df, columns=["size_traffic"], drop_first=True)
        use_cols += [c for c in df.columns if c.startswith("size_traffic_")]

    for col in ("src_ip", "dst_ip"):
        if col in df.columns:
            df = df.drop(columns=[col])

    X = df[use_cols]
    y = df["target_traffic"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=CONFIG.random_state,
    )

    LOGGER.info(
        "Train/Test: %d/%d | elephants: %.2f%% / %.2f%% | features: %s",
        len(X_train),
        len(X_test),
        100 * y_train.mean(),
        100 * y_test.mean(),
        use_cols,
    )
    return X_train, X_test, y_train, y_test


def get_models():
    return {
        "log_reg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, n_jobs=-1)),
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


def run_baselines() -> None:
    X_train, X_test, y_train, y_test = load_data()
    models = get_models()

    results = []
    for name, model in models.items():
        LOGGER.info("Training %s", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append(build_result(name, y_test, y_pred))
        LOGGER.info(
            "\n%s",
            classification_report(y_test, y_pred, zero_division=0, digits=4),
        )

    print("# Baseline results (paper-faithful: Chebyshev + paper features)\n")
    print(format_markdown_table(results))


if __name__ == "__main__":
    run_baselines()
