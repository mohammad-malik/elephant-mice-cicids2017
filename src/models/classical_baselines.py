"""Train classical ML baselines for elephant vs. mice classification."""
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
DATA_PATH = CONFIG.labeled_csv
FEATURE_COLS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "total_bytes",
    "total_pkts",
]


def load_data(data_path: Path = DATA_PATH):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    X = df[FEATURE_COLS]
    y = df["target_traffic"]
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=CONFIG.random_state
    )


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


def run_baselines() -> None:
    X_train, X_test, y_train, y_test = load_data()
    models = get_models()

    markdown_results = []
    for name, model in models.items():
        LOGGER.info("Training %s", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        markdown_results.append(build_result(name, y_test, y_pred))
        LOGGER.info("\n%s", classification_report(y_test, y_pred, zero_division=0))

    print("# Baseline results (elephant vs. mice)\n")
    print(format_markdown_table(markdown_results))


if __name__ == "__main__":
    run_baselines()
