"""Train classical ML baselines for elephant vs. mice classification.

Design:
- Use NFStream-style flow features derived from CIC-IDS2017.
- Elephant vs. mice labels already constructed upstream (target_traffic).
- Prefer a fixed-size experiment set (~55k flows, ~5% elephants) to match the
  original paper's regime; fall back to on-the-fly downsampling if needed.
"""

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

# Upstream:
# - CONFIG.labeled_csv -> elephant_mice_flows.csv (full, 2.8M+ rows)
# - scripts/01_prepare_data.sh also writes elephant_mice_flows_small.csv (~55,726 rows)
FULL_DATA_PATH: Path = CONFIG.labeled_csv
SMALL_DATA_PATH: Path = FULL_DATA_PATH.parent / "elephant_mice_flows_small.csv"

# Target size if we ever need to downsample from the full set.
TARGET_N = 55_726

# Features aligned with the flow representation used in the paper.
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


def _load_base_frame() -> pd.DataFrame:
    """Load the experiment dataset.

    Priority:
    1. If elephant_mice_flows_small.csv exists, use it directly (deterministic).
    2. Otherwise, load the full labeled CSV and downsample once (stratified).
    """
    if SMALL_DATA_PATH.exists():
        df = pd.read_csv(SMALL_DATA_PATH)
        df.columns = df.columns.str.strip()
        LOGGER.info(
            "Loaded experiment set from %s: %d rows (elephants: %.2f%%)",
            SMALL_DATA_PATH,
            len(df),
            df["target_traffic"].mean() * 100.0,
        )
        return df

    # Fallback: build an experiment-sized sample from full data
    df = pd.read_csv(FULL_DATA_PATH)
    df.columns = df.columns.str.strip()

    required = FEATURE_COLS + ["target_traffic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in labeled CSV: {missing}")

    df = df.dropna(subset=FEATURE_COLS)

    n_total = len(df)
    if n_total <= TARGET_N:
        LOGGER.info(
            "Full dataset smaller than target; using all %d rows (elephants: %.2f%%)",
            n_total,
            df["target_traffic"].mean() * 100.0,
        )
        return df

    df_major = df[df["target_traffic"] == 0]
    df_minor = df[df["target_traffic"] == 1]

    ratio = df_minor.shape[0] / n_total
    n_minor = max(1, int(TARGET_N * ratio))
    n_major = TARGET_N - n_minor

    n_major = min(n_major, df_major.shape[0])
    n_minor = min(n_minor, df_minor.shape[0])

    df_major_sample = df_major.sample(n=n_major, random_state=CONFIG.random_state)
    df_minor_sample = df_minor.sample(n=n_minor, random_state=CONFIG.random_state)

    df_small = pd.concat([df_major_sample, df_minor_sample], ignore_index=True)

    LOGGER.info(
        "Downsampled from %d to %d rows (elephants: %.2f%%)",
        n_total,
        len(df_small),
        df_small["target_traffic"].mean() * 100.0,
    )

    return df_small


def load_data():
    df = _load_base_frame()

    # Ensure feature columns exist in whatever we loaded
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataset: {missing}")

    X = df[FEATURE_COLS]
    y = df["target_traffic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=CONFIG.random_state,
    )

    LOGGER.info(
        "Train/Test split: %d train / %d test "
        "(elephants train: %.2f%%, test: %.2f%%)",
        len(X_train),
        len(X_test),
        y_train.mean() * 100.0,
        y_test.mean() * 100.0,
    )

    return X_train, X_test, y_train, y_test


def get_models():
    """Classical ML baselines as in the paper."""
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
        "dt": DecisionTreeClassifier(
            random_state=CONFIG.random_state,
        ),
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

        LOGGER.info(
            "\n%s",
            classification_report(
                y_test,
                y_pred,
                zero_division=0,
                digits=4,
            ),
        )

    print("# Baseline results (elephant vs. mice)\n")
    print(format_markdown_table(markdown_results))


if __name__ == "__main__":
    run_baselines()
