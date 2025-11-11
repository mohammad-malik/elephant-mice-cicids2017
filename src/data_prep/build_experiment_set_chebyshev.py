"""Sample a 55,726-flow experiment set matching the paper's regime."""
from pathlib import Path

import pandas as pd

from src.config import CONFIG
from src.utils.logging_utils import get_logger

LOGGER = get_logger("build_experiment_set_chebyshev")
IN_PATH = CONFIG.chebyshev_csv
OUT_PATH = CONFIG.paper_small_csv
TARGET_N = 55_726


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df.columns = df.columns.str.strip()

    if len(df) > TARGET_N:
        g0 = df[df["target_traffic"] == 0]
        g1 = df[df["target_traffic"] == 1]
        r = len(g1) / len(df)
        n1 = max(1, int(TARGET_N * r))
        n0 = TARGET_N - n1
        n0 = min(n0, len(g0))
        n1 = min(n1, len(g1))
        df = pd.concat(
            [
                g0.sample(n=n0, random_state=CONFIG.random_state),
                g1.sample(n=n1, random_state=CONFIG.random_state),
            ],
            ignore_index=True,
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    LOGGER.info(
        "Experiment set: %d rows | elephants: %.2f%%",
        len(df),
        100.0 * df["target_traffic"].mean(),
    )


if __name__ == "__main__":
    main()
