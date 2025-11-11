"""Label flows via the paper’s Chebyshev size rule."""
from pathlib import Path

import pandas as pd

from src.config import CONFIG
from src.utils.logging_utils import get_logger

LOGGER = get_logger("label_elephants_chebyshev")
IN_PATH = CONFIG.paper_schema_csv
OUT_PATH = CONFIG.chebyshev_csv


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df.columns = df.columns.str.strip()

    s = df["bidirectional_bytes"].astype(float)
    mean, std = s.mean(), s.std(ddof=0)
    threshold = mean + 3.0 * std

    df["target_traffic"] = (s >= threshold).astype(int)
    df["size_traffic"] = df["target_traffic"].map({0: "mice", 1: "elephant"})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    pct = 100.0 * df["target_traffic"].mean()
    LOGGER.info(
        "Chebyshev threshold: %.2f bytes | elephants: %.2f%% | rows: %d",
        threshold,
        pct,
        len(df),
    )


if __name__ == "__main__":
    main()
