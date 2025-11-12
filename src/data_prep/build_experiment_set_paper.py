"""Sample a 55,726-flow paper base for downstream labeling."""

from typing import Dict

import pandas as pd

from src.config import CONFIG
from src.utils.logging_utils import get_logger

LOGGER = get_logger("build_experiment_set_paper")
IN_PATH = CONFIG.paper_schema_csv
OUT_PATH = CONFIG.paper_base_csv
TARGET_N = 55_726


def _log_byte_stats(series: pd.Series) -> None:
    clean = series.dropna()
    if clean.empty:
        LOGGER.warning("No bidirectional bytes to compute statistics on the paper base.")
        return

    stats: Dict[str, float] = {
        "mean": clean.mean(),
        "std": clean.std(ddof=1),
        "min": clean.min(),
        "p50": clean.quantile(0.5),
        "p90": clean.quantile(0.9),
        "p99": clean.quantile(0.99),
        "p99.9": clean.quantile(0.999),
    }
    LOGGER.info(
        "Paper base bytes stats (n=%d): mean=%.2f | std=%.2f | min=%.2f | p50=%.2f | p90=%.2f | p99=%.2f | p99.9=%.2f",
        len(clean),
        stats["mean"],
        stats["std"],
        stats["min"],
        stats["p50"],
        stats["p90"],
        stats["p99"],
        stats["p99.9"],
    )


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df.columns = df.columns.str.strip()
    orig_len = len(df)

    if orig_len > TARGET_N:
        df = df.sample(n=TARGET_N, random_state=CONFIG.random_state)
        LOGGER.info("Sampled %d rows from %d → %s", TARGET_N, orig_len, OUT_PATH.name)
    else:
        LOGGER.info("Dataset already %d rows; no sampling performed", orig_len)

    _log_byte_stats(df["bidirectional_bytes"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    LOGGER.info("Wrote paper base to %s (%d rows)", OUT_PATH, len(df))


if __name__ == "__main__":
    main()
