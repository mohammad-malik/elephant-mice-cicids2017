"""Label flows via the paper’s Chebyshev rule, optionally aggregating on (src_ip,dst_ip,src_port,dst_port)."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import CONFIG
from src.utils.logging_utils import get_logger

LOGGER = get_logger("label_elephants_chebyshev")
IN_PATH = CONFIG.paper_base_csv
INTERMEDIATE_OUT_PATH = CONFIG.chebyshev_csv
PROCESSED_OUT_PATH = CONFIG.paper_small_csv
THRESHOLD_PATH = Path("artifacts") / "threshold.json"
KEY_COLS = ["src_ip", "dst_ip", "src_port", "dst_port"]
FEATURE_COLUMNS = list(CONFIG.feature_columns)
MODEL_COLUMNS = [*FEATURE_COLUMNS, "target_traffic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label elephant flows via the paper's Chebyshev size rule."
    )
    parser.add_argument(
        "--per-flow",
        action="store_true",
        help="Skip 4-tuple aggregation and apply the threshold per flow (legacy).",
    )
    return parser.parse_args()


def _threshold(series: pd.Series) -> tuple[float, float, float]:
    clean = series.dropna()
    if clean.empty:
        raise ValueError("No data to compute Chebyshev threshold")
    mu = clean.mean()
    sigma = clean.std(ddof=1)
    return mu + 3.0 * sigma, mu, sigma


def _add_flow_bytes(df: pd.DataFrame) -> pd.Series:
    series = pd.to_numeric(df["bidirectional_bytes"], errors="coerce").fillna(0.0)
    df["bidirectional_bytes"] = series
    return series


def _aggregate_4tuple(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(KEY_COLS, as_index=False, observed=False)["bidirectional_bytes"]
        .sum()
        .rename(columns={"bidirectional_bytes": "bytes_4tuple"})
    )


def _log_counts(
    counts: dict[int, int], method: str, tuple_count: int, threshold: float, total: int
) -> None:
    elephants = counts.get(1, 0)
    LOGGER.info(
        "Chebyshev threshold (%s): %.2f bytes | elephants: %.2f%% (%d/%d flows) | tuples=%d",
        method,
        threshold,
        100.0 * elephants / total if total else 0.0,
        elephants,
        total,
        tuple_count,
    )
    LOGGER.info("Label counts: %s", counts)


def _persist_threshold(
    threshold: float,
    mu: float,
    sigma: float,
    counts: dict[int, int],
    total: int,
    method: str,
    tuple_count: int,
) -> None:
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_PATH.write_text(
        json.dumps(
            {
                "method": method,
                "threshold_bytes": threshold,
                "mean_bytes": mu,
                "std_bytes": sigma,
                "elephant_fraction": float(counts.get(1, 0) / total if total else 0.0),
                "num_flows": total,
                "num_tuples": tuple_count,
            },
            indent=2,
        )
    )


def _write_outputs(df: pd.DataFrame) -> None:
    if missing := [col for col in MODEL_COLUMNS if col not in df.columns]:
        raise ValueError(f"Missing required model columns: {missing}")

    INTERMEDIATE_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERMEDIATE_OUT_PATH, index=False)
    LOGGER.info("Wrote labeled flows (full schema) to %s", INTERMEDIATE_OUT_PATH)

    PROCESSED_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_model = df[MODEL_COLUMNS].copy()
    df_model.to_csv(PROCESSED_OUT_PATH, index=False)
    LOGGER.info(
        "Wrote modeling subset (%d rows, %d cols) to %s",
        len(df_model),
        len(df_model.columns),
        PROCESSED_OUT_PATH,
    )


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(IN_PATH)
    df.columns = df.columns.str.strip()
    series = _add_flow_bytes(df)

    agg = _aggregate_4tuple(df)
    threshold, mu, sigma = _threshold(agg["bytes_4tuple"])
    tuple_count = len(agg)

    if args.per_flow:
        hook = "flow"
        df["size_traffic"] = series
        df["target_traffic"] = (series >= threshold).astype(int)
    else:
        hook = "4tuple"
        agg["target_traffic"] = (agg["bytes_4tuple"] >= threshold).astype(int)
        df = df.merge(
            agg[[*KEY_COLS, "bytes_4tuple", "target_traffic"]],
            on=KEY_COLS,
            how="left",
        )
        df["target_traffic"] = df["target_traffic"].fillna(0).astype(int)
        df["size_traffic"] = df["bytes_4tuple"].fillna(df["bidirectional_bytes"])
        df = df.drop(columns=["bytes_4tuple"])

    counts = df["target_traffic"].value_counts().to_dict()
    total = len(df)
    _log_counts(counts, hook, tuple_count, threshold, total)
    _persist_threshold(threshold, mu, sigma, counts, total, hook, tuple_count)

    _write_outputs(df)


if __name__ == "__main__":
    main(parse_args())
