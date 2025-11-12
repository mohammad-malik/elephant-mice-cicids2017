"""Periodic Chebyshev threshold recomputation on a sliding time window."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Final

import pandas as pd

from src.config import CONFIG
from src.utils.logging_utils import get_logger

LOGGER = get_logger("online_threshold_updater")

KEY_COLS: Final[list[str]] = ["src_ip", "dst_ip", "src_port", "dst_port"]
BYTES_COL: Final[str] = "bidirectional_bytes"
FIRST_SEEN_COL: Final[str] = "bidirectional_first_seen_ms"
THRESHOLD_PATH: Final[Path] = Path("artifacts") / "threshold.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute the Chebyshev elephant-flow threshold on a sliding window."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=CONFIG.paper_schema_csv,
        help="CSV file containing the paper-aligned GLF schema.",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="7d",
        help="Sliding window size (pandas Timedelta syntax, e.g., '7d', '12h').",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="300s",
        help="Sleep interval between refresh cycles (e.g., '300s', '5m').",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=THRESHOLD_PATH,
        help="Output JSON path for the refreshed threshold.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Compute the threshold a single time and exit (no periodic loop).",
    )
    return parser.parse_args()


def _to_timedelta(value: str) -> pd.Timedelta:
    try:
        return pd.to_timedelta(value)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise argparse.ArgumentTypeError(f"Invalid timedelta value: {value}") from exc


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Source CSV does not exist: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [col for col in [*KEY_COLS, BYTES_COL, FIRST_SEEN_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def _prepare_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _compute_threshold(agg: pd.DataFrame) -> tuple[float, float, float]:
    clean = agg["bytes_4tuple"].dropna()
    if clean.empty:
        raise ValueError("No data available to compute threshold in current window.")
    mu = float(clean.mean())
    sigma = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    return float(mu + 3.0 * sigma), mu, sigma


def _persist(
    destination: Path,
    *,
    method: str,
    threshold: float,
    mu: float,
    sigma: float,
    window_arg: str,
    asof_ms: int,
    tuple_count: int,
    flow_count: int,
    elephant_fraction: float,
    source_path: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "threshold_bytes": threshold,
        "mean_bytes": mu,
        "std_bytes": sigma,
        "window": window_arg,
        "asof": asof_ms,
        "num_tuples": tuple_count,
        "num_flows": flow_count,
        "elephant_fraction": elephant_fraction,
        "source_path": str(source_path),
    }
    destination.write_text(json.dumps(payload, indent=2))
    LOGGER.info(
        "Persisted threshold=%.2f (μ=%.2f, σ=%.2f) | elephants=%.4f | tuples=%d | flows=%d",
        threshold,
        mu,
        sigma,
        elephant_fraction,
        tuple_count,
        flow_count,
    )


def _refresh_once(source: Path, window_arg: str, destination: Path) -> None:
    df = _load_csv(source)
    LOGGER.info("Loaded %d rows from %s", len(df), source)

    window = _to_timedelta(window_arg)
    if window <= pd.Timedelta(0):
        raise ValueError(f"Window must be positive, got {window_arg!r}")

    first_seen = _prepare_numeric(df[FIRST_SEEN_COL])
    if first_seen.isna().all():
        raise ValueError("No valid timestamps found in source data.")

    latest_ms = int(first_seen.max())
    window_ms = int(window / pd.Timedelta(milliseconds=1))
    cutoff_ms = latest_ms - window_ms
    mask = first_seen >= cutoff_ms
    window_df = df.loc[mask].copy()
    if window_df.empty:
        LOGGER.warning(
            "No flows found within window=%s (latest_ms=%d, cutoff_ms=%d). Skipping write.",
            window_arg,
            latest_ms,
            cutoff_ms,
        )
        return

    window_df[BYTES_COL] = _prepare_numeric(window_df[BYTES_COL]).fillna(0.0)

    agg = (
        window_df.groupby(KEY_COLS, as_index=False, observed=False)[BYTES_COL]
        .sum()
        .rename(columns={BYTES_COL: "bytes_4tuple"})
    )

    threshold, mu, sigma = _compute_threshold(agg)

    elephants = int((agg["bytes_4tuple"] >= threshold).sum())
    tuple_count = len(agg)
    flow_count = len(window_df)
    elephant_fraction = float(elephants / tuple_count) if tuple_count else 0.0

    _persist(
        destination,
        method="4tuple",
        threshold=threshold,
        mu=mu,
        sigma=sigma,
        window_arg=window_arg,
        asof_ms=latest_ms,
        tuple_count=tuple_count,
        flow_count=flow_count,
        elephant_fraction=elephant_fraction,
        source_path=source,
    )


def main() -> None:
    args = _parse_args()
    interval_td = _to_timedelta(args.interval)
    if interval_td <= pd.Timedelta(0):
        raise ValueError(f"Interval must be positive, got {args.interval!r}")
    interval_seconds = float(interval_td.total_seconds())

    LOGGER.info(
        "Starting threshold updater | source=%s | window=%s | interval=%.3fs | destination=%s",
        args.source,
        args.window,
        interval_seconds,
        args.destination,
    )

    try:
        while True:
            _refresh_once(args.source, args.window, args.destination)
            if args.run_once:
                break
            LOGGER.info("Sleeping for %.3f seconds before next refresh.", interval_seconds)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        LOGGER.info("Threshold updater interrupted by user.")


if __name__ == "__main__":
    main()


