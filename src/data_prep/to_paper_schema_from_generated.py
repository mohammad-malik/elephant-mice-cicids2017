"""Transform GeneratedLabelledFlows (CIC-IDS2017) into the paper/NFStream schema."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.utils.logging_utils import get_logger

LOGGER = get_logger("to_paper_schema_from_generated")
IN_PATH = CONFIG.merged_csv
OUT_PATH = CONFIG.paper_schema_csv

DTYPES = {
    "Flow ID": "string",
    "Source IP": "string",
    "Destination IP": "string",
    "Source Port": "Int64",
    "Destination Port": "Int64",
    "Protocol": "Int64",
    "Flow Duration": "Int64",
    "Total Fwd Packets": "Int64",
    "Total Backward Packets": "Int64",
    "Total Length of Fwd Packets": "float64",
    "Total Length of Bwd Packets": "float64",
    "Label": "string",
}


def _read_merged(path: str) -> pd.DataFrame:
    kwargs = dict(
        encoding="utf-8",
        dtype=DTYPES,
        low_memory=False,
    )
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        LOGGER.warning("Decoding %s as latin1 due to UTF-8 error", Path(path).name)
        kwargs["encoding"] = "latin1"
        return pd.read_csv(path, **kwargs)


def _ensure_protocol(df: pd.DataFrame) -> None:
    if "protocol" in df.columns:
        df["protocol"] = pd.to_numeric(df["protocol"], errors="coerce").astype("Int64")


def _coerce_numeric(df: pd.DataFrame, col: str, dtype: str = "float64") -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(dtype)


def _as_seconds(dur_raw: pd.Series, scale: float) -> pd.Series:
    x = pd.to_numeric(dur_raw, errors="coerce")
    x = x.clip(lower=0)
    sec = x / scale
    return sec.replace(0, np.nan)


def _median_rel_err(pred: pd.Series, truth: pd.Series) -> float:
    mask = np.isfinite(pred) & np.isfinite(truth) & (truth != 0)
    if mask.sum() < 100:
        return np.inf
    r = (pred[mask] - truth[mask]).abs() / truth[mask].abs()
    return float(r.median())


def choose_duration_seconds(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "Flow Duration" not in df.columns:
        raise KeyError("Flow Duration column missing from GLF input")

    candidates = {
        "us": _as_seconds(df["Flow Duration"], 1e6),
        "ms": _as_seconds(df["Flow Duration"], 1e3),
        "s": _as_seconds(df["Flow Duration"], 1.0),
    }

    bytes_tot = (
        pd.to_numeric(df.get("Total Length of Fwd Packets"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("Total Length of Bwd Packets"), errors="coerce").fillna(0)
    )
    pkts_tot = (
        pd.to_numeric(df.get("Total Fwd Packets"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("Total Backward Packets"), errors="coerce").fillna(0)
    )

    src_bytes_s = pd.to_numeric(df.get("Flow Bytes/s"), errors="coerce")
    src_pkts_s = pd.to_numeric(df.get("Flow Packets/s"), errors="coerce")

    scores = {}
    for name, dur_s in candidates.items():
        pred_bps = bytes_tot / dur_s
        pred_pps = pkts_tot / dur_s
        e1 = _median_rel_err(pred_bps, src_bytes_s) if "Flow Bytes/s" in df.columns else np.inf
        e2 = _median_rel_err(pred_pps, src_pkts_s) if "Flow Packets/s" in df.columns else np.inf
        scores[name] = np.nanmin([e1, e2])

    best = min(scores, key=lambda k: scores[k])
    if np.isfinite(scores[best]) and scores[best] < 0.5:
        chosen = best
    else:
        med = pd.to_numeric(df["Flow Duration"], errors="coerce").median()
        if pd.isna(med):
            chosen = "us"
        elif med > 1e8:
            chosen = "us"
        elif med > 1e5:
            chosen = "ms"
        else:
            chosen = "s"

    return candidates[chosen], chosen


def main() -> None:
    df = _read_merged(IN_PATH)
    df.columns = df.columns.str.strip()

    duration_s, duration_unit = choose_duration_seconds(df)
    if duration_unit in ("ms", "s"):
        LOGGER.warning(
            "Flow Duration unit chosen by heuristic: %s; source rates inconsistent",
            duration_unit,
        )
    else:
        LOGGER.info("Detected Flow Duration unit: %s", duration_unit)

    df.rename(
        columns={
            "Source IP": "src_ip",
            "Destination IP": "dst_ip",
            "Source Port": "src_port",
            "Destination Port": "dst_port",
            "Protocol": "protocol",
            "Total Fwd Packets": "TotalFwdPkts",
            "Total Backward Packets": "TotalBwdPkts",
            "Total Length of Fwd Packets": "TotFwdBytes",
            "Total Length of Bwd Packets": "TotBwdBytes",
            "Flow Duration": "FlowDuration",
            "Timestamp": "Timestamp",
        },
        inplace=True,
    )

    for col in ("src_port", "dst_port"):
        if col not in df.columns:
            raise KeyError(f"Required port column missing: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df["TotalFwdPkts"] = _coerce_numeric(df, "TotalFwdPkts", "Int64")
    df["TotalBwdPkts"] = _coerce_numeric(df, "TotalBwdPkts", "Int64")
    df["TotFwdBytes"] = _coerce_numeric(df, "TotFwdBytes", "float64").fillna(0.0)
    df["TotBwdBytes"] = _coerce_numeric(df, "TotBwdBytes", "float64").fillna(0.0)
    df["FlowDuration"] = pd.to_numeric(df["FlowDuration"], errors="coerce")

    _ensure_protocol(df)

    df["bidirectional_packets"] = df["TotalFwdPkts"] + df["TotalBwdPkts"]
    df["src2dst_packets"] = df["TotalFwdPkts"]
    df["dst2src_packets"] = df["TotalBwdPkts"]
    df["src2dst_bytes"] = df["TotFwdBytes"]
    df["dst2src_bytes"] = df["TotBwdBytes"]
    df["bidirectional_bytes"] = df["src2dst_bytes"] + df["dst2src_bytes"]
    df["bidirectional_duration_ms"] = duration_s * 1000.0

    ts = pd.to_datetime(df["Timestamp"], errors="coerce", infer_datetime_format=True)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    ts_int = ts.astype("int64")
    ts_int = ts_int.where(ts.notna(), np.nan)
    first_ms = ts_int / 1_000_000.0
    df["src2dst_first_seen_ms"] = first_ms
    df["src2dst_last_seen_ms"] = df["src2dst_first_seen_ms"] + df["bidirectional_duration_ms"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    LOGGER.info(
        "Wrote %s (%d rows) | duration units=%s",
        OUT_PATH,
        len(df),
        duration_unit,
    )


if __name__ == "__main__":
    main()
