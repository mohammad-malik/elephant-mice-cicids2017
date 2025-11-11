"""Transform merged CIC-IDS2017 flows into the paper/NFStream schema."""
from pathlib import Path

import pandas as pd

from src.config import CONFIG
from src.utils.logging_utils import get_logger

LOGGER = get_logger("to_paper_schema")
IN_PATH = CONFIG.merged_csv
OUT_PATH = CONFIG.paper_schema_csv


def to_ms(flow_duration: pd.Series) -> pd.Series:
    """Convert CIC-IDS2017 durations to milliseconds if they appear to be in µs."""
    med = flow_duration.median()
    if med > 1_000_000:
        return (flow_duration / 1000.0).round()
    return flow_duration


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df.columns = df.columns.str.strip()

    rename = {
        "Source IP": "src_ip",
        "Destination IP": "dst_ip",
        "Source Port": "src_port",
        "Destination Port": "dst_port",
        "Flow Duration": "flow_duration_raw",
        "Total Length of Fwd Packets": "len_fwd",
        "Total Length of Bwd Packets": "len_bwd",
    }
    present = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(columns=present)

    if "dst_port" not in df.columns:
        raise KeyError("Destination port column is required")

    if "src_port" not in df.columns:
        df["src_port"] = df["dst_port"].copy()
        LOGGER.warning("Source port missing; copying destination port values to src_port")

    for col, default in {
        "src_ip": "0.0.0.0",
        "dst_ip": "0.0.0.0",
    }.items():
        if col not in df.columns:
            df[col] = default
            LOGGER.warning("%s column missing; using default %s", col, default)

    required = [
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "flow_duration_raw",
        "len_fwd",
        "len_bwd",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after rename: {missing}")

    df["len_fwd"] = pd.to_numeric(df["len_fwd"], errors="coerce")
    df["len_bwd"] = pd.to_numeric(df["len_bwd"], errors="coerce")
    df["bidirectional_bytes"] = df["len_fwd"] + df["len_bwd"]

    flow_ms = to_ms(pd.to_numeric(df["flow_duration_raw"], errors="coerce"))
    out = pd.DataFrame(
        {
            "src_ip": df["src_ip"],
            "dst_ip": df["dst_ip"],
            "src_port": df["src_port"].astype(float),
            "dst_port": df["dst_port"].astype(float),
            "bidirectional_first_seen_ms": 0.0,
            "bidirectional_last_seen_ms": flow_ms,
            "bidirectional_bytes": df["bidirectional_bytes"],
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    LOGGER.info("Wrote %s (%d rows)", OUT_PATH, len(out))


if __name__ == "__main__":
    main()
