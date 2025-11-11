"""Select and derive core flow-size features from CIC-IDS2017."""
import pandas as pd

from src.config import CONFIG
from src.utils.logging_utils import get_logger

IN_PATH = CONFIG.merged_csv
OUT_PATH = CONFIG.feature_csv
LOGGER = get_logger("clean_features")

# Candidate columns we would like to use
CANDIDATE_FEATURE_COLS = [
    "Source Port",
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
]

def build_features():
    df = pd.read_csv(IN_PATH)

    # Normalize column names to avoid hidden spaces
    df.columns = [c.strip() for c in df.columns]

    # Some CIC-IDS2017 variants use 'Src Port' / 'Dst Port'
    rename_map = {}
    if "Src Port" in df.columns and "Source Port" not in df.columns:
        rename_map["Src Port"] = "Source Port"
    if "Dst Port" in df.columns and "Destination Port" not in df.columns:
        rename_map["Dst Port"] = "Destination Port"

    if rename_map:
        df = df.rename(columns=rename_map)

    existing = [c for c in CANDIDATE_FEATURE_COLS if c in df.columns]

    if len(existing) == 0:
        raise ValueError(f"No expected feature columns found. Got: {df.columns.tolist()}")

    # Drop rows with NaN in the features we actually have
    df = df.dropna(subset=existing)

    # Build aggregate features
    if "Total Length of Fwd Packets" in df.columns and "Total Length of Bwd Packets" in df.columns:
        df["total_bytes"] = (
            df["Total Length of Fwd Packets"] +
            df["Total Length of Bwd Packets"]
        )
    elif "total_bytes" not in df.columns:
        raise ValueError("Missing fields to compute total_bytes.")

    if "Total Fwd Packets" in df.columns and "Total Backward Packets" in df.columns:
        df["total_pkts"] = (
            df["Total Fwd Packets"] +
            df["Total Backward Packets"]
        )
    elif "total_pkts" not in df.columns:
        raise ValueError("Missing fields to compute total_pkts.")

    # Keep only what is meaningful
    keep_cols = sorted(
        set(existing + ["total_bytes", "total_pkts"] + (["Label"] if "Label" in df.columns else []))
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[keep_cols].to_csv(OUT_PATH, index=False)

    print("Using feature columns:", keep_cols)

if __name__ == "__main__":
    build_features()
