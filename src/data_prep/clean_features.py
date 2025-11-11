"""Select and derive core flow-size features from CIC-IDS2017."""
import pandas as pd

from src.config import CONFIG
from src.utils.logging_utils import get_logger

IN_PATH = CONFIG.merged_csv
OUT_PATH = CONFIG.feature_csv
LOGGER = get_logger("clean_features")

FEATURE_COLS = [
    "Source Port",
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
]


def build_features(in_path=IN_PATH, out_path=OUT_PATH) -> None:
    LOGGER.info("Loading merged CSV: %s", in_path)
    df = pd.read_csv(in_path)
    df = df.dropna(subset=FEATURE_COLS)

    df["total_bytes"] = (
        df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
    )
    df["total_pkts"] = df["Total Fwd Packets"] + df["Total Backward Packets"]

    keep = FEATURE_COLS + ["total_bytes", "total_pkts"]
    if "Label" in df.columns:
        keep.append("Label")
    df = df[keep]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    LOGGER.info("Saved feature table to %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    build_features()
