"""Merge all CIC-IDS2017 CSVs into one file."""
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR, CONFIG
from src.utils.logging_utils import get_logger

OUT_PATH = CONFIG.merged_csv
LOGGER = get_logger("merge_cicids2017")


def merge_all(raw_dir: Path = RAW_DIR, out_path: Path = OUT_PATH) -> None:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Did you download CIC-IDS2017?"
        )

    frames = []
    for path in csv_files:
        LOGGER.info("Reading %s", path.name)
        df = pd.read_csv(path)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    LOGGER.info("Wrote merged dataset to %s (%d rows)", out_path, len(merged))


if __name__ == "__main__":
    merge_all()
