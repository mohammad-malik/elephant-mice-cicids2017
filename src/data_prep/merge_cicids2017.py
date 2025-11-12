"""Merge all CIC-IDS2017 GLF CSVs into one file."""
from pathlib import Path

import pandas as pd

from src.config import RAW_GLF_DIR, CONFIG
from src.utils.logging_utils import get_logger

OUT_PATH = CONFIG.merged_csv
LOGGER = get_logger("merge_cicids2017")

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


def _read_csv(path: Path) -> pd.DataFrame:
    kwargs = dict(
        encoding="utf-8",
        dtype=DTYPES,
        low_memory=False,
    )
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        LOGGER.warning("Decoding %s as latin1 due to UTF-8 error", path.name)
        kwargs["encoding"] = "latin1"
        kwargs.pop("errors", None)
        return pd.read_csv(path, **kwargs)


def merge_all(raw_dir: Path = RAW_GLF_DIR, out_path: Path = OUT_PATH) -> None:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Did you download CIC-IDS2017?"
        )

    frames = []
    for path in csv_files:
        LOGGER.info("Reading %s", path.name)
        df = _read_csv(path)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    LOGGER.info("Wrote merged dataset to %s (%d rows)", out_path, len(merged))


if __name__ == "__main__":
    merge_all()
