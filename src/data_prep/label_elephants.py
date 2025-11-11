"""Label flows as elephants (1) or mice (0) based on byte quantile."""
import pandas as pd

from src.config import CONFIG
from src.utils.logging_utils import get_logger

IN_PATH = CONFIG.feature_csv
OUT_PATH = CONFIG.labeled_csv
LOGGER = get_logger("label_elephants")


def label_by_quantile(elephant_percent: float = CONFIG.elephant_percent) -> float:
    df = pd.read_csv(IN_PATH)
    df.columns = df.columns.str.strip()

    q = 1.0 - elephant_percent / 100.0
    threshold = df["total_bytes"].quantile(q)

    df["target_traffic"] = (df["total_bytes"] >= threshold).astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    LOGGER.info(
        "Wrote labeled dataset to %s (%.2f%% elephants)",
        OUT_PATH,
        df["target_traffic"].mean() * 100,
    )
    LOGGER.info("Elephant threshold (bytes): %.2f", threshold)
    return threshold


if __name__ == "__main__":
    label_by_quantile()
