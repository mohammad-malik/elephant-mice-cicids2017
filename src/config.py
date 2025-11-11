"""Central configuration for paths and labeling thresholds."""
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "CIC-IDS2017"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"


@dataclass(frozen=True)
class PipelineConfig:
    elephant_percent: float = 5.0
    random_state: int = 67

    @property
    def merged_csv(self) -> Path:
        return INTERMEDIATE_DIR / "cicids2017_merged.csv"

    @property
    def feature_csv(self) -> Path:
        return INTERMEDIATE_DIR / "cicids2017_features.csv"

    @property
    def labeled_csv(self) -> Path:
        return PROCESSED_DIR / "elephant_mice_flows.csv"


CONFIG = PipelineConfig()
