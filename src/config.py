"""Central configuration for paths and labeling thresholds."""
import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"

DEFAULT_RAW_GLF_DIR = DATA_DIR / "raw" / "CIC-IDS2017"
RAW_GLF_DIR = Path(os.environ.get("RAW_GLF_DIR", DEFAULT_RAW_GLF_DIR))

PAPER_SCHEMA_CSV = INTERMEDIATE_DIR / "cicids2017_paper_schema_glf.csv"
PAPER_BASE_CSV = INTERMEDIATE_DIR / "cicids2017_paper_base_glf.csv"
LABELED_CSV = INTERMEDIATE_DIR / "cicids2017_labeled_chebyshev_glf.csv"
EXPERIMENT_CSV = PROCESSED_DIR / "elephant_mice_flows_paper_small.csv"


@dataclass(frozen=True)
class PipelineConfig:
    elephant_percent: float = 5.0
    random_state: int = 67

    @property
    def merged_csv(self) -> Path:
        return INTERMEDIATE_DIR / "cicids2017_merged.csv"

    @property
    def paper_schema_csv(self) -> Path:
        return PAPER_SCHEMA_CSV

    @property
    def paper_base_csv(self) -> Path:
        return PAPER_BASE_CSV

    @property
    def chebyshev_csv(self) -> Path:
        return LABELED_CSV

    @property
    def paper_small_csv(self) -> Path:
        return EXPERIMENT_CSV


CONFIG = PipelineConfig()
