"""First-flow inference that blends ML prediction with counting threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

import joblib
import pandas as pd

from src.config import CONFIG
from src.utils.logging_utils import get_logger

LOGGER = get_logger("online_predict")

FEATURE_COLUMNS: Final[list[str]] = list(CONFIG.feature_columns)
KEY_COLS: Final[list[str]] = ["src_ip", "dst_ip", "src_port", "dst_port"]
BYTES_COL: Final[str] = "bidirectional_bytes"
FIRST_SEEN_COL: Final[str] = "bidirectional_first_seen_ms"
DEFAULT_MODEL_PATH: Final[Path] = Path("artifacts") / "model.joblib"
DEFAULT_THRESHOLD_PATH: Final[Path] = Path("artifacts") / "threshold.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine ML prediction with byte-count thresholding for a single flow."
    )
    parser.add_argument(
        "--flow",
        required=True,
        help="Flow payload as JSON string or path to a JSON file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the persisted scikit-learn model pipeline (joblib).",
    )
    parser.add_argument(
        "--threshold-path",
        type=Path,
        default=DEFAULT_THRESHOLD_PATH,
        help="Path to the Chebyshev threshold JSON.",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        help="Optional CSV containing historical flows (defaults to the threshold's source_path).",
    )
    return parser.parse_args()


def _load_flow(flow_arg: str) -> dict[str, Any]:
    candidate = Path(flow_arg)
    if candidate.exists():
        payload = json.loads(candidate.read_text())
    else:
        payload = json.loads(flow_arg)
    if not isinstance(payload, dict):
        raise ValueError("Flow payload must be a JSON object.")
    return payload


def _load_model(model_path: Path) -> tuple[Any, list[str]]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run training and export the model first."
        )
    LOGGER.info("Loading model from %s", model_path)
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        features = list(artifact.get("features", FEATURE_COLUMNS))
    else:
        model = artifact
        features = FEATURE_COLUMNS
    return model, features


def _load_threshold(threshold_path: Path) -> dict[str, Any]:
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Threshold artifact not found at {threshold_path}. Run threshold updater first."
        )
    payload = json.loads(threshold_path.read_text())
    required = {"threshold_bytes", "window", "asof"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Threshold JSON missing keys: {missing}")
    return payload


def _ml_predict(model, features: list[str], flow: dict[str, Any]) -> int:
    missing = [col for col in features if col not in flow]
    if missing:
        raise ValueError(f"Flow payload missing required model features: {missing}")
    X = pd.DataFrame([{col: flow[col] for col in features}])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    pred = model.predict(X)
    return int(pred[0])


def _load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [col for col in [*KEY_COLS, BYTES_COL, FIRST_SEEN_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"History CSV is missing required columns: {missing}")
    return df


def _counting_flag(
    flow: dict[str, Any],
    threshold: dict[str, Any],
    history_path: Path | None,
) -> tuple[int | None, float]:
    window_arg = threshold.get("window")
    if not window_arg or history_path is None:
        return None, 0.0

    if not history_path.exists():
        LOGGER.warning("History path %s does not exist. Counting flag unavailable.", history_path)
        return None, 0.0

    df = _load_history(history_path)
    window = pd.to_timedelta(window_arg)
    asof_ms = int(threshold["asof"])
    window_ms = int(window / pd.Timedelta(milliseconds=1))
    cutoff_ms = asof_ms - window_ms
    df[FIRST_SEEN_COL] = pd.to_numeric(df[FIRST_SEEN_COL], errors="coerce")
    window_df = df[df[FIRST_SEEN_COL] >= cutoff_ms].copy()
    if window_df.empty:
        LOGGER.warning(
            "History path %s returned no flows within window=%s. Counting flag unavailable.",
            history_path,
            window_arg,
        )
        return None, 0.0

    for col in KEY_COLS:
        if col not in flow:
            LOGGER.warning("Flow missing %s; counting flag unavailable.", col)
            return None, 0.0

    window_df[BYTES_COL] = pd.to_numeric(window_df[BYTES_COL], errors="coerce").fillna(0.0)

    key_values = tuple(flow[col] for col in KEY_COLS)
    mask = pd.Series(True, index=window_df.index)
    for col, value in zip(KEY_COLS, key_values, strict=True):
        mask &= window_df[col] == value

    historical_bytes = float(window_df.loc[mask, BYTES_COL].sum())
    current_bytes_value = pd.to_numeric([flow.get(BYTES_COL, 0.0)], errors="coerce")[0]
    current_bytes = float(current_bytes_value) if pd.notna(current_bytes_value) else 0.0
    accumulated = historical_bytes + current_bytes
    threshold_value = float(threshold["threshold_bytes"])
    flag = int(accumulated >= threshold_value)
    return flag, accumulated


def main() -> None:
    args = _parse_args()
    flow = _load_flow(args.flow)
    threshold = _load_threshold(args.threshold_path)

    model, features = _load_model(args.model_path)
    ml_pred = _ml_predict(model, features, flow)

    history_path = args.history_path
    if history_path is None:
        source_path = threshold.get("source_path")
        if source_path:
            history_path = Path(source_path)

    counting_flag, accumulated_bytes = _counting_flag(flow, threshold, history_path)
    final_flag = counting_flag if counting_flag is not None else ml_pred

    result = {
        "ml_pred": ml_pred,
        "counting_flag": counting_flag,
        "final_flag": final_flag,
        "accumulated_bytes": accumulated_bytes,
        "threshold_bytes": threshold.get("threshold_bytes"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


