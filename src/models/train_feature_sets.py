"""Run classical baselines across all feature sets and aggregate results."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.models.classical_baselines import run_baselines
from src.models.evaluate import format_markdown_table
from src.utils.logging_utils import get_logger

LOGGER = get_logger("train_feature_sets")
REPORTS_PATH = Path("reports") / "results_feature_sets.md"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline models across baseline / non-leaky / early feature sets."
    )
    parser.add_argument(
        "--export-model",
        action="store_true",
        help="Export the best-performing model for each feature set.",
    )
    args = parser.parse_args()

    comparison_lines: list[str] = [
        "# Feature set comparison\n",
        "This document compares the baseline (paper-faithful), non-leaky, and early-available "
        "feature sets using identical cross-validation settings.\n",
    ]

    feature_sets = ["baseline", "non_leaky", "early"]
    for feature_set in feature_sets:
        LOGGER.info("=" * 72)
        LOGGER.info("Training feature_set=%s", feature_set)
        LOGGER.info("=" * 72)
        aggregated_results, _ = run_baselines(
            feature_set=feature_set, export_model=args.export_model
        )
        comparison_lines.append(f"\n## Feature set: `{feature_set}`\n")
        comparison_lines.append(format_markdown_table(aggregated_results))
        comparison_lines.append("")

    REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.write_text("\n".join(comparison_lines))
    LOGGER.info("Wrote comparison report to %s", REPORTS_PATH)
    print(f"\n✓ Combined feature-set results saved to {REPORTS_PATH}")


if __name__ == "__main__":
    main()

