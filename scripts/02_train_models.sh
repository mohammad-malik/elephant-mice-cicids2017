#!/usr/bin/env bash
set -euo pipefail

python src/models/classical_baselines.py > reports/results_baselines.md
