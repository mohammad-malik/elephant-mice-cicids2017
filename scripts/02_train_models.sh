#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PWD}"
export PYTHONPATH

python src/models/classical_baselines.py > reports/results_baselines.md
