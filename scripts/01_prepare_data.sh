#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PWD}"
export PYTHONPATH

python src/data_prep/merge_cicids2017.py
python src/data_prep/clean_features.py
python src/data_prep/label_elephants.py
python src/data_prep/build_experiment_set.py
