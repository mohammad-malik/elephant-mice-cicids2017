#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PWD}"
export PYTHONPATH

python src/data_prep/merge_cicids2017.py
python src/data_prep/to_paper_schema.py
python src/data_prep/label_elephants_chebyshev.py
python src/data_prep/build_experiment_set_chebyshev.py
