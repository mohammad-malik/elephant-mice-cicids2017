#!/usr/bin/env bash
set -euo pipefail

python src/data_prep/merge_cicids2017.py
python src/data_prep/clean_features.py
python src/data_prep/label_elephants.py
