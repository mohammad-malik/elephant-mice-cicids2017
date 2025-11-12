#!/usr/bin/env bash
set -euo pipefail

RAW_GLF_DIR="${RAW_GLF_DIR:-data/raw/CIC-IDS2017}"
export RAW_GLF_DIR

PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PWD}"
export PYTHONPATH

python src/data_prep/merge_cicids2017.py      # writes ..._generated_glf.csv
python src/data_prep/to_paper_schema_from_generated.py       # writes ..._paper_schema_glf.csv
python src/data_prep/build_experiment_set_paper.py           # sample to 55,726 → ..._paper_base.csv
python src/data_prep/label_elephants_chebyshev.py --per-flow # online labels → ..._paper_small.csv
