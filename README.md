# Elephant vs. Mice Flow Classification on CIC-IDS2017

This repository replicates the **methodology** (not the proprietary dataset) of the IEEE “Elephant and Mice Flow” study by rebuilding the entire pipeline with the public [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) flow dataset. CIC-IDS2017 already contains bidirectional NetFlow-like features, so no packet capture or NFStream usage is required (NFStream can be applied later for live capture scenarios and is mentioned in the report only as an alternative data source).

## Project layout

```
elephant-mice-cicids2017/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/
│  │  └─ CIC-IDS2017 # unzip the GeneratedLabelledFlows (GLF) archive here
│  ├─ intermediate/
│  │  ├─ cicids2017_merged.csv           # merged GLF rows
│  │  ├─ cicids2017_paper_schema_glf.csv # NFStream-like schema
│  │  ├─ cicids2017_paper_base_glf.csv   # 55,726-row working set
│  │  └─ cicids2017_labeled_chebyshev_glf.csv
│  └─ processed/
│     └─ elephant_mice_flows_paper_small.csv
├─ src/
│  ├─ config.py
│  ├─ data_prep/
│  │  ├─ merge_cicids2017.py
│  │  ├─ to_paper_schema_from_generated.py
│  │  ├─ to_paper_schema.py
│  │  ├─ label_elephants_chebyshev.py
│  │  └─ build_experiment_set_paper.py
│  ├─ models/
│  │  ├─ classical_baselines.py
│  │  └─ evaluate.py
│  └─ utils/
│     └─ logging_utils.py
├─ notebooks/
│  ├─ 01_explore_cicids2017.ipynb
│  ├─ 02_elephant_threshold_selection.ipynb
│  └─ 03_train_baselines.ipynb
├─ scripts/
│  ├─ 00_download_instructions.txt
│  ├─ 01_prepare_data.sh
│  └─ 02_train_models.sh
└─ reports/
   ├─ results_baselines.md
   └─ plots/
```

## Data acquisition

1. The GeneratedLabelledFlows (GLF) release provides the GLF-formatted CIC-IDS2017 tables. Download and unzip the archive locally.
2. Place every CSV inside `data/raw/CIC-IDS2017/` so `scripts/01_prepare_data.sh` can read them in bulk.
3. No data files are tracked in this repo—only the scripts that process them.

See `scripts/00_download_instructions.txt` for a concise reminder.

## Pipeline overview

| Stage | Script/Notebook | Description |
| --- | --- | --- |
| Merge raw CSVs | `src/data_prep/merge_cicids2017.py` | Concatenates every GLF daily capture into `data/intermediate/cicids2017_merged.csv` with typed columns and strict decoding modes. |
| Paper schema | `src/data_prep/to_paper_schema_from_generated.py` | Maps the merged file to NFStream-like fields, coerces ports/protocols, converts Flow Duration to milliseconds, and asserts the Flow Bytes/s sanity check. |
| Paper base | `src/data_prep/build_experiment_set_paper.py` | Samples ~55,726 flows from the schema, logs bidirectional byte quantiles, and writes `data/intermediate/cicids2017_paper_base_glf.csv`. |
| Chebyshev labeling | `src/data_prep/label_elephants_chebyshev.py` | Aggregates bytes per (`src_ip`,`dst_ip`,`src_port`,`dst_port`), thresholds at `μ+3σ`, propagates the label to every flow, writes metadata to `artifacts/threshold.json`, and outputs `data/intermediate/cicids2017_labeled_chebyshev_glf.csv`. |
| Baselines | `src/models/classical_baselines.py` | Trains LR, SVM, RF, DT, KNN, LDA, NB models using the five paper-aligned features (ports, first/last seen, bytes) with class priors logged. |
| Evaluation helper | `src/models/evaluate.py` | Shared metrics/report helpers. |
| Notebooks 01–03 | `notebooks/` | Generate the descriptive tables/figures needed for the report. |

The final table used for modeling lives at `data/processed/elephant_mice_flows_paper_small.csv`, while `data/intermediate/` now contains the merged file, GLF schema, sampled paper base, and the Chebyshev-labeled table for downstream analyses or alternative thresholds.

## Workflow

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare data (assumes CSVs are in place):

```bash
bash scripts/01_prepare_data.sh
```

Train and log model baselines:

```bash
bash scripts/02_train_models.sh
cat reports/results_baselines.md
```

Each stage writes deterministic outputs into `data/intermediate/` and `data/processed/`, enabling notebook reuse and reproducibility.

## Methodological notes

- The original paper used a private enterprise backbone trace; this project **mirrors the methodology** using public flows.
- Elephant detection is operationally defined as “bytes ≥ mean + 3·std” (Chebyshev) on the **per 4-tuple** aggregated `bidirectional_bytes`, matching the paper’s size framing and reproducing roughly 5% elephants over the 55,726-row working set.
- Baseline models follow the same family (classical ML with small feature set). Hyperparameters are intentionally conservative to keep comparisons fair and transparent.
- Notebook outputs (9 figures, 3 tables) are tailored for academic reporting: dataset profiling, threshold justification, and classifier benchmarking.

## Next steps

1. Export plots from the notebooks into `reports/plots/` for insertion into papers or slide decks.
2. Extend `src/models/` with deep-learning or streaming classifiers if you need online detection.
3. Replace the GLF dataset with your own NetFlow export by dropping new CSVs into `data/raw/CIC-IDS2017/` (or pointing `RAW_GLF_DIR` elsewhere) and rerunning the preparation scripts.

This README intentionally states that we reproduce the **structure and approach** of the paper rather than exact numerical scores, making expectations clear for collaborators and reviewers.
