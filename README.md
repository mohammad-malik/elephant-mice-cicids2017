# Elephant vs. Mice Flow Classification on CIC-IDS2017

This repository replicates the **methodology** (not the proprietary dataset) of the IEEE “Elephant and Mice Flow” study by rebuilding the entire pipeline with the public [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) flow dataset. CIC-IDS2017 already contains bidirectional NetFlow-like features, so no packet capture or NFStream usage is required (NFStream can be applied later for live capture scenarios and is mentioned in the report only as an alternative data source).

## Project layout

```text
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
│  ├─ 02_train_models.sh
│  └─ 03_train_feature_sets.sh
└─ reports/
   ├─ results_baselines.md
   ├─ results_feature_sets.md
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
| Labeling (Chebyshev + quantile) | `src/data_prep/label_elephants_chebyshev.py` | Computes both the μ+3σ Chebyshev cutoff **and** the 99th-percentile bytes threshold from aggregated 4-tuples, writes `target_traffic` and `target_99pct`, logs stats to `artifacts/threshold.json`, and saves the labeled table to `data/intermediate/cicids2017_labeled_chebyshev_glf.csv` (plus the trimmed modeling subset in `data/processed/`). |
| Baselines | `src/models/classical_baselines.py` | Runs LR, SVM, RF, DT, KNN, LDA, NB over selected feature sets (`baseline`, `non_leaky`, `early`) via 5-fold stratified group CV, aggregates accuracy/precision/recall/F1/**AUC-ROC/AUC-PR**, and saves confusion matrix + ROC/PR plots to `reports/plots/`. |
| Evaluation helper | `src/models/evaluate.py` | Shared metrics/report helpers. |
| Online threshold | `src/online/threshold_updater.py` | Recomputes μ, σ, and the Chebyshev cutoff on a sliding window (default 7 days) and refreshes `artifacts/threshold.json` on an interval so the “periodic/online” narrative remains true in deployment. |
| Online inference | `src/online/predict.py` | Loads a persisted scikit-learn model plus the refreshed threshold to deliver an early ML prediction and a counting-based override for a single incoming flow JSON. |
| Notebooks 01-03 | `notebooks/` | Generate the descriptive tables/figures needed for the report. |

Modeling currently reads from `data/intermediate/cicids2017_labeled_chebyshev_glf.csv` (contains both label columns) while `data/processed/elephant_mice_flows_paper_small.csv` remains available for legacy five-feature analyses.

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

Compare all feature sets (baseline vs. non-leaky vs. early) with the richer metrics/plots:

```bash
bash scripts/03_train_feature_sets.sh
cat reports/results_feature_sets.md
ls reports/plots/  # baseline_*.png, non_leaky_*.png, early_*.png
```

Each stage writes deterministic outputs into `data/intermediate/` and `data/processed/`, enabling notebook reuse and reproducibility.

### Optional online loop

To mirror the paper’s “periodic/online” threshold refresh, run:

```bash
python -m src.online.threshold_updater --source data/intermediate/cicids2017_paper_schema_glf.csv --window 7d --interval 300s
```

Use `--run-once` to recompute a single time (handy for CI) or leave it running to continually update `artifacts/threshold.json`. The updater records the window, cutoff, μ, σ, and as-of timestamp so downstream consumers know the context of the current threshold.

For a first-flow decision that matches the paper’s narrative (model + counting override):

```bash
python -m src.online.predict --flow path/to/flow.json --model-path artifacts/model.joblib --threshold-path artifacts/threshold.json
```

The predictor returns `ml_pred`, the `counting_flag` (if the 4-tuple’s bytes exceed the refreshed threshold after adding the current flow), and a `final_flag` that favors the counting signal when present.

### Quick demo without CIC-IDS2017

If you just want to exercise the pipeline without downloading CIC-IDS2017, run:

```bash
python scripts/generate_synthetic_data.py
python -m src.models.classical_baselines --export-model
python -m src.online.threshold_updater --source data/intermediate/cicids2017_paper_schema_glf.csv --window 7d --interval 300s --run-once
python -m src.online.predict --flow artifacts/sample_flow.json --model-path artifacts/model.joblib --threshold-path artifacts/threshold.json --history-path data/intermediate/cicids2017_paper_schema_glf.csv
```

The helper script populates small synthetic CSVs and a `sample_flow.json`, allowing you to train, export the model, refresh the Chebyshev threshold, and emit an online prediction end-to-end.

## Methodological notes

- The original paper used a private enterprise backbone trace; this project **mirrors the methodology** using public flows.
- Elephant detection uses two complementary definitions:
  - **Chebyshev (μ + 3·σ)** → `target_traffic` for paper-faithful reproduction (≈0.09% positives).
  - **99th percentile** → `target_99pct` for non-leaky & early-available experiments (≈1% positives, avoids label leakage from total bytes).
- Baseline models follow the same classical family but now expose a `--feature-set` selector (`baseline`, `non_leaky`, `early`) so experiments can exclude bytes or mimic early-flow observability. All runs use 5-fold stratified group CV, report accuracy/precision/recall/F1/**AUC-ROC/AUC-PR**, and emit confusion + ROC/PR plots under `reports/plots/`.
- Notebook outputs cover dataset profiling, threshold justification (Chebyshev vs. quantile), and classifier benchmarking for all feature sets.

## Next steps

1. Export plots from the notebooks into `reports/plots/` for insertion into papers or slide decks.
2. Extend `src/models/` with deep-learning or streaming classifiers if you need online detection.
3. Replace the GLF dataset with your own NetFlow export by dropping new CSVs into `data/raw/CIC-IDS2017/` (or pointing `RAW_GLF_DIR` elsewhere) and rerunning the preparation scripts.

This README intentionally states that we reproduce the **structure and approach** of the paper rather than exact numerical scores, making expectations clear for collaborators and reviewers.
