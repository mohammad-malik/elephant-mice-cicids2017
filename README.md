# Elephant vs. Mice Flow Classification on CIC-IDS2017

This repository replicates the **methodology** (not the proprietary dataset) of the IEEE “Elephant and Mice Flow” study by rebuilding the entire pipeline with the public [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) flow dataset. CIC-IDS2017 already contains bidirectional NetFlow-like features, so no packet capture or NFStream usage is required (NFStream can be applied later for live capture scenarios and is mentioned in the report only as an alternative data source).

## Project layout

```
elephant-mice-cicids2017/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/
│  │  └─ CIC-IDS2017/          # download CSVs here
│  ├─ intermediate/
│  │  ├─ cicids2017_merged.csv # merge result
│  │  └─ cicids2017_features.csv
│  └─ processed/
│     └─ elephant_mice_flows.csv
├─ src/
│  ├─ config.py
│  ├─ data_prep/
│  │  ├─ merge_cicids2017.py
│  │  ├─ clean_features.py
│  │  └─ label_elephants.py
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

1. Request access to CIC-IDS2017 from UNB or download the mirrored Kaggle archive. The dataset arrives as daily CSV files.
2. Place every CSV inside `data/raw/CIC-IDS2017/`. The filenames are read directly, so keep their original names (e.g., `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`).
3. No files are tracked here—only scripts describing how to process them.

See `scripts/00_download_instructions.txt` for a concise reminder.

## Pipeline overview

| Stage | Script/Notebook | Description |
| --- | --- | --- |
| Merge raw CSVs | `src/data_prep/merge_cicids2017.py` | Concatenates all daily captures into one table.
| Feature cleaning | `src/data_prep/clean_features.py` | Keeps flow-size features analogous to the paper (ports, durations, packet and byte counts) and derives totals.
| Label creation | `src/data_prep/label_elephants.py` | Marks elephants as the **top 5%** flows by total bytes (configurable).
| Baselines | `src/models/classical_baselines.py` | Trains LR, SVM, RF, DT, KNN, LDA, NB models to classify elephant vs. mice.
| Evaluation helper | `src/models/evaluate.py` | Shared metrics/report helpers.
| Notebooks 01–03 | `notebooks/` | Generate EDA, threshold selection, and modeling figures/tables for publication.

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
- Elephant detection is operationally defined as “top X% of flows when ordered by total bytes.” The default cutoff is 5%, matching the paper’s reported elephant prevalence.
- Baseline models follow the same family (classical ML with small feature set). Hyperparameters are intentionally conservative to keep comparisons fair and transparent.
- Notebook outputs (9 figures, 3 tables) are tailored for academic reporting: dataset profiling, threshold justification, and classifier benchmarking.

## Next steps

1. Export plots from the notebooks into `reports/plots/` for insertion into papers or slide decks.
2. Extend `src/models/` with deep-learning or streaming classifiers if you need online detection.
3. Replace CIC-IDS2017 with your own NetFlow export by dropping new CSVs into `data/raw/` and running the same pipeline.

This README intentionally states that we reproduce the **structure and approach** of the paper rather than exact numerical scores, making expectations clear for collaborators and reviewers.
