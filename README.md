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
│  │  └─ cicids2017_paper_schema.csv
│  └─ processed/
│     ├─ elephant_mice_flows_chebyshev.csv
│     └─ elephant_mice_flows_paper_small.csv
├─ src/
│  ├─ config.py
│  ├─ data_prep/
│  │  ├─ merge_cicids2017.py
│  │  ├─ to_paper_schema.py
│  │  ├─ label_elephants_chebyshev.py
│  │  └─ build_experiment_set_chebyshev.py
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
| Merge raw CSVs | `src/data_prep/merge_cicids2017.py` | Concatenates every daily CIC-IDS2017 capture into one master table. |
| Paper schema | `src/data_prep/to_paper_schema.py` | Maps the merged file to NFStream-like fields (`src_port`, `dst_port`, `bidirectional_bytes`, first/last seen) and normalizes durations to milliseconds. |
| Chebyshev labeling | `src/data_prep/label_elephants_chebyshev.py` | Applies the paper’s size-only rule (bytes ≥ mean + 3·std) to mark elephants, adding `target_traffic` and `size_traffic`. |
| Experiment set | `src/data_prep/build_experiment_set_chebyshev.py` | Samples ~55,726 flows with the same elephant/mice ratio as the full set, facilitating deterministic experiments. |
| Baselines | `src/models/classical_baselines.py` | Trains LR, SVM, RF, DT, KNN, LDA, NB models using the paper-aligned features, optionally including `size_traffic` as a dummy. |
| Evaluation helper | `src/models/evaluate.py` | Shared metrics/report helpers. |
| Notebooks 01–03 | `notebooks/` | Generate the descriptive tables/figures needed for the report. |

The final table used for modeling lives at `data/processed/elephant_mice_flows_paper_small.csv`, while the intermediate schema and Chebyshev-labeled tables remain in `data/intermediate/` and `data/processed/`, respectively, to support downstream analyses or alternative thresholds.

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
- Elephant detection is operationally defined as “bytes ≥ mean + 3·std” (Chebyshev) on `bidirectional_bytes`, matching the paper’s size-only rule and reproducing the ~5% elephant prevalence over ~55,726 sampled flows.
- Baseline models follow the same family (classical ML with small feature set). Hyperparameters are intentionally conservative to keep comparisons fair and transparent.
- Notebook outputs (9 figures, 3 tables) are tailored for academic reporting: dataset profiling, threshold justification, and classifier benchmarking.

## Next steps

1. Export plots from the notebooks into `reports/plots/` for insertion into papers or slide decks.
2. Extend `src/models/` with deep-learning or streaming classifiers if you need online detection.
3. Replace CIC-IDS2017 with your own NetFlow export by dropping new CSVs into `data/raw/` and running the same pipeline.

This README intentionally states that we reproduce the **structure and approach** of the paper rather than exact numerical scores, making expectations clear for collaborators and reviewers.
