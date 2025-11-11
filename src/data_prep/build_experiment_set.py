"""Generate a smaller experiment set from the full elephant/mice flows dataset."""
import pandas as pd
from pathlib import Path

IN_PATH = Path("data/processed/elephant_mice_flows.csv")
OUT_PATH = Path("data/processed/elephant_mice_flows_small.csv")

# Match their order of magnitude
TARGET_N = 55_726

def build_experiment_set():
    df = pd.read_csv(IN_PATH)

    # Keep only columns we actually use as features + label
    # (they ultimately used a subset; you’re allowed to drop IPs)
    drop_cols = {"Label"}  # CIC attack label not used
    cols = [c for c in df.columns if c not in drop_cols]
    df = df[cols]

    # If we have more rows than target, stratified sample to match size
    if len(df) > TARGET_N:
        df_major = df[df["target_traffic"] == 0]
        df_minor = df[df["target_traffic"] == 1]

        # preserve existing elephant ratio (already ~5.01%)
        ratio = df_minor.shape[0] / df.shape[0]
        n_minor = max(1, int(TARGET_N * ratio))
        n_major = TARGET_N - n_minor

        df_major_sample = df_major.sample(n=n_major, random_state=42)
        df_minor_sample = df_minor.sample(n=n_minor, random_state=42)

        df_small = pd.concat([df_major_sample, df_minor_sample],
                             ignore_index=True)
    else:
        df_small = df

    df_small.to_csv(OUT_PATH, index=False)
    print(f"Built experiment set: {len(df_small)} rows, "
          f"{df_small['target_traffic'].mean() * 100:.2f}% elephants")

if __name__ == "__main__":
    build_experiment_set()
