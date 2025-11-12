"""Generate small synthetic datasets so examples can run without CIC-IDS2017."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    base = Path("data")
    intermediate = base / "intermediate"
    processed = base / "processed"
    intermediate.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    num_records = 200

    src_ips = [f"10.0.0.{i}" for i in range(1, 6)]
    dst_ips = [f"192.168.1.{i}" for i in range(1, 6)]

    src_ports = rng.integers(1_000, 65_000, size=num_records, dtype=int)
    dst_ports = rng.integers(1_000, 65_000, size=num_records, dtype=int)
    first_seen = 1_700_000_000_000 + rng.integers(0, 86_400_000, size=num_records, dtype=int)
    durations = rng.integers(1_000, 300_000, size=num_records, dtype=int)
    last_seen = first_seen + durations
    bytes_vals = rng.lognormal(mean=12, sigma=1.0, size=num_records)

    schema_df = pd.DataFrame(
        {
            "src_ip": rng.choice(src_ips, size=num_records),
            "dst_ip": rng.choice(dst_ips, size=num_records),
            "src_port": src_ports,
            "dst_port": dst_ports,
            "bidirectional_bytes": bytes_vals,
            "bidirectional_first_seen_ms": first_seen,
            "src2dst_first_seen_ms": first_seen,
            "src2dst_last_seen_ms": last_seen,
        }
    )

    agg = (
        schema_df.groupby(["src_ip", "dst_ip", "src_port", "dst_port"], as_index=False)[
            "bidirectional_bytes"
        ]
        .sum()
        .rename(columns={"bidirectional_bytes": "bytes_4tuple"})
    )
    mu = agg["bytes_4tuple"].mean()
    sigma = agg["bytes_4tuple"].std(ddof=1)
    threshold = mu + 3 * sigma
    agg["target"] = (agg["bytes_4tuple"] >= threshold).astype(int)

    schema_with_target = schema_df.merge(
        agg[["src_ip", "dst_ip", "src_port", "dst_port", "target"]],
        on=["src_ip", "dst_ip", "src_port", "dst_port"],
        how="left",
    )
    schema_with_target["target"] = schema_with_target["target"].fillna(0).astype(int)

    processed_df = schema_with_target[
        [
            "src_port",
            "dst_port",
            "src2dst_first_seen_ms",
            "src2dst_last_seen_ms",
            "bidirectional_bytes",
            "target",
        ]
    ].rename(columns={"target": "target_traffic"})

    schema_df.to_csv(intermediate / "cicids2017_paper_schema_glf.csv", index=False)
    schema_with_target.to_csv(intermediate / "cicids2017_labeled_chebyshev_glf.csv", index=False)
    processed_df.to_csv(processed / "elephant_mice_flows_paper_small.csv", index=False)

    print("Synthetic datasets created at:")
    print(f"  {intermediate / 'cicids2017_paper_schema_glf.csv'}")
    print(f"  {intermediate / 'cicids2017_labeled_chebyshev_glf.csv'}")
    print(f"  {processed / 'elephant_mice_flows_paper_small.csv'}")


if __name__ == "__main__":
    main()


