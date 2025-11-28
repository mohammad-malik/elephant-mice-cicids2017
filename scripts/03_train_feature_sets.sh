#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Training classical baselines across feature sets ==="
python -m src.models.train_feature_sets "$@"

cat <<'EOF'

Individual runs:
  python -m src.models.classical_baselines --feature-set=baseline
  python -m src.models.classical_baselines --feature-set=non_leaky
  python -m src.models.classical_baselines --feature-set=early
EOF

