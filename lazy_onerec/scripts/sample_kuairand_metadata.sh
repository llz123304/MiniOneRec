#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# Edit this section to configure metadata sampling.
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="lazy_onerec/KuaiRand-1K"
CATALOG="${DATA_ROOT}/data/video_features_basic_1k.csv"
CAPTIONS="${DATA_ROOT}/kuairand_video_captions.csv"
CATEGORIES="${DATA_ROOT}/kuairand_video_categories.csv"
OUTPUT_DIR="${DATA_ROOT}/samples"
SAMPLE_SIZE=20
SEED=42

args=(
  --catalog "${CATALOG}"
  --captions "${CAPTIONS}"
  --categories "${CATEGORIES}"
  --output-dir "${OUTPUT_DIR}"
  --sample-size "${SAMPLE_SIZE}"
  --seed "${SEED}"
)

# Arguments supplied at invocation time are appended last and override defaults.
exec "${PYTHON_BIN}" -m lazy_onerec.src.sample_kuairand_metadata "${args[@]}" "$@"
