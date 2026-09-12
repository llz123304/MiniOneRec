#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# Edit this section to configure MiniOneRec SID conversion.
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT="data/Amazon/index/Industrial_and_Scientific.index.json"
OUTPUT="lazy_onerec/output/converted_sid/sid_index.json"
CODEBOOK_SIZES=(256 256 256)

args=(
  --input "${INPUT}"
  --output "${OUTPUT}"
  --codebook-sizes "${CODEBOOK_SIZES[@]}"
)

# Arguments supplied at invocation time are appended last and override defaults.
exec "${PYTHON_BIN}" -m lazy_onerec.sid.minionerec_adapter "${args[@]}" "$@"
