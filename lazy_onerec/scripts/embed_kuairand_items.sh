#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# Edit this section to configure item embedding.
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
REVISION=""
DATA_ROOT="lazy_onerec/KuaiRand/1K/KuaiRand-1K"
CAPTIONS="lazy_onerec/KuaiRand/kuairand_video_captions.csv"
CATEGORIES="lazy_onerec/KuaiRand/kuairand_video_categories.csv"
SCOPE="clicked"  # clicked | catalog
WORK_DIR=""
OUTPUT_DIR="lazy_onerec/output/embeddings/qwen-qwen3-embedding-0-6b-clicked"

BATCH_SIZE=128
WRITE_BATCH_SIZE=8192
MAX_LENGTH=256
OUTPUT_DIM=""  # Empty keeps the model's native dimension; use 512 for MRL.
MODEL_DTYPE="auto"  # auto | float16 | bfloat16 | float32
STORAGE_DTYPE="float16"  # float16 | float32
DEVICE="auto"
TRUST_REMOTE_CODE=true
NORMALIZE=true

PREPARE_ONLY=false
REBUILD_TEXTS=false
OVERWRITE=false
LIMIT=""  # Set a positive integer for a smoke run.

args=(
  --model-name "${MODEL_NAME}"
  --data-root "${DATA_ROOT}"
  --captions "${CAPTIONS}"
  --categories "${CATEGORIES}"
  --scope "${SCOPE}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --write-batch-size "${WRITE_BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
  --model-dtype "${MODEL_DTYPE}"
  --storage-dtype "${STORAGE_DTYPE}"
  --device "${DEVICE}"
)

[[ -n "${REVISION}" ]] && args+=(--revision "${REVISION}")
[[ -n "${WORK_DIR}" ]] && args+=(--work-dir "${WORK_DIR}")
[[ -n "${OUTPUT_DIM}" ]] && args+=(--output-dim "${OUTPUT_DIM}")
[[ -n "${LIMIT}" ]] && args+=(--limit "${LIMIT}")

if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  args+=(--trust-remote-code)
else
  args+=(--no-trust-remote-code)
fi

if [[ "${NORMALIZE}" == "true" ]]; then
  args+=(--normalize)
else
  args+=(--no-normalize)
fi

[[ "${PREPARE_ONLY}" == "true" ]] && args+=(--prepare-only)
[[ "${REBUILD_TEXTS}" == "true" ]] && args+=(--rebuild-texts)
[[ "${OVERWRITE}" == "true" ]] && args+=(--overwrite)

# Arguments supplied at invocation time are appended last and override defaults.
exec "${PYTHON_BIN}" -m lazy_onerec.src.embed_kuairand_items "${args[@]}" "$@"
