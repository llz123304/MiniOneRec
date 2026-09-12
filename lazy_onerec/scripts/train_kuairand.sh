#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# Edit this section to configure KuaiRand training.
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="lazy_onerec/KuaiRand/1K/KuaiRand-1K"
SID_ARTIFACT="lazy_onerec/output/kuairand_sid/sid_index.json"
OUTPUT_DIR="lazy_onerec/output/kuairand_model"

# Dataset parameters.
SAMPLE=-1  # -1 uses all training samples.
MIN_HISTORY=3
MAX_HISTORY=128

# Model parameters. Codebook sizes come from SID_ARTIFACT.
D_MODEL=768
GID_DIM=128
N_LAYERS=6
N_CONTEXT_LAYERS=2
N_HEADS=12
N_KV_HEADS=2
KV_SHARE_EVERY=2
POSITION_ENCODING="rope"  # rope | learned

# Optimization parameters.
NUM_EPOCHS=10
BATCH_SIZE=256
MICRO_BATCH_SIZE=32
LEARNING_RATE=1e-3
WEIGHT_DECAY=0.01
WARMUP_STEPS=100
LOGGING_STEPS=10
SEED=42
BF16=false

args=(
  --data-root "${DATA_ROOT}"
  --sid-artifact "${SID_ARTIFACT}"
  --output-dir "${OUTPUT_DIR}"
  --sample "${SAMPLE}"
  --min-history "${MIN_HISTORY}"
  --max-history "${MAX_HISTORY}"
  --d-model "${D_MODEL}"
  --gid-dim "${GID_DIM}"
  --n-layers "${N_LAYERS}"
  --n-context-layers "${N_CONTEXT_LAYERS}"
  --n-heads "${N_HEADS}"
  --n-kv-heads "${N_KV_HEADS}"
  --kv-share-every "${KV_SHARE_EVERY}"
  --position-encoding "${POSITION_ENCODING}"
  --num-epochs "${NUM_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-steps "${WARMUP_STEPS}"
  --logging-steps "${LOGGING_STEPS}"
  --seed "${SEED}"
)

[[ "${BF16}" == "true" ]] && args+=(--bf16)

# Arguments supplied at invocation time are appended last and override defaults.
exec "${PYTHON_BIN}" -m lazy_onerec.src.train_kuairand "${args[@]}" "$@"
