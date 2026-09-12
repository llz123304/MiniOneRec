#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# Edit this section to configure SID construction.
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID=0  # Used by rq-vae and rq-kmeans-plus.
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
METHOD="constrained-rq-kmeans"  # rq-kmeans | constrained-rq-kmeans | rq-vae | rq-kmeans-plus
EMBEDDINGS="lazy_onerec/output/embeddings/qwen-qwen3-embedding-0-6b-clicked/item_embeddings.npy"
ITEM_IDS="lazy_onerec/output/embeddings/qwen-qwen3-embedding-0-6b-clicked/item_ids.npy"
OUTPUT_DIR="lazy_onerec/output/kuairand_sid"
CODEBOOK_SIZES=(256 256 256)
REQUIRE_UNIQUE=false

# K-means parameters.
MAX_ITER=100
BEAM_SIZE=1

# RQ-VAE and RQ-Kmeans+ parameters.
LATENT_DIM=32
HIDDEN_DIMS=(512 256 128)
EPOCHS=500
BATCH_SIZE=2048
LEARNING_RATE=""  # Empty uses the method default.
WEIGHT_DECAY=0.0
DEVICE="cuda"
BETA=0.25
QUANT_LOSS_WEIGHT=1.0
NO_KMEANS_INIT=false
KMEANS_ITERS=100
SINKHORN_EPSILONS=(0.0 0.0 0.0)
SINKHORN_ITERS=50
EVAL_EVERY=10
SEED=42

args=(
  --method "${METHOD}"
  --embeddings "${EMBEDDINGS}"
  --item-ids "${ITEM_IDS}"
  --output-dir "${OUTPUT_DIR}"
  --codebook-sizes "${CODEBOOK_SIZES[@]}"
  --max-iter "${MAX_ITER}"
  --beam-size "${BEAM_SIZE}"
  --latent-dim "${LATENT_DIM}"
  --hidden-dims "${HIDDEN_DIMS[@]}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --weight-decay "${WEIGHT_DECAY}"
  --device "${DEVICE}"
  --beta "${BETA}"
  --quant-loss-weight "${QUANT_LOSS_WEIGHT}"
  --kmeans-iters "${KMEANS_ITERS}"
  --sinkhorn-epsilons "${SINKHORN_EPSILONS[@]}"
  --sinkhorn-iters "${SINKHORN_ITERS}"
  --eval-every "${EVAL_EVERY}"
  --seed "${SEED}"
)

[[ -n "${LEARNING_RATE}" ]] && args+=(--learning-rate "${LEARNING_RATE}")
[[ "${REQUIRE_UNIQUE}" == "true" ]] && args+=(--require-unique)
[[ "${NO_KMEANS_INIT}" == "true" ]] && args+=(--no-kmeans-init)

# Arguments supplied at invocation time are appended last and override defaults.
exec "${PYTHON_BIN}" -m lazy_onerec.sid.build_sid "${args[@]}" "$@"
