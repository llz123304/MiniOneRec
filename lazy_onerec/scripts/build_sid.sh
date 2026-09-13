#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

# Edit this section to configure SID construction.
python_bin="${python_bin:-python3}"
gpu_id=0  # Used by rq-vae and rq-kmeans-plus.
export CUDA_VISIBLE_DEVICES="${gpu_id}"
method="${method:-rq-kmeans}"  # rq-kmeans | constrained-rq-kmeans | rq-vae | rq-kmeans-plus
embeddings="lazy_onerec/output/embeddings/qwen-qwen3-embedding-0-6b-catalog-raw/item_embeddings.npy"
item_ids="lazy_onerec/output/embeddings/qwen-qwen3-embedding-0-6b-catalog-raw/item_ids.npy"
require_unique=false

# Select one experiment preset.
preset="${preset:-512-512-512-cosine}"
case "${preset}" in
  "256-256-256-euclidean")
    codebook_sizes=(256 256 256)
    distance_metric="euclidean"
    ;;
  "256-256-256-cosine")
    codebook_sizes=(256 256 256)
    distance_metric="cosine"
    ;;
  "512-512-512-cosine")
    codebook_sizes=(512 512 512)
    distance_metric="cosine"
    ;;
  "256-512-1024-cosine")
    codebook_sizes=(256 512 1024)
    distance_metric="cosine"
    ;;
  "1024-512-256-euclidean")
    codebook_sizes=(1024 512 256)
    distance_metric="euclidean"
    ;;
  "1024-512-256-cosine")
    codebook_sizes=(1024 512 256)
    distance_metric="cosine"
    ;;
  *)
    echo "unknown SID preset: ${preset}" >&2
    exit 2
    ;;
esac

output_dir="lazy_onerec/output/kuairand_sid/${method}-${preset}"

# K-means parameters.
max_iter=100
beam_size=1

# RQ-VAE and RQ-Kmeans+ parameters.
latent_dim=32
hidden_dims=(512 256 128)
epochs=500
batch_size=2048
learning_rate=""  # Empty uses the method default.
weight_decay=0.0
device="cuda"
beta=0.25
quant_loss_weight=1.0
no_kmeans_init=false
kmeans_iters=100
sinkhorn_epsilons=(0.0 0.0 0.0)
sinkhorn_iters=50
eval_every=10
seed=42

args=(
  --method "${method}"
  --embeddings "${embeddings}"
  --item-ids "${item_ids}"
  --output-dir "${output_dir}"
  --codebook-sizes "${codebook_sizes[@]}"
  --distance-metric "${distance_metric}"
  --max-iter "${max_iter}"
  --beam-size "${beam_size}"
  --latent-dim "${latent_dim}"
  --hidden-dims "${hidden_dims[@]}"
  --epochs "${epochs}"
  --batch-size "${batch_size}"
  --weight-decay "${weight_decay}"
  --device "${device}"
  --beta "${beta}"
  --quant-loss-weight "${quant_loss_weight}"
  --kmeans-iters "${kmeans_iters}"
  --sinkhorn-epsilons "${sinkhorn_epsilons[@]}"
  --sinkhorn-iters "${sinkhorn_iters}"
  --eval-every "${eval_every}"
  --seed "${seed}"
)

[[ -n "${learning_rate}" ]] && args+=(--learning-rate "${learning_rate}")
[[ "${require_unique}" == "true" ]] && args+=(--require-unique)
[[ "${no_kmeans_init}" == "true" ]] && args+=(--no-kmeans-init)

# Arguments supplied at invocation time are appended last and override defaults.
exec "${python_bin}" -m lazy_onerec.sid.build_sid "${args[@]}" "$@"
