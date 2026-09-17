#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

# Edit this section to configure SID construction.
python_bin="${python_bin:-python3}"
gpu_id="${GPU_ID:-0}"  # Used by rq-vae and rq-kmeans-plus.
export CUDA_VISIBLE_DEVICES="${gpu_id}"
method="${SID_METHOD:-rq-kmeans}"  # rq-kmeans | constrained-rq-kmeans | rq-vae | rq-kmeans-plus

# This stage consumes existing embeddings; it does not load an embedding model.
# EMBEDDING_MODEL selects the expected output of embed_kuairand_items.sh.
embedding_model="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
embedding_slug="$(
  "${python_bin}" -m lazy_onerec.src.pipeline_artifacts \
    slug "${embedding_model}"
)"
embedding_dir="${EMBEDDING_DIR:-lazy_onerec/output/embeddings/${embedding_slug}}"
embeddings="${EMBEDDINGS_PATH:-${embedding_dir}/item_embeddings.npy}"
item_ids="${EMBEDDING_ITEM_IDS:-${embedding_dir}/item_ids.npy}"
require_unique=false

# SID experiment parameters.
read -r -a codebook_sizes <<< "${SID_CODEBOOK_SIZES:-512 512 512}"
distance_metric="${SID_DISTANCE_METRIC:-cosine}"  # euclidean | cosine

codebook_tag="$(IFS=-; echo "${codebook_sizes[*]}")"
default_output_dir="lazy_onerec/output/kuairand_sid"
default_output_dir+="/${method}-${codebook_tag}-${distance_metric}"
output_dir="${SID_OUTPUT_DIR:-${default_output_dir}}"

# Method-specific K-means parameters.
beam_size=1  # rq-kmeans
max_iter=100  # constrained-rq-kmeans

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
)

case "${method}" in
  rq-kmeans)
    args+=(--beam-size "${beam_size}")
    ;;
  constrained-rq-kmeans)
    args+=(--max-iter "${max_iter}" --seed "${seed}")
    ;;
  rq-vae|rq-kmeans-plus)
    args+=(
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
    [[ "${no_kmeans_init}" == "true" ]] && args+=(--no-kmeans-init)
    ;;
  *)
    echo "unsupported SID method: ${method}" >&2
    exit 2
    ;;
esac

[[ "${require_unique}" == "true" ]] && args+=(--require-unique)

exec "${python_bin}" -m lazy_onerec.sid.build_sid "${args[@]}"
