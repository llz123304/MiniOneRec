#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

# Edit this section to configure KuaiRand training.
python_bin="${python_bin:-python3}"
gpu_id=0  # Physical GPU index from nvidia-smi.
export CUDA_VISIBLE_DEVICES="${gpu_id}"
data_root="lazy_onerec/KuaiRand-1K"
sid_artifact="lazy_onerec/output/kuairand_sid/rq-kmeans-512-512-512-cosine/sid_index.json"
output_dir="lazy_onerec/output/kuairand_model"

# Dataset parameters.
sample=-1  # -1 uses all training samples.
min_history=3
max_history=128

# Model parameters. Codebook sizes come from sid_artifact.
d_model=768
gid_dim=128
n_layers=6
n_context_layers=2
n_heads=12
n_kv_heads=2
kv_sharing=true
kv_share_every=2
position_encoding="rope"  # rope | learned

# Optimization parameters. Training is a single chronological pass.
batch_size=256
micro_batch_size=32
learning_rate=1e-3
weight_decay=0.01
warmup_steps=100
logging_steps=10
seed=42
bf16=false

args=(
  --data-root "${data_root}"
  --sid-artifact "${sid_artifact}"
  --output-dir "${output_dir}"
  --sample "${sample}"
  --min-history "${min_history}"
  --max-history "${max_history}"
  --d-model "${d_model}"
  --gid-dim "${gid_dim}"
  --n-layers "${n_layers}"
  --n-context-layers "${n_context_layers}"
  --n-heads "${n_heads}"
  --n-kv-heads "${n_kv_heads}"
  --kv-share-every "${kv_share_every}"
  --position-encoding "${position_encoding}"
  --batch-size "${batch_size}"
  --micro-batch-size "${micro_batch_size}"
  --learning-rate "${learning_rate}"
  --weight-decay "${weight_decay}"
  --warmup-steps "${warmup_steps}"
  --logging-steps "${logging_steps}"
  --seed "${seed}"
)

if [[ "${kv_sharing}" == "true" ]]; then
  args+=(--kv-sharing)
else
  args+=(--no-kv-sharing)
fi

[[ "${bf16}" == "true" ]] && args+=(--bf16)

exec "${python_bin}" -m lazy_onerec.src.train_kuairand "${args[@]}"
