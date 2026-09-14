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
click_history_length=128
long_view_history_length=128  # Also used by long-view duration.
like_history_length=64
deep_interact_history_length=32
hate_history_length=16
warmup_days=3  # First days used as history only, no samples.
test_days=3    # Last days form the test split; middle days train.

# Model parameters. Codebook sizes come from sid_artifact.
d_model=256
d_ff=1024
gid_dim=64
user_id_dim=128
categorical_dim=8
continuous_dim=16
duration_dim=8
qformer_layers=1
click_query_tokens=16
long_view_query_tokens=16
long_view_duration_query_tokens=16
like_query_tokens=8
deep_interact_query_tokens=4
hate_query_tokens=2
n_layers=6
n_context_layers=2
n_heads=4
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
  --click-history-length "${click_history_length}"
  --long-view-history-length "${long_view_history_length}"
  --like-history-length "${like_history_length}"
  --deep-interact-history-length "${deep_interact_history_length}"
  --hate-history-length "${hate_history_length}"
  --warmup-days "${warmup_days}"
  --test-days "${test_days}"
  --d-model "${d_model}"
  --d-ff "${d_ff}"
  --gid-dim "${gid_dim}"
  --user-id-dim "${user_id_dim}"
  --categorical-dim "${categorical_dim}"
  --continuous-dim "${continuous_dim}"
  --duration-dim "${duration_dim}"
  --qformer-layers "${qformer_layers}"
  --click-query-tokens "${click_query_tokens}"
  --long-view-query-tokens "${long_view_query_tokens}"
  --long-view-duration-query-tokens "${long_view_duration_query_tokens}"
  --like-query-tokens "${like_query_tokens}"
  --deep-interact-query-tokens "${deep_interact_query_tokens}"
  --hate-query-tokens "${hate_query_tokens}"
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
