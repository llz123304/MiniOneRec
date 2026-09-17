#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

# Edit this section to configure SID generation evaluation.
python_bin="${python_bin:-python3}"
gpu_id="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
data_root="${DATA_ROOT:-lazy_onerec/KuaiRand-1K}"

# SID artifact used to decode targets. These defaults match build_sid.sh.
sid_method="${SID_METHOD:-rq-kmeans}"
read -r -a sid_codebook_sizes <<< "${SID_CODEBOOK_SIZES:-512 512 512}"
sid_distance_metric="${SID_DISTANCE_METRIC:-cosine}"
codebook_tag="$(IFS=-; printf '%s' "${sid_codebook_sizes[*]}")"
default_sid_dir="lazy_onerec/output/kuairand_sid"
default_sid_dir+="/${sid_method}-${codebook_tag}-${sid_distance_metric}"
sid_dir="${SID_DIR:-${default_sid_dir}}"
sid_artifact="${SID_ARTIFACT:-${sid_dir}/sid_index.json}"

positive_target="${POSITIVE_TARGET:-all}"  # all | click | long-view
num_train_epochs="${NUM_TRAIN_EPOCHS:-1}"
default_checkpoint="lazy_onerec/output/kuairand_model"
[[ "${positive_target}" != "all" ]] && default_checkpoint+="/target-${positive_target}"
default_checkpoint+="/epochs-${num_train_epochs}"
checkpoint="${MODEL_CHECKPOINT:-${default_checkpoint}}"
output="${EVALUATION_OUTPUT:-${checkpoint}/test_sid_metrics.json}"

sample=-1
batch_size=64
num_workers=8
prefetch_factor=4
warmup_days=3
test_days=3
min_history=3
beam_size=10
kv_cache=true
bf16=true
device="cuda"  # cuda | mps | cpu
seed=42

args=(
  --data-root "${data_root}"
  --sid-artifact "${sid_artifact}"
  --checkpoint "${checkpoint}"
  --output "${output}"
  --sample "${sample}"
  --positive-target "${positive_target}"
  --batch-size "${batch_size}"
  --num-workers "${num_workers}"
  --prefetch-factor "${prefetch_factor}"
  --warmup-days "${warmup_days}"
  --test-days "${test_days}"
  --min-history "${min_history}"
  --beam-size "${beam_size}"
  --device "${device}"
  --seed "${seed}"
)

if [[ "${kv_cache}" == "true" ]]; then
  args+=(--kv-cache)
else
  args+=(--no-kv-cache)
fi

[[ "${bf16}" == "true" ]] && args+=(--bf16)

exec "${python_bin}" -m lazy_onerec.src.evaluate_kuairand "${args[@]}"
