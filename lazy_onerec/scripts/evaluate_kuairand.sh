#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

# Edit this section to configure SID generation evaluation.
python_bin="${python_bin:-python3}"
gpu_id=0
export CUDA_VISIBLE_DEVICES="${gpu_id}"
data_root="lazy_onerec/KuaiRand-1K"
sid_artifact="lazy_onerec/output/kuairand_sid/rq-kmeans-512-512-512-cosine/sid_index.json"
checkpoint="lazy_onerec/output/kuairand_model"
output="lazy_onerec/output/kuairand_model/test_sid_metrics.json"

sample=-1
batch_size=4
num_workers=0
warmup_days=3
test_days=3
min_history=3
beam_size=10
device="cuda"  # cuda | mps | cpu
seed=42

exec "${python_bin}" -m lazy_onerec.src.evaluate_kuairand \
  --data-root "${data_root}" \
  --sid-artifact "${sid_artifact}" \
  --checkpoint "${checkpoint}" \
  --output "${output}" \
  --sample "${sample}" \
  --batch-size "${batch_size}" \
  --num-workers "${num_workers}" \
  --warmup-days "${warmup_days}" \
  --test-days "${test_days}" \
  --min-history "${min_history}" \
  --beam-size "${beam_size}" \
  --device "${device}" \
  --seed "${seed}"
