#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

# Edit this section to configure metadata sampling.
python_bin="${python_bin:-python3}"
data_root="lazy_onerec/KuaiRand-1K"
catalog="${data_root}/data/video_features_basic_1k.csv"
captions="${data_root}/kuairand_video_captions.csv"
categories="${data_root}/kuairand_video_categories.csv"
output_dir="${data_root}/samples"
sample_size=20
seed=42

args=(
  --catalog "${catalog}"
  --captions "${captions}"
  --categories "${categories}"
  --output-dir "${output_dir}"
  --sample-size "${sample_size}"
  --seed "${seed}"
)

# Arguments supplied at invocation time are appended last and override defaults.
exec "${python_bin}" -m lazy_onerec.src.sample_kuairand_metadata "${args[@]}" "$@"
