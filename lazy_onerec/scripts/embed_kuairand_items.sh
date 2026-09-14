#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

# Edit this section to configure item embedding.
python_bin="${python_bin:-python3}"
gpu_id=0  # Physical GPU index from nvidia-smi.
export CUDA_VISIBLE_DEVICES="${gpu_id}"

# Recommended default: strong Chinese/multilingual quality, 1024-d native
# embeddings, and Matryoshka truncation support (for example output_dim=512).
model_name="Qwen/Qwen3-Embedding-0.6B"

# Other supported choices (uncomment exactly one and comment the default):
# model_name="BAAI/bge-m3"                         # Multilingual, 1024 dimensions.
# model_name="BAAI/bge-large-zh-v1.5"             # Chinese-focused, 1024 dimensions.
# model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct" # Higher cost and memory usage.
# model_name="intfloat/multilingual-e5-large-instruct" # Multilingual; adds passage prefix.
# model_name="moka-ai/m3e-base"                    # Lightweight Chinese baseline, 768 dimensions.
#
# A local Hugging Face model directory is also accepted:
# model_name="/data/sdb2/llz/hf_models/Qwen3-Embedding-0.6B"

revision=""
data_root="lazy_onerec/KuaiRand-1K"
captions="${data_root}/kuairand_video_captions.csv"
categories="${data_root}/kuairand_video_categories.csv"
work_dir=""
output_dir=""  # Empty selects output/embeddings/<model>-catalog-raw.

batch_size=512
write_batch_size=8192
max_length=256
output_dim=""  # Empty keeps the model's native dimension; use 512 for MRL.
model_dtype="auto"  # auto | float16 | bfloat16 | float32
storage_dtype="float16"  # float16 | float32
device="cuda"
trust_remote_code=true
normalize=false

prepare_only=false
rebuild_texts=false
overwrite=false
limit=""  # Set a positive integer for a smoke run.

args=(
  --model-name "${model_name}"
  --data-root "${data_root}"
  --captions "${captions}"
  --categories "${categories}"
  --batch-size "${batch_size}"
  --write-batch-size "${write_batch_size}"
  --max-length "${max_length}"
  --model-dtype "${model_dtype}"
  --storage-dtype "${storage_dtype}"
  --device "${device}"
)

[[ -n "${revision}" ]] && args+=(--revision "${revision}")
[[ -n "${work_dir}" ]] && args+=(--work-dir "${work_dir}")
[[ -n "${output_dir}" ]] && args+=(--output-dir "${output_dir}")
[[ -n "${output_dim}" ]] && args+=(--output-dim "${output_dim}")
[[ -n "${limit}" ]] && args+=(--limit "${limit}")

if [[ "${trust_remote_code}" == "true" ]]; then
  args+=(--trust-remote-code)
else
  args+=(--no-trust-remote-code)
fi

if [[ "${normalize}" == "true" ]]; then
  args+=(--normalize)
else
  args+=(--no-normalize)
fi

[[ "${prepare_only}" == "true" ]] && args+=(--prepare-only)
[[ "${rebuild_texts}" == "true" ]] && args+=(--rebuild-texts)
[[ "${overwrite}" == "true" ]] && args+=(--overwrite)

exec "${python_bin}" -m lazy_onerec.src.embed_kuairand_items "${args[@]}"
