#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

# Edit these defaults or override them with the matching environment variables.
# Model, training, and evaluation details remain in their own scripts.
python_bin="${python_bin:-python3}"
gpu_id="${GPU_ID:-0}"
data_root="${DATA_ROOT:-lazy_onerec/KuaiRand-1K}"
output_root="${PIPELINE_ROOT:-lazy_onerec/output}"

embedding_model="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
# embedding_model="/data/sdb2/llz/hf_models/Qwen3-Embedding-0.6B"
embedding_revision="${EMBEDDING_REVISION:-}"

sid_method="${SID_METHOD:-rq-kmeans}"
read -r -a sid_codebook_sizes <<< "${SID_CODEBOOK_SIZES:-512 512 512}"
sid_distance_metric="${SID_DISTANCE_METRIC:-cosine}"
positive_target="${POSITIVE_TARGET:-all}"  # all | click | long-view
num_train_epochs="${NUM_TRAIN_EPOCHS:-1}"

# Print all resolved paths and commands without running any stage.
dry_run="${PIPELINE_DRY_RUN:-false}"

slugify() {
  "${python_bin}" -m lazy_onerec.src.pipeline_artifacts slug "$1"
}

stage_complete() {
  "${python_bin}" -m lazy_onerec.src.pipeline_artifacts check \
    --stage "$1" \
    --directory "$2" \
    "${artifact_args[@]}"
}

mark_stage() {
  "${python_bin}" -m lazy_onerec.src.pipeline_artifacts mark \
    --stage "$1" \
    --directory "$2" \
    "${artifact_args[@]}"
}

print_command() {
  printf '[command]'
  printf ' %q' "$@"
  printf '\n'
}

run_stage() {
  local stage="$1"
  local directory="$2"
  shift 2
  if [[ "${dry_run}" == "true" ]]; then
    echo "[dry-run] ${stage}: ${directory}"
    print_command "$@"
    return
  fi
  if stage_complete "${stage}" "${directory}"; then
    echo "[skip] ${stage}: ${directory}"
    return
  fi

  echo "[run] ${stage}: ${directory}"
  print_command "$@"
  "$@"
  mark_stage "${stage}" "${directory}"
}

validate_config() {
  if [[ "${#sid_codebook_sizes[@]}" -ne 3 ]]; then
    echo "exactly three SID codebook sizes are required" >&2
    exit 2
  fi
  case "${sid_method}" in
    rq-kmeans|balanced-kmeans|constrained-rq-kmeans|rq-vae|rq-kmeans-plus) ;;
    *)
      echo "unsupported sid_method=${sid_method}" >&2
      exit 2
      ;;
  esac
  case "${sid_distance_metric}" in
    cosine|euclidean) ;;
    *)
      echo "unsupported sid_distance_metric=${sid_distance_metric}" >&2
      exit 2
      ;;
  esac
  case "${positive_target}" in
    all|click|long-view) ;;
    *)
      echo "unsupported positive_target=${positive_target}" >&2
      exit 2
      ;;
  esac
  if [[ ! "${num_train_epochs}" =~ ^[1-9][0-9]*$ ]]; then
    echo "num_train_epochs must be a positive integer" >&2
    exit 2
  fi
  if [[ "${dry_run}" != "true" && "${dry_run}" != "false" ]]; then
    echo "PIPELINE_DRY_RUN must be true or false" >&2
    exit 2
  fi
  for size in "${sid_codebook_sizes[@]}"; do
    if [[ ! "${size}" =~ ^[1-9][0-9]*$ ]]; then
      echo "SID codebook sizes must be positive integers" >&2
      exit 2
    fi
  done
}

validate_config

embedding_slug="$(slugify "${embedding_model}")"
codebook_tag="$(IFS=-; printf '%s' "${sid_codebook_sizes[*]}")"
sid_name="${sid_method}-${codebook_tag}-${sid_distance_metric}"
target_suffix=""
if [[ "${positive_target}" != "all" ]]; then
  target_suffix="/target-${positive_target}"
fi
epoch_suffix="/epochs-${num_train_epochs}"

text_dir="${output_root}/kuairand_items/catalog"
embedding_dir="${output_root}/embeddings/${embedding_slug}"
sid_dir="${output_root}/kuairand_sid/${embedding_slug}/${sid_name}"
model_dir="${output_root}/models/${embedding_slug}/${sid_name}"
model_dir+="${target_suffix}${epoch_suffix}"
evaluation_dir="${output_root}/evaluations/${embedding_slug}/${sid_name}"
evaluation_dir+="${target_suffix}${epoch_suffix}"

sid_artifact="${sid_dir}/sid_index.json"
evaluation_output="${evaluation_dir}/test_sid_metrics.json"

artifact_args=(
  --data-root "${data_root}"
  --embedding-model "${embedding_model}"
  --embedding-revision "${embedding_revision}"
  --text-dir "${text_dir}"
  --embedding-dir "${embedding_dir}"
  --sid-method "${sid_method}"
  --sid-codebook-sizes "${sid_codebook_sizes[@]}"
  --sid-distance-metric "${sid_distance_metric}"
  --sid-dir "${sid_dir}"
  --positive-target "${positive_target}"
  --num-train-epochs "${num_train_epochs}"
  --model-dir "${model_dir}"
  --evaluation-dir "${evaluation_dir}"
)

echo "[pipeline] embedding_model=${embedding_model}"
echo "[pipeline] sid=${sid_method} ${sid_codebook_sizes[*]} ${sid_distance_metric}"
echo "[pipeline] positive_target=${positive_target}"
echo "[pipeline] num_train_epochs=${num_train_epochs}"
echo "[pipeline] embedding_dir=${embedding_dir}"
echo "[pipeline] sid_dir=${sid_dir}"
echo "[pipeline] model_dir=${model_dir}"
echo "[pipeline] evaluation_dir=${evaluation_dir}"

run_stage \
  embedding "${embedding_dir}" \
  env \
  "python_bin=${python_bin}" \
  "GPU_ID=${gpu_id}" \
  "DATA_ROOT=${data_root}" \
  "EMBEDDING_MODEL=${embedding_model}" \
  "EMBEDDING_REVISION=${embedding_revision}" \
  "EMBEDDING_WORK_DIR=${text_dir}" \
  "EMBEDDING_OUTPUT_DIR=${embedding_dir}" \
  bash lazy_onerec/scripts/embed_kuairand_items.sh

run_stage \
  sid "${sid_dir}" \
  env \
  "python_bin=${python_bin}" \
  "GPU_ID=${gpu_id}" \
  "SID_METHOD=${sid_method}" \
  "SID_CODEBOOK_SIZES=${sid_codebook_sizes[*]}" \
  "SID_DISTANCE_METRIC=${sid_distance_metric}" \
  "EMBEDDING_DIR=${embedding_dir}" \
  "SID_OUTPUT_DIR=${sid_dir}" \
  bash lazy_onerec/scripts/build_sid.sh

run_stage \
  train "${model_dir}" \
  env \
  "python_bin=${python_bin}" \
  "GPU_ID=${gpu_id}" \
  "DATA_ROOT=${data_root}" \
  "SID_ARTIFACT=${sid_artifact}" \
  "POSITIVE_TARGET=${positive_target}" \
  "NUM_TRAIN_EPOCHS=${num_train_epochs}" \
  "MODEL_OUTPUT_DIR=${model_dir}" \
  bash lazy_onerec/scripts/train_kuairand.sh

run_stage \
  evaluate "${evaluation_dir}" \
  env \
  "python_bin=${python_bin}" \
  "GPU_ID=${gpu_id}" \
  "DATA_ROOT=${data_root}" \
  "SID_ARTIFACT=${sid_artifact}" \
  "POSITIVE_TARGET=${positive_target}" \
  "NUM_TRAIN_EPOCHS=${num_train_epochs}" \
  "MODEL_CHECKPOINT=${model_dir}" \
  "EVALUATION_OUTPUT=${evaluation_output}" \
  bash lazy_onerec/scripts/evaluate_kuairand.sh

echo "[done] embedding=${embedding_dir}"
echo "[done] sid=${sid_artifact}"
echo "[done] model=${model_dir}"
echo "[done] metrics=${evaluation_output}"
