#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

# Edit this block. Model/training/evaluation details remain in their own scripts.
python_bin="${python_bin:-python3}"
gpu_id=0
data_root="lazy_onerec/KuaiRand-1K"
output_root="${PIPELINE_ROOT:-lazy_onerec/output}"

embedding_model="Qwen/Qwen3-Embedding-0.6B"
# embedding_model="/data/sdb2/llz/hf_models/Qwen3-Embedding-0.6B"
embedding_revision=""
embedding_name=""  # Empty derives a stable name from embedding_model.

sid_method="rq-kmeans"  # rq-kmeans | constrained-rq-kmeans | rq-vae | rq-kmeans-plus
sid_codebook_sizes=(512 512 512)
sid_distance_metric="cosine"  # euclidean | cosine

# Print all resolved paths and commands without running any stage.
dry_run="${PIPELINE_DRY_RUN:-false}"

slugify() {
  "${python_bin}" - "$1" <<'PY'
import re
import sys

value = sys.argv[1].rstrip("/")
print(re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-"))
PY
}

stage_complete() {
  "${python_bin}" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

stage = sys.argv[1]
directory = Path(sys.argv[2])

def nonempty(name):
    path = directory / name
    return path.is_file() and path.stat().st_size > 0

complete = False
try:
    if stage == "embedding":
        config = json.loads(
            (directory / "embedding_config.json").read_text(encoding="utf-8")
        )
        progress = json.loads(
            (directory / "progress.json").read_text(encoding="utf-8")
        )
        complete = (
            config.get("completed") is True
            and progress.get("next_index") == config.get("num_items")
            and nonempty("item_embeddings.npy")
            and nonempty("item_ids.npy")
        )
    elif stage == "sid":
        complete = all(
            nonempty(name)
            for name in (
                "sid_index.json",
                "codes.npy",
                "codebooks.npz",
                "sid_metrics.json",
            )
        )
    elif stage == "train":
        complete = nonempty("config.json") and any(
            nonempty(name)
            for name in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
        )
    elif stage == "evaluate":
        metrics = json.loads(
            (directory / "test_sid_metrics.json").read_text(encoding="utf-8")
        )
        required = {
            "sid0_hr_at_10",
            "sid0_mrr_at_10",
            "sid1_hr_at_10",
            "sid1_mrr_at_10",
            "sid2_hr_at_10",
            "sid2_mrr_at_10",
            "overall_hr_at_10",
            "overall_mrr_at_10",
            "invalid_sid_rate",
        }
        complete = required.issubset(metrics)
    else:
        raise ValueError(f"unknown stage: {stage}")
except (FileNotFoundError, json.JSONDecodeError, OSError):
    complete = False

raise SystemExit(0 if complete else 1)
PY
}

write_manifest() {
  local directory="$1"
  local stage="$2"
  mkdir -p "${directory}"
  "${python_bin}" - \
    "${directory}/pipeline_stage.json" \
    "${stage}" \
    "${embedding_model}" \
    "${embedding_revision}" \
    "${sid_method}" \
    "${sid_codebook_sizes[*]}" \
    "${sid_distance_metric}" \
    "${embedding_dir}" \
    "${sid_dir}" \
    "${model_dir}" \
    "${evaluation_dir}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "format": "lazy-onerec-pipeline-stage",
    "version": 1,
    "stage": sys.argv[2],
    "embedding": {
        "model": sys.argv[3],
        "revision": sys.argv[4] or None,
        "output_dir": sys.argv[8],
    },
    "sid": {
        "method": sys.argv[5],
        "codebook_sizes": [int(value) for value in sys.argv[6].split()],
        "distance_metric": sys.argv[7],
        "output_dir": sys.argv[9],
    },
    "model_dir": sys.argv[10],
    "evaluation_dir": sys.argv[11],
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
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
  if stage_complete "${stage}" "${directory}"; then
    echo "[skip] ${stage}: ${directory}"
    return
  fi

  echo "[run] ${stage}: ${directory}"
  print_command "$@"
  if [[ "${dry_run}" == "true" ]]; then
    return
  fi

  "$@"
  if ! stage_complete "${stage}" "${directory}"; then
    echo "${stage} finished without complete artifacts: ${directory}" >&2
    exit 1
  fi
}

if [[ "${#sid_codebook_sizes[@]}" -ne 3 ]]; then
  echo "exactly three SID codebook sizes are required" >&2
  exit 2
fi
case "${sid_method}" in
  rq-kmeans|constrained-rq-kmeans|rq-vae|rq-kmeans-plus) ;;
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
for size in "${sid_codebook_sizes[@]}"; do
  if [[ ! "${size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SID codebook sizes must be positive integers" >&2
    exit 2
  fi
done

if [[ -z "${embedding_name}" ]]; then
  embedding_name="$(slugify "${embedding_model}")-catalog-raw"
fi
if [[ -n "${embedding_revision}" ]]; then
  embedding_name="${embedding_name}-$(slugify "${embedding_revision}")"
fi

codebook_tag="$(IFS=-; printf '%s' "${sid_codebook_sizes[*]}")"
sid_name="${sid_method}-${codebook_tag}-${sid_distance_metric}"

text_dir="${output_root}/kuairand_items/catalog"
embedding_dir="${output_root}/embeddings/${embedding_name}"
sid_dir="${output_root}/kuairand_sid/${embedding_name}/${sid_name}"
model_dir="${output_root}/models/${embedding_name}/${sid_name}"
evaluation_dir="${output_root}/evaluations/${embedding_name}/${sid_name}"

embeddings="${embedding_dir}/item_embeddings.npy"
embedding_item_ids="${embedding_dir}/item_ids.npy"
sid_artifact="${sid_dir}/sid_index.json"
evaluation_output="${evaluation_dir}/test_sid_metrics.json"

echo "[pipeline] embedding_model=${embedding_model}"
echo "[pipeline] sid=${sid_method} ${sid_codebook_sizes[*]} ${sid_distance_metric}"
echo "[pipeline] embedding_dir=${embedding_dir}"
echo "[pipeline] sid_dir=${sid_dir}"
echo "[pipeline] model_dir=${model_dir}"
echo "[pipeline] evaluation_dir=${evaluation_dir}"

run_stage \
  embedding "${embedding_dir}" \
  env \
  "python_bin=${python_bin}" \
  "LAZY_GPU_ID=${gpu_id}" \
  "LAZY_DATA_ROOT=${data_root}" \
  "LAZY_EMBEDDING_MODEL=${embedding_model}" \
  "LAZY_EMBEDDING_REVISION=${embedding_revision}" \
  "LAZY_EMBEDDING_WORK_DIR=${text_dir}" \
  "LAZY_EMBEDDING_OUTPUT_DIR=${embedding_dir}" \
  bash lazy_onerec/scripts/embed_kuairand_items.sh
if [[ "${dry_run}" != "true" ]]; then
  write_manifest "${embedding_dir}" embedding
fi

run_stage \
  sid "${sid_dir}" \
  env \
  "python_bin=${python_bin}" \
  "LAZY_GPU_ID=${gpu_id}" \
  "LAZY_SID_METHOD=${sid_method}" \
  "LAZY_SID_CODEBOOK_SIZES=${sid_codebook_sizes[*]}" \
  "LAZY_SID_DISTANCE_METRIC=${sid_distance_metric}" \
  "LAZY_EMBEDDINGS_PATH=${embeddings}" \
  "LAZY_EMBEDDING_ITEM_IDS=${embedding_item_ids}" \
  "LAZY_SID_OUTPUT_DIR=${sid_dir}" \
  bash lazy_onerec/scripts/build_sid.sh
if [[ "${dry_run}" != "true" ]]; then
  write_manifest "${sid_dir}" sid
fi

run_stage \
  train "${model_dir}" \
  env \
  "python_bin=${python_bin}" \
  "LAZY_GPU_ID=${gpu_id}" \
  "LAZY_DATA_ROOT=${data_root}" \
  "LAZY_SID_ARTIFACT=${sid_artifact}" \
  "LAZY_MODEL_OUTPUT_DIR=${model_dir}" \
  bash lazy_onerec/scripts/train_kuairand.sh
if [[ "${dry_run}" != "true" ]]; then
  write_manifest "${model_dir}" train
fi

run_stage \
  evaluate "${evaluation_dir}" \
  env \
  "python_bin=${python_bin}" \
  "LAZY_GPU_ID=${gpu_id}" \
  "LAZY_DATA_ROOT=${data_root}" \
  "LAZY_SID_ARTIFACT=${sid_artifact}" \
  "LAZY_MODEL_CHECKPOINT=${model_dir}" \
  "LAZY_EVALUATION_OUTPUT=${evaluation_output}" \
  bash lazy_onerec/scripts/evaluate_kuairand.sh
if [[ "${dry_run}" != "true" ]]; then
  write_manifest "${evaluation_dir}" evaluate
fi

echo "[done] embedding=${embedding_dir}"
echo "[done] sid=${sid_artifact}"
echo "[done] model=${model_dir}"
echo "[done] metrics=${evaluation_output}"
