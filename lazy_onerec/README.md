# LazyOneRec

[English](README.md) | [简体中文](README_zh.md)

```text
KuaiRand video text -> item embedding -> three-level SID -> next-SID model
```

Python code is organized under `data/`, `model/`, `sid/`, and `src/`. Configure
each command at the top of its Shell file under `scripts/`. The package root
contains documentation and dependency metadata only.

## Environment

Versions: Python 3.10, PyTorch 2.3.1, CUDA 12.1.

```bash
cd /data/sdb2/llz/code/onerec/MiniOneRec

# Create and activate the environment
uv venv --python 3.10 .venv
source .venv/bin/activate

# Install the CUDA 12.1 PyTorch build
uv pip install torch==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Install the remaining dependencies
uv pip install -r lazy_onerec/requirements.txt

# Verify PyTorch and GPU access
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Select the physical GPU in each GPU-backed Shell script:

```bash
gpu_id=0
```

## Data

```text
lazy_onerec/KuaiRand-1K/
├── data/
│   ├── video_features_basic_1k.csv
│   ├── log_standard_4_08_to_4_21_1k.csv
│   ├── log_standard_4_22_to_5_08_1k.csv
│   └── log_random_4_22_to_5_08_1k.csv
├── kuairand_video_captions.csv
└── kuairand_video_categories.csv
```

## Build Item Embeddings

Set these values in `embed_kuairand_items.sh`:

```bash
model_name="Qwen/Qwen3-Embedding-0.6B"
output_dim=""         # Native Qwen dimension is 1024; use 512 for MRL
batch_size=512
device="cuda"
normalize=false       # Save raw vectors; SID cosine mode normalizes them
```

Run:

```bash
lazy_onerec/scripts/embed_kuairand_items.sh
```

Output:

```text
lazy_onerec/output/embeddings/qwen-qwen3-embedding-0-6b-catalog-raw/
├── item_ids.npy
├── item_embeddings.npy
├── embedding_config.json
└── progress.json
```

The script resumes automatically. To rebuild:

```bash
rebuild_texts=true
overwrite=true
```

## Build Semantic IDs

Set these values in `build_sid.sh`:

```bash
method="rq-kmeans"
codebook_sizes=(512 512 512)
distance_metric="cosine"  # euclidean | cosine
```

`method` supports `rq-kmeans`, `constrained-rq-kmeans`, `rq-vae`, and
`rq-kmeans-plus`.

`rq-kmeans` supports a different power-of-two codebook at each level; for
example, `256-512-1024` uses `nbits=[8,9,10]`. Cosine mode normalizes inputs
and uses cosine assignment in neural quantizers.
With pre-normalized embeddings, FAISS `rq-kmeans` may produce identical
Euclidean and cosine results; use constrained or neural methods for that
distance comparison.

Run:

```bash
lazy_onerec/scripts/build_sid.sh
```

Output:

```text
lazy_onerec/output/kuairand_sid/<method>-<K1>-<K2>-<K3>-<distance>/
├── sid_index.json
├── codes.npy
├── codebooks.npz
└── sid_metrics.json
```

`sid_metrics.json` contains full SID-space utilization, collision rate,
singleton-cluster ratio, maximum and mean cluster sizes, P50/P90/P95/P99, and
SID-distribution entropy.

```text
full SID-space utilization = effective_cluster_count / (K1 * K2 * K3)
```

## Train

Set model and optimization parameters in `train_kuairand.sh`. Set `bf16=true`
when BF16 is supported. Every exposure in the two standard recommendation logs
is a target SID sample. Its context contains up to 128 preceding exposures with
their click and interaction feedback. The random-exposure log is not used. This
treats standard exposure as weak positive feedback and imitates the logging
policy's exposure distribution.

`user_features_1k.csv` is loaded once at startup. Each sample references 26
categorical features and four `log1p`-standardized continuous features by
`user_id`.
Each target exposure also carries four categorical request features: tab,
hour, day of week, and the time-gap bucket since the previous exposure.

Training dates are visited in ascending order. Batches are shuffled within each
date, and individual batches never cross date boundaries. A date's incomplete
final batch is retained, so gradient accumulation may span adjacent dates.

```bash
lazy_onerec/scripts/train_kuairand.sh
```

Default model output:

```text
lazy_onerec/output/kuairand_model/
```

## Checks

```bash
lazy_onerec/scripts/smoke_test.sh
```
