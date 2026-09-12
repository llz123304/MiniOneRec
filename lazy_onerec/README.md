# LazyOneRec

[English](README.md) | [简体中文](README_zh.md)

```text
KuaiRand video text -> item embedding -> three-level SID -> next-SID model
```

Python code is under `lazy_onerec/src/`. Configure each command at the top of
its Shell file under `lazy_onerec/scripts/`.

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

Generate metadata samples:

```bash
lazy_onerec/scripts/sample_kuairand_metadata.sh
```

## Build Item Embeddings

Set these values in `embed_kuairand_items.sh`:

```bash
model_name="Qwen/Qwen3-Embedding-0.6B"
scope="catalog"       # All videos; clicked processes clicked videos only
output_dim=""         # Native Qwen dimension is 1024; use 512 for MRL
batch_size=1024
device="cuda"
normalize=true
```

Run:

```bash
lazy_onerec/scripts/embed_kuairand_items.sh
```

Output:

```text
lazy_onerec/output/embeddings/<model>-<scope>/
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
method="constrained-rq-kmeans"
codebook_sizes=(256 256 256)
```

`method` supports `rq-kmeans`, `constrained-rq-kmeans`, `rq-vae`, and
`rq-kmeans-plus`.

Run:

```bash
lazy_onerec/scripts/build_sid.sh
```

Output:

```text
lazy_onerec/output/kuairand_sid/
├── sid_index.json
├── codes.npy
├── codebooks.npz
└── sid_metrics.json
```

`sid_metrics.json` contains collision rate, singleton-cluster ratio, maximum
and mean cluster sizes, P50/P90/P95/P99, and SID-distribution entropy.

## Train

Set model and optimization parameters in `train_kuairand.sh`. Set `bf16=true`
when BF16 is supported.

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
lazy_onerec/scripts/embed_kuairand_items.sh --help
lazy_onerec/scripts/build_sid.sh --help
```
