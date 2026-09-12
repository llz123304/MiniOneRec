# LazyOneRec

[English](README.md) | [简体中文](README_zh.md)

Standalone pipeline:

```text
KuaiRand video text -> item embedding -> three-level SID -> next-SID model
```

Python implementations are under `lazy_onerec/src/`. All executable entry
points and their parameters are under `lazy_onerec/scripts/`.

## 1. Environment

Recommended versions:

| Component | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.3.1 |
| CUDA | 12.1 |
| Transformers | 4.51.3 |
| Sentence Transformers | 4.1.0 |

Run on the GPU server:

```bash
cd /data/sdb2/llz/code/onerec/MiniOneRec

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

uv venv --python 3.10 .venv
source .venv/bin/activate

uv pip install torch==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install -r lazy_onerec/requirements.txt
```

Verify the environment:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected output includes:

```text
torch: 2.3.1
cuda: 12.1
available: True
```

To select a physical GPU, edit the embedding, SID, or training shell script:

```bash
gpu_id=0
```

Use the index reported by `nvidia-smi`. The selected physical GPU is exposed
inside the process as `cuda:0`, so keep the default `device="cuda"`.

For a CUDA 11.8 server, replace `cu121` with `cu118` in the PyTorch index URL.
For macOS or CPU-only environments, use:

```bash
uv pip install torch==2.3.1
uv pip install -r lazy_onerec/requirements.txt
```

For non-CUDA execution, also set `device="cpu"` or `"mps"` in the relevant
shell scripts.

## 2. Data

Keep the following layout on the server:

```text
/data/sdb2/llz/code/onerec/MiniOneRec/
└── lazy_onerec/KuaiRand-1K/
    ├── data/
    │   ├── video_features_basic_1k.csv
    │   ├── log_standard_4_08_to_4_21_1k.csv
    │   ├── log_standard_4_22_to_5_08_1k.csv
    │   └── log_random_4_22_to_5_08_1k.csv
    ├── kuairand_video_captions.csv
    └── kuairand_video_categories.csv
```

Optional: generate 20 aligned metadata samples. Edit parameters at the top of
`lazy_onerec/scripts/sample_kuairand_metadata.sh`.

```bash
lazy_onerec/scripts/sample_kuairand_metadata.sh
```

## 3. Build Item Embeddings

Edit the configuration at the top of
`lazy_onerec/scripts/embed_kuairand_items.sh`. Common parameters:

```bash
model_name="Qwen/Qwen3-Embedding-0.6B"
scope="clicked"       # clicked | catalog
output_dim=""         # Empty: native Qwen dimension (1024); optionally use 512
batch_size=128
device="cuda"
normalize=true
```

Run:

```bash
lazy_onerec/scripts/embed_kuairand_items.sh
```

Default output:

```text
lazy_onerec/output/embeddings/qwen-qwen3-embedding-0-6b-clicked/
├── item_ids.npy
├── item_embeddings.npy
├── embedding_config.json
└── progress.json
```

The script resumes automatically. To regenerate both text and embeddings, set
`rebuild_texts=true` and `overwrite=true` for one run, then reset both to
`false`.
Text preparation and model encoding show live progress, throughput, and ETA.

## 4. Build Semantic IDs

Edit `lazy_onerec/scripts/build_sid.sh`:

```bash
method="constrained-rq-kmeans"
codebook_sizes=(256 256 256)
```

Available methods:

```text
rq-kmeans
constrained-rq-kmeans
rq-vae
rq-kmeans-plus
```

Run:

```bash
lazy_onerec/scripts/build_sid.sh
```

SID construction shows live codebook-level or training-batch progress. Neural
evaluation and final SID encoding use separate progress bars.

Output:

```text
lazy_onerec/output/kuairand_sid/
├── sid_index.json
├── codes.npy
├── codebooks.npz
└── sid_metrics.json
```

`sid_metrics.json` evaluates clusters formed by complete SIDs. It reports the
singleton-cluster ratio, maximum and mean cluster sizes, P50/P90/P95/P99, and
raw and normalized SID-distribution entropy, effective-cluster perplexity, and
collision counts and rates. Quantiles exclude unused codes.

```text
singleton-cluster ratio = size-one complete-SID clusters / effective clusters
mean cluster size = total items / effective clusters
SID distribution entropy = -sum(p_i * ln(p_i))
normalized entropy = SID distribution entropy / ln(effective clusters)
```

## 5. Train

Edit data, model, and optimization settings at the top of
`lazy_onerec/scripts/train_kuairand.sh`. Enable BF16 when supported:

```bash
bf16=true
```

Run:

```bash
lazy_onerec/scripts/train_kuairand.sh
```

Default model output:

```text
lazy_onerec/output/kuairand_model/
```

## 6. Checks

```bash
lazy_onerec/scripts/smoke_test.sh
lazy_onerec/scripts/embed_kuairand_items.sh --help
lazy_onerec/scripts/build_sid.sh --help
```
