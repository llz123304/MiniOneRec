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
GPU_ID=0
```

Use the index reported by `nvidia-smi`. The selected physical GPU is exposed
inside the process as `cuda:0`, so keep `DEVICE` set to `"auto"` or `"cuda"`.

For a CUDA 11.8 server, replace `cu121` with `cu118` in the PyTorch index URL.
For macOS or CPU-only environments, use:

```bash
uv pip install torch==2.3.1
uv pip install -r lazy_onerec/requirements.txt
```

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
MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
SCOPE="clicked"       # clicked | catalog
OUTPUT_DIM=""         # Empty: native Qwen dimension (1024); optionally use 512
BATCH_SIZE=128
DEVICE="auto"         # Uses CUDA automatically when available
NORMALIZE=true
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
`REBUILD_TEXTS=true` and `OVERWRITE=true` for one run, then reset both to
`false`.

## 4. Build Semantic IDs

Edit `lazy_onerec/scripts/build_sid.sh`:

```bash
METHOD="constrained-rq-kmeans"
CODEBOOK_SIZES=(256 256 256)
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

Output:

```text
lazy_onerec/output/kuairand_sid/
├── sid_index.json
├── codes.npy
└── codebooks.npz
```

## 5. Train

Edit data, model, and optimization settings at the top of
`lazy_onerec/scripts/train_kuairand.sh`. Enable BF16 when supported:

```bash
BF16=true
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
