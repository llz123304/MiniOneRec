# LazyOneRec

[English](README.md) | [简体中文](README_zh.md)

Standalone code for:

1. encoding KuaiRand item text;
2. building three-level semantic IDs;
3. training a custom next-SID recommender.

Python programs are under `lazy_onerec/src/`. Shell entry points are under
`lazy_onerec/scripts/`.

Run all commands from the MiniOneRec repository root:

```bash
cd /Users/bytedance/Desktop/llz/sentiment/MiniOneRec
```

## Environment

Use the conservative stack: Python 3.10, `uv`, and PyTorch 2.3.1.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10 .venv
source .venv/bin/activate
```

Install PyTorch for CUDA 12.1:

```bash
uv pip install torch==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8, replace `cu121` with `cu118`. For macOS or CPU-only:

```bash
uv pip install torch==2.3.1
```

Install the remaining dependencies:

```bash
uv pip install -r lazy_onerec/requirements.txt
```

Verify the environment:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected PyTorch version:

```text
2.3.1
```

## Data

Expected local paths:

```text
lazy_onerec/KuaiRand/
├── 1K/KuaiRand-1K/data/
├── kuairand_video_captions.csv
└── kuairand_video_categories.csv
```

Create a small aligned metadata sample:

```bash
python -m lazy_onerec.src.sample_kuairand_metadata \
  --sample-size 20 \
  --seed 42
```

## 1. Build Item Embeddings

Prepare the clicked-video text cache:

```bash
python -m lazy_onerec.src.embed_kuairand_items \
  --scope clicked \
  --prepare-only
```

Encode with Qwen3:

```bash
python -m lazy_onerec.src.embed_kuairand_items \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --scope clicked \
  --output-dim 512 \
  --batch-size 128 \
  --device cuda
```

Use another encoder by changing `--model-name`:

```text
BAAI/bge-m3
BAAI/bge-large-zh-v1.5
Alibaba-NLP/gte-Qwen2-1.5B-instruct
intfloat/multilingual-e5-large-instruct
moka-ai/m3e-base
```

Encode the complete 1K catalog instead of clicked videos:

```bash
python -m lazy_onerec.src.embed_kuairand_items \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --scope catalog \
  --output-dim 512 \
  --batch-size 128 \
  --device cuda
```

The script resumes automatically. Use `--overwrite` to restart encoding and
`--rebuild-texts` to rebuild the text cache.

Default Qwen clicked-video outputs:

```text
lazy_onerec/output/
├── kuairand_items/clicked/
│   ├── item_ids.npy
│   ├── item_texts.jsonl
│   └── text_manifest.json
└── embeddings/qwen-qwen3-embedding-0-6b-clicked/
    ├── item_ids.npy
    ├── item_embeddings.npy
    ├── embedding_config.json
    └── progress.json
```

Small pipeline test:

```bash
python -m lazy_onerec.src.embed_kuairand_items \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --scope clicked \
  --limit 100 \
  --output-dir lazy_onerec/output/embed-smoke
```

## 2. Build Semantic IDs

Available methods:

```text
rq-kmeans
constrained-rq-kmeans
rq-vae
rq-kmeans-plus
```

Run one method:

```bash
lazy_onerec/scripts/build_sid.sh \
  --method constrained-rq-kmeans \
  --embeddings lazy_onerec/output/embeddings/qwen-qwen3-embedding-0-6b-clicked/item_embeddings.npy \
  --item-ids lazy_onerec/output/embeddings/qwen-qwen3-embedding-0-6b-clicked/item_ids.npy \
  --codebook-sizes 256 256 256 \
  --output-dir lazy_onerec/output/kuairand_sid
```

For neural methods, add:

```text
--epochs 500 --batch-size 2048 --device cuda
```

SID outputs:

```text
sid_index.json
codes.npy
codebooks.npz
```

Convert an existing MiniOneRec SID index:

```bash
lazy_onerec/scripts/convert_minionerec_sid.sh \
  --input data/Amazon/index/Industrial_and_Scientific.index.json \
  --codebook-sizes 256 256 256 \
  --output lazy_onerec/output/converted_sid/sid_index.json
```

## 3. Train

```bash
lazy_onerec/scripts/train_kuairand.sh \
  --data-root lazy_onerec/KuaiRand/1K/KuaiRand-1K \
  --sid-artifact lazy_onerec/output/kuairand_sid/sid_index.json \
  --output-dir lazy_onerec/output/kuairand_model \
  --max-history 128 \
  --bf16
```

The training data uses standard-log click sequences. Random-exposure logs are
reserved for unbiased evaluation and later reinforcement learning.

## Checks

Show all embedding options:

```bash
python -m lazy_onerec.src.embed_kuairand_items --help
```

Show all SID options:

```bash
lazy_onerec/scripts/build_sid.sh --help
```

Run the model smoke test:

```bash
lazy_onerec/scripts/smoke_test.sh
```
