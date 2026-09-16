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

## Run the Full Pipeline

Set these parameters at the top of `run_kuairand_pipeline.sh`:

```bash
embedding_model="Qwen/Qwen3-Embedding-0.6B"
sid_method="rq-kmeans"
sid_codebook_sizes=(512 512 512)
sid_distance_metric="cosine"
positive_target="all"  # all | click | long-view
```

Model, training, and inference settings remain in `train_kuairand.sh` and
`evaluate_kuairand.sh`. Set `embedding_name` explicitly when two embedding
paths share the same final directory name.

Then run embedding, SID construction, training, and evaluation in order:

```bash
lazy_onerec/scripts/run_kuairand_pipeline.sh
```

Output directories preserve the embedding and SID lineage:

```text
lazy_onerec/output/
├── embeddings/<embedding-model>/
├── kuairand_sid/<embedding-model>/<sid-method-codebooks-distance>/
├── models/<embedding-model>/<sid-method-codebooks-distance>[/target-<mode>]/
└── evaluations/<embedding-model>/<sid-method-codebooks-distance>[/target-<mode>]/
```

Complete artifacts are skipped automatically. Incomplete embedding output
resumes from its saved progress. `all` keeps the original path; `click` and
`long-view` use a `target-<mode>` subdirectory. Each stage writes
`pipeline_stage.json` with
the selected model, SID configuration, and upstream directories. Preview the
resolved paths and commands without running them:

```bash
PIPELINE_DRY_RUN=true lazy_onerec/scripts/run_kuairand_pipeline.sh
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
when BF16 is supported. `positive_target` selects target exposures:

```text
all       any click/long-view/like/follow/comment/forward/profile-enter
click     is_click=1 only
long-view long_view=1 only
```

All modes exclude `is_hate=1`. Random-exposure logs are not used. Other
standard exposures remain in the chronological timeline for request gaps and
history boundaries but are not generation targets.

History strictly uses events with `time_ms < target_time` and forms six
independent sequences: click GID 128, long-view GID 128, long-view duration
128, like GID 64, deep-interaction GID 32, and hate GID 16, for 496 behavior
positions. Deep interaction merges follow, comment, forward, and profile
enter. Long-view duration uses the same events and mask as long-view GID but
produces separate tokens. Its bucket is:

```text
min(round(sqrt(duration_ms / 1000)), 99)
```

`user_features_1k.csv` is loaded once at startup. Each sample references 26
categorical features and four `log1p`-standardized continuous features by
`user_id`. User ID uses a 128-dimensional embedding; other categorical
features use 8 dimensions. Each target exposure also carries four categorical
request features: tab, hour, day of week, and the time-gap bucket since the
previous exposure. Their embeddings are concatenated with the user features
and jointly projected into two 256-dimensional tokens. Together with the
uncompressed behavior sequences, the raw context contains 498 tokens.

Each behavior sequence has its own one-layer Q-Former, learnable queries, and
local positional embeddings; Q-Former parameters are not shared across
sequences. The default query counts are 16 for click, 16 for long-view GID,
16 for long-view duration, 8 for like, 4 for deep interaction, and 2 for
hate. The 62 compressed behavior tokens plus two user/request tokens give
the Context Encoder a total length of 64.

The default backbone width is 256 with four attention heads and a
1024-dimensional FFN. The shared GID embedding is 64-dimensional, while the
long-view duration embedding is 8-dimensional; both are projected to the
backbone width.

Decoder per-token QKV and per-token SwiGLU are enabled by default. Each target
position has independent Self-Attention Q/K/V, Cross-Attention Q, and SwiGLU
weights. Context K/V and attention output projections remain shared. Disable
them with `per_token_qkv=false` or `per_token_ffn=false` in the training script.

Natural dates are derived from `time_ms` in the `Asia/Shanghai` timezone; the
source `date` field is not used for splitting. The first three dates provide
history only, middle dates train, and the final three dates test. Training
dates are visited in ascending order. Batches are shuffled within each date
and never cross date boundaries. A date's incomplete final batch is retained,
and the final incomplete gradient-accumulation window still performs an
optimizer step.

```bash
lazy_onerec/scripts/train_kuairand.sh
```

Default model output:

```text
lazy_onerec/output/kuairand_model/
```

## Evaluate

Set the checkpoint and SID artifact paths in `evaluate_kuairand.sh`, then run:

```bash
lazy_onerec/scripts/evaluate_kuairand.sh
```

Evaluation defaults to BF16, batch size 64, eight DataLoader workers, cached
Context K/V and Decoder Self-Attention K/V. Reduce `batch_size` first if GPU
memory is insufficient. Set `kv_cache=false` to disable caching for result
comparisons.

Evaluation uses unconstrained Top-10 beam search over each complete codebook,
then checks whether each generated complete path exists in the SID index. It
reports only:

```text
sid0_hr_at_10
sid0_mrr_at_10
sid1_hr_at_10
sid1_mrr_at_10
sid2_hr_at_10
sid2_mrr_at_10
overall_hr_at_10
overall_mrr_at_10
invalid_sid_rate
```

Per-level and overall metrics are all computed from the same ranked complete
SID paths produced by unconstrained beam search. `invalid_sid_rate` is the
fraction of generated paths absent from the SID index. No teacher-forced
evaluation or item-level metrics are used.

## Checks

```bash
lazy_onerec/scripts/smoke_test.sh
```
