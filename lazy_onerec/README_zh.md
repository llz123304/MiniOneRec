# LazyOneRec

[English](README.md) | [简体中文](README_zh.md)

```text
KuaiRand 视频文本 -> Item Embedding -> 三层 SID -> next-SID 模型
```

Python 代码位于 `lazy_onerec/src/`，运行参数统一在
`lazy_onerec/scripts/` 的 Shell 文件顶部修改。

## 环境

环境版本：Python 3.10、PyTorch 2.3.1、CUDA 12.1。

```bash
cd /data/sdb2/llz/code/onerec/MiniOneRec

# 创建并激活环境
uv venv --python 3.10 .venv
source .venv/bin/activate

# 安装 CUDA 12.1 版本的 PyTorch
uv pip install torch==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖
uv pip install -r lazy_onerec/requirements.txt

# 检查 PyTorch 和 GPU
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

各 GPU 脚本通过以下参数选择物理卡：

```bash
gpu_id=0
```

## 数据

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

生成元数据样本：

```bash
lazy_onerec/scripts/sample_kuairand_metadata.sh
```

## 生成 Item Embedding

在 `embed_kuairand_items.sh` 中设置：

```bash
model_name="Qwen/Qwen3-Embedding-0.6B"
scope="catalog"       # 全部视频；clicked 仅处理点击过的视频
output_dim=""         # Qwen 原生 1024 维；设为 512 使用 MRL 降维
batch_size=512
device="cuda"
normalize=false       # 保存原始向量；SID 余弦模式会在构建时归一化
```

执行：

```bash
lazy_onerec/scripts/embed_kuairand_items.sh
```

输出：

```text
lazy_onerec/output/embeddings/qwen-qwen3-embedding-0-6b-catalog-raw/
├── item_ids.npy
├── item_embeddings.npy
├── embedding_config.json
└── progress.json
```

脚本支持断点续传。重新生成时设置：

```bash
rebuild_texts=true
overwrite=true
```

## 生成 SID

在 `build_sid.sh` 中设置：

```bash
method="rq-kmeans"
preset="512-512-512-cosine"
```

`method` 支持 `rq-kmeans`、`constrained-rq-kmeans`、`rq-vae` 和
`rq-kmeans-plus`。

`preset` 支持：

```text
256-256-256-euclidean
256-256-256-cosine
512-512-512-cosine
256-512-1024-cosine
1024-512-256-euclidean
1024-512-256-cosine
```

`rq-kmeans` 支持每层使用不同的 2 的幂码本，例如 `256-512-1024`
对应 `nbits=[8,9,10]`。余弦模式会归一化输入，并在神经量化器中使用
余弦距离。
当 embedding 已经 L2 归一化时，FAISS `rq-kmeans` 的欧氏与余弦结果
可能相同；距离对比优先使用 constrained 或神经方法。

执行：

```bash
lazy_onerec/scripts/build_sid.sh
```

输出：

```text
lazy_onerec/output/kuairand_sid/<method>-<preset>/
├── sid_index.json
├── codes.npy
├── codebooks.npz
└── sid_metrics.json
```

`sid_metrics.json` 包含碰撞率、孤点簇占比、最大/平均簇大小、
P50/P90/P95/P99 和 SID 分布熵。

## 训练

在 `train_kuairand.sh` 中设置模型与训练参数。支持 BF16 时设置
`bf16=true`。

```bash
lazy_onerec/scripts/train_kuairand.sh
```

模型默认保存到：

```text
lazy_onerec/output/kuairand_model/
```

## 检查

```bash
lazy_onerec/scripts/smoke_test.sh
lazy_onerec/scripts/embed_kuairand_items.sh --help
lazy_onerec/scripts/build_sid.sh --help
```
