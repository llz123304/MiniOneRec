# LazyOneRec

[English](README.md) | [简体中文](README_zh.md)

独立实现以下流程：

1. 编码 KuaiRand 物品文本；
2. 生成三层语义 ID；
3. 训练自定义 next-SID 推荐模型。

Python 代码统一放在 `lazy_onerec/src/`，Shell 入口统一放在
`lazy_onerec/scripts/`。

所有命令均从 MiniOneRec 根目录执行：

```bash
cd /Users/bytedance/Desktop/llz/sentiment/MiniOneRec
```

## 环境

使用较成熟的组合：Python 3.10、`uv` 和 PyTorch 2.3.1。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10 .venv
source .venv/bin/activate
```

CUDA 12.1 环境安装 PyTorch：

```bash
uv pip install torch==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

CUDA 11.8 将 `cu121` 改为 `cu118`。macOS 或纯 CPU 环境执行：

```bash
uv pip install torch==2.3.1
```

安装其余依赖：

```bash
uv pip install -r lazy_onerec/requirements.txt
```

检查环境：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

PyTorch 版本应为：

```text
2.3.1
```

## 数据

本地数据应位于：

```text
lazy_onerec/KuaiRand/
├── 1K/KuaiRand-1K/data/
├── kuairand_video_captions.csv
└── kuairand_video_categories.csv
```

生成 20 条对齐的数据样本：

```bash
python -m lazy_onerec.src.sample_kuairand_metadata \
  --sample-size 20 \
  --seed 42
```

## 1. 生成物品 Embedding

先构建被点击视频的文本缓存：

```bash
python -m lazy_onerec.src.embed_kuairand_items \
  --scope clicked \
  --prepare-only
```

使用 Qwen3 编码：

```bash
python -m lazy_onerec.src.embed_kuairand_items \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --scope clicked \
  --output-dim 512 \
  --batch-size 128 \
  --device cuda
```

使用其他模型时只需替换 `--model-name`：

```text
BAAI/bge-m3
BAAI/bge-large-zh-v1.5
Alibaba-NLP/gte-Qwen2-1.5B-instruct
intfloat/multilingual-e5-large-instruct
moka-ai/m3e-base
```

如需编码完整 1K 视频目录：

```bash
python -m lazy_onerec.src.embed_kuairand_items \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --scope catalog \
  --output-dim 512 \
  --batch-size 128 \
  --device cuda
```

脚本默认自动断点续传。使用 `--overwrite` 重新编码，使用
`--rebuild-texts` 重新生成文本缓存。

Qwen clicked 模式的默认输出：

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

小规模测试：

```bash
python -m lazy_onerec.src.embed_kuairand_items \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --scope clicked \
  --limit 100 \
  --output-dir lazy_onerec/output/embed-smoke
```

## 2. 生成 SID

支持四种方法：

```text
rq-kmeans
constrained-rq-kmeans
rq-vae
rq-kmeans-plus
```

执行一种方法：

```bash
lazy_onerec/scripts/build_sid.sh \
  --method constrained-rq-kmeans \
  --embeddings lazy_onerec/output/embeddings/qwen-qwen3-embedding-0-6b-clicked/item_embeddings.npy \
  --item-ids lazy_onerec/output/embeddings/qwen-qwen3-embedding-0-6b-clicked/item_ids.npy \
  --codebook-sizes 256 256 256 \
  --output-dir lazy_onerec/output/kuairand_sid
```

使用 RQ-VAE 或 RQ-Kmeans+ 时补充：

```text
--epochs 500 --batch-size 2048 --device cuda
```

输出文件：

```text
sid_index.json
codes.npy
codebooks.npz
```

转换已有 MiniOneRec SID：

```bash
lazy_onerec/scripts/convert_minionerec_sid.sh \
  --input data/Amazon/index/Industrial_and_Scientific.index.json \
  --codebook-sizes 256 256 256 \
  --output lazy_onerec/output/converted_sid/sid_index.json
```

## 3. 训练模型

```bash
lazy_onerec/scripts/train_kuairand.sh \
  --data-root lazy_onerec/KuaiRand/1K/KuaiRand-1K \
  --sid-artifact lazy_onerec/output/kuairand_sid/sid_index.json \
  --output-dir lazy_onerec/output/kuairand_model \
  --max-history 128 \
  --bf16
```

训练使用标准推荐日志中的点击序列。随机曝光日志保留给无偏评测和后续
强化学习。

## 检查命令

查看 embedding 参数：

```bash
python -m lazy_onerec.src.embed_kuairand_items --help
```

查看 SID 参数：

```bash
lazy_onerec/scripts/build_sid.sh --help
```

运行模型冒烟测试：

```bash
lazy_onerec/scripts/smoke_test.sh
```
