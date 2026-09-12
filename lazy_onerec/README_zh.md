# LazyOneRec

[English](README.md) | [简体中文](README_zh.md)

独立完成三步流程：

```text
KuaiRand 视频文本 -> Item Embedding -> 三层 SID -> next-SID 模型
```

Python 实现在 `lazy_onerec/src/`，所有执行入口和参数配置在
`lazy_onerec/scripts/`。

## 1. 环境

推荐环境：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.3.1 |
| CUDA | 12.1 |
| Transformers | 4.51.3 |
| Sentence Transformers | 4.1.0 |

在 GPU 服务器上执行：

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

检查环境：

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

预期至少包含：

```text
torch: 2.3.1
cuda: 12.1
available: True
```

需要指定物理 GPU 时，在 embedding、SID 或训练 Shell 顶部修改：

```bash
gpu_id=0
```

编号以 `nvidia-smi` 为准。设置后该物理 GPU 在进程内映射为 `cuda:0`，
因此 `device` 保持默认的 `"cuda"`，不要再改成 `"cuda:1"`。

CUDA 11.8 服务器只需将安装地址中的 `cu121` 改为 `cu118`。macOS 或
CPU 环境使用：

```bash
uv pip install torch==2.3.1
uv pip install -r lazy_onerec/requirements.txt
```

非 CUDA 环境还需将相关 Shell 中的 `device` 改为 `"cpu"` 或 `"mps"`。

## 2. 数据

服务器目录必须保持为：

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

可选：生成 20 条对齐样本。参数在
`lazy_onerec/scripts/sample_kuairand_metadata.sh` 顶部修改。

```bash
lazy_onerec/scripts/sample_kuairand_metadata.sh
```

## 3. 生成 Item Embedding

编辑 `lazy_onerec/scripts/embed_kuairand_items.sh` 顶部配置。常用参数：

```bash
model_name="Qwen/Qwen3-Embedding-0.6B"
scope="clicked"       # clicked | catalog
output_dim=""         # 空值为 Qwen 原生 1024 维；可设为 512
batch_size=128
device="cuda"
normalize=true
```

执行：

```bash
lazy_onerec/scripts/embed_kuairand_items.sh
```

默认输出：

```text
lazy_onerec/output/embeddings/qwen-qwen3-embedding-0-6b-clicked/
├── item_ids.npy
├── item_embeddings.npy
├── embedding_config.json
└── progress.json
```

脚本支持断点续传。需要重新生成文本和 embedding 时，将
`rebuild_texts=true`、`overwrite=true`，运行一次后改回 `false`。
文本构建和模型编码均显示动态进度、处理速度和预计剩余时间。

## 4. 生成 SID

编辑 `lazy_onerec/scripts/build_sid.sh`：

```bash
method="constrained-rq-kmeans"
codebook_sizes=(256 256 256)
```

`method` 可选：

```text
rq-kmeans
constrained-rq-kmeans
rq-vae
rq-kmeans-plus
```

执行：

```bash
lazy_onerec/scripts/build_sid.sh
```

SID 构建会显示码本层级或训练 batch 的动态进度；神经方法的周期性评估
和最终 SID 编码也有独立进度条。

输出：

```text
lazy_onerec/output/kuairand_sid/
├── sid_index.json
├── codes.npy
├── codebooks.npz
└── sid_metrics.json
```

`sid_metrics.json` 统计完整 SID 形成的最终簇，包括孤点簇占比、最大簇、
平均簇大小、P50/P90/P95/P99、SID 分布熵、归一化分布熵，以及碰撞
数量和碰撞率。分位数仅基于实际出现的簇计算，不包含未使用码字。

```text
孤点簇占比 = 大小为 1 的完整 SID 簇数 / 有效完整 SID 簇数
平均簇大小 = 物品总数 / 有效完整 SID 簇数
SID 分布熵 = -Σ p_i × ln(p_i)，其中 p_i = 第 i 个簇大小 / 物品总数
归一化分布熵 = SID 分布熵 / ln(有效完整 SID 簇数)
```

## 5. 训练

编辑 `lazy_onerec/scripts/train_kuairand.sh` 顶部的数据、模型和训练参数。
GPU 支持 BF16 时设置：

```bash
bf16=true
```

执行：

```bash
lazy_onerec/scripts/train_kuairand.sh
```

默认模型输出目录：

```text
lazy_onerec/output/kuairand_model/
```

## 6. 检查

```bash
lazy_onerec/scripts/smoke_test.sh
lazy_onerec/scripts/embed_kuairand_items.sh --help
lazy_onerec/scripts/build_sid.sh --help
```
