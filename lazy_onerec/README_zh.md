# LazyOneRec

[English](README.md) | [简体中文](README_zh.md)

```text
KuaiRand 视频文本 -> Item Embedding -> 三层 SID -> next-SID 模型
```

Python 代码按 `data/`、`model/`、`sid/` 和 `src/` 分层，运行参数统一
在 `scripts/` 的 Shell 文件顶部修改。根目录只保留文档和依赖文件。

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

## 一键执行

在 `run_kuairand_pipeline.sh` 顶部设置以下参数：

```bash
embedding_model="Qwen/Qwen3-Embedding-0.6B"
sid_method="rq-kmeans"
sid_codebook_sizes=(512 512 512)
sid_distance_metric="cosine"
positive_target="all"  # all | click | long-view
num_train_epochs=1
```

模型结构、训练和推理参数仍分别使用 `train_kuairand.sh` 与
`evaluate_kuairand.sh` 中的配置。若两个 Embedding 路径具有相同目录名，
可显式设置 `embedding_name` 区分。

然后依次执行 Embedding、SID、训练和评估：

```bash
lazy_onerec/scripts/run_kuairand_pipeline.sh
```

输出目录按 Embedding 模型和 SID 配置关联：

```text
lazy_onerec/output/
├── embeddings/<embedding-model>/
├── kuairand_sid/<embedding-model>/<sid-method-codebooks-distance>/
├── models/<embedding-model>/<sid-config>[/target-<mode>]/epochs-N/
└── evaluations/<embedding-model>/<sid-config>[/target-<mode>]/epochs-N/
```

只有完整产物的阶段 manifest 与当前流水线配置一致时才会自动跳过；
Embedding 未完成时继续断点编码。`all` 不增加目标类型子目录，
`click` 和 `long-view` 使用
`target-<mode>` 子目录。
所有训练轮数都使用 `epochs-N` 子目录，包括 `epochs-1`。
每个阶段写入 `pipeline_stage.json`，记录使用的模型、SID 配置和上下游
目录。只检查目录与命令而不执行：

```bash
PIPELINE_DRY_RUN=true lazy_onerec/scripts/run_kuairand_pipeline.sh
```

## 生成 Item Embedding

在 `embed_kuairand_items.sh` 中设置：

```bash
model_name="Qwen/Qwen3-Embedding-0.6B"
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
lazy_onerec/output/embeddings/qwen-qwen3-embedding-0-6b/
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
codebook_sizes=(512 512 512)
distance_metric="cosine"  # euclidean | cosine
```

`method` 支持 `rq-kmeans`、`balanced-kmeans`、
`constrained-rq-kmeans`、`rq-vae` 和 `rq-kmeans-plus`。

`rq-kmeans` 支持每层使用不同的 2 的幂码本，例如 `256-512-1024`
对应 `nbits=[8,9,10]`。余弦模式会归一化输入，并在神经量化器中使用
余弦距离。
当 embedding 已经 L2 归一化时，FAISS `rq-kmeans` 的欧氏与余弦结果
可能相同；距离对比优先使用 constrained 或神经方法。

`balanced-kmeans` 实现 OneRec 的贪心均衡算法。每轮按中心顺序从尚未
分配的 item 中选择最近的固定配额，因此每簇大小为
`floor(N/K)` 或 `ceil(N/K)`。距离、Top-K、中心和 residual 更新默认
使用 CUDA；`balanced_distance_mode=auto` 会在显存足够时缓存
`N x K` 距离矩阵，否则自动使用分块流式 Top-K。该方法保证容量均衡，
但不保证最小费用流意义下的全局最优分配。

执行：

```bash
lazy_onerec/scripts/build_sid.sh
```

输出：

```text
lazy_onerec/output/kuairand_sid/<method>-<K1>-<K2>-<K3>-<distance>/
├── sid_index.json
├── codes.npy
├── codebooks.npz
└── sid_metrics.json
```

`sid_metrics.json` 包含完整 SID 空间利用率、碰撞率、孤点簇占比、
最大/平均簇大小、P50/P90/P95/P99 和 SID 分布熵。

```text
完整 SID 空间利用率 = effective_cluster_count / (K1 × K2 × K3)
```

## 训练

在 `train_kuairand.sh` 中设置模型与训练参数。支持 BF16 时设置
`bf16=true`。`positive_target` 控制目标曝光：

```text
all       click/long-view/like/follow/comment/forward/profile-enter 任一为 1
click     仅 is_click=1
long-view 仅 long_view=1
```

三种模式都排除 `is_hate=1`。随机曝光日志不参与训练，其他标准曝光仍
保留在时间线中用于计算请求间隔和历史边界，但不作为生成目标。

`num_train_epochs` 控制训练轮数，默认值为 `1`。一轮训练严格按日期
从早到晚遍历一次；多轮训练会再次从最早训练日开始，因此只建议用于
收敛性实验，不再属于严格在线单遍训练。

历史严格使用 `time_ms < target_time` 的行为，并独立构建 6 条序列：
click GID 128、long-view GID 128、long-view duration 128、like GID
64、deep-interaction GID 32 和 hate GID 16，共 496 个行为位置。
deep interaction 合并 follow、comment、forward 和 profile enter。
long-view duration 与 long-view GID 使用相同事件和 mask，但分别生成
token；其时长桶为：

```text
min(round(sqrt(duration_ms / 1000)), 99)
```

`user_features_1k.csv` 在启动时读取一次。每条样本通过 `user_id`
引用 26 个类别特征和 4 个经过 `log1p` 标准化的连续特征。
User ID 使用 128 维 embedding，其余类别特征统一使用 8 维。每条
目标曝光还包含 4 个请求类别特征：`tab`、小时、星期和距上次曝光的
时间间隔桶。User 与 Request 特征全部拼接后，统一投影为两个 256 维
token。与未压缩的行为序列拼接后，原始 Context 长度为 498。

每条行为序列分别使用一套独立的一层 Q-Former、learnable query 和
序列内部位置 embedding，不共享 Q-Former 参数。默认 query 数依次为：
click 16、long-view GID 16、long-view duration 16、like 8、deep
interaction 4、hate 2。压缩后得到 62 个行为 token，加上两个
User/Request token，Context Encoder 的实际输入长度为 64。

默认主干宽度为 256，使用 4 个 attention heads 和 1024 维 FFN。
五条 GID 序列共享 64 维 GID embedding，long-view duration 使用
8 维 embedding，之后分别投影到主干宽度。

Decoder 默认启用 per-token QKV 和 per-token SwiGLU。每个目标位置
分别使用独立的 Self-Attention Q/K/V、Cross-Attention Q 和 SwiGLU
参数；Context K/V 与 attention 输出投影保持共享。可在训练脚本中将
`per_token_qkv` 或 `per_token_ffn` 设为 `false` 关闭。

自然日统一由 `time_ms` 按 `Asia/Shanghai` 时区生成，不使用日志中的
源 `date` 字段切分。前 3 天仅构建历史，中间日期训练，最后 3 天测试。
训练日期固定升序，每天内部随机组 batch，单个 batch 不会跨日期。
每天不足 `micro_batch_size` 的尾部仍作为小 batch 训练；训练末尾不足
一次梯度累积的窗口也会执行 optimizer step。

```bash
lazy_onerec/scripts/train_kuairand.sh
```

模型默认保存到：

```text
lazy_onerec/output/kuairand_model/epochs-1/
```

## 评估

在 `evaluate_kuairand.sh` 中设置 checkpoint 和 SID artifact 路径，然后执行：

```bash
lazy_onerec/scripts/evaluate_kuairand.sh
```

评估默认使用 BF16、batch size 64、8 个 DataLoader workers，并缓存
Context K/V 和 Decoder Self-Attention K/V。显存不足时优先减小
`batch_size`；可设置 `kv_cache=false` 关闭缓存进行结果核对。

评估使用无约束 Top-10 Beam Search，从每层完整码本中生成原生 SID，
然后检查完整路径是否存在于 SID 索引。仅输出：

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

分层与总体指标均从同一组无约束 Beam Search 完整 SID 排名中统计；
`invalid_sid_rate` 表示生成路径不在 SID 索引中的比例。不使用 teacher
forcing，也不计算 item 粒度指标。

## 检查

```bash
lazy_onerec/scripts/smoke_test.sh
```
