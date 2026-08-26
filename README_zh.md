<div align="center">


<img src="./assets/logo.png" width="500em" ></img> 

**面向规模化生成式推荐的开源框架**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)
<a href="https://arxiv.org/abs/2510.24431"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red"></a>

<a href="https://arxiv.org/abs/2510.24431">📄 技术报告</a> | <a href="https://huggingface.co/kkknight/MiniOneRec">🤗 Huggingface</a> | <a href="https://modelscope.cn/models/k925238839/MiniOneRec">🤖  Modelscope</a>

[English](./README.md) | 简体中文
</div>

**MiniOneRec** 是首个完全开源的**生成式推荐（generative recommendation）**框架，提供覆盖 **语义 ID（SID）构建**、**监督微调（SFT）** 以及**面向推荐的强化学习（RL）** 的端到端完整流程。

---

## 📢 更新公告

- 2026-05-13 — 我们引入了全新的 TS-Rec 代码库，遵循 [Fine-grained Semantics Integration for Large Language Model-based Recommendation](https://arxiv.org/pdf/2602.22632) 提出的方法。衷心感谢各位贡献者为本次更新付出的宝贵努力与支持。

- 2026-01-04 — 关于基于 Instruct 模型复现结果与我们报告指标之间可能存在的差异，请检查评测日志中的 CC 指标是否非零（参见 calc.py）。若非零，说明模型仍在大量生成无效物品，约束解码（constrained decoding）未能成功。我们怀疑该问题可能与 transformer 库等依赖的版本有关，目前仍在排查以提供通用解决方案。在此期间，你可以将 Instruct 模型切换为 base 模型（如 Qwen2.5-base）以规避该问题。

- 2025-12-04 — 我们更新了新脚本以支持处理 Amazon23 数据集。

- 2025-12-01 — 我们修复了 data.py 中一个可能导致 SID–item 对齐任务提前看到答案的 bug。此前我们曾尝试用部分轨迹来引导完整的 SID–item 生成，该问题不影响模型性能。

- 2025-11-20 — **RQ-Kmeans+** 的 SID 构建方法已更新（最早由 **GPR** 提出，这是首个开源复现）。

- 2025-11-19 — 我们基于 Accelerate 实现了多 GPU 并行的文本转 embedding 方法，相较原始版本效率显著提升：rq/text2emb/amazon_text2emb.py

- 2025-11-19 — **constrained-RQ-Kmeans** 的 SID 构建方法已更新。

- 2025-11-07 — 感谢大家提交 issue！根据反馈，我们发布了新的实现。若运行代码时遇到任何问题，请先更新到并参考**最新版本**。

- 2025-11-07 — 现在你可以在 SFT 阶段冻结 LLM 参数，只训练新增 SID 词表的 embedding。

- 2025-10-31 — 你现在可以直接下载我们 MiniOneRec 模型的实现**检查点（checkpoints）**。

- 2025-10-31 — **RQ-Kmeans** 的 SID 构建方法已更新。

---

## 🛠️ 关键技术
<div align="center">
<img src="./assets/minionerec_framework.png" width=100% ></img> 
</div>

- **SID 构建：MiniOneRec 首先将每个商品转化为一个紧凑、语义有意义的 token。** 它将商品的标题与描述拼接，输入冻结的文本编码器，再用三层 RQ-VAE 对得到的 embedding 进行量化。

- **SFT：将所有商品改写为 SID 后，先对模型进行监督式训练。** 它将按时间顺序排列的用户历史视为一个 token 序列，通过 next-token 预测学习生成用户下一个可能消费商品的 SID。关键在于，该阶段与一组语言对齐目标（在自然语言与 SID 空间之间双向映射）联合训练，使推荐器既能继承大语言模型中蕴含的世界知识，又能将该知识落地到离散的物品编码上。

- **面向推荐的 RL：在 SFT 之后，MiniOneRec 进一步通过基于 GRPO 的面向推荐的 RL 阶段进行打磨。** 对每个 prompt 生成多个候选推荐，在组内对奖励进行归一化以稳定梯度，并用 KL 惩罚使更新后的策略保持接近参考策略。由于动作空间是一个封闭的物品 SID 列表，系统切换为约束 beam search，保证每个 beam 都唯一且有效，从而大幅提升采样效率与多样性。奖励信号本身融合了一个二值正确性项和一个感知排名（rank-aware）的分项——后者对高概率但错误的物品施以更重的惩罚，并且可以叠加协同过滤（collaborative-filtering）分数。整套流程共同作用，使 MiniOneRec 能够耦合稠密的语言知识，构建出一个高性能、轻量级的生成式推荐系统。

---

## 📊 评测

<div align="center">
<img src="./assets/minionerec_main_result.png" width=100% ></img> 
</div>

---

## 🗂️ 仓库结构总览

| 文件 / 目录               | 说明                                                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `sft.sh`                  | 启动监督微调（SFT）阶段的 Shell 脚本                                                                           |
| `sft.py`                  | SFT 训练循环的 Python 实现                                                                                     |
| `sft_gpr.py`              | 受 GPR 启发的 SFT，含价值感知微调（VAFT）：基于模拟物品价值实现加权损失                                        |
| `rl.sh`                   | 启动强化学习（RL）阶段的 Shell 脚本                                                                            |
| `rl.py`                   | RL 训练循环的 Python 实现                                                                                      |
| `rl_gpr.py`               | 受 GPR 启发的 RL，含层次增强策略优化（HEPO）                                                                   |
| `minionerec_trainer.py`   | MiniOneRec 训练器——专为生成式推荐定制的基于 GRPO 的训练器                                                      |
| `configs/`                | YAML 配置文件                                                                                                  |
| `evaluate.sh`             | 一键离线 Top-K 评测脚本                                                                                        |
| `evaluate.py`             | 用于计算 HR@K 和 NDCG@K 的评测工具                                                                             |
| `LogitProcessor.py`       | 用于约束解码的 Logit 处理器（Python 实现）                                                                     |
| `data.py`                 | SFT 与 RL 训练的数据管线                                                                                       |
| `convert_dataset.py`      | 将 RQ 训练好的数据集转换为「先 SFT 后 RL」的格式                                                               |
| `convert_dataset_gpr.py`  | 受 GPR 启发的数据集转换器：注入模拟的异构 token（U/E/I/O）以模拟统一输入表示                                   |
| `data/amazon18_data_process.sh`     | 将 Amazon18 数据过滤并预处理为 RQ-ready 格式的 Shell 脚本                                             |
| `data/amazon18_data_process.py`     | Amazon18 数据预处理管线的 Python 实现                                                                 |
| `data/amazon18_data_process_gpr.py` | 受 GPR 启发的 Amazon18 预处理：为统一输入表示提取异构特征                                             |
| `data/amazon23_data_process.sh`     | 将 Amazon23 数据过滤并预处理为 RQ-ready 格式的 Shell 脚本                                             |
| `data/amazon23_data_process.py`     | Amazon23 数据预处理管线的 Python 实现                                                                 |
| `rq/text2emb/amazon_text2emb.sh`    | 通过 emb_model 为 Amazon 数据集生成物品 embedding（标题+描述）的 Shell 脚本                           |
| `rq/text2emb/amazon_text2emb.py`    | 上述 embedding 生成的 Python 实现                                                                     |
| `rq/text2emb/amazon_text2emb_gpr.py`| 受 GPR 启发的文本转 embedding                                                                         |
| `rq/generate_indices.py`  | 训练完 RQ-VAE 模型后生成 SID 文件                                                                              |
| `rq/rqvae.sh`             | 在 Amazon 物品 embedding 上训练 RQ-VAE 的 Shell 脚本                                                           |
| `rq/rqvae.py`             | RQ-VAE 训练的 Python 实现                                                                                      |
| `rq/rqkmeans_faiss.py`    | 基于 faiss 的 RQ-Kmeans 训练 Python 实现                                                                       |
| `rq/rqkmeans_constrained.py`        | Constrained RQ-Kmeans 的 Python 实现                                                                  |
| `rq/rqkmeans_constrained.sh`        | 在 Amazon 物品 embedding 上训练 constrained RQ-Kmeans 的 Shell 脚本                                   |
| `rq/rqkmeans_plus.py`     | RQ-Kmeans+ 的 Python 实现                                                                                      |
| `rq/rqkmeans_plus.sh`     | 在 Amazon 物品 embedding 上训练 RQ-Kmeans+ 的 Shell 脚本                                                       |
| `rq/generate_indices_plus.py`       | 训练完 RQ-Kmeans+ 模型后生成 SID 文件                                                                 |
| `rq/generate_indices_plus.sh`       | 训练完 RQ-Kmeans+ 模型后生成 SID 文件的 Shell 脚本                                                    |
| `requirements.txt`        | Python 依赖列表                                                                                                |

---

## 🚀 快速开始

使用我们提供的预训练 Industrial/Office SID 即可快速上手！
仅需 4–8 张 A100/H100 GPU 即可完成复现。

### 1. 创建独立的 Python 环境

```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```

### 2. 安装所需依赖包

```bash
pip install -r requirements.txt
```

### 3. SFT

```bash
bash sft.sh
```

### 4. 面向推荐的 RL

```bash
bash rl.sh
```

### 5. 运行评测脚本

```bash
bash evaluate.sh
```

---

## 📜 完整流程详解

### 0. 前置条件
- GPU：<例如 4–8 × A100/H100 80 GB 或同等规格>
- Python：3.11

### 1. 环境搭建
- **1.1 克隆仓库**
```
git clone https://github.com/AkaliKong/MiniOneRec.git
cd MiniOneRec
```
- **1.2 创建并激活 conda 环境**
```
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```
- **1.3 安装依赖**
```
pip install -r requirements.txt
```

### 2. 数据准备

- **2.1 下载原始数据集（可选）**  
  从官方页面获取：
  [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)、
  [Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)、
  [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)。
  注意：Industrial 和 Office 数据集包含在 Amazon 2018 中；Amazon 2014 与 2023 版本需要对 data/amazon18_data_process.py 做少量修改。
- **2.2 过滤与预处理**
```
bash data/amazon18_data_process.sh \
     --dataset  your_dataset_type \ # 例如 Industrial
     --user_k 5 \
     --item_k 5 \
     --st_year 2017 \
     --st_month 10 \
     --ed_year 2018 \
     --ed_month 11 \
     --output_path ./data/Amazon18
```
- **2.3 将物品文本编码为 embedding**
```
bash rq/amazon_text2emb.sh \
     --dataset your_dataset_type \ # 例如 Industrial 
     --root your_processed_dataset_path \
     --plm_name qwen \
     --plm_checkpoint your_emb_model_path
```

### 3. SID 构建

从 3.1.1、3.1.2、3.1.3 或 3.1.4 中任选其一。

- **3.1.1 在 embedding 上训练 RQ-VAE**
```
bash rq/rqvae.sh \
      --data_path xxx/data/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
      --ckpt_dir ./output/Industrial_and_Scientific \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480
```

- **3.1.2 在 embedding 上训练 RQ-Kmeans**

```
conda install faiss-gpu
python rqkmeans_faiss.py --dataset Industrial_and_Scientific # 基于语义 embedding 的 RQ-Kmeans 方法碰撞率相对较高。
```

- **3.1.3 在 embedding 上训练 constrained RQ-Kmeans**
对于冲突的物品，我们额外增加一层进行去重；同时使用均衡约束以确保 SID 分布均匀。
```
pip install k_means_constrained
pip install polars
bash rqkmeans_constrained.sh
```

- **3.1.4 在 embedding 上训练 RQ-Kmeans+**
```
pip install k_means_constrained
pip install polars
bash rqkmeans_constrained.sh
bash rqkmeans_plus.sh
```

- **3.2 生成索引（仅 RQ-VAE 与 RQ-Kmeans+ 需要）**
```
python rq/generate_indices.py
# 或
bash rq/generate_indices_plus.sh
```

- **3.3 转换数据集格式**
```
python convert_dataset.py \
     --dataset_name Industrial_and_Scientific \
     --data_dir /path/to/Industrial_and_Scientific \
     --output_dir /path/to/ourput_dir \

```

### 4. SFT

```
bash sft.sh \
     --base_model your_model_path \
     --output_dir your_ourput_dir \
     --sid_index_path your_.index.json_path \
     --item_meta_path your_.item.json_path
```

### 5. 面向推荐的 RL
> （可选）对于生产级规模的数据集，考虑到强化学习的成本与边际收益递减，你可以只用数万量级的相对小子集来执行 RL 阶段。
```
bash rl.sh \
     --model_path your_model_path \
     --output_dir output_dir \
```

### 6. 离线评测

```
bash evaluate.sh \
     --exp_name your_model_path 
```

---

## 🤖 支持的 LLM 提供商

MiniOneRec 支持多个 LLM 提供商用于文本增强任务（例如用户偏好与物品特征抽取）。在你的 `api_info` 字典中配置提供商：

| 提供商 | `provider` 取值 | 默认 Base URL | 示例模型 |
|----------|-----------------|------------------|----------------|
| OpenAI | `"openai"` | — | `text-davinci-003` |
| DeepSeek | `"deepseek"` | `https://api.deepseek.com` | `deepseek-chat` |
| [MiniMax](https://www.minimaxi.com) | `"minimax"` | `https://api.minimax.io/v1` | `MiniMax-M2.7`、`MiniMax-M2.5` |

**示例 —— 使用 MiniMax：**

```python
api_info = {
    "provider": "minimax",
    "api_key_list": ["your-minimax-api-key"],
    "base_url": "https://api.minimax.io/v1",  # 可选，此为默认值
}
get_res_batch("MiniMax-M2.7", prompt_list, max_tokens=512, api_info=api_info)
```

---

## 📝 即将推出的功能

我们正积极扩展 MiniOneRec 的能力。以下增强已列入路线图：
* ⏱️ **更多 SID 构建算法**：即将支持 R-VQ、RQ-Kmeans、RQ-OPQ 与 RQ-VAE-v2（PLUM）。
* ⚙️ **MiniOneRec-Think**：无缝整合对话、推理与个性化推荐的模块，为复杂交互场景提供一站式解决方案。
* 🔍 **更广的数据集支持**：新增更多流行公开数据集（包括 Yelp），以进一步验证算法的通用性。

---

## 🏫 参与机构  <!-- omit in toc -->

本项目由以下机构共同开发：

- <img src="assets/lds.png" width="28px"> [LDS](https://data-science.ustc.edu.cn/_upload/tpl/15/04/5380/template5380/index.html)
- <img src="assets/alphalab.jpg" width="28px"> [AlphaLab](https://alphalab-ustc.github.io/index.html)
- <img src="assets/next.jpg" width="28px"> [NExT](https://www.nextcenter.org/)
 
---

## 🧩 贡献

我们欢迎并感谢一切贡献！如果你有改进 MiniOneRec 的想法，欢迎随时提交 pull request（PR）。

---
## 🙏 致谢

本仓库复用或改编了以下开源项目的部分代码。我们衷心感谢其作者与贡献者：

- [ReRe](https://github.com/sober-clever/ReRe)
- [LC-Rec](https://github.com/zhengbw0324/LC-Rec)

---

## 🔖 引用 <!-- omit in toc -->

如果你觉得我们的代码/论文/模型有帮助，欢迎引用我们的论文 📝 并给我们点个 star ⭐️！

```bib
@misc{MiniOneRec,
      title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation}, 
      author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
      year={2025},
      eprint={2510.24431},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
}

@article{ReRe,
      title={Reinforced Preference Optimization for Recommendation}, 
      author={Junfei Tan and Yuxin Chen and An Zhang and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Xiang Wang},
      journal={arXiv preprint arXiv:2510.12211},
      year={2025},
}

@inproceedings{RecZero,
      title={Think before Recommendation: Autonomous Reasoning-enhanced Recommender}, 
      author={Xiaoyu Kong and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Jiancan Wu and Xiang Wang},
      year={2025},
      booktitle={NeurIPS},
}

```

---

<div align="center">
我们欢迎社区的贡献！🤝
</div>
