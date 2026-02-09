# GPT (Generative Pre-trained Transformer) 实现

本项目实现了一个教程向的 GPT 模型，用于展示 GPT 架构的核心概念和实现细节。

## 目录结构

```
2-4/
├── data/
│   ├── liwengduiyun.txt      # 原始数据（笠翁对韵）
│   ├── train.txt             # 预处理后的训练集
│   ├── val.txt               # 预处理后的验证集
│   ├── vocab.json            # 词汇表
│   └── stats.json            # 数据统计信息
├── data_process.py           # 数据预处理脚本（独立运行）
├── train.py                  # 训练脚本（支持 TensorBoard）
├── eval.py                   # 评估/推理脚本（交互式生成）
├── config.py                 # 配置文件
├── gpt.py                    # GPT 主模型
├── blocks/
│   └── gpt_block.py          # GPT Block (核心组件)
├── layers/
│   ├── multi_head_attention.py    # 多头注意力
│   ├── scale_dot_product_attention.py  # 缩放点积注意力
│   └── ffn.py                # 前馈神经网络 (GELU)
├── embedding/
│   └── gpt_embedding.py      # GPT 嵌入层 (可学习位置编码)
├── runs/                     # TensorBoard 日志目录
└── README.md
```

## 快速开始

### 1. 数据预处理

```bash
cd 2-4
python data_process.py
```

输出示例：
```
==================================================
GPT 数据预处理
==================================================

[1/6] 读取原始数据...
  原始字符数: 9053

[2/6] 清洗文本...
  清洗后字符数: 8460
  移除比例: 6.6%

[3/6] 滑动窗口切分...
  窗口大小: 64
  滑动步长: 32
  生成样本数: 264

[4/6] 划分数据集...
  训练样本数: 238
  验证样本数: 26
...
```

预处理后的数据保存在 `data/` 目录下，可以直接查看：
```bash
cat data/train.txt | head -3
cat data/stats.json
```

### 2. 训练模型

```bash
python train.py
```

训练过程会：
- 加载预处理好的数据
- 训练 GPT 模型
- 在验证集上评估
- 使用 TensorBoard 记录 Loss 和 Perplexity
- 保存最佳模型到 `gpt.pth`

### 3. 查看训练曲线（可选）

```bash
pip install tensorboard  # 如果未安装
tensorboard --logdir=runs
```

然后在浏览器中打开 http://localhost:6006

### 4. 交互式生成

```bash
python eval.py
```

输入任意文字作为提示词，模型会生成后续内容：
```
提示词> 天对地
--- 生成结果 ---
温度=0.5: 天对地，雨对风。大陆对长空。山花对海树...
温度=1.0: 天对地雨对风大陆对长空山花海树赤日苍穹...
温度=1.5: 天对地，雨风。大陆长空山花海树赤对苍穹...
```

**eval.py 支持的命令：**
- `temp <值>` - 设置采样温度
- `topk <值>` - 设置 Top-K 采样
- `maxlen <值>` - 设置最大生成长度
- `quit` / `exit` - 退出

---

## GPT vs 原始 Transformer 的关键差异

### 1. 架构差异

| 特性 | 原始 Transformer | GPT |
|------|-----------------|-----|
| 架构类型 | Encoder-Decoder | **Decoder-only** |
| Cross-Attention | 有 | **无** |
| 任务 | 序列到序列（翻译） | 自回归语言建模 |

### 2. LayerNorm 位置

| 类型 | 计算顺序 | 优点 |
|------|---------|------|
| Post-LN (原始) | Attention → Add → LN | 原始设计 |
| **Pre-LN (GPT-2)** | **LN → Attention → Add** | **训练更稳定** |

### 3. 位置编码

| 类型 | 实现方式 | 特点 |
|------|---------|------|
| 固定正弦编码 | 数学公式计算 | 可推广到更长序列 |
| **可学习位置嵌入** | **nn.Embedding** | **让模型自己学习** |

### 4. 激活函数

| 模型 | 激活函数 | 特点 |
|------|---------|------|
| 原始 Transformer | ReLU | 简单，但有"死亡神经元"问题 |
| **GPT-2/3** | **GELU** | **更平滑，效果更好** |

---

## 数据处理流程

```
原始文本 (liwengduiyun.txt)
    │
    ▼
┌─────────────────────────────┐
│  通用清洗                    │
│  - 移除空行                  │
│  - 移除短行 (<10字符)         │
│  - 移除特殊字符行             │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  滑动窗口切分                 │
│  - window_size=64            │
│  - stride=32                 │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  划分数据集                   │
│  - 训练集 90%                │
│  - 验证集 10%                │
└─────────────────────────────┘
    │
    ▼
保存到 data/ 目录
```

---

## 模型配置

默认配置（`config.py`）：

```python
# 模型结构
d_model = 256        # 隐藏维度
n_head = 8           # 注意力头数
max_len = 128        # 最大序列长度
ffn_hidden = 1024    # FFN 隐藏层维度
n_blocks = 4         # GPT Block 数量
drop_prob = 0.1      # Dropout 概率

# 训练参数
batch_size = 16      # 批次大小
lr = 3e-4            # 学习率
epochs = 200         # 训练轮数

# 数据处理
window_size = 64     # 滑动窗口大小
stride = 32          # 滑动步长
val_ratio = 0.1      # 验证集比例
```

---

## 评估指标

### 困惑度 (Perplexity)

困惑度是语言模型最常用的评估指标：

```
Perplexity = exp(CrossEntropyLoss)
```

- 困惑度越低，模型对数据的预测越好
- 可以理解为模型在预测下一个词时的"困惑程度"
- 例如 PPL=10 表示模型平均在 10 个候选词中选择

### TensorBoard 可视化

训练过程中会记录：
- **Loss**: 训练集和验证集的交叉熵损失
- **Perplexity**: 训练集和验证集的困惑度

---

## 生成策略

### 温度采样 (Temperature Sampling)

```python
probs = softmax(logits / temperature)
next_token = sample(probs)
```

- 温度 < 1.0: 更确定性，倾向于选择高概率词
- 温度 = 1.0: 标准采样
- 温度 > 1.0: 更随机，增加多样性

### Top-K 采样

只从概率最高的 K 个词中采样，避免选择低概率的奇怪词。

---

## 扩展阅读

- [GPT-1 论文](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf): Improving Language Understanding by Generative Pre-Training
- [GPT-2 论文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): Language Models are Unsupervised Multitask Learners
- [GPT-3 论文](https://arxiv.org/abs/2005.14165): Language Models are Few-Shot Learners

---

## 注意事项

1. 本实现是教程向的简化版本，用于理解 GPT 架构
2. 训练数据量较小（笠翁对韵约 8000 字），生成效果有限
3. TensorBoard 是可选依赖，未安装时会自动跳过
4. 生产环境需要考虑 KV Cache 等优化技术
