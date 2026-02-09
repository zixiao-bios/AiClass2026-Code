"""
GPT 嵌入层 (GPT Embedding)

GPT 的嵌入层包含两部分：
1. Token Embedding: 将 token ID 映射为向量
2. Position Embedding: 可学习的位置嵌入（区别于原始 Transformer 的固定正弦编码）

最终嵌入 = Token Embedding + Position Embedding
"""

import torch
from torch import nn


class GPTEmbedding(nn.Module):
    """GPT 嵌入层
    
    与原始 Transformer 的主要区别：
    - 使用可学习的位置嵌入（nn.Embedding）而非固定的正弦位置编码
    - 这让模型能够自己学习最适合的位置表示
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, pad_idx, device):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型的嵌入维度
            max_len: 最大序列长度
            drop_prob: Dropout 概率
            pad_idx: 填充 token 的索引
            device: 计算设备
        """
        super().__init__()
        # TODO: 初始化
        # 1. 保存 device
        # 2. Token 嵌入层:
        # 3. 位置嵌入层
        # 4. Dropout 层

    def forward(self, x):
        """
        Args:
            x: 输入的 token 索引，形状为 [batch_size, seq_len]
        
        Returns:
            嵌入表示，形状为 [batch_size, seq_len, d_model]
        """
        # TODO: 实现嵌入层前向传播
        # 1. 计算 token 嵌入
        # 2. 生成位置索引 [0, 1, 2, ..., seq_len-1]
        # 3. 计算位置嵌入
        # 4. 将 token 嵌入和位置嵌入相加
        # 5. 应用 Dropout
        pass
