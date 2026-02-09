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
        super(GPTEmbedding, self).__init__()
        
        self.device = device

        # 1. Token 嵌入层
        # 将离散的 token ID 映射为连续的向量
        self.tok_emb = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=d_model, 
            padding_idx=pad_idx
        )
        
        # 2. 位置嵌入层（可学习）
        # GPT 的关键特点：使用可学习的位置嵌入而非固定编码
        self.pos_emb = nn.Embedding(
            num_embeddings=max_len, 
            embedding_dim=d_model
        )

        # 3. Dropout 层
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        Args:
            x: 输入的 token 索引，形状为 [batch_size, seq_len]
        
        Returns:
            嵌入表示，形状为 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = x.size()
        
        # 计算 token 嵌入
        tok_emb = self.tok_emb(x)
        # tok_emb: [batch_size, seq_len, d_model]

        # 生成位置索引 [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(0, seq_len, device=self.device)
        # positions: [seq_len]
        
        # 计算位置嵌入
        pos_emb = self.pos_emb(positions)
        # pos_emb: [seq_len, d_model]

        # 将 token 嵌入和位置嵌入相加（广播机制自动扩展 batch 维度）
        embedding = tok_emb + pos_emb
        
        # 应用 Dropout
        return self.dropout(embedding)
