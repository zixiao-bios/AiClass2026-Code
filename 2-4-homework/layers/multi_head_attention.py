"""
多头注意力机制 (Multi-Head Attention)

将注意力计算拆分到多个"头"上并行执行，让模型能从不同的表示子空间学习信息。
公式: MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_o
      其中 head_i = Attention(Q * W_q_i, K * W_k_i, V * W_v_i)
"""

import torch
import torch.nn as nn

from layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    """多头自注意力机制
    
    GPT 中只使用自注意力（Self-Attention），即 Q、K、V 来自同一个输入。
    通过因果掩码（Causal Mask）确保每个位置只能看到它之前的内容。
    """
    
    def __init__(self, d_model=512, n_head=8):
        """
        Args:
            d_model: 模型的隐藏维度
            n_head: 注意力头的数量
        """
        super().__init__()
        # TODO: 初始化
        # 1. 保存 d_model, n_head, head_dim = d_model // n_head
        # 2. 定义四个线性变换层: W_q, W_k, W_v, W_o
        # 3. 初始化 ScaleDotProductAttention
    
    def _split_heads(self, tensor):
        """将张量按注意力头数拆分
        
        Args:
            tensor: 形状为 [batch_size, seq_len, d_model]
        
        Returns:
            形状为 [batch_size, n_head, seq_len, head_dim]
        """
        # TODO: 实现多头拆分
        # 1. 重塑维度: [batch, seq_len, d_model] -> [batch, seq_len, n_head, head_dim]
        # 2. 转置: [batch, seq_len, n_head, head_dim] -> [batch, n_head, seq_len, head_dim]
        pass

    def forward(self, x, mask=None):
        """
        GPT 使用自注意力，所以 Q、K、V 都来自同一个输入 x
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]
            mask: 因果掩码，形状为 [batch_size, seq_len, seq_len]
        
        Returns:
            输出张量，形状为 [batch_size, seq_len, d_model]
        """
        # TODO: 实现多头注意力前向传播
        # 1. 线性投影生成 Q、K、V
        # 2. 拆分为多头
        # 3. mask 扩展到多头维度
        # 4. 计算缩放点积注意力
        # 5. 合并多头并进行输出投影
        pass
