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
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # 维度校验：确保 d_model 能被 n_head 整除
        assert self.head_dim * n_head == d_model, "d_model 必须能被 n_head 整除"

        # 定义四个线性变换层
        self.W_q = nn.Linear(d_model, d_model)  # Query 投影
        self.W_k = nn.Linear(d_model, d_model)  # Key 投影
        self.W_v = nn.Linear(d_model, d_model)  # Value 投影
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影
        
        # 缩放点积注意力计算模块
        self.attention = ScaleDotProductAttention()
    
    def _split_heads(self, tensor):
        """将张量按注意力头数拆分
        
        Args:
            tensor: 形状为 [batch_size, seq_len, d_model]
        
        Returns:
            形状为 [batch_size, n_head, seq_len, head_dim]
        """
        batch_size, seq_len, d_model = tensor.size()
        
        # 重塑维度：[batch, seq_len, d_model] -> [batch, seq_len, n_head, head_dim]
        tensor = tensor.view(batch_size, seq_len, self.n_head, self.head_dim)
        
        # 转置：[batch, seq_len, n_head, head_dim] -> [batch, n_head, seq_len, head_dim]
        return tensor.transpose(1, 2)

    def forward(self, x, mask=None):
        """
        GPT 使用自注意力，所以 Q、K、V 都来自同一个输入 x
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]
            mask: 因果掩码，形状为 [batch_size, seq_len, seq_len]
        
        Returns:
            输出张量，形状为 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # === 步骤 1: 线性投影生成 Q、K、V ===
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # === 步骤 2: 拆分为多头 ===
        Q = self._split_heads(Q)  # [batch, n_head, seq_len, head_dim]
        K = self._split_heads(K)
        V = self._split_heads(V)

        # mask 也需要扩展到多头维度
        # [batch, seq_len, seq_len] -> [batch, n_head, seq_len, seq_len]
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        # === 步骤 3: 计算缩放点积注意力 ===
        attn_output = self.attention(Q, K, V, mask)

        # === 步骤 4: 合并多头并进行输出投影 ===
        # [batch, n_head, seq_len, head_dim] -> [batch, seq_len, n_head, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # 展平最后两个维度：[batch, seq_len, d_model]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # 最终线性投影
        output = self.W_o(attn_output)

        return output
