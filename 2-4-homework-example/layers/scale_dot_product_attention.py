"""
缩放点积注意力 (Scaled Dot-Product Attention)

这是 Transformer 和 GPT 共用的基础注意力计算模块。
计算公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
"""

import torch
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """缩放点积注意力
    
    根据指定的 Q、K、V 计算注意力输出，支持多头注意力和 Mask 功能。
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query，形状为 [batch_size, num_heads, seq_len, head_dim]
            K: Key，形状为 [batch_size, num_heads, seq_len, head_dim]
            V: Value，形状为 [batch_size, num_heads, seq_len, head_dim]
            mask: 因果掩码，形状为 [batch_size, num_heads, seq_len, seq_len]
        
        Returns:
            注意力加权后的输出，形状为 [batch_size, num_heads, seq_len, head_dim]
        """
        # 获取尺寸信息
        batch_size, num_heads, seq_len, head_dim = K.size()
        
        # 1. 计算注意力分数矩阵
        # Q @ K^T: [batch, heads, seq_len, head_dim] × [batch, heads, head_dim, seq_len]
        #        -> [batch, heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(head_dim, dtype=torch.float32, device=Q.device)
        )

        # 2. 应用因果掩码（GPT 的核心：防止看到未来的 token）
        if mask is not None:
            # 将掩码为 0 的位置设置为负无穷，softmax 后趋近于 0
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 3. Softmax 归一化得到注意力权重
        attn_weights = torch.softmax(scores, dim=-1)

        # 4. 加权求和得到输出
        # [batch, heads, seq_len, seq_len] × [batch, heads, seq_len, head_dim]
        # -> [batch, heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        return attn_output
