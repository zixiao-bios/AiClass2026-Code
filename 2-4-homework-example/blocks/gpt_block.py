"""
GPT Block (GPT 解码器块)

GPT Block 是 GPT 模型的核心组件，与原始 Transformer Decoder Block 的主要区别：
1. 没有 Cross-Attention（因为没有 Encoder）
2. 使用 Pre-LN（先 LayerNorm 再计算）而非 Post-LN

结构：
    x -> LN -> Self-Attention -> + -> LN -> FFN -> +
    |__________________________|   |______________|
           残差连接                    残差连接
"""

from torch import nn

from layers.multi_head_attention import MultiHeadAttention
from layers.ffn import FFN


class GPTBlock(nn.Module):
    """GPT 解码器块
    
    使用 Pre-LayerNorm 结构（GPT-2 引入的改进）：
    - 先进行 LayerNorm
    - 再进行 Attention/FFN 计算
    - 最后残差连接
    
    这种结构有助于训练稳定性，特别是在深层模型中。
    """

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """
        Args:
            d_model: 模型的隐藏维度
            ffn_hidden: FFN 的隐藏层维度
            n_head: 注意力头的数量
            drop_prob: Dropout 概率
        """
        super(GPTBlock, self).__init__()
        
        # 1. 第一个 LayerNorm（用于 Self-Attention 之前）
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. 因果自注意力机制（Masked Self-Attention）
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        
        # 3. 第一个 Dropout
        self.dropout1 = nn.Dropout(p=drop_prob)
        
        # 4. 第二个 LayerNorm（用于 FFN 之前）
        self.norm2 = nn.LayerNorm(d_model)
        
        # 5. 前馈神经网络（使用 GELU 激活函数）
        self.ffn = FFN(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        
        # 6. 第二个 Dropout
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask):
        """
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]
            mask: 因果掩码，形状为 [batch_size, seq_len, seq_len]
        
        Returns:
            输出张量，形状为 [batch_size, seq_len, d_model]
        """
        # ===== Self-Attention 子层 (Pre-LN) =====
        # 保存输入用于残差连接
        residual = x
        
        # 先 LayerNorm（Pre-LN 的核心）
        x = self.norm1(x)
        
        # 计算因果自注意力
        x = self.self_attention(x, mask=mask)
        
        # Dropout
        x = self.dropout1(x)
        
        # 残差连接
        x = x + residual

        # ===== FFN 子层 (Pre-LN) =====
        # 保存输入用于残差连接
        residual = x
        
        # 先 LayerNorm
        x = self.norm2(x)
        
        # 前馈神经网络
        x = self.ffn(x)
        
        # Dropout
        x = self.dropout2(x)
        
        # 残差连接
        x = x + residual

        return x
