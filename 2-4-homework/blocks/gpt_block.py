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
        super().__init__()
        # TODO: 初始化
        # 1. 第一个 LayerNorm（用于 Self-Attention 之前）
        # 2. MultiHeadAttention
        # 3. 第一个 Dropout
        # 4. 第二个 LayerNorm（用于 FFN 之前）
        # 5. FFN
        # 6. 第二个 Dropout

    def forward(self, x, mask):
        """
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]
            mask: 因果掩码，形状为 [batch_size, seq_len, seq_len]
        
        Returns:
            输出张量，形状为 [batch_size, seq_len, d_model]
        """
        # TODO: 实现 Pre-LN 结构的前向传播
        # Self-Attention 子层:
        # 1. 保存输入用于残差连接
        # 2. LayerNorm -> Self-Attention -> Dropout
        # 3. 残差连接
        #
        # FFN 子层:
        # 4. 保存输入用于残差连接
        # 5. LayerNorm -> FFN -> Dropout
        # 6. 残差连接
        pass
