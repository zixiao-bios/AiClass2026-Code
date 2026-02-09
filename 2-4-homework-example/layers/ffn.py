"""
前馈神经网络 (Feed-Forward Network)

Transformer/GPT 中的位置前馈网络，本质上是一个两层的 MLP。
GPT 使用 GELU 激活函数（而不是原始 Transformer 的 ReLU）。

结构: Linear -> GELU -> Dropout -> Linear
"""

from torch import nn


class FFN(nn.Module):
    """位置前馈神经网络
    
    GPT 的关键改进之一：使用 GELU 激活函数替代 ReLU。
    GELU 更平滑，在实践中表现更好。
    """
    
    def __init__(self, d_model, hidden, drop_prob=0.1):
        """
        Args:
            d_model: 输入和输出的维度
            hidden: 隐藏层维度（通常是 d_model 的 4 倍）
            drop_prob: Dropout 概率
        """
        super(FFN, self).__init__()
        
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        
        # GPT 使用 GELU 激活函数（区别于原始 Transformer 的 ReLU）
        # GELU(x) = x * Φ(x)，其中 Φ 是标准正态分布的累积分布函数
        self.gelu = nn.GELU()
        
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]
        
        Returns:
            输出张量，形状为 [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
