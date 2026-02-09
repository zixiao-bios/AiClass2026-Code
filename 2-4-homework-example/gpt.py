"""
GPT 模型 (Generative Pre-trained Transformer)

GPT 是一个 Decoder-only 的 Transformer 架构，用于自回归语言建模。
给定前面的 token 序列，预测下一个 token。

与原始 Transformer 的主要区别：
1. 只有 Decoder，没有 Encoder（因此没有 Cross-Attention）
2. 使用 Pre-LayerNorm（先归一化再计算）
3. 使用可学习的位置嵌入
4. 使用 GELU 激活函数
5. 在最后一层后有一个额外的 LayerNorm
"""

import torch
from torch import nn

from blocks.gpt_block import GPTBlock
from embedding.gpt_embedding import GPTEmbedding


class GPT(nn.Module):
    """GPT 模型
    
    结构：
    Input -> Embedding -> [GPT Block] x N -> Final LayerNorm -> Linear -> Output
    
    使用因果掩码（Causal Mask）确保每个位置只能看到它之前的 token。
    """

    def __init__(self, vocab_size, d_model, n_head, max_len, ffn_hidden, 
                 n_blocks, drop_prob, pad_idx, device):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型的隐藏维度
            n_head: 多头注意力的头数
            max_len: 最大序列长度
            ffn_hidden: FFN 的隐藏层维度
            n_blocks: GPT Block 的数量
            drop_prob: Dropout 概率
            pad_idx: 填充 token 的索引
            device: 计算设备
        """
        super().__init__()
        
        self.pad_idx = pad_idx
        self.device = device

        # 1. GPT 嵌入层（Token Embedding + Learnable Position Embedding）
        self.embedding = GPTEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            drop_prob=drop_prob,
            pad_idx=pad_idx,
            device=device
        )

        # 2. GPT Blocks 堆叠
        self.blocks = nn.ModuleList([
            GPTBlock(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                drop_prob=drop_prob
            )
            for _ in range(n_blocks)
        ])

        # 3. 最终的 LayerNorm（GPT-2 引入的改进）
        # 在所有 Block 之后、输出层之前添加一个 LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

        # 4. 输出层：将隐藏状态映射到词汇表大小
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        """
        Args:
            x: 输入的 token 索引，形状为 [batch_size, seq_len]
        
        Returns:
            logits: 输出的 logits，形状为 [batch_size, seq_len, vocab_size]
        """
        # 1. 生成因果掩码
        mask = self.make_causal_mask(x)

        # 2. 通过嵌入层
        x = self.embedding(x)
        # x: [batch_size, seq_len, d_model]

        # 3. 通过所有 GPT Blocks
        for block in self.blocks:
            x = block(x, mask)

        # 4. 最终的 LayerNorm
        x = self.final_norm(x)

        # 5. 输出层
        logits = self.output_layer(x)
        # logits: [batch_size, seq_len, vocab_size]

        return logits

    def make_causal_mask(self, x):
        """生成因果掩码（下三角矩阵）
        
        因果掩码确保每个位置只能看到它自己和之前的位置，
        这是 GPT 自回归生成的关键。
        
        例如，对于序列长度为 4：
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len]
        
        Returns:
            因果掩码，形状为 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len = x.shape
        
        # 创建下三角矩阵
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        # mask: [seq_len, seq_len]
        
        # 扩展到 batch 维度
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        # mask: [batch_size, seq_len, seq_len]
        
        return mask

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token=None):
        """自回归生成文本
        
        Args:
            idx: 起始 token 索引，形状为 [batch_size, seq_len]
            max_new_tokens: 最大生成 token 数量
            temperature: 采样温度，越高越随机，越低越确定
            top_k: 如果设置，只从概率最高的 k 个 token 中采样
            eos_token: 结束 token 的索引
        
        Returns:
            生成的 token 序列，形状为 [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # 获取模型输出
            logits = self(idx)
            
            # 只取最后一个时间步的 logits
            logits = logits[:, -1, :] / temperature
            # logits: [batch_size, vocab_size]
            
            # 可选：top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # 转换为概率分布
            probs = torch.softmax(logits, dim=-1)
            
            # 采样下一个 token
            next_token = torch.multinomial(probs, num_samples=1)
            # next_token: [batch_size, 1]
            
            # 如果遇到结束 token，停止生成
            if eos_token is not None and next_token.item() == eos_token:
                break
            
            # 拼接到序列中
            idx = torch.cat([idx, next_token], dim=1)
            # shape: [batch_size, seq_len + 1]
        
        return idx
