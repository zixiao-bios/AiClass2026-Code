"""
GPT 训练脚本

从预处理好的数据文件中读取训练集和验证集，
训练 GPT 模型，并使用 TensorBoard 记录训练过程。

使用方法：
    1. 先运行 data_process.py 预处理数据
    2. 运行本脚本进行训练
    3. 使用 tensorboard --logdir=2-4/runs 查看训练曲线
"""

import os
import json
import time
import math
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter
from config import *
from gpt import GPT


def load_processed_data():
    """加载预处理好的数据
    
    文件格式：每个样本一行，样本之间用空行分隔
    
    Returns:
        train_samples: 训练样本列表
        val_samples: 验证样本列表
        vocab: 词汇表字典
    """
    # 检查文件是否存在
    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f'找不到训练数据文件: {train_file}\n'
            f'请先运行 python data_process.py 预处理数据'
        )
    
    # 读取训练集（跳过空行）
    with open(train_file, 'r', encoding='utf-8') as f:
        train_samples = [line.strip() for line in f if line.strip()]
    
    # 读取验证集
    with open(val_file, 'r', encoding='utf-8') as f:
        val_samples = [line.strip() for line in f if line.strip()]
    
    # 读取词汇表
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    return train_samples, val_samples, vocab


def samples_to_tensor(samples, vocab, max_len):
    """将样本列表转换为模型输入的 tensor
    
    为每个样本添加 <bos> 和 <eos>，并进行填充/截断。
    
    Args:
        samples: 样本列表（字符串）
        vocab: 词汇表
        max_len: 最大序列长度
    
    Returns:
        tensor: [num_samples, max_len]
    """
    pad_idx = vocab['<pad>']
    bos_idx = vocab['<bos>']
    eos_idx = vocab['<eos>']
    unk_idx = vocab['<unk>']
    
    tensors = []
    for sample in samples:
        # 字符转 ID
        ids = [vocab.get(char, unk_idx) for char in sample]
        
        # 添加 <bos> 和 <eos>
        ids = [bos_idx] + ids + [eos_idx]
        
        # 截断或填充
        if len(ids) > max_len:
            ids = ids[:max_len-1] + [eos_idx]  # 保留 eos
        else:
            ids = ids + [pad_idx] * (max_len - len(ids))
        
        tensors.append(torch.tensor(ids, dtype=torch.long))
    
    return torch.stack(tensors)


def compute_perplexity(loss):
    """计算困惑度
    
    Perplexity = exp(CrossEntropyLoss)
    困惑度越低，模型对数据的预测越好。
    
    Args:
        loss: 交叉熵损失值
    
    Returns:
        困惑度值
    """
    return math.exp(loss)


def evaluate(model, dataloader, criterion, vocab, device):
    """在验证集上评估模型
    
    Args:
        model: GPT 模型
        dataloader: 验证集 DataLoader
        criterion: 损失函数
        vocab: 词汇表
        device: 计算设备
    
    Returns:
        avg_loss: 平均损失
        perplexity: 困惑度
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            
            # 输入: x[:, :-1], 目标: x[:, 1:]
            input_ids = x[:, :-1]
            target_ids = x[:, 1:]
            
            # 前向传播
            logits = model(input_ids)
            
            # 计算损失
            loss = criterion(
                logits.reshape(-1, len(vocab)),
                target_ids.reshape(-1)
            )
            
            # 累积损失（按 token 数加权）
            num_tokens = (target_ids != vocab['<pad>']).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = compute_perplexity(avg_loss)
    
    return avg_loss, perplexity


def main():
    # ==================== 加载数据 ====================
    print('=' * 60)
    print('GPT 训练')
    print('=' * 60)
    
    print('\n[1/4] 加载预处理数据...')
    train_samples, val_samples, vocab = load_processed_data()
    id2token = {i: token for token, i in vocab.items()}
    
    print(f'  训练样本数: {len(train_samples)}')
    print(f'  验证样本数: {len(val_samples)}')
    print(f'  词汇表大小: {len(vocab)}')
    
    # 转换为 tensor
    print('\n[2/4] 准备数据集...')
    train_tensor = samples_to_tensor(train_samples, vocab, max_len)
    val_tensor = samples_to_tensor(val_samples, vocab, max_len)
    
    print(f'  训练数据形状: {train_tensor.shape}')
    print(f'  验证数据形状: {val_tensor.shape}')
    
    # 创建 DataLoader
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # ==================== 初始化模型 ====================
    print('\n[3/4] 初始化模型...')
    model = GPT(
        vocab_size=len(vocab),
        d_model=d_model,
        n_head=n_head,
        max_len=max_len,
        ffn_hidden=ffn_hidden,
        n_blocks=n_blocks,
        drop_prob=drop_prob,
        pad_idx=vocab['<pad>'],
        device=device
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f'  模型参数量: {num_params:,}')
    print(f'  计算设备: {device}')
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    
    # TensorBoard
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    print(f'  TensorBoard 日志: {tensorboard_dir}')
    
    # ==================== 训练循环 ====================
    print('\n[4/4] 开始训练...')
    print(f'  Epochs: {epochs}')
    print(f'  Batch Size: {batch_size}')
    print(f'  Learning Rate: {lr}')
    print('-' * 60)
    
    # TODO: 实现训练循环
    # 提示：
    # - 遍历 epochs
    # - 对于每个 batch：前向传播、计算损失、反向传播、更新参数
    # - 每个 epoch 结束后调用 evaluate() 评估验证集
    # - 使用 writer.add_scalars() 记录 Loss 和 Perplexity
    # - 保存最佳模型到 model_file
    
    # ==================== 训练完成 ====================
    if writer is not None:
        writer.close()
    
    print('-' * 60)
    print('训练完成！')
    print(f'\n查看训练曲线:')
    print(f'  tensorboard --logdir={tensorboard_dir}')


if __name__ == '__main__':
    # 选择计算设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    main()
