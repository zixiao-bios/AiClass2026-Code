"""
GPT 文本处理模块

GPT 是一个语言模型，只处理单一语言的文本。
这里我们使用中文文本进行训练，按字符分词。
"""

import torch
import json


def tokenize(text: str) -> list[str]:
    """中文分词：按字符分割
    
    Args:
        text: 输入文本
    
    Returns:
        字符列表
    """
    return list(text)


def add_special_tokens(tokens, bos, eos, pad, pad_len):
    """添加特殊 token
    
    为序列添加起始符、结束符，并进行填充或截断。
    
    Args:
        tokens: token 列表
        bos: 起始 token
        eos: 结束 token
        pad: 填充 token
        pad_len: 目标长度
    
    Returns:
        处理后的 token 列表
    """
    tokens = [bos] + tokens + [eos]
    
    if len(tokens) < pad_len:
        # 长度不足，填充
        tokens += [pad] * (pad_len - len(tokens))
    else:
        # 长度超过，截断（保留 eos）
        tokens = tokens[:pad_len]
        if tokens[-1] != eos:
            tokens[-1] = eos
    
    return tokens


def make_vocab(dataset):
    """根据数据集构建词汇表
    
    Args:
        dataset: 文本列表
    
    Returns:
        vocab: token 到 id 的映射字典
    """
    # 特殊 token
    special_tokens = ['<bos>', '<eos>', '<pad>', '<unk>']
    
    # 初始化词汇表
    vocab = {token: i for i, token in enumerate(special_tokens)}
    
    # 遍历数据集，收集所有字符
    for text in dataset:
        tokens = tokenize(text)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    
    # 保存词汇表
    with open('2-4/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=4, ensure_ascii=False)
    
    print(f'词汇表大小: {len(vocab)}')
    return vocab


def process_text(text, vocab, pad_len):
    """将文本转换为模型输入的 tensor
    
    Args:
        text: 输入文本
        vocab: 词汇表
        pad_len: 填充长度
    
    Returns:
        token 索引的 tensor，形状为 [pad_len]
    """
    # 分词
    tokens = tokenize(text)
    
    # 添加特殊 token
    tokens = add_special_tokens(tokens, '<bos>', '<eos>', '<pad>', pad_len)
    
    # 转换为 id
    ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    
    # 转换为 tensor
    return torch.tensor(ids, dtype=torch.long)


def idx_to_text(idx_list, id2token):
    """将 id 列表转换为文本
    
    Args:
        idx_list: id 列表
        id2token: id 到 token 的映射
    
    Returns:
        文本字符串
    """
    # 过滤掉特殊 token
    special_tokens = {'<bos>', '<eos>', '<pad>', '<unk>'}
    chars = [id2token[i] for i in idx_list if id2token.get(i, '<unk>') not in special_tokens]
    return ''.join(chars)
