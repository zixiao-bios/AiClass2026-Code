"""
GPT 数据预处理脚本

独立运行，处理原始文本数据并保存结果。
处理流程：
1. 读取原始数据
2. 通用清洗（移除空行、短行、特殊字符行）
3. 滑动窗口切分
4. 划分训练集/验证集
5. 构建词汇表
6. 保存结果到文件

使用方法：
    python data_process.py
"""

import json
import random
import os

from config import min_sample_length, sentence_stride, val_ratio, min_line_length


def read_file(file_path):
    """读取原始文本文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def clean_text(text):
    """通用文本清洗
    
    清洗规则（不针对特定数据集）：
    1. 移除空行
    2. 移除过短的行（标题、序号等通常很短）
    3. 移除只包含特殊字符的行
    4. 移除全角空格等空白字符
    
    Args:
        text: 原始文本
    
    Returns:
        清洗后的文本（连续的字符串）
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # 移除首尾空白（包括全角空格）
        line = line.strip().strip('　')
        
        # 1. 移除空行
        if not line:
            continue
        
        # 2. 移除过短的行（标题、序号等）
        if len(line) < min_line_length:
            continue
        
        # 3. 移除只包含特殊字符的行
        special_chars = set('*-=#_~·　 ')
        if all(c in special_chars for c in line):
            continue
        
        cleaned_lines.append(line)
    
    # 将所有行连接成一个连续的字符串
    return ''.join(cleaned_lines)


def create_samples(text, min_length, stride):
    """按句子边界切分文本
    
    规则：
    1. 每个样本必须以句子开头（前一个字符是句号或文本开头）
    2. 每个样本必须以句号结尾
    3. 长度不足 min_length 时，继续加入下一句
    4. 达到 min_length 后，在下一个句号处截断
    
    Args:
        text: 清洗后的文本
        min_length: 最小样本长度
        stride: 滑动步长（按句子数滑动）
    
    Returns:
        样本列表
    """
    # 定义句子结束符
    sentence_ends = {'。', '！', '？', '；'}
    
    # 先按句号分割成句子列表
    sentences = []
    current_sentence = []
    
    for char in text:
        current_sentence.append(char)
        if char in sentence_ends:
            sentences.append(''.join(current_sentence))
            current_sentence = []
    
    # 处理最后一个不完整的句子（如果有的话，丢弃）
    # 因为我们要求必须以句号结尾
    
    if not sentences:
        return []
    
    samples = []
    i = 0  # 起始句子索引
    
    while i < len(sentences):
        # 从第 i 个句子开始，累积到 min_length
        sample_sentences = []
        current_length = 0
        j = i
        
        # 累积句子直到达到 min_length
        while j < len(sentences):
            sample_sentences.append(sentences[j])
            current_length += len(sentences[j])
            j += 1
            
            # 达到最小长度后，当前句子已经是句号结尾，可以停止
            if current_length >= min_length:
                break
        
        # 如果收集到了句子，创建样本
        if sample_sentences:
            sample = ''.join(sample_sentences)
            samples.append(sample)
        
        # 滑动到下一个起始位置（按句子数滑动）
        # stride 表示每次滑动几个句子
        i += max(1, stride)
    
    return samples


def split_dataset(samples, val_ratio, seed=42):
    """划分训练集和验证集
    
    Args:
        samples: 样本列表
        val_ratio: 验证集比例
        seed: 随机种子（保证可复现）
    
    Returns:
        train_samples, val_samples
    """
    # 设置随机种子
    random.seed(seed)
    
    # 复制并打乱
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    # 计算验证集大小
    val_size = int(len(shuffled) * val_ratio)
    val_size = max(1, val_size)  # 至少保留1个验证样本
    
    # 划分
    val_samples = shuffled[:val_size]
    train_samples = shuffled[val_size:]
    
    return train_samples, val_samples


def build_vocab(samples):
    """构建词汇表
    
    从所有样本中收集字符，构建字符到ID的映射。
    
    Args:
        samples: 样本列表
    
    Returns:
        vocab: 字符到ID的字典
    """
    # 特殊 token
    special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
    
    # 初始化词汇表
    vocab = {token: i for i, token in enumerate(special_tokens)}
    
    # 收集所有字符
    for sample in samples:
        for char in sample:
            if char not in vocab:
                vocab[char] = len(vocab)
    
    return vocab


def save_samples(file_path, samples):
    """保存样本到文件（每个样本用空行分隔）"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(sample + '\n')
            f.write('\n')  # 空行分隔


def save_vocab(file_path, vocab):
    """保存词汇表到 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)


def save_stats(file_path, stats):
    """保存统计信息到 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def main():
    """主函数：执行完整的数据处理流程"""
    
    # 确保数据目录存在
    data_dir = '2-4/data'
    os.makedirs(data_dir, exist_ok=True)
    
    # 原始数据路径
    raw_file = os.path.join(data_dir, 'liwengduiyun.txt')
    
    print('=' * 50)
    print('GPT 数据预处理')
    print('=' * 50)
    
    # 1. 读取原始数据
    print('\n[1/6] 读取原始数据...')
    raw_text = read_file(raw_file)
    raw_char_count = len(raw_text)
    print(f'  原始字符数: {raw_char_count}')
    
    # 2. 通用清洗
    print('\n[2/6] 清洗文本...')
    cleaned_text = clean_text(raw_text)
    cleaned_char_count = len(cleaned_text)
    print(f'  清洗后字符数: {cleaned_char_count}')
    print(f'  移除比例: {(1 - cleaned_char_count/raw_char_count)*100:.1f}%')
    
    # 3. 按句子边界切分
    print('\n[3/6] 按句子边界切分...')
    print(f'  最小样本长度: {min_sample_length}')
    print(f'  句子滑动步长: {sentence_stride}')
    samples = create_samples(cleaned_text, min_sample_length, sentence_stride)
    print(f'  生成样本数: {len(samples)}')
    
    # 4. 划分训练集/验证集
    print('\n[4/6] 划分数据集...')
    print(f'  验证集比例: {val_ratio*100:.0f}%')
    train_samples, val_samples = split_dataset(samples, val_ratio)
    print(f'  训练样本数: {len(train_samples)}')
    print(f'  验证样本数: {len(val_samples)}')
    
    # 5. 构建词汇表
    print('\n[5/6] 构建词汇表...')
    vocab = build_vocab(train_samples + val_samples)
    print(f'  词汇表大小: {len(vocab)}')
    
    # 6. 保存结果
    print('\n[6/6] 保存结果...')
    
    train_file = os.path.join(data_dir, 'train.txt')
    val_file = os.path.join(data_dir, 'val.txt')
    vocab_file = os.path.join(data_dir, 'vocab.json')
    stats_file = os.path.join(data_dir, 'stats.json')
    
    save_samples(train_file, train_samples)
    print(f'  训练集: {train_file}')
    
    save_samples(val_file, val_samples)
    print(f'  验证集: {val_file}')
    
    save_vocab(vocab_file, vocab)
    print(f'  词汇表: {vocab_file}')
    
    # 保存统计信息
    stats = {
        '原始字符数': raw_char_count,
        '清洗后字符数': cleaned_char_count,
        '训练样本数': len(train_samples),
        '验证样本数': len(val_samples),
        '词汇表大小': len(vocab),
        '最小样本长度': min_sample_length,
        '句子滑动步长': sentence_stride,
        '验证集比例': val_ratio
    }
    save_stats(stats_file, stats)
    print(f'  统计信息: {stats_file}')
    
    # 打印样本示例
    print('\n' + '=' * 50)
    print('样本示例（前3个训练样本）')
    print('=' * 50)
    for i, sample in enumerate(train_samples[:3]):
        print(f'\n样本 {i+1}:')
        print(f'  {sample[:50]}...' if len(sample) > 50 else f'  {sample}')
    
    print('\n' + '=' * 50)
    print('数据预处理完成！')
    print('=' * 50)


if __name__ == '__main__':
    main()
