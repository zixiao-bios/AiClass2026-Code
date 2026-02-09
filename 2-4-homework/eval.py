"""
GPT 评估/推理脚本

加载训练好的 GPT 模型，进行交互式文本生成。
输入任意文字作为提示词，模型会预测并生成后续内容。

使用方法：
    python eval.py
"""

import os
import json
import torch

from gpt import GPT
from config import *


def load_model(vocab, device):
    """加载训练好的模型
    
    Args:
        vocab: 词汇表
        device: 计算设备
    
    Returns:
        加载好权重的模型
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f'找不到模型文件: {model_file}\n'
            f'请先运行 python train.py 训练模型'
        )
    
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
    
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()
    
    return model


def generate_text(model, vocab, id2token, prompt, max_tokens=50, 
                  temperature=1.0, top_k=None, device=None):
    """根据提示词生成文本
    
    Args:
        model: GPT 模型
        vocab: 词汇表
        id2token: ID 到字符的映射
        prompt: 提示词（字符串）
        max_tokens: 最大生成字符数
        temperature: 采样温度（越高越随机，越低越确定）
        top_k: Top-K 采样参数（只从概率最高的 K 个中采样）
        device: 计算设备
    
    Returns:
        生成的完整文本
    """
    bos_idx = vocab['<bos>']
    eos_idx = vocab['<eos>']
    unk_idx = vocab['<unk>']
    
    # 将提示词转换为 token IDs
    if prompt:
        ids = [vocab.get(char, unk_idx) for char in prompt]
        ids = [bos_idx] + ids
    else:
        ids = [bos_idx]
    
    # 转换为 tensor
    input_ids = torch.tensor([ids], device=device)
    
    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # 转换回文本
    special_tokens = {'<bos>', '<eos>', '<pad>', '<unk>'}
    chars = []
    for idx in output_ids[0].tolist():
        token = id2token.get(idx, '')
        if token == '<eos>':
            break
        if token not in special_tokens:
            chars.append(token)
    
    return ''.join(chars)


def main():
    # ==================== 加载词汇表和模型 ====================
    print('=' * 60)
    print('GPT 文本生成')
    print('=' * 60)
    
    # 检查词汇表
    if not os.path.exists(vocab_file):
        print(f'错误: 找不到词汇表文件 {vocab_file}')
        print('请先运行 python data_process.py 预处理数据')
        return
    
    # 加载词汇表
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    id2token = {i: token for token, i in vocab.items()}
    
    print(f'\n词汇表大小: {len(vocab)}')
    
    # 加载模型
    try:
        model = load_model(vocab, device)
        print(f'模型加载成功: {model_file}')
        print(f'计算设备: {device}')
    except FileNotFoundError as e:
        print(f'\n错误: {e}')
        return
    
    # ==================== 交互式生成 ====================
    print('\n' + '=' * 60)
    print('交互式文本生成')
    print('=' * 60)
    print('输入提示词，按回车生成文本')
    print('输入空行让模型自由生成')
    print('输入 "quit" 或 "exit" 退出')
    print('输入 "help" 查看更多选项')
    print('=' * 60)
    
    # 默认参数
    current_temp = 1.0
    current_top_k = None
    current_max_tokens = 50
    
    while True:
        try:
            prompt = input('\n提示词> ').strip()
            
            # 退出命令
            if prompt.lower() in ['quit', 'exit', 'q']:
                print('再见！')
                break
            
            # 帮助命令
            if prompt.lower() == 'help':
                print('\n可用命令:')
                print('  temp <值>     设置温度 (当前: {:.1f})'.format(current_temp))
                print('  topk <值>     设置 Top-K (当前: {})'.format(current_top_k or '无'))
                print('  maxlen <值>   设置最大生成长度 (当前: {})'.format(current_max_tokens))
                print('  quit/exit     退出程序')
                continue
            
            # 设置温度
            if prompt.lower().startswith('temp '):
                try:
                    current_temp = float(prompt.split()[1])
                    print(f'温度已设置为: {current_temp}')
                except (IndexError, ValueError):
                    print('用法: temp <值>，例如: temp 0.8')
                continue
            
            # 设置 Top-K
            if prompt.lower().startswith('topk '):
                try:
                    val = prompt.split()[1]
                    current_top_k = None if val.lower() == 'none' else int(val)
                    print(f'Top-K 已设置为: {current_top_k or "无"}')
                except (IndexError, ValueError):
                    print('用法: topk <值>，例如: topk 10 或 topk none')
                continue
            
            # 设置最大长度
            if prompt.lower().startswith('maxlen '):
                try:
                    current_max_tokens = int(prompt.split()[1])
                    print(f'最大生成长度已设置为: {current_max_tokens}')
                except (IndexError, ValueError):
                    print('用法: maxlen <值>，例如: maxlen 100')
                continue
            
            # 生成文本
            print('\n--- 生成结果 ---')
            
            # 使用不同温度生成
            if current_temp == 1.0 and current_top_k is None:
                # 默认模式：展示不同温度的效果
                for temp in [0.5, 1.0, 1.5]:
                    text = generate_text(
                        model, vocab, id2token, prompt,
                        max_tokens=current_max_tokens,
                        temperature=temp,
                        top_k=current_top_k,
                        device=device
                    )
                    print(f'温度={temp}: {text}')
            else:
                # 自定义模式
                text = generate_text(
                    model, vocab, id2token, prompt,
                    max_tokens=current_max_tokens,
                    temperature=current_temp,
                    top_k=current_top_k,
                    device=device
                )
                print(f'生成: {text}')
            
        except KeyboardInterrupt:
            print('\n\n再见！')
            break
        except Exception as e:
            print(f'\n发生错误: {e}')


if __name__ == '__main__':
    # 选择计算设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    main()
