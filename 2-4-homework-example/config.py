"""
GPT 模型配置文件

包含模型结构参数、训练超参数和数据处理参数
"""

# ==================== 模型结构参数 ====================
d_model = 256        # 模型的隐藏维度（嵌入维度）
n_head = 8           # 多头注意力的头数
max_len = 128        # 最大序列长度（需要 >= window_size + 2）
ffn_hidden = 1024    # FFN 隐藏层维度（通常是 d_model 的 4 倍）
n_blocks = 4         # GPT Block 的数量
drop_prob = 0.1      # Dropout 概率

# ==================== 训练参数 ====================
batch_size = 16      # 批次大小
lr = 3e-4            # 学习率（GPT 常用的学习率）
epochs = 20          # 训练轮数
log_interval = 1     # 每隔多少个 epoch 打印一次日志

# ==================== 数据处理参数 ====================
min_sample_length = 64  # 最小样本长度（不足时继续加入下一句）
sentence_stride = 1     # 滑动步长（每次滑动几个句子，1表示最大重叠）
val_ratio = 0.1         # 验证集比例
min_line_length = 10    # 最小行长度（低于此长度的行会被过滤）

# ==================== 文件路径 ====================
data_dir = '2-4/data'
train_file = f'{data_dir}/train.txt'
val_file = f'{data_dir}/val.txt'
vocab_file = f'{data_dir}/vocab.json'
model_file = '2-4/gpt.pth'
tensorboard_dir = '2-4/runs'
