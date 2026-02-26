"""
model.py - 策略网络模型定义

本文件定义了策略梯度算法中使用的策略网络（Policy Network）。
策略网络是一个神经网络，它的作用是：
    输入：当前环境的状态（state）
    输出：在该状态下执行每个动作的概率分布

在 CartPole 问题中：
    - 状态（state）是一个 4 维向量，包含：小车位置、小车速度、杆的角度、杆的角速度
    - 动作（action）有 2 个：向左推（0）、向右推（1）
    - 所以网络输入维度为 4，输出维度为 2（每个动作对应一个概率值）

策略网络的核心思想：
    强化学习中的"策略"（Policy）就是一个从状态到动作的映射。
    我们用神经网络来参数化这个策略，即 π(a|s; θ)，
    其中 θ 是神经网络的参数，s 是状态，a 是动作。
    网络输出的是一个概率分布，表示在状态 s 下选择各个动作的概率。
"""

import torch                       # PyTorch 深度学习框架，用于构建和训练神经网络
import torch.nn.functional as F    # PyTorch 的函数式接口，提供激活函数、损失函数等


class PolicyNet(torch.nn.Module):
    """
    策略网络（Policy Network）

    这是一个简单的两层全连接神经网络（也叫多层感知机 MLP），结构如下：

        输入层 (state_dim)  →  隐藏层 (hidden_dim)  →  输出层 (action_dim)
              [4维]        ReLU激活     [128维]     Softmax激活   [2维]

    - 输入层：接收环境状态向量（维度 = state_dim）
    - 隐藏层：对输入进行非线性变换，提取特征（维度 = hidden_dim）
    - 输出层：输出每个动作的概率（维度 = action_dim），概率之和为 1

    继承自 torch.nn.Module：
        这是 PyTorch 中所有神经网络模型的基类。
        继承它可以让我们方便地管理网络参数、进行前向传播、保存/加载模型等。
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        """
        初始化策略网络的各层。

        参数：
            state_dim  (int): 状态空间的维度，即输入层的神经元数量。
                              在 CartPole 中为 4（小车位置、小车速度、杆角度、杆角速度）。
            hidden_dim (int): 隐藏层的神经元数量，控制网络的表达能力。
                              值越大，网络能学习到越复杂的模式，但也更容易过拟合、训练更慢。
                              本项目中设为 128。
            action_dim (int): 动作空间的维度，即输出层的神经元数量。
                              在 CartPole 中为 2（向左推、向右推）。
        """
        # 调用父类 torch.nn.Module 的初始化方法，这是 PyTorch 的固定写法
        super(PolicyNet, self).__init__()

        # 第一层全连接层（也叫线性层）：将状态向量映射到隐藏层
        # Linear(in_features, out_features) 会创建一个 y = xW^T + b 的线性变换
        # 其中 W 的形状为 (hidden_dim, state_dim)，b 的形状为 (hidden_dim,)
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)

        # 第二层全连接层：将隐藏层的特征映射到动作空间
        # 输出维度等于动作数量，每个输出值对应一个动作的"得分"（logit）
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        前向传播：定义数据在网络中的流动方式。

        参数：
            x (Tensor): 输入的状态张量，形状为 (batch_size, state_dim)。
                        batch_size 是批次大小（一次处理多少个状态），
                        state_dim 是每个状态的维度（CartPole 中为 4）。

        返回：
            Tensor: 动作概率分布，形状为 (batch_size, action_dim)。
                    每一行表示一个状态下各动作的概率，且概率之和为 1。

        数据流动过程：
            1. 输入 x (形状: batch_size × 4)
            2. 经过第一层全连接层 fc1，得到 (batch_size × 128) 的张量
            3. 经过 ReLU 激活函数，将负值置为 0，保留正值（引入非线性）
            4. 经过第二层全连接层 fc2，得到 (batch_size × 2) 的张量（原始得分/logits）
            5. 经过 Softmax 函数，将得分转换为概率分布（所有值在 0~1 之间，且和为 1）
        """
        # 第一步：通过第一层全连接层，然后用 ReLU 激活函数
        # ReLU(x) = max(0, x)，它的作用是引入非线性，使得网络能够学习非线性的映射关系
        # 如果没有激活函数，多层线性层叠加后仍然是一个线性变换，无法拟合复杂的策略
        x = F.relu(self.fc1(x))

        # 第二步：通过第二层全连接层，得到每个动作的原始得分（logits）
        # 然后用 Softmax 函数将得分转换为概率分布
        # Softmax 公式：softmax(x_i) = e^(x_i) / Σ(e^(x_j))
        # dim=1 表示在第 1 个维度（动作维度）上做 softmax，
        # 这样每个样本（状态）的动作概率之和为 1
        return F.softmax(self.fc2(x), dim=1)
