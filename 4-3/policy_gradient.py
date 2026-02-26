"""
policy_gradient.py - 策略梯度（REINFORCE）算法实现

本文件实现了 REINFORCE 算法，这是最经典的策略梯度（Policy Gradient）方法。

=== 策略梯度算法的核心思想 ===

1. 什么是策略梯度？
   - 在强化学习中，我们希望找到一个"策略"（Policy），使得智能体在环境中获得的总奖励最大。
   - 策略 π(a|s; θ) 是一个参数化的概率分布：给定状态 s，输出选择动作 a 的概率。
   - 策略梯度方法直接对策略的参数 θ 进行优化，通过梯度上升来最大化期望回报。

2. REINFORCE 算法的更新公式：
   ∇θ J(θ) ≈ Σ_t [ ∇θ log π(a_t | s_t; θ) × G_t ]

   其中：
   - J(θ) 是目标函数（期望总回报），我们希望最大化它
   - π(a_t | s_t; θ) 是在状态 s_t 下选择动作 a_t 的概率
   - G_t 是从时刻 t 开始的折扣累计回报（return）
   - log π 是策略概率的对数

3. 直观理解：
   - 如果一个动作带来了较高的回报 G_t（G_t > 0），就增大选择这个动作的概率
   - 如果一个动作带来了较低的回报，就减小选择这个动作的概率
   - 回报越高，概率增大的幅度越大

4. 为什么用对数概率 log π？
   - 数学上：∇θ π = π × ∇θ log π，使用对数可以简化梯度计算
   - 数值上：概率值在 0~1 之间，取对数后范围更大，数值更稳定
   - 实际上：PyTorch 的自动微分可以直接对 log π 求梯度

=== 折扣因子 γ（gamma）===

折扣因子 γ ∈ [0, 1] 用于计算折扣累计回报：
    G_t = R_t + γ × R_{t+1} + γ² × R_{t+2} + ... + γ^(T-t) × R_T

- γ = 0：只关注即时奖励，完全不考虑未来
- γ = 1：同等看待当前和未来的奖励
- γ = 0.98（本项目使用）：更注重当前奖励，但也考虑未来奖励，未来越远影响越小
"""

import torch                      # PyTorch 深度学习框架

from model import PolicyNet       # 导入我们定义的策略网络模型


class PolicyGradient:
    """
    策略梯度（REINFORCE）算法的完整实现。

    该类封装了：
    1. 策略网络的创建和管理
    2. 基于策略网络的动作选择（探索与利用的平衡）
    3. 基于收集到的轨迹数据进行策略更新

    使用流程：
        1. 创建 PolicyGradient 对象
        2. 在环境中与智能体交互，调用 take_action() 选择动作
        3. 收集一批轨迹数据后，调用 update() 更新策略网络
        4. 重复步骤 2-3 直到策略收敛
    """

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device='cpu'):
        """
        初始化策略梯度算法的所有组件。

        参数：
            state_dim     (int):   状态空间维度。CartPole 中为 4。
            hidden_dim    (int):   策略网络隐藏层维度。本项目中为 128。
            action_dim    (int):   动作空间维度。CartPole 中为 2（左推/右推）。
            learning_rate (float): 学习率，控制每次参数更新的步长。
                                   值太大会导致训练不稳定，值太小会导致训练太慢。
                                   本项目中为 2e-3（即 0.002）。
            gamma         (float): 折扣因子，范围 [0, 1]，控制对未来奖励的重视程度。
                                   本项目中为 0.98。
            device        (str):   计算设备，'cpu' 或 'cuda'（GPU）。
                                   默认使用 CPU，如果有 GPU 可以加速训练。
        """
        # 创建策略网络，并将其放到指定设备上（CPU 或 GPU）
        # .to(device) 会将网络的所有参数移到指定设备的内存中
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)

        # 创建 Adam 优化器，用于更新策略网络的参数
        # Adam 是一种自适应学习率的优化算法，结合了 Momentum 和 RMSProp 的优点：
        #   - Momentum：积累历史梯度的动量，加速收敛
        #   - RMSProp：根据梯度的大小自适应调整学习率
        # self.policy_net.parameters() 返回网络中所有可训练的参数
        # lr=learning_rate 设置学习率
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)

        # 折扣因子 γ，用于计算折扣累计回报 G_t
        self.gamma = gamma

        # 保存设备信息，后续创建张量时需要放到同一设备上
        self.device = device

    def take_action(self, state):
        """
        根据当前状态，使用策略网络选择一个动作。

        这个方法体现了策略梯度的"随机策略"特性：
        - 策略网络输出每个动作的概率
        - 按照这个概率分布进行随机采样来选择动作
        - 而不是总是选择概率最大的动作（那样就是确定性策略了）

        为什么要随机采样而不是选最大概率的动作？
        - 探索（Exploration）：随机性使智能体有机会尝试不同的动作，发现更好的策略
        - 如果总是选概率最大的动作（贪心），可能会陷入局部最优
        - 随着训练的进行，好的动作概率会越来越高，探索会自然减少

        参数：
            state: 当前环境的状态，是一个长度为 state_dim 的数组/列表。
                   CartPole 中为 [小车位置, 小车速度, 杆角度, 杆角速度]。

        返回：
            int: 选择的动作编号。CartPole 中为 0（左推）或 1（右推）。
        """
        # 将状态转换为 PyTorch 张量
        # [state] 外面套一层列表，是为了增加一个 batch 维度
        # 例如 state = [0.1, 0.2, 0.3, 0.4]
        # 转换后 tensor 的形状为 (1, 4)，其中 1 是 batch_size，4 是 state_dim
        # 策略网络要求输入必须有 batch 维度
        state = torch.tensor([state], dtype=torch.float).to(self.device)

        # 将状态输入策略网络，得到各动作的概率分布
        # probs 的形状为 (1, action_dim)，例如 tensor([[0.6, 0.4]])
        # 表示选择动作 0 的概率为 0.6，选择动作 1 的概率为 0.4
        probs = self.policy_net(state)

        # 创建一个"分类分布"（Categorical Distribution）对象
        # 分类分布是离散概率分布的一种，适用于从多个类别中随机选择一个
        # 例如 probs = [0.6, 0.4]，那么有 60% 的概率采样到 0，40% 的概率采样到 1
        action_dist = torch.distributions.Categorical(probs)

        # 从分类分布中随机采样一个动作
        # sample() 会根据概率分布进行随机采样
        # 概率越高的动作越容易被采样到，但不保证一定选到概率最高的
        action = action_dist.sample()

        # .item() 将单元素张量转换为 Python 标量（int）
        # 例如 tensor(1) → 1
        return action.item()

    def update(self, tracj_list):
        """
        使用收集到的轨迹数据更新策略网络。

        这是 REINFORCE 算法的核心：
        1. 对于每条轨迹中的每个时间步，计算折扣累计回报 G_t
        2. 计算策略梯度：∇θ log π(a_t|s_t) × G_t
        3. 使用梯度下降更新网络参数

        注意：虽然理论上策略梯度是梯度上升（最大化回报），
        但在代码实现中，我们对损失函数取负号，然后用梯度下降来等价实现。
        即：最大化 J(θ) 等价于 最小化 -J(θ)

        参数：
            tracj_list (list): 一批轨迹数据的列表，每个元素是一个字典，包含：
                - 'states':  该轨迹中所有时间步的状态列表
                - 'actions': 该轨迹中所有时间步的动作列表
                - 'rewards': 该轨迹中所有时间步的即时奖励列表

                例如一条长度为 3 的轨迹：
                {
                    'states':  [s0, s1, s2],
                    'actions': [a0, a1, a2],
                    'rewards': [r0, r1, r2]
                }
        """
        # 累计所有轨迹的总损失
        # 使用多条轨迹的平均梯度可以减小梯度估计的方差，使训练更稳定
        tot_loss = 0

        for transition_dict in tracj_list:
            # 从轨迹字典中提取奖励、状态和动作序列
            reward_list = transition_dict['rewards']   # 每一步获得的即时奖励
            state_list = transition_dict['states']     # 每一步的环境状态
            action_list = transition_dict['actions']   # 每一步执行的动作

            # 初始化折扣累计回报 G
            # G_t 的递推公式：G_t = R_t + γ × G_{t+1}
            # 所以从后往前计算更高效（反向遍历）
            #
            # 举例（假设 γ=0.98，轨迹长度为 3，奖励序列为 [1, 1, 1]）：
            #   G_2 = R_2 = 1                           （最后一步）
            #   G_1 = R_1 + γ × G_2 = 1 + 0.98 × 1 = 1.98
            #   G_0 = R_0 + γ × G_1 = 1 + 0.98 × 1.98 = 2.9404
            G = 0

            # 在计算新的梯度之前，先清除优化器中之前累积的梯度
            # PyTorch 默认会累积梯度（这在某些场景下有用），
            # 但在这里我们需要每次用新的梯度进行更新
            self.optimizer.zero_grad()

            # 从最后一个时间步开始，向前遍历整条轨迹
            # reversed(range(len(reward_list))) 生成逆序索引：[T-1, T-2, ..., 1, 0]
            # 这样可以高效地用递推公式计算 G_t
            for i in reversed(range(len(reward_list))):
                # 取出第 i 个时间步的即时奖励
                reward = reward_list[i]

                # 将第 i 个时间步的状态转换为张量
                # 形状为 (1, state_dim)，增加 batch 维度以匹配网络输入要求
                state = torch.tensor([state_list[i]],
                                    dtype=torch.float).to(self.device)

                # 将第 i 个时间步的动作转换为张量
                # .view(-1, 1) 将其变形为列向量 (1, 1)，方便后面用 gather 函数索引
                action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)

                # 计算当前状态下选择该动作的对数概率 log π(a_t | s_t)
                #
                # 详细步骤：
                # 1. self.policy_net(state) 输出所有动作的概率，形状 (1, action_dim)
                #    例如：tensor([[0.6, 0.4]])
                # 2. .gather(1, action) 从概率张量中取出 action 对应的那个概率
                #    gather(dim, index)：沿着 dim 维度，按 index 取值
                #    如果 action = tensor([[1]])，则取出 index=1 的概率 0.4
                #    结果形状为 (1, 1)
                # 3. torch.log() 对概率取自然对数
                #    log(0.4) ≈ -0.916
                log_prob = torch.log(self.policy_net(state).gather(1, action))

                # 利用递推公式计算折扣累计回报 G_t
                # G_t = R_t + γ × G_{t+1}
                # 因为我们是从后往前遍历的，所以当前的 G 就是 G_{t+1}
                G = self.gamma * G + reward

                # 计算当前时间步的损失并累加到总损失中
                # 损失 = -log π(a_t | s_t) × G_t
                # 负号是因为：我们要最大化期望回报，而 PyTorch 的优化器默认做梯度下降（最小化）
                # 所以取负号，把"最大化"转换为"最小化"
                #
                # 直觉理解：
                # - 如果 G_t > 0（这个动作带来了正回报），-log_prob × G_t 是正的，
                #   梯度下降会增大 log_prob（即增大选择该动作的概率）
                # - G_t 越大，增大的幅度越大（更好的动作得到更多强化）
                tot_loss += -log_prob * G

        # 反向传播：计算总损失对网络所有参数的梯度
        # PyTorch 的自动微分引擎会沿着计算图从 tot_loss 一路反向传播，
        # 计算出每个参数的梯度 ∂tot_loss/∂θ
        tot_loss.backward()

        # 使用优化器根据计算出的梯度更新网络参数
        # θ_new = θ_old - learning_rate × ∇θ tot_loss
        # 由于 tot_loss = -Σ log_prob × G，梯度下降最小化 tot_loss
        # 等价于梯度上升最大化 Σ log_prob × G（即期望回报）
        self.optimizer.step()
