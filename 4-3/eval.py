"""
eval.py - 模型评估脚本

本文件用于加载训练好的策略网络模型，并在 CartPole 环境中运行评估。

与训练脚本 (train.py) 的区别：
1. 不进行参数更新（不需要优化器和损失函数）
2. 使用确定性策略（选择概率最大的动作），而不是随机采样
3. 使用 model.eval() 切换到评估模式
4. 设置更长的最大步数（2000步），以充分展示训练效果

评估的目的：
- 验证训练好的模型是否能有效控制倒立摆
- 在训练时我们使用随机采样动作（探索），评估时使用确定性策略（利用）
- 一个好的策略应该能让杆子长时间不倒（接近或达到最大步数限制）
"""

import torch                     # PyTorch 深度学习框架，用于加载模型和做推理
import gymnasium as gym          # OpenAI Gymnasium，提供 CartPole 环境

from model import PolicyNet      # 导入策略网络模型的类定义


# ======================== 配置参数 ========================

hidden_dim = 128                 # 隐藏层维度，必须与训练时使用的值一致
                                 # 否则加载模型权重时会因为维度不匹配而报错

model_weights = 'xx.pth'         # 训练好的模型权重文件路径
                                 # 需要替换为实际的文件名，例如 'pg_01.pth'
                                 # 这个文件由 train.py 训练结束后保存

# ======================== 环境初始化 ========================

# 创建 CartPole 环境
# render_mode="human"：开启图形化渲染，可以看到小车和杆子的实时动画
# max_episode_steps=2000：将最大步数从默认的 500 增加到 2000
#   训练时默认 500 步，评估时增加到 2000 步，可以更好地观察模型的长期控制能力
#   如果模型训练得好，杆子应该能在 2000 步内一直保持不倒
env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=2000)

# 获取状态空间和动作空间的维度
# state_dim = 4（小车位置、速度、杆角度、角速度）
state_dim = env.observation_space.shape[0]

# action_dim = 2（向左推、向右推）
action_dim = env.action_space.n

# ======================== 模型加载 ========================

# 创建策略网络实例
# 网络结构必须与训练时完全一致（相同的 state_dim, hidden_dim, action_dim）
model = PolicyNet(state_dim, hidden_dim, action_dim)

# 加载训练好的权重参数
# torch.load() 从 .pth 文件中反序列化参数字典
# model.load_state_dict() 将参数字典中的值赋给模型对应的参数
# 这里要求文件中的参数名和形状与当前模型完全匹配
model.load_state_dict(torch.load(model_weights))

# 将模型切换到评估模式
# 评估模式会影响某些层的行为：
#   - Dropout 层：训练时随机丢弃神经元，评估时不丢弃
#   - BatchNorm 层：训练时使用批次统计量，评估时使用全局统计量
# 虽然本项目的策略网络没有用到这些层，但这是一个良好的编程习惯
# 与之对应的是 model.train()，用于切换回训练模式
model.eval()

# ======================== 运行评估 ========================

# 重置环境，获取初始状态
# 返回 (observation, info)，我们只需要 observation
state, _ = env.reset()

# 初始化总奖励计数器
# 在 CartPole 中，每存活一步得 +1 分，所以总奖励就等于存活的总步数
tot_reward = 0

# 标志位，表示当前 episode 是否结束
finish = False

# 运行一个完整的 episode
while not finish:
    # 将状态转换为 PyTorch 张量
    # [state] 外面套一层列表增加 batch 维度：(4,) → (1, 4)
    # dtype=torch.float 指定数据类型为 32 位浮点数
    state = torch.tensor([state], dtype=torch.float)

    # 将状态输入策略网络，获得各动作的概率分布
    # probs 的形状为 (1, 2)，例如 tensor([[0.95, 0.05]])
    # 表示向左推的概率 95%，向右推的概率 5%
    probs = model(state)

    # === 动作选择策略 ===
    # 评估时有两种选择动作的方式：

    # 方式一：从概率分布中随机采样（与训练时相同）
    # 这种方式保留了一定的随机性，但评估结果可能每次不同
    # action_dist = torch.distributions.Categorical(probs)
    # action = action_dist.sample()

    # 方式二：选择概率最大的动作（贪心策略/确定性策略）
    # torch.argmax(probs) 返回概率最大的动作的索引
    # 例如 probs = [[0.95, 0.05]]，argmax 返回 0（向左推）
    # 这种方式是确定性的，每次对相同状态都会做出相同决策
    # 评估时通常使用这种方式，因为：
    #   1. 不需要探索了（训练阶段已经充分探索）
    #   2. 结果可复现
    #   3. 代表了策略学到的"最佳判断"
    action = torch.argmax(probs)

    # 在环境中执行选定的动作
    # action.item() 将张量转换为 Python 标量（int 类型）
    # env.step() 返回：新状态、即时奖励、是否终止、是否截断、额外信息
    state, reward, terminated, truncated, _ = env.step(action.item())

    # 判断 episode 是否结束
    finish = terminated or truncated

    # 累计总奖励
    tot_reward += reward

# ======================== 评估结束 ========================

# 关闭环境，释放资源
env.close()

# 打印最终的总奖励
# 总奖励 = 存活步数，越高越好
# 如果接近 2000（最大步数限制），说明模型训练得非常好
# 如果只有几十或几百，说明模型还需要更多训练
print(f'Total reward: {tot_reward}')
