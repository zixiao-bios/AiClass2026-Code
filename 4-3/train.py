"""
train.py - 策略梯度算法的训练脚本

本文件是整个项目的训练入口，负责：
1. 设置超参数（学习率、折扣因子、训练轮次等）
2. 创建 CartPole 环境和策略梯度智能体
3. 运行训练循环：与环境交互 → 收集轨迹 → 更新策略
4. 记录训练过程（使用 TensorBoard 可视化）
5. 保存训练好的模型

=== CartPole 环境介绍 ===

CartPole（倒立摆）是强化学习中最经典的入门环境之一：
- 一根杆子通过铰链连接在一辆小车上
- 小车可以在一维轨道上左右移动
- 目标：通过左右推小车，让杆子尽可能长时间保持竖直不倒

状态空间（4维连续）：
    [0] 小车位置    范围：(-4.8, 4.8)
    [1] 小车速度    范围：(-∞, +∞)
    [2] 杆的角度    范围：(-0.418 rad, 0.418 rad)，约 ±24°
    [3] 杆的角速度  范围：(-∞, +∞)

动作空间（2个离散动作）：
    0: 向左推小车
    1: 向右推小车

奖励：每个时间步存活就得 +1 分

终止条件：
    - 杆的角度超过 ±12°（terminated）
    - 小车位置超出 ±2.4（terminated）
    - 达到最大步数 500 步（truncated，即被截断）

目标：获得尽可能高的总奖励（最高 500）

=== 训练策略说明 ===

本训练脚本使用了"多轨迹批量更新"策略：
- 每次收集 N 条轨迹（N=5）后，一起更新一次策略网络
- 这样做的好处是：用多条轨迹的平均梯度来代替单条轨迹的梯度，
  可以减小梯度估计的方差，使训练更加稳定
"""

import tensorboardX              # TensorBoard 数据记录库，用于记录训练指标并可视化
import gymnasium as gym          # OpenAI Gymnasium，提供标准化的强化学习环境接口
import torch                     # PyTorch 深度学习框架，这里用于保存模型

from policy_gradient import PolicyGradient  # 导入我们实现的策略梯度算法

# ======================== 超参数设置 ========================
# 超参数是训练前需要人为设定的参数，它们会显著影响训练效果

run_name = 'pg_01'        # 本次训练的名称，用于区分不同的实验
                           # 模型保存和 TensorBoard 日志都会用到这个名称

learning_rate = 2e-3       # 学习率（0.002）
                           # 控制每次参数更新的步长大小
                           # 太大：训练不稳定，loss 震荡甚至发散
                           # 太小：训练速度慢，可能陷入局部最优
                           # 2e-3 是一个对 CartPole 比较合适的经验值

num_episodes = 2000        # 总训练轮次（episode 数量）
                           # 一个 episode 就是从环境重置到终止的一次完整交互
                           # CartPole 通常 1000~2000 个 episode 就能收敛

hidden_dim = 128           # 策略网络隐藏层的神经元数量
                           # 控制网络的容量（表达能力）
                           # CartPole 问题较简单，128 已足够

gamma = 0.98               # 折扣因子 γ
                           # 计算折扣累计回报时对未来奖励的衰减系数
                           # 0.98 意味着 50 步后的奖励权重衰减为 0.98^50 ≈ 0.36

N = 5                      # 每次策略更新时使用的轨迹数量
                           # 每收集 N 条轨迹后进行一次策略网络更新
                           # N 越大，梯度估计越准确，但每次更新需要更多交互数据

# ======================== 环境与智能体初始化 ========================

# 创建 CartPole-v1 环境
# render_mode="human" 表示开启图形化渲染窗口，可以实时看到小车和杆子的动画
# 注意：开启渲染会显著拖慢训练速度，正式训练时可以去掉这个参数
env = gym.make("CartPole-v1", render_mode="human")

# 获取状态空间和动作空间的维度
# env.observation_space.shape 返回状态的形状，CartPole 中为 (4,)
# env.observation_space.shape[0] 取第一个维度，得到 4
state_dim = env.observation_space.shape[0]

# env.action_space.n 返回离散动作的数量，CartPole 中为 2
action_dim = env.action_space.n

# 创建策略梯度智能体
# 传入状态维度、隐藏层维度、动作维度、学习率和折扣因子
agent = PolicyGradient(state_dim, hidden_dim, action_dim, learning_rate, gamma)

# 创建 TensorBoard 日志记录器
# 训练数据会保存到 runs/pg_01/ 目录下
# 之后可以在终端运行 `tensorboard --logdir runs` 启动 TensorBoard 查看训练曲线
summary_writer = tensorboardX.SummaryWriter(f'runs/{run_name}')

# ======================== 训练主循环 ========================
# 总共训练 num_episodes 个 episode，每 N 个 episode 为一组进行更新
# 所以外层循环次数为 num_episodes / N = 2000 / 5 = 400 次

for i in range(int(num_episodes / N)):
    # data_N 用于存储当前这一组（N 条）轨迹的数据
    data_N = []

    # episode_reward 用于累计当前这一组所有 episode 的总奖励
    # 最后除以 N 得到平均奖励，用于监控训练进度
    episode_reward = 0

    # 内层循环：收集 N 条轨迹
    for j in range(N):
        # 创建一个字典来存储单条轨迹的数据
        # 每条轨迹由一系列 (状态, 动作, 奖励) 三元组组成
        data = {
            'states': [],     # 存储每一步的状态
            'actions': [],    # 存储每一步选择的动作
            'rewards': [],    # 存储每一步获得的即时奖励
        }

        # 重置环境，获得初始状态 s_0
        # env.reset() 返回 (observation, info)，我们只需要 observation
        # info 包含一些额外信息（如随机种子），这里用 _ 忽略
        state, _ = env.reset()

        # finish 标志位，表示当前 episode 是否结束
        finish = False

        # 运行一个完整的 episode（从初始状态到终止状态）
        while not finish:
            # 第一步：使用策略网络选择动作
            # agent.take_action(state) 会将 state 输入策略网络，
            # 得到动作概率分布，然后从中随机采样一个动作
            action = agent.take_action(state)

            # 第二步：在环境中执行该动作，获得反馈
            # env.step(action) 返回 5 个值：
            #   next_state:  执行动作后的新状态
            #   reward:      获得的即时奖励（CartPole 中每步存活得 +1）
            #   terminated:  是否因为"失败"而终止（杆倒了/小车出界）
            #   truncated:   是否因为"达到最大步数"而截断（达到 500 步）
            #   info:        额外信息（这里不需要，用 _ 忽略）
            next_state, reward, terminated, truncated, _ = env.step(action)

            # 第三步：将当前时间步的数据记录到轨迹中
            # 注意：记录的是当前状态 state（不是 next_state），
            # 因为我们需要的是"在状态 state 下执行动作 action 获得奖励 reward"这个三元组
            data['states'].append(state)
            data['actions'].append(action)
            data['rewards'].append(reward)

            # 更新状态：将当前状态设为下一步的状态
            state = next_state

            # 累计本次 episode 的奖励
            episode_reward += reward

            # 判断 episode 是否结束
            # terminated=True：环境判定失败（杆倒了或小车出界）
            # truncated=True：达到最大步数限制（500步）
            finish = terminated or truncated

        # 将这条完整轨迹添加到当前批次中
        data_N.append(data)

    # 收集完 N 条轨迹后，使用这批数据更新策略网络
    # 这里是 REINFORCE 算法的核心：根据轨迹数据计算策略梯度，然后更新网络参数
    agent.update(data_N)

    # 打印训练进度：当前 episode 编号和平均奖励
    # (i + 1) * N 是当前已完成的总 episode 数
    # episode_reward / N 是最近 N 个 episode 的平均奖励
    # 平均奖励接近 500 说明策略已经很好了（因为 CartPole 最多 500 步）
    print(f'Episode {(i + 1) * N}, reward: {episode_reward / N}')

    # 将平均奖励写入 TensorBoard 日志
    # 第一个参数 'reward' 是指标名称
    # 第二个参数是指标值（平均奖励）
    # 第三个参数 i 是横轴坐标（第几次更新）
    summary_writer.add_scalar('reward', episode_reward / N, i)

# ======================== 训练结束，保存与清理 ========================

# 保存策略网络的权重参数到文件
# .state_dict() 返回模型所有参数的字典
# torch.save() 将参数字典序列化保存到 .pth 文件中
# 之后可以用 model.load_state_dict(torch.load('pg_01.pth')) 加载
torch.save(agent.policy_net.state_dict(), f'{run_name}.pth')

# 关闭 TensorBoard 日志记录器，确保数据完整写入磁盘
summary_writer.close()

# 关闭环境，释放资源（关闭渲染窗口等）
env.close()
