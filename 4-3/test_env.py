"""
test_env.py - CartPole 环境测试与探索脚本

本文件用于帮助学生了解和熟悉 Gymnasium（OpenAI Gym 的继任者）提供的 CartPole 环境。

通过运行本脚本，你可以了解到：
1. 如何创建和初始化一个强化学习环境
2. 环境的状态（observation）是什么样的
3. 环境的动作空间（action space）有哪些可选动作
4. 如何与环境进行交互（执行动作、获取反馈）
5. 一个完整的 episode（回合）是如何运行的

本脚本使用"随机策略"来控制小车（即随机选择动作），
因此杆子很快就会倒下（通常只能存活 10~50 步）。
这正好说明了为什么我们需要强化学习来训练一个更好的策略！

=== Gymnasium 框架简介 ===

Gymnasium 是一个标准化的强化学习环境接口库，它提供了：
- 统一的 API：所有环境都使用相同的 reset()、step() 接口
- 丰富的环境：包括经典控制、Atari 游戏、机器人仿真等
- 易于使用：几行代码就能创建并运行一个环境

核心交互流程（强化学习的基本循环）：
    env.reset()     →  获得初始状态 s_0
    env.step(a_0)   →  执行动作 a_0，获得 (s_1, r_0, terminated, truncated, info)
    env.step(a_1)   →  执行动作 a_1，获得 (s_2, r_1, terminated, truncated, info)
    ...
    直到 terminated 或 truncated 为 True，一个 episode 结束
"""

import gymnasium as gym    # 导入 Gymnasium 库，它提供了各种强化学习标准环境


# ======================== 第一部分：创建环境 ========================

# 使用 gym.make() 创建 CartPole-v1 环境
# "CartPole-v1" 是环境的 ID，Gymnasium 会根据这个 ID 找到对应的环境类并实例化
# render_mode="human" 开启人类可视化模式，会弹出一个窗口显示小车和杆子的实时动画
# 其他可选的 render_mode：
#   - "rgb_array"：返回 RGB 像素数组（用于录制视频）
#   - None：不渲染（训练时推荐，速度最快）
env = gym.make("CartPole-v1", render_mode="human")

# ======================== 第二部分：探索环境属性 ========================

# 重置环境到初始状态
# env.reset() 返回一个元组 (observation, info)：
#   - observation (numpy.ndarray): 初始状态向量，CartPole 中是一个长度为 4 的数组
#   - info (dict): 环境的额外信息（通常在 CartPole 中为空字典）
# 这里用 _ 忽略 info
obs, _ = env.reset()

# 查看观测值（状态）的数据类型
# 输出：<class 'numpy.ndarray'>
# 状态是一个 numpy 数组，可以直接转换为 PyTorch 张量
print(type(obs))

# 查看观测值的形状
# 输出：(4,)
# 这是一个一维数组，包含 4 个浮点数
print(obs.shape)

# 打印具体的观测值
# 输出示例：[ 0.0273956  -0.00611216  0.03585979  0.0197368 ]
# 四个数值分别代表：
#   obs[0]: 小车位置（0 表示在轨道中心）
#   obs[1]: 小车速度（正值表示向右移动）
#   obs[2]: 杆的角度（0 表示竖直，正值表示向右倾斜，单位：弧度）
#   obs[3]: 杆的角速度（正值表示向右转动）
# 初始状态的各值都在 [-0.05, 0.05] 范围内随机生成
print(obs)

# ======================== 第三部分：探索动作空间 ========================

# 查看动作空间的类型和大小
# 输出：Discrete(2)
# Discrete(2) 表示离散动作空间，有 2 个可选动作：0 和 1
#   动作 0：向左施加力（推小车向左）
#   动作 1：向右施加力（推小车向右）
print(env.action_space)

# 从动作空间中随机采样一个动作
# env.action_space.sample() 均匀随机地返回 0 或 1
# 这就是"随机策略"——不考虑当前状态，随便选一个动作
action = env.action_space.sample()

# 打印采样到的动作
# 输出：0 或 1
print(action)

# ======================== 第四部分：随机策略运行一个完整 episode ========================

# 累计奖励，用于衡量这个 episode 的表现
# 在 CartPole 中，每一步存活就得 +1 分，所以总奖励 = 存活步数
tot_reward = 0

# 标志位：当前 episode 是否已结束
finished = False

# 运行一个完整的 episode（从初始状态到终止状态的完整过程）
while not finished:
    # 随机选择一个动作（这就是"随机策略"）
    # 随机策略是最简单的策略，不使用任何学习到的知识
    # 它的表现通常很差，在 CartPole 中平均只能存活约 20~30 步
    action = env.action_space.sample()

    # 在环境中执行该动作，获得环境的反馈
    # env.step(action) 返回一个包含 5 个元素的元组：
    #
    #   obs (numpy.ndarray):  执行动作后的新状态（4维向量）
    #   reward (float):       获得的即时奖励（CartPole 中每步存活 +1.0）
    #   terminated (bool):    是否因为"任务失败"而终止
    #                         True 的情况：杆的角度 > ±12° 或 小车位置 > ±2.4
    #   truncated (bool):     是否因为"达到步数上限"而截断
    #                         True 的情况：已经执行了 500 步（CartPole-v1 的默认上限）
    #   info (dict):          额外信息，这里用 _ 忽略
    #
    # terminated 和 truncated 的区别非常重要：
    #   - terminated=True 表示"自然终止"——环境因为违反约束条件而结束（失败了）
    #   - truncated=True 表示"人为截断"——环境因为达到时间限制而结束（可能还没失败）
    obs, reward, terminated, truncated, _ = env.step(action)

    # 累加即时奖励
    tot_reward += reward

    # 当 terminated 或 truncated 任一为 True 时，episode 结束
    finished = terminated or truncated

# ======================== 第五部分：输出结果 ========================

# 打印终止原因
# terminated=True, truncated=False：杆倒了或小车出界（失败终止）
# terminated=False, truncated=True：达到 500 步上限（时间截断，表示成功撑过了全程！）
print(f"terminated: {terminated}, truncated: {truncated}")

# 打印总奖励（即存活的总步数）
# 随机策略通常只能获得 10~50 的奖励
# 训练好的策略可以获得 500（满分，即撑满了整个 episode）
print(f"reward: {tot_reward}")

# 关闭环境，释放资源（关闭渲染窗口、释放内存等）
env.close()
