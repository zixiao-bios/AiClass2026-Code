import torch
import matplotlib.pyplot as plt


# ===================== 1. 构建人工数据集 =====================
# 假设真实的数据关系是 y = 3x + 2，并加入一些随机噪声

# 生成从-10到10的100个点
X = torch.linspace(-10, 10, 100)
# X.shape = (100)

X = X.unsqueeze(1)
# X.shape = (100, 1)

# 生成 y，加入随机噪声
noise = torch.randn(100, 1)
# noise.shape = (100, 1)
y = 3 * X + 2 + noise * 0
# y.shape = (100, 1)

# 画出示例数据
plt.scatter(X.numpy(), y.numpy())
plt.show()


# ===================== 2. 手动定义模型参数 =====================
# y = wx + b
# 随机初始化参数，并设置 requires_grad=True 以便自动求导
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f'初始参数: w = {w.item():.4f}, b = {b.item():.4f}')


# ===================== 3. 定义前向传播和损失函数 =====================
def forward(x):
    """前向传播: y = wx + b"""
    return x * w + b


def mse_loss(predictions, targets):
    """均方误差损失函数: MSE = mean((predictions - targets)^2)"""
    return ((predictions - targets) ** 2).mean()


# ===================== 4. 训练模型 =====================
learning_rate = 0.01
epochs = 1000  # 训练10个回合

for epoch in range(epochs):
    # 使用模型预测输出
    predictions = forward(X)
    
    # 计算损失
    loss = mse_loss(predictions, y)
    
    # 计算梯度
    loss.backward()
    
    # 手动更新参数 (SGD)
    # 使用 .data 直接修改底层数据，绕过自动求导系统
    w.data -= learning_rate * w.grad
    b.data -= learning_rate * b.grad
    
    # 梯度清零，防止累积
    w.grad.zero_()
    b.grad.zero_()

    # 打印训练过程中的损失值
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

print(f'\n训练后参数: w = {w.item():.4f}, b = {b.item():.4f}')
print(f'真实参数: w = 3, b = 2')


# ===================== 5. 查看结果 =====================
# 使用训练好的模型预测
predicted = forward(X)

# 原始数据
plt.scatter(X.numpy(), y.numpy(), label='Original Data')

# 预测数据
plt.plot(X.numpy(), predicted.detach().numpy(), color='red', label='Fitted Line')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title(f'Linear Regression (Scratch): y = {w.item():.2f}x + {b.item():.2f}')
plt.show()
