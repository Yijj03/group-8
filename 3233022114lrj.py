import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD, Adagrad, Adam, Adamax, RMSprop, Rprop
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取并预处理数据
df = pd.read_csv('train.csv')
x, y = df['x'].values, df['y'].values
mask = np.isfinite(x) & np.isfinite(y)
x, y = x[mask], y[mask]

# 数据标准化
x_normalized = (x - np.mean(x)) / np.std(x)
y_normalized = (y - np.mean(y)) / np.std(y)

x_tensor = torch.FloatTensor(x_normalized).view(-1, 1)
y_tensor = torch.FloatTensor(y_normalized).view(-1, 1)


# 定义线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.normal_(self.linear.weight, mean=0, std=1)
        nn.init.normal_(self.linear.bias, mean=0, std=1)

    def forward(self, x):
        return self.linear(x)


# 训练函数
def train_model(optimizer_class, optimizer_name, lr=0.01, epochs=1000):
    model = LinearModel()
    criterion = nn.MSELoss()
    optimizer = optimizer_class(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    final_w = model.linear.weight.item()
    final_b = model.linear.bias.item()

    print(f'{optimizer_name:8} | 最终参数: w={final_w:.4f}, b={final_b:.4f}, Loss: {losses[-1]:.6f}')

    return losses, final_w, final_b


# 主程序
if __name__ == "__main__":
    # 比较不同优化器
    optimizers = [(SGD, 'SGD'), (Adagrad, 'Adagrad'), (Adam, 'Adam'),
                  (Adamax, 'Adamax'), (RMSprop, 'RMSprop'), (Rprop, 'Rprop')]

    results = {}
    print("优化器比较结果:")
    for optimizer_class, optimizer_name in optimizers:
        losses, final_w, final_b = train_model(optimizer_class, optimizer_name)
        results[optimizer_name] = {'losses': losses, 'final_w': final_w, 'final_b': final_b}

    # 绘制损失曲线
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    for name, result in results.items():
        plt.plot(result['losses'], label=name, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('不同优化器的损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制最终损失对比
    plt.subplot(2, 2, 2)
    final_losses = [result['losses'][-1] for result in results.values()]
    plt.bar(results.keys(), final_losses, color='skyblue', alpha=0.7)
    plt.ylabel('Final Loss')
    plt.title('最终损失对比')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # 绘制拟合效果
    plt.subplot(2, 2, 3)
    plt.scatter(x, y, alpha=0.3, s=10, label='原始数据')

    best_optimizer = min(results.keys(), key=lambda x: results[x]['losses'][-1])
    best_result = results[best_optimizer]

    # 计算原始尺度参数
    w_original = best_result['final_w'] * (np.std(y) / np.std(x))
    b_original = np.mean(y) + best_result['final_b'] * np.std(y) - best_result['final_w'] * np.std(y) * np.mean(
        x) / np.std(x)

    x_plot = np.linspace(x.min(), x.max(), 100)
    y_plot = w_original * x_plot + b_original
    plt.plot(x_plot, y_plot, 'red', linewidth=2,
             label=f'最佳拟合 ({best_optimizer})\ny={w_original:.2f}x+{b_original:.2f}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('最佳拟合结果')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 参数变化轨迹
    plt.subplot(2, 2, 4)
    for name, result in list(results.items())[:4]:
        plt.plot(result['losses'][:100], label=name, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('前100轮训练损失')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n最佳优化器: {best_optimizer}")
    print(f"拟合方程: y = {w_original:.4f}x + {b_original:.4f}")