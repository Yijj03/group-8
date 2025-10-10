import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 基础设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
torch.manual_seed(42)  # 固定随机种子，保证结果可复现
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 1. 数据加载与预处理（移除 sklearn，手动实现归一化）
def load_and_preprocess_data(csv_path='train.csv'):
    """
    加载、清洗数据，并手动进行Z-score归一化。
    返回：原始清洗数据、归一化后的张量、以及归一化参数（用于后续反归一化）。
    """
    # 生成模拟数据（若文件不存在）
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("警告：未找到 'train.csv' 文件。将生成一组模拟数据用于演示。")
        x = np.linspace(0, 100, 1000)
        y = 2 * x + 3 + np.random.normal(0, 10, size=x.shape)
        df = pd.DataFrame({'x': x, 'y': y})
        df.to_csv(csv_path, index=False)
        print("已生成模拟数据 'train.csv'。")

    # 数据清洗
    df_clean = df.dropna().drop_duplicates()

    # IQR异常值处理
    def remove_outliers(df, col):
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    df_clean = remove_outliers(df_clean, 'x')
    df_clean = remove_outliers(df_clean, 'y')

    # MODIFICATION: 手动实现 Z-score 归一化 (x - mu) / sigma
    # 计算均值和标准差
    x = df_clean['x'].values.reshape(-1, 1)
    y = df_clean['y'].values.reshape(-1, 1)

    x_mean, x_std = np.mean(x), np.std(x)
    y_mean, y_std = np.mean(y), np.std(y)

    # 防止标准差为0导致除零错误
    x_std = 1.0 if x_std == 0 else x_std
    y_std = 1.0 if y_std == 0 else y_std

    # 归一化
    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std

    print("\n数据已进行手动 Z-score 归一化。")

    # 转换为张量
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(device)

    # 返回原始清洗数据、归一化张量和归一化参数
    return df_clean, x_tensor, y_tensor, (x_mean, x_std, y_mean, y_std)


# 2. 模型定义（正态分布初始化权重和偏置）
class LinearRegressionModel(nn.Module):
    def __init__(self, mean=0.0, std=0.1):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        # MODIFICATION 1: 正态分布初始化参数
        nn.init.normal_(self.linear.weight, mean=mean, std=std)  # 权重w正态分布
        nn.init.normal_(self.linear.bias, mean=mean, std=std)  # 偏置b正态分布

    def forward(self, x):
        return self.linear(x)


# 3. 通用训练函数（支持不同优化器、记录参数变化）
def train_model(x, y, optimizer_name, lr=0.01, epochs=2000):
    # 初始化模型
    model = LinearRegressionModel().to(device)
    criterion = nn.MSELoss()
    # 根据名称选择优化器
    optimizer_map = {
        'Adagrad': optim.Adagrad(model.parameters(), lr=lr),
        'Adam': optim.Adam(model.parameters(), lr=lr),
        'Adamax': optim.Adamax(model.parameters(), lr=lr),
        'SGD': optim.SGD(model.parameters(), lr=lr)
    }
    optimizer = optimizer_map[optimizer_name]

    # 记录训练过程：损失、w、b
    logs = {
        'loss': [],
        'w': [],
        'b': [],
        'lr': []
    }

    for epoch in range(epochs):
        # 前向传播
        y_pred = model(x)
        loss = criterion(y_pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录数据
        logs['loss'].append(loss.item())
        logs['w'].append(model.linear.weight.item())
        logs['b'].append(model.linear.bias.item())
        logs['lr'].append(optimizer.param_groups[0]['lr'])

    return model, logs


# 4. 可视化工具函数（统一风格）
def plot_metric_comparison(logs_dict, metric='loss', title='', ylabel=''):
    """对比不同优化器的指标（loss/w/b）"""
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'orange']
    for i, (opt_name, logs) in enumerate(logs_dict.items()):
        plt.plot(range(len(logs[metric])), logs[metric],
                 color=colors[i], label=opt_name, linewidth=2)
    plt.xlabel('训练轮数（Epoch）')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_param_trace(logs, param1='w', param2='b', title=''):
    """绘制参数（w/b）调节轨迹"""
    plt.figure(figsize=(8, 6))
    plt.plot(logs[param1], logs[param2], color='purple', linewidth=2, alpha=0.8)
    # 标记起点和终点
    plt.scatter(logs[param1][0], logs[param2][0], color='red', s=100, label='初始值', zorder=5)
    plt.scatter(logs[param1][-1], logs[param2][-1], color='green', s=100, label='最终值', zorder=5)
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# 5. 核心实验
if __name__ == "__main__":
    # 加载数据 (获取归一化参数)
    df_clean, x_tensor, y_tensor, (x_mean, x_std, y_mean, y_std) = load_and_preprocess_data()

    # 实验1：三种优化器性能对比（固定lr=0.01，epochs=2000）
    optimizers_to_test = ['Adam', 'Adagrad', 'SGD']
    logs_dict = {}  # 存储所有优化器的训练日志

    print("\n=== 开始对比三种优化器 ===")
    for opt in optimizers_to_test:
        print(f"\n训练优化器：{opt}")
        model, logs = train_model(x_tensor, y_tensor, optimizer_name=opt)
        logs_dict[opt] = logs
        print(f"最终损失：{logs['loss'][-1]:.6f}，最终w：{logs['w'][-1]:.4f}，最终b：{logs['b'][-1]:.4f}")

    # 可视化1：三种优化器的Loss变化对比
    plot_metric_comparison(
        logs_dict,
        metric='loss',
        title='三种优化器的Loss变化对比',
        ylabel='Loss（MSE）'
    )

    # 可视化2：三种优化器的w变化对比
    plot_metric_comparison(
        logs_dict,
        metric='w',
        title='三种优化器的权重w变化对比',
        ylabel='权重w'
    )

    # 可视化3：三种优化器的b变化对比
    plot_metric_comparison(
        logs_dict,
        metric='b',
        title='三种优化器的偏置b变化对比',
        ylabel='偏置b'
    )

    # 实验2：参数（w/b）调节轨迹（以性能最好的Adam为例）
    best_opt = 'Adam'  # 可根据实验结果替换为实际最好的优化器
    best_logs = logs_dict[best_opt]

    # 可视化4：w和b的调节轨迹
    plot_param_trace(
        best_logs,
        param1='w',
        param2='b',
        title=f'{best_opt}优化器的参数（w-b）调节轨迹'
    )

    # 实验3：学习率（lr）调节可视化
    lr_values = [0.001, 0.01, 0.1]
    lr_logs_dict = {}

    print("\n=== 开始对比不同学习率 ===")
    for lr in lr_values:
        print(f"\n学习率：{lr}")
        model, logs = train_model(x_tensor, y_tensor, optimizer_name='Adam', lr=lr)
        lr_logs_dict[f'lr={lr}'] = logs

    # 可视化5：不同学习率的Loss变化
    plot_metric_comparison(
        lr_logs_dict,
        metric='loss',
        title='不同学习率的Loss变化对比（Adam优化器）',
        ylabel='Loss（MSE）'
    )

    # 实验4：迭代次数（epochs）调节可视化
    epoch_values = [500, 1000, 2000]
    epoch_logs_dict = {}

    print("\n=== 开始对比不同迭代次数 ===")
    for epochs in epoch_values:
        print(f"\n迭代次数：{epochs}")
        model, logs = train_model(x_tensor, y_tensor, optimizer_name='Adam', epochs=epochs)
        epoch_logs_dict[f'epochs={epochs}'] = logs

    # 可视化6：不同迭代次数的Loss变化
    plot_metric_comparison(
        epoch_logs_dict,
        metric='loss',
        title='不同迭代次数的Loss变化对比（Adam优化器）',
        ylabel='Loss（MSE）'
    )

    # 保存性能最好的模型及其归一化参数
    best_model = train_model(x_tensor, y_tensor, optimizer_name='Adam')[0]
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,
    }, 'best_linear_regression_model.pth')
    print("\n性能最好的模型及手动归一化参数已保存为：best_linear_regression_model.pth")