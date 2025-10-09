import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler


# Windows系统专用字体配置（确保中文显示正常）
def configure_windows_fonts():
    # Windows系统自带中文字体路径（100%存在）
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/dengxian.ttf"  # 等线
    ]

    # 加载第一个可用的字体
    for font_path in font_paths:
        if os.path.exists(font_path):
            font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
            matplotlib.rcParams["font.family"] = font_prop.get_name()
            print(f"已加载中文字体: {font_prop.get_name()}")
            break
    else:
        print("警告: 未找到系统中文字体，可能显示异常")

    matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# Windows系统专用后端设置
def configure_windows_backend():
    # 优先使用TkAgg后端（Windows兼容性好）
    try:
        matplotlib.use('TkAgg', force=True)
        print("已启用TkAgg后端（Windows专用）")
    except Exception as e:
        print(f"TkAgg后端启用失败，使用默认后端: {e}")


# 初始化Windows环境配置
configure_windows_fonts()
configure_windows_backend()


def load_data(file_path):
    """读取CSV数据，处理缺失值并归一化"""
    try:
        df = pd.read_csv(file_path)
        df = df.dropna()  # 移除缺失值
        print(f"成功读取数据，共{len(df)}条记录（已处理缺失值）")

        # 提取特征和目标值
        # 假设数据集中有特征列和目标列，这里使用前n-1列作为特征，最后一列作为目标
        if len(df.columns) >= 2:
            x = df.iloc[:, :-1].values  # 所有行，除了最后一列
            y = df.iloc[:, -1].values.reshape(-1, 1)  # 最后一列作为目标
        else:
            raise ValueError("数据集至少需要包含两列（特征和目标）")

        x_original = x.copy()  # 保存原始数据用于绘图
        y_original = y.copy()

        # 数据归一化（解决数值溢出问题）
        scaler_x = MinMaxScaler()
        x_scaled = scaler_x.fit_transform(x)
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y)

        # 转换为PyTorch张量
        x_tensor = torch.FloatTensor(x_scaled)
        y_tensor = torch.FloatTensor(y_scaled)

        return x_tensor, y_tensor, x_original, y_original, scaler_x, scaler_y
    except Exception as e:
        print(f"数据读取错误: {e}")
        raise


# 定义线性回归模型，使用正态分布初始化参数
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim=1):
        super(LinearRegressionModel, self).__init__()
        # 定义一个线性层，输入维度根据数据自动确定，输出维度1
        self.linear = nn.Linear(input_dim, 1)

        # 使用正态分布初始化权重和偏置
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.linear.bias, mean=0.0, std=0.1)

    def forward(self, x):
        return self.linear(x)


def train_model(x, y, optimizer_name, learning_rate=0.01, epochs=1000):
    """使用指定优化器训练线性回归模型"""
    # 获取输入维度
    input_dim = x.shape[1]

    # 初始化模型、损失函数和优化器
    model = LinearRegressionModel(input_dim)
    criterion = nn.MSELoss()  # 均方误差损失

    # 根据名称选择优化器
    optimizers = {
        'Adagrad': optim.Adagrad(model.parameters(), lr=learning_rate),
        'Adam': optim.Adam(model.parameters(), lr=learning_rate),
        'Adamax': optim.Adamax(model.parameters(), lr=learning_rate),
        'ASGD': optim.ASGD(model.parameters(), lr=learning_rate),
        'LBFGS': optim.LBFGS(model.parameters(), lr=learning_rate),
        'RMSprop': optim.RMSprop(model.parameters(), lr=learning_rate),
        'Rprop': optim.Rprop(model.parameters(), lr=learning_rate),
        'SGD': optim.SGD(model.parameters(), lr=learning_rate)
    }

    if optimizer_name not in optimizers:
        raise ValueError(f"未知的优化器: {optimizer_name}")

    optimizer = optimizers[optimizer_name]

    losses = []
    weights = []  # 存储权重w的变化
    biases = []  # 存储偏置b的变化

    for i in range(epochs):
        # 对于LBFGS优化器，需要特殊处理
        if optimizer_name == 'LBFGS':
            def closure():
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                return loss

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.step(closure)
        else:
            # 前向传播：计算预测值
            y_pred = model(x)

            # 计算损失
            loss = criterion(y_pred, y)

            # 反向传播和优化
            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数

        # 记录损失和参数
        losses.append(loss.item())
        w = model.linear.weight.detach().numpy()
        weights.append(w.flatten())  # 存储所有权重
        b = model.linear.bias.item()
        biases.append(b)

        # 每100轮打印一次
        if i % 100 == 0:
            print(f"{optimizer_name} 迭代 {i:4d} | 损失: {loss.item():.6f} | w: {w.flatten()} | b: {b:.6f}")

    # 获取最终参数
    final_w = model.linear.weight.detach().numpy()
    final_b = model.linear.bias.item()
    print(f"{optimizer_name} 训练完成! 最终损失: {losses[-1]:.6f}, 最优w: {final_w.flatten()}, 最优b: {final_b:.6f}")

    return model, final_w, final_b, losses, weights, biases


def plot_optimizer_comparison(optimizer_results):
    """比较不同优化器的性能"""
    plt.figure(figsize=(12, 6))

    for name, result in optimizer_results.items():
        plt.plot(range(len(result['losses'])), result['losses'], label=name, linewidth=2, alpha=0.7)

    plt.title('不同优化器的损失变化曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.yscale('log')  # 使用对数尺度更好地观察差异
    plt.grid(True, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_parameter_changes(weights, biases, optimizer_name):
    """绘制权重w和偏置b随迭代次数的变化"""
    # 绘制权重变化
    plt.figure(figsize=(12, 6))
    weights_np = np.array(weights)

    for i in range(weights_np.shape[1]):
        plt.plot(range(len(weights)), weights_np[:, i], label=f'w{i + 1}', linewidth=2, alpha=0.7)

    plt.title(f'{optimizer_name}优化器权重参数变化')
    plt.xlabel('迭代次数')
    plt.ylabel('权重值')
    plt.grid(True, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 绘制偏置变化
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(biases)), biases, 'r-', linewidth=2, alpha=0.7)
    plt.title(f'{optimizer_name}优化器偏置参数b变化')
    plt.xlabel('迭代次数')
    plt.ylabel('偏置值')
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_epoch_learning_rate_effects(x, y, base_optimizer='Adam'):
    """绘制不同epoch和学习率对模型性能的影响"""
    # 测试不同学习率
    learning_rates = [0.001, 0.01, 0.1, 0.2]
    plt.figure(figsize=(12, 6))

    for lr in learning_rates:
        _, _, _, losses, _, _ = train_model(x, y, base_optimizer, learning_rate=lr, epochs=500)
        plt.plot(range(len(losses)), losses, label=f'学习率={lr}', linewidth=2, alpha=0.7)

    plt.title(f'{base_optimizer}优化器不同学习率的损失变化')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.yscale('log')
    plt.grid(True, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 测试不同迭代次数
    epochs_list = [100, 300, 500, 1000]
    final_losses = []

    for epochs in epochs_list:
        _, _, _, losses, _, _ = train_model(x, y, base_optimizer, learning_rate=0.01, epochs=epochs)
        final_losses.append(losses[-1])

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(epochs_list)), final_losses, color='skyblue')
    plt.xticks(range(len(epochs_list)), epochs_list)
    plt.title(f'{base_optimizer}优化器不同迭代次数的最终损失')
    plt.xlabel('迭代次数')
    plt.ylabel('最终损失值')
    plt.grid(axis='y', alpha=0.7)

    # 在柱状图上添加数值标签
    for i, v in enumerate(final_losses):
        plt.text(i, v, f'{v:.6f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_regression_line(x_original, y_original, model, scaler_x, scaler_y, optimizer_name):
    """绘制原始数据与回归直线"""
    # 转换为PyTorch张量并进行预测
    x_scaled = torch.FloatTensor(scaler_x.transform(x_original))
    with torch.no_grad():  # 不需要计算梯度
        y_pred_scaled = model(x_scaled)

    # 转换回原始尺度
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled.numpy())

    # 获取模型参数用于显示
    w = model.linear.weight.detach().numpy().flatten()
    b = model.linear.bias.item()

    plt.figure(figsize=(10, 6))
    plt.scatter(x_original[:, 0], y_original, alpha=0.6, label='数据点', edgecolors='k', s=30)
    plt.plot(x_original[:, 0], y_pred_original, 'r-', linewidth=2,
             label=f'回归直线: y = {w[0]:.4f}x + {b:.4f}')
    plt.title(f'{optimizer_name}优化器的回归结果')
    plt.xlabel('x值')
    plt.ylabel('y值')
    plt.grid(True, alpha=0.7)
    plt.legend()
    plt.show()


def main():
    # 读取数据（确保train.csv在当前目录）
    x_tensor, y_tensor, x_original, y_original, scaler_x, scaler_y = load_data('train.csv')

    # 选择三种不同的优化器进行比较
    selected_optimizers = ['SGD', 'Adam', 'RMSprop']
    optimizer_results = {}

    # 训练并收集每个优化器的结果
    for optimizer in selected_optimizers:
        print(f"\n===== 开始使用{optimizer}优化器训练 =====")
        model, w, b, losses, weights, biases = train_model(
            x_tensor, y_tensor, optimizer, learning_rate=0.01, epochs=1000)

        optimizer_results[optimizer] = {
            'model': model,
            'w': w,
            'b': b,
            'losses': losses,
            'weights': weights,
            'biases': biases
        }

        # 绘制该优化器的参数变化
        plot_parameter_changes(weights, biases, optimizer)

        # 绘制回归直线
        if x_original.shape[1] == 1:  # 仅对单特征数据绘制回归直线
            plot_regression_line(x_original, y_original, model, scaler_x, scaler_y, optimizer)

    # 比较不同优化器的性能
    plot_optimizer_comparison(optimizer_results)

    # 绘制epoch和学习率的影响
    plot_epoch_learning_rate_effects(x_tensor, y_tensor)


if __name__ == "__main__":
    main()
