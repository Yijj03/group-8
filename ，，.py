import matplotlib

matplotlib.use('TkAgg')  # 或尝试 'Qt5Agg'，根据你的系统选择
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置中文字体（关键：解决中文显示乱码问题）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
import matplotlib.pyplot as plt
# 指定中文字体，比如 "SimHei"（黑体）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 1. 数据加载与预处理
def load_and_preprocess_data(file_path='train.csv'):
    """加载并预处理数据集"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件 {file_path} 不存在，请确保文件在正确路径下")

    # 读取数据
    df = pd.read_csv(file_path)

    # 假设最后一列是目标变量（回归任务）
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 划分训练集和测试集（8:2分割，固定随机种子确保结果可复现）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化（回归任务必需，加速模型收敛）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为PyTorch张量（float32为默认推荐类型，节省内存）
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 转为列向量
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train, X_test, y_train, y_test


# 2. 线性回归模型定义
class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        # 小权重初始化（避免梯度爆炸，加速收敛）
        self.w = nn.Parameter(torch.randn(input_size, 1) * 0.01)
        self.b = nn.Parameter(torch.randn(1) * 0.01)

    def forward(self, x):
        # 线性回归公式：y = x * w + b
        return torch.matmul(x, self.w) + self.b


# 3. 模型训练函数
def train_model(model, optimizer, criterion, X_train, y_train, X_test, y_test, epochs=100):
    """训练模型并记录训练过程（损失+参数变化）"""
    train_losses = []  # 记录每轮训练损失
    test_losses = []   # 记录每轮测试损失
    w_history = []     # 记录权重w的变化
    b_history = []     # 记录偏置b的变化

    # 记录初始参数（第0轮）
    w_history.append(model.w.detach().numpy().copy())
    b_history.append(model.b.detach().numpy().copy())

    for epoch in range(epochs):
        # 训练模式（启用梯度计算）
        model.train()
        optimizer.zero_grad()  # 清空上一轮梯度

        # 前向传播：计算模型预测值
        outputs = model(X_train)
        loss = criterion(outputs, y_train)  # 计算训练损失

        # 反向传播：计算梯度并更新参数
        loss.backward()
        optimizer.step()

        # 记录训练损失
        train_losses.append(loss.item())

        # 测试模式（禁用梯度计算，节省内存）
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())

        # 记录当前轮参数
        w_history.append(model.w.detach().numpy().copy())
        b_history.append(model.b.detach().numpy().copy())

        # 每10轮打印一次训练信息（方便监控进度）
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'w_history': w_history,
        'b_history': b_history,
        'model': model
    }


# 4. 可视化函数（核心：确保图形可正常显示）
def plot_optimizer_comparison(results, optimizer_names):
    """对比不同优化器的训练/测试损失"""
    plt.figure(figsize=(12, 6))  # 设置画布大小

    # 子图1：训练损失对比
    plt.subplot(1, 2, 1)
    for i, result in enumerate(results):
        plt.plot(result['train_losses'], label=optimizer_names[i])
    plt.title('不同优化器的训练损失')
    plt.xlabel('Epoch（迭代轮次）')
    plt.ylabel('MSE损失值')
    plt.legend()  # 显示图例

    # 子图2：测试损失对比
    plt.subplot(1, 2, 2)
    for i, result in enumerate(results):
        plt.plot(result['test_losses'], label=optimizer_names[i])
    plt.title('不同优化器的测试损失')
    plt.xlabel('Epoch（迭代轮次）')
    plt.ylabel('MSE损失值')
    plt.legend()

    plt.tight_layout()  # 自动调整子图间距，避免标签重叠
    plt.savefig('optimizer_comparison.png', dpi=300)  # 保存图片（高分辨率）
    plt.show()  # 弹窗显示图片


def visualize_parameter_changes(param_history, param_name='权重w'):
    """可视化单个参数（w或b）的迭代变化过程（动画）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], 'b-', linewidth=2)  # 初始化线条
    ax.set_xlabel('迭代步骤（Epoch+1）')
    ax.set_ylabel('参数值')
    ax.set_title(f'{param_name}的迭代调节过程')

    # 处理参数数据（w取平均值，b直接取数值）
    if param_name == '权重w':
        data = [np.mean(param) for param in param_history]  # 多特征时，权重取平均便于可视化
    else:
        data = [param[0] for param in param_history]  # 偏置b是标量，直接提取

    # 设置坐标轴范围（避免动画中线条超出视野）
    ax.set_xlim(0, len(data))
    ax.set_ylim(min(data) * 1.1, max(data) * 1.1)

    # 动画更新函数（每帧绘制前n步的参数）
    def update(frame):
        line.set_data(range(frame + 1), data[:frame + 1])
        return line,

    # 创建动画（关键：interval=50为帧间隔，单位ms）
    ani = FuncAnimation(
        fig, update, frames=len(data), interval=50, blit=True, repeat=False
    )

    # 保存动画（可选：解决部分环境弹窗不显示的问题）
    ani.save(f'{param_name}_animation.gif', writer='pillow', fps=20)
    plt.show()
    return ani


def visualize_hyperparameter_effects(X_train, y_train, X_test, y_test, param_type='epochs'):
    """可视化超参数（epoch或学习率）对模型性能的影响"""
    input_size = X_train.shape[1]
    criterion = nn.MSELoss()
    results = []
    param_values = []

    if param_type == 'epochs':
        # 测试不同迭代轮次的影响
        param_values = [50, 100, 200, 300, 400]
        fixed_lr = 0.01  # 固定学习率
        for epochs in param_values:
            model = LinearModel(input_size)
            optimizer = optim.Adam(model.parameters(), lr=fixed_lr)  # 固定用Adam优化器
            result = train_model(model, optimizer, criterion, X_train, y_train, X_test, y_test, epochs)
            results.append(result)
        x_label = 'Epoch（迭代轮次）'
        title = '不同迭代轮次对模型测试损失的影响'
        save_name = 'epochs_effect.png'

    else:  # param_type == 'learning_rate'
        # 测试不同学习率的影响
        param_values = [0.001, 0.005, 0.01, 0.05, 0.1]
        fixed_epochs = 200  # 固定迭代轮次
        for lr in param_values:
            model = LinearModel(input_size)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            result = train_model(model, optimizer, criterion, X_train, y_train, X_test, y_test, fixed_epochs)
            results.append(result)
        x_label = '学习率'
        title = '不同学习率对模型测试损失的影响'
        save_name = 'learning_rate_effect.png'

    # 绘制结果
    plt.figure(figsize=(10, 6))
    for i, result in enumerate(results):
        plt.plot(result['test_losses'], label=f'{x_label} = {param_values[i]}', linewidth=2)

    plt.title(title)
    plt.xlabel('训练迭代步骤')
    plt.ylabel('测试损失（MSE）')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()


# 5. 主函数（程序入口）
def main():
    # 步骤1：加载数据（需确保train.csv在代码同一文件夹下）
    print("正在加载数据集...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    input_size = X_train.shape[1]
    print(f"数据加载完成！特征数量：{input_size}，训练集样本数：{len(X_train)}，测试集样本数：{len(X_test)}")

    # 步骤2：定义损失函数（回归任务用MSE）
    criterion = nn.MSELoss()

    # 步骤3：选择3种优化器对比
    optimizer_classes = [optim.Adagrad, optim.Adam, optim.Adamax]
    optimizer_names = ['Adagrad', 'Adam', 'Adamax']
    results = []  # 存储不同优化器的训练结果

    # 步骤4：训练模型（每种优化器独立训练）
    print("\n开始训练模型，对比不同优化器性能...")
    for i, opt_class in enumerate(optimizer_classes):
        print(f"\n=== 使用 {optimizer_names[i]} 优化器训练 ===")
        model = LinearModel(input_size)  # 每个优化器用新模型，避免参数干扰
        optimizer = opt_class(model.parameters(), lr=0.01)  # 固定学习率0.01
        result = train_model(model, optimizer, criterion, X_train, y_train, X_test, y_test, epochs=200)
        results.append(result)

    # 步骤5：可视化1：优化器性能对比（生成静态图）
    print("\n正在生成优化器性能对比图...")
    plot_optimizer_comparison(results, optimizer_names)

    # 步骤6：可视化2：参数迭代过程（生成动画，以Adam为例）
    print("\n正在生成参数迭代动画...")
    adam_index = optimizer_names.index('Adam')  # 找到Adam优化器的结果索引
    # 可视化权重w
    visualize_parameter_changes(results[adam_index]['w_history'], param_name='权重w')
    # 可视化偏置b
    visualize_parameter_changes(results[adam_index]['b_history'], param_name='偏置b')

    # 步骤7：可视化3：超参数影响（生成静态图）
    print("\n正在生成超参数影响图...")
    visualize_hyperparameter_effects(X_train, y_train, X_test, y_test, param_type='epochs')  # 迭代轮次影响
    visualize_hyperparameter_effects(X_train, y_train, X_test, y_test, param_type='learning_rate')  # 学习率影响

    print("\n所有图形已生成完成！图片和动画文件已保存到当前文件夹。")


# 运行程序（确保代码被直接执行时才触发）
if __name__ == "__main__":
    main()