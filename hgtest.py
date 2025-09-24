import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
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
        x = df['x'].values.reshape(-1, 1)
        y = df['y'].values.reshape(-1, 1)
        x_original = x.copy()  # 保存原始数据用于绘图

        # 数据归一化（解决数值溢出问题）
        scaler_x = MinMaxScaler()
        x_scaled = scaler_x.fit_transform(x)
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y)

        return x_scaled, y_scaled, x_original, y, scaler_x, scaler_y
    except Exception as e:
        print(f"数据读取错误: {e}")
        raise


def compute_loss(y_true, y_pred):
    """计算均方误差损失"""
    return np.mean((y_true - y_pred) ** 2)


def train_linear_model(x, y, learning_rate=0.01, epochs=1000):
    """训练线性回归模型（优化数值稳定性）"""
    n = len(x)
    w = 0.0  # 初始权重
    b = 0.0  # 初始偏置
    losses = []

    for i in range(epochs):
        # 前向传播
        y_pred = w * x + b

        # 计算损失
        loss = compute_loss(y, y_pred)
        losses.append(loss)

        # 计算梯度（添加微小值防止数值异常）
        dw = (-2 / n) * np.sum(x * (y - y_pred + 1e-10))
        db = (-2 / n) * np.sum((y - y_pred) + 1e-10)

        # 更新参数
        w -= learning_rate * dw
        b -= learning_rate * db

        # 每100轮打印一次
        if i % 100 == 0:
            print(f"迭代 {i:4d} | 损失: {loss:.6f} | w: {w:.6f} | b: {b:.6f}")

    print(f"训练完成! 最终损失: {losses[-1]:.6f}, 最优w: {w:.6f}, 最优b: {b:.6f}")
    return w, b, losses


def plot_parameter_vs_loss(x, y, optimal_w, optimal_b):
    """绘制w和b与损失的关系图"""
    # 生成合理的参数范围（确保曲线可见）
    w_min, w_max = optimal_w * 0.5, optimal_w * 1.5
    w_values = np.linspace(w_min, w_max, 200)
    w_losses = [compute_loss(y, w * x + optimal_b) for w in w_values]

    b_min, b_max = optimal_b * 0.5, optimal_b * 1.5
    b_values = np.linspace(b_min, b_max, 200)
    b_losses = [compute_loss(y, optimal_w * x + b) for b in b_values]

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 绘制w-loss曲线
    ax1.plot(w_values, w_losses, 'b-', linewidth=2)
    ax1.axvline(optimal_w, color='r', linestyle='--', label=f'最优w: {optimal_w:.4f}')
    ax1.set_title('w与损失值的关系')
    ax1.set_xlabel('w值')
    ax1.set_ylabel('损失值')
    ax1.grid(True, alpha=0.7)
    ax1.legend()

    # 绘制b-loss曲线
    ax2.plot(b_values, b_losses, 'g-', linewidth=2)
    ax2.axvline(optimal_b, color='r', linestyle='--', label=f'最优b: {optimal_b:.4f}')
    ax2.set_title('b与损失值的关系')
    ax2.set_xlabel('b值')
    ax2.set_ylabel('损失值')
    ax2.grid(True, alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_training_curve(losses):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses, 'b-', linewidth=2)
    plt.title('训练过程中的损失变化')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.grid(True, alpha=0.7)
    plt.show()


def plot_regression_line(x_original, y_original, w, b, scaler_x, scaler_y):
    """绘制原始数据与回归直线"""
    # 计算原始尺度的预测值
    x_scaled = scaler_x.transform(x_original)
    y_pred_scaled = w * x_scaled + b
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_original, y_original, alpha=0.6, label='数据点', edgecolors='k', s=30)
    plt.plot(x_original, y_pred_original, 'r-', linewidth=2,
             label=f'回归直线: y = {w:.4f}x + {b:.4f}')
    plt.title('数据点与回归直线')
    plt.xlabel('x值')
    plt.ylabel('y值')
    plt.grid(True, alpha=0.7)
    plt.legend()
    plt.show()


def main():
    # 读取数据（确保train.csv在当前目录）
    x, y, x_original, y_original, scaler_x, scaler_y = load_data('train.csv')

    # 训练模型
    w, b, losses = train_linear_model(x, y)

    # 绘制所有图表
    plot_training_curve(losses)
    plot_parameter_vs_loss(x, y, w, b)
    plot_regression_line(x_original, y_original, w, b, scaler_x, scaler_y)


if __name__ == "__main__":
    main()
