import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


# 1. 读取数据（优化了文件路径处理）
def load_data():
    """从CSV文件加载数据，支持用户输入路径"""
    # 尝试默认路径
    default_paths = ['train.csv', './train.csv', '../train.csv']

    for path in default_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"成功读取数据: {path}，形状: {df.shape}")
                # 检查必要的列是否存在
                required_columns = ['x', 'y']
                if not all(col in df.columns for col in required_columns):
                    raise ValueError(f"数据文件必须包含'{required_columns[0]}'和'{required_columns[1]}'列")

                X = df['x'].values.reshape(-1, 1)
                y = df['y'].values
                return X, y
            except Exception as e:
                print(f"读取{path}时出错: {e}")

    # 如果默认路径都找不到，提示用户输入
    while True:
        file_path = input("请输入train.csv文件的路径（可拖拽文件到此处）: ").strip().replace('"', '')
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"成功读取数据: {file_path}，形状: {df.shape}")

                required_columns = ['x', 'y']
                if not all(col in df.columns for col in required_columns):
                    print(f"错误：数据文件必须包含'{required_columns[0]}'和'{required_columns[1]}'列")
                    continue

                X = df['x'].values.reshape(-1, 1)
                y = df['y'].values
                return X, y
            except Exception as e:
                print(f"读取文件时出错: {e}，请重新输入")
        else:
            print(f"错误：找不到文件 {file_path}，请重新输入")


# 2. 线性回归模型（y=wx+b）
class LinearRegression:
    def __init__(self):
        self.w = np.random.randn()  # 随机初始化权重
        self.b = np.random.randn()  # 随机初始化偏置

    def predict(self, x):
        """预测函数：y = wx + b"""
        return self.w * x + self.b

    def compute_loss(self, y_pred, y_true):
        """计算均方误差损失"""
        return np.mean((y_pred - y_true) ** 2)

    def train(self, x, y, learning_rate=0.01, epochs=1000):
        """使用梯度下降训练模型"""
        losses = []  # 记录损失变化
        ws = []  # 记录权重w变化
        bs = []  # 记录偏置b变化
        n = len(x)  # 样本数量

        for epoch in range(epochs):
            # 计算预测值
            y_pred = self.predict(x)
            current_loss = self.compute_loss(y_pred, y)

            # 计算梯度
            dw = (2 / n) * np.sum((y_pred - y) * x)  # w的梯度
            db = (2 / n) * np.sum(y_pred - y)  # b的梯度

            # 更新参数
            self.w -= learning_rate * dw
            self.b -= learning_rate * db

            # 记录训练过程
            losses.append(current_loss)
            ws.append(self.w)
            bs.append(self.b)

            # 定期输出训练进度
            if (epoch + 1) % 100 == 0:
                print(f"迭代次数: {epoch + 1}/{epochs}, 损失: {current_loss:.6f}, w: {self.w:.6f}, b: {self.b:.6f}")

        return losses, ws, bs


# 3. 可视化函数
def visualize_results(losses, ws, bs, X_test, y_test, model):
    """绘制w与loss、b与loss的关系图及拟合结果"""
    plt.style.use('seaborn-v0_8')  # 设置绘图风格

    # 创建一个2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('线性回归模型训练结果', fontsize=16)

    # 1. 损失随迭代次数变化
    axes[0, 0].plot(losses, color='blue')
    axes[0, 0].set_title('损失值随迭代次数变化', fontsize=12)
    axes[0, 0].set_xlabel('迭代次数')
    axes[0, 0].set_ylabel('损失值 (MSE)')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)

    # 2. w与loss的关系
    axes[0, 1].plot(ws, losses, color='green')
    axes[0, 1].set_title('权重w与损失值的关系', fontsize=12)
    axes[0, 1].set_xlabel('权重w')
    axes[0, 1].set_ylabel('损失值 (MSE)')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)

    # 3. b与loss的关系
    axes[1, 0].plot(bs, losses, color='red')
    axes[1, 0].set_title('偏置b与损失值的关系', fontsize=12)
    axes[1, 0].set_xlabel('偏置b')
    axes[1, 0].set_ylabel('损失值 (MSE)')
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)

    # 4. 最终拟合结果
    axes[1, 1].scatter(X_test, y_test, color='blue', alpha=0.6, label='测试数据')
    axes[1, 1].plot(X_test, model.predict(X_test), color='red', linewidth=2,
                    label=f'拟合直线: y={model.w:.4f}x + {model.b:.4f}')
    axes[1, 1].set_title('线性回归拟合结果', fontsize=12)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('linear_regression_results.png', dpi=300, bbox_inches='tight')
    print("可视化结果已保存为 'linear_regression_results.png'")
    plt.show()


# 主函数
def main():
    # 读取数据
    X, y = load_data()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 初始化模型
    model = LinearRegression()
    print(f"初始参数: w={model.w:.6f}, b={model.b:.6f}")

    # 训练模型（可根据实际数据调整学习率和迭代次数）
    losses, ws, bs = model.train(
        X_train.flatten(),  # 将二维数组转为一维
        y_train,
        learning_rate=0.001,  # 学习率
        epochs=2000  # 迭代次数
    )

    print(f"\n训练完成！最终参数: w={model.w:.6f}, b={model.b:.6f}")

    # 可视化结果
    visualize_results(losses, ws, bs, X_test, y_test, model)


if __name__ == "__main__":
    main()
