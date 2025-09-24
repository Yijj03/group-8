import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取并清洗数据
df = pd.read_csv('train.csv')
x, y = df['x'].values, df['y'].values
mask = np.isfinite(x) & np.isfinite(y)
x, y = x[mask], y[mask]

# 线性回归
x_mean, y_mean = np.mean(x), np.mean(y)
w = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b = y_mean - w * x_mean

# 预测和评估
y_pred = w * x + b
mse = np.mean((y - y_pred) ** 2)
r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2)

print(f"参数: w={w:.4f}, b={b:.4f}")
print(f"MSE: {mse:.4f}, R²: {r2:.4f}")

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 拟合图
axes[0,0].scatter(x, y, alpha=0.7, s=10)
x_line = np.linspace(x.min(), x.max(), 100)
axes[0,0].plot(x_line, w*x_line+b, 'r-', linewidth=2)
axes[0,0].set_title('线性回归拟合')

# 2. 残差图
axes[0,1].scatter(y_pred, y-y_pred, alpha=0.7, s=10)
axes[0,1].axhline(0, color='red', linestyle='--')
axes[0,1].set_title('残差分析')

# 3. 参数搜索（使用pcolormesh避免警告）
w_range = np.linspace(w-0.3, w+0.3, 30)
b_range = np.linspace(b-3, b+3, 30)
W, B = np.meshgrid(w_range, b_range)
MSE = np.array([[np.mean((y - (w_val*x + b_val))**2) for w_val in w_range] for b_val in b_range])

im = axes[1,0].pcolormesh(W, B, MSE, cmap='viridis', shading='auto')
axes[1,0].scatter(w, b, c='red', s=50, marker='*')
axes[1,0].set_title('参数搜索空间')
plt.colorbar(im, ax=axes[1,0])

# 4. 损失曲线
axes[1,1].plot(b_range, [np.mean((y - (w*x + b_val))**2) for b_val in b_range], 'b-')
axes[1,1].axvline(b, color='red', linestyle='--')
axes[1,1].set_title('损失函数曲线')

plt.tight_layout()
plt.show()