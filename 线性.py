import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('train.csv')
df=df.dropna()
x_data = df['x'].values
y_data = df['y'].values
#print(df)
print(df.describe())
print(f"数据量: {len(x_data)}")

# 线性回归模型：y = wx + b
def predict(x, w, b):
    return w * x + b

def compute_loss(w, b, x_data, y_data):
    """计算均方误差损失"""
    total_loss = 0
    for i in range(len(x_data)):
        y_pred = predict(x_data[i], w, b)
        total_loss += (y_pred - y_data[i]) ** 2
    return total_loss / len(x_data)

# 分析w与loss的关系（固定b=0）
print("分析w与loss的关系（固定b=0）")
w_list = []
loss_list_w = []

for w in np.arange(0.5, 2.5, 0.01):  # w的搜索范围
    current_loss = compute_loss(w, 0, x_data, y_data)
    w_list.append(w)
    loss_list_w.append(current_loss)
#print("loss_list结果",loss_list_w)
# 找到最优w
best_w_index = np.argmin(loss_list_w)
best_w = w_list[best_w_index]
min_loss_w = loss_list_w[best_w_index]

print(f"固定b=0时，最优w: {best_w:.4f}, 最小loss: {min_loss_w:.4f}")

# 分析b与loss的关系（固定w=最优值）
print("\n分析b与loss的关系（固定w=最优值）")
b_list = []
loss_list_b = []

for b in np.arange(-5, 5, 0.1):  # b的搜索范围
    current_loss = compute_loss(best_w, b, x_data, y_data)
    b_list.append(b)
    loss_list_b.append(current_loss)

# 找到最优b
best_b_index = np.argmin(loss_list_b)
best_b = b_list[best_b_index]
min_loss_b = loss_list_b[best_b_index]

print(f"固定w={best_w:.4f}时，最优b: {best_b:.4f}, 最小loss: {min_loss_b:.4f}")

# 绘制图形
plt.figure(figsize=(15, 5))

# 图1：w与loss的关系
plt.subplot(1, 2, 1)
plt.plot(w_list, loss_list_w, 'blue', linewidth=2)
plt.axvline(x=best_w, color='red', linestyle='--', label=f'最优w={best_w:.4f}')
plt.xlabel('w')
plt.ylabel('Loss')
#plt.title('w与Loss的关系（固定b=0）')
plt.legend()
plt.grid(True)

# 图2：b与loss的关系
plt.subplot(1, 2, 2)
plt.plot(b_list, loss_list_b, 'green', linewidth=2)
plt.axvline(x=best_b, color='red', linestyle='--', label=f'最优b={best_b:.4f}')
plt.xlabel('b')
plt.ylabel('Loss')
#plt.title('b与Loss的关系（固定w=最优值）')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()