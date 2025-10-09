# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from sklearn.metrics import mean_squared_error
#
#
# df = pd.read_csv(r'C:\Users\DELL\Downloads\train.csv')
# df_clean = df.dropna(subset=['x', 'y'])
# X_np, y_np = df_clean['x'].values, df_clean['y'].values
#
# X_train = torch.tensor(X_np, dtype=torch.float32).view(-1, 1)
# y_train = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)
#
#
# model = nn.Linear(1, 1)
# criterion = nn.MSELoss()  # 均方误差损失
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
#
#
# num_epochs = 1000
# for epoch in range(num_epochs):
#     # 正向传播
#     y_pred = model(X_train)
#     loss = criterion(y_pred, y_train)
#
#     optimizer.zero_grad()  # 梯度清零
#     loss.backward()  # 反向传播
#     optimizer.step()
#
#
# w_final = model.weight.item()
# b_final = model.bias.item()
# final_loss = mean_squared_error(y_np, w_final * X_np + b_final)
#
#
# plt.subplot(1,2,1)
# ws = np.linspace(w_final - 2, w_final + 2, 100)
# losses_w = [mean_squared_error(y_np, wi * X_np + b_final) for wi in ws]
# plt.plot(ws, losses_w)
# plt.title('w and loss')
#
#
# plt.subplot(1,2,2)
# bs = np.linspace(b_final - 2, b_final + 2, 100)
# losses_b = [mean_squared_error(y_np, w_final * X_np + bi) for bi in bs]
# plt.plot(bs, losses_b)
# plt.title('b and loss')
#
# plt.tight_layout()
# plt.show()

#第二次作业
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
#
# df = pd.read_csv(r'C:\Users\DELL\Downloads\train.csv')
# df_clean = df.dropna(subset=['x', 'y'])
#
# X_train = torch.tensor(df_clean['x'].values, dtype=torch.float32)
# y_train = torch.tensor(df_clean['y'].values, dtype=torch.float32)
#
# def forward(x):
#     return w * x
#
# w = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
#
# def loss(x, y):
#     y_pred = forward(x)
#     return (y_pred - y) ** 2
#
# learning_rate = 0.0001
# num_epochs = 100
# print("predict (before training)", 4, forward(torch.tensor([4.0])).item())
#
# for epoch in range(num_epochs):
#     total_loss = 0
#     for x, y in zip(X_train, y_train):
#         l = loss(x, y)
#         l.backward()
#         print(f'\tgrad: {x.item():.4f}, {y.item():.4f}, {w.grad.item():.4f}')
#         w.data = w.data - learning_rate * w.grad.data
#         w.grad.data.zero_()
#         total_loss += l.item()
#     print(f"progress: {epoch}, {total_loss / len(X_train)}")
#
# print("predict (after training)", 4, forward(torch.tensor([4.0])).item())
#
# #可视化
# w_final = w.item()
# b_final = 0
#
# X_np = X_train.numpy().flatten()
# y_np = y_train.numpy().flatten()
#
# # 绘制 w 与损失的关系
# plt.subplot(1, 2, 1)
# ws = np.linspace(w_final - 2, w_final + 2, 100)
# losses_w = np.mean((y_np - (ws[:, None] * X_np))**2, axis=1)
# plt.plot(ws, losses_w)
# plt.title('w and loss')
# plt.xlabel('w')
# plt.ylabel('Loss')
#
# # 绘制 b 与损失的关系
# plt.subplot(1, 2, 2)
# bs = np.linspace(-2, 2, 100)
# losses_b = np.mean((y_np - (w_final * X_np + bs[:, None]))**2, axis=1)
# plt.plot(bs, losses_b)
# plt.title('b and loss')
# plt.xlabel('b')
# plt.ylabel('Loss')
#
# plt.tight_layout()
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim


df = pd.read_csv(r'C:\Users\DELL\Downloads\train.csv')
df_clean = df.dropna(subset=['x', 'y'])
X_train = torch.tensor(df_clean['x'].values, dtype=torch.float32).reshape(-1, 1)  # 适配优化器输入
y_train = torch.tensor(df_clean['y'].values, dtype=torch.float32).reshape(-1, 1)


w = torch.normal(0, 1, size=(1,), dtype=torch.float32, requires_grad=True)  # 均值0，标准差1
b = torch.normal(0, 1, size=(1,), dtype=torch.float32, requires_grad=True)


# 模型与损失函数
def forward(x):
    return w * x + b


loss_fn = torch.nn.MSELoss()

#选择三种优化器
optimizers = {
    'SGD': optim.SGD([w, b], lr=0.0001),
    'Adam': optim.Adam([w, b], lr=0.001),
    'Adagrad': optim.Adagrad([w, b], lr=0.1)
}

num_epochs = 100
results = {name: {'loss': [], 'w': [], 'b': []} for name in optimizers}  # 存储训练记录


for opt_name, optimizer in optimizers.items():
    w.data = torch.normal(0, 1, size=(1,))
    b.data = torch.normal(0, 1, size=(1,))

    print(f"\n===== 优化器: {opt_name} =====")
    print(f"初始预测 (x=4): {forward(torch.tensor([[4.0]])).item():.4f}")

    for epoch in range(num_epochs):
        y_pred = forward(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #训练过程
        results[opt_name]['loss'].append(loss.item())
        results[opt_name]['w'].append(w.item())
        results[opt_name]['b'].append(b.item())
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}, 损失: {loss.item():.4f}")

    print(f"最终预测 (x=4): {forward(torch.tensor([[4.0]])).item():.4f}")

#优化器性能对比（损失曲线）
plt.figure(figsize=(10, 4))
for name in optimizers:
    plt.plot(results[name]['loss'], label=name)
plt.title('diff youhuaqi')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#可视化：w和b的调节过程
plt.figure(figsize=(12, 5))
# w的变化
plt.subplot(1, 2, 1)
for name in optimizers:
    plt.plot(results[name]['w'], label=name)
plt.title('w')
plt.xlabel('Epoch')
plt.ylabel('w值')
plt.legend()

# b的变化
plt.subplot(1, 2, 2)
for name in optimizers:
    plt.plot(results[name]['b'], label=name)
plt.title('b')
plt.xlabel('Epoch')
plt.ylabel('b值')
plt.legend()
plt.tight_layout()
plt.show()

#学习率和epoch的影响
lrs = [0.0001, 0.001, 0.01]
lr_loss = []
for lr in lrs:
    w_temp = torch.normal(0, 1, size=(1,), requires_grad=True)
    b_temp = torch.normal(0, 1, size=(1,), requires_grad=True)
    opt = optim.Adam([w_temp, b_temp], lr=lr)
    losses = []
    for _ in range(num_epochs):
        loss = loss_fn(w_temp * X_train + b_temp, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    lr_loss.append(losses)

plt.figure(figsize=(10, 4))
for i, lr in enumerate(lrs):
    plt.plot(lr_loss[i], label=f'学习率={lr}')
plt.title('lr vs Adam')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 不同epoch（延长训练轮数观察）
epochs = [50, 100, 200]
epoch_loss = []
for ep in epochs:
    w_temp = torch.normal(0, 1, size=(1,), requires_grad=True)
    b_temp = torch.normal(0, 1, size=(1,), requires_grad=True)
    opt = optim.Adam([w_temp, b_temp], lr=0.001)
    losses = []
    for _ in range(ep):
        loss = loss_fn(w_temp * X_train + b_temp, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    epoch_loss.append(losses)

plt.figure(figsize=(10, 4))
for i, ep in enumerate(epochs):
    plt.plot(epoch_loss[i], label=f'Epochs={ep}')
plt.title('train vs Adam')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()