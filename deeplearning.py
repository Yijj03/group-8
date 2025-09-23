import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 读取数据并处理缺失值（简化版）
df = pd.read_csv(r'C:\Users\DELL\Downloads\train.csv')
df_clean = df.dropna(subset=['x', 'y'])  # 直接删除x或y列有缺失值的行
print(f"原始数据行数: {len(df)}, 处理后数据行数: {len(df_clean)}")

# 准备特征和目标值
X, y = df_clean['x'].values.reshape(-1, 1), df_clean['y'].values

# 训练模型
model = LinearRegression().fit(X, y)
w, b = model.coef_[0], model.intercept_
loss = mean_squared_error(y, model.predict(X))

# 可视化参数与损失关系
plt.figure(figsize=(12, 5))

# 绘制w与loss关系
plt.subplot(121)
ws = np.linspace(w-2, w+2, 100)
plt.plot(ws, [mean_squared_error(y, wi*X.flatten()+b) for wi in ws])
plt.scatter(w, loss)
plt.title('w and loss')

# 绘制b与loss关系
plt.subplot(122)
bs = np.linspace(b-2, b+2, 100)
plt.plot(bs, [mean_squared_error(y, w*X.flatten()+bi) for bi in bs])
plt.scatter(b, loss)
plt.title('b and loss')

plt.tight_layout()
plt.show()
