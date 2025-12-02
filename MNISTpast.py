import matplotlib
matplotlib.use("TkAgg")   # 或 "Qt5Agg"，强制换后端

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

# ===================== 1. Windows多进程兼容配置 =====================
# Windows系统下，多进程需要设置启动方式（避免冲突）
if torch.cuda.is_available():
    torch.multiprocessing.set_start_method('spawn', force=True)
else:
    # CPU情况下禁用多进程（避免报错）
    multiprocessing.set_start_method('spawn', force=True)

# ===================== 2. 环境配置与超参数 =====================
# 设备配置（自动检测GPU/CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数（可直接调整）
batch_size = 64
learning_rate = 0.001
epochs = 10
num_classes = 10  # MNIST为10分类

# ===================== 3. 数据加载与预处理 =====================
# 数据预处理流水线
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor（0-1范围）
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值和标准差（标准化）
])

# 加载MNIST数据集（自动下载）
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 关键修复：num_workers=0（禁用多进程，Windows系统推荐）
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# ===================== 4. GoogleNet 模型定义 =====================
class InceptionA(nn.Module):
    """Inception模块（GoogleNet核心组件）"""

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # 分支1：1x1卷积（直接降维）
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        # 分支2：1x1降维 + 5x5卷积（padding=2保持尺寸）
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        # 分支3：1x1降维 + 2个3x3卷积（替代5x5，减少计算量）
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        # 分支4：平均池化 + 1x1卷积（保持通道数一致）
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        # 四个分支并行计算
        branch1x1 = self.branch1x1(x)

        branch5x5 = F.relu(self.branch5x5_1(x))
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = F.relu(self.branch3x3_1(x))
        branch3x3 = F.relu(self.branch3x3_2(branch3x3))
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # 通道维度拼接（16+24+24+24=88通道输出）
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class GoogleNet(nn.Module):
    """GoogleNet完整模型（适配MNIST）"""

    def __init__(self):
        super(GoogleNet, self).__init__()
        # 初始卷积层（1->10通道，28x28->24x24）
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # Inception模块（接收10通道，输出88通道）
        self.incep1 = InceptionA(in_channels=10)
        # 卷积层（88->20通道，12x12->8x8）
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        # 第二个Inception模块（接收20通道，输出88通道）
        self.incep2 = InceptionA(in_channels=20)
        # 最大池化层（步长2，尺寸减半）
        self.mp = nn.MaxPool2d(2)
        # 全连接层（88通道 * 4x4尺寸 = 1408，输出10类）
        self.fc = nn.Linear(1408, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        # 第一层：Conv1 -> MaxPool -> ReLU
        x = self.mp(F.relu(self.conv1(x)))  # 28x28 -> 24x24 -> 12x12
        # 第一层Inception
        x = self.incep1(x)  # 12x12x10 -> 12x12x88
        # 第二层：Conv2 -> MaxPool -> ReLU
        x = self.mp(F.relu(self.conv2(x)))  # 12x12 -> 8x8 -> 4x4
        # 第二层Inception
        x = self.incep2(x)  # 4x4x20 -> 4x4x88
        # 展平特征图（batch_size, 4*4*88=1408）
        x = x.view(batch_size, -1)
        # 全连接层输出
        x = self.fc(x)
        return x


# ===================== 5. ResNet 模型定义 =====================
class ResidualBlock(nn.Module):
    """残差块（恒等映射，通道数不变）"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # 两个3x3卷积（padding=1保持尺寸不变）
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 残差分支：Conv1 -> ReLU -> Conv2
        residual = F.relu(self.conv1(x))
        residual = self.conv2(residual)
        # Shortcut连接：输入x + 残差（元素-wise相加）
        out = F.relu(x + residual)
        return out


class ResNet(nn.Module):
    """ResNet完整模型（适配MNIST）"""

    def __init__(self):
        super(ResNet, self).__init__()
        # 初始卷积层（1->16通道，28x28->24x24）
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        # 残差块1（16通道）
        self.rblock1 = ResidualBlock(16)
        # 卷积层（16->32通道，12x12->8x8）
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        # 残差块2（32通道）
        self.rblock2 = ResidualBlock(32)
        # 最大池化层
        self.mp = nn.MaxPool2d(2)
        # 全连接层（32通道 * 4x4尺寸 = 512，输出10类）
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        # 第一层：Conv1 -> MaxPool -> ReLU
        x = self.mp(F.relu(self.conv1(x)))  # 28x28 -> 24x24 -> 12x12
        # 第一个残差块
        x = self.rblock1(x)  # 12x12x16 -> 12x12x16
        # 第二层：Conv2 -> MaxPool -> ReLU
        x = self.mp(F.relu(self.conv2(x)))  # 12x12 -> 8x8 -> 4x4
        # 第二个残差块
        x = self.rblock2(x)  # 4x4x32 -> 4x4x32
        # 展平特征图（batch_size, 4*4*32=512）
        x = x.view(batch_size, -1)
        # 全连接层输出
        x = self.fc(x)
        return x


# ===================== 6. 训练与测试函数 =====================
def train(model, train_loader, criterion, optimizer, epoch):
    """单轮训练函数"""
    model.train()  # 训练模式（启用Dropout、BatchNorm更新）
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据移至设备（GPU/CPU）
        data, target = data.to(device), target.to(device)

        # 前向传播：计算模型输出
        output = model(data)
        # 计算损失
        loss = criterion(output, target)

        # 反向传播：更新参数
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新权重

        total_loss += loss.item()

        # 打印训练进度（每300个batch打印一次）
        if (batch_idx + 1) % 300 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 返回本轮平均损失
    avg_train_loss = total_loss / len(train_loader)
    return avg_train_loss


def test(model, test_loader, criterion):
    """测试函数（无梯度计算）"""
    model.eval()  # 测试模式（禁用Dropout、固定BatchNorm）
    total_loss = 0.0
    correct = 0  # 正确预测的样本数

    with torch.no_grad():  # 禁用梯度计算，节省内存
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 累计测试损失
            total_loss += criterion(output, target).item()
            # 预测类别（取概率最大的类）
            pred = output.argmax(dim=1, keepdim=True)
            # 累计正确数
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 计算平均损失和准确率
    avg_test_loss = total_loss / len(test_loader)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print(
        f'\nTest Set - Average Loss: {avg_test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.2f}%)\n')
    return avg_test_loss, test_accuracy


# ===================== 7. 模型训练与评估 =====================
def train_and_evaluate(model_name):
    """完整训练+评估流程"""
    # 初始化模型
    if model_name == "googlenet":
        model = GoogleNet().to(device)
    elif model_name == "resnet":
        model = ResNet().to(device)
    else:
        raise ValueError("仅支持 'googlenet' 或 'resnet'")

    # 损失函数（交叉熵损失，适用于多分类）
    criterion = nn.CrossEntropyLoss()
    # 优化器（Adam，自适应学习率）
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练过程数据
    train_losses = []
    test_losses = []
    test_accuracies = []

    # 开始训练
    print(f"\n========================================")
    print(f"开始训练 {model_name.upper()} 模型")
    print(f"========================================\n")
    for epoch in range(epochs):
        # 训练
        train_loss = train(model, train_loader, criterion, optimizer, epoch)
        # 测试
        test_loss, test_acc = test(model, test_loader, criterion)
        # 保存结果
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    return model, train_losses, test_losses, test_accuracies


# ===================== 8. 结果可视化 =====================
def plot_comparison(g_train_loss, g_test_loss, g_test_acc, r_train_loss, r_test_loss, r_test_acc):
    """绘制训练/测试损失和准确率对比图"""
    epochs_range = np.arange(1, epochs + 1)

    # 创建画布（1行2列子图）
    plt.figure(figsize=(14, 6))

    # 子图1：损失对比
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, g_train_loss, label='GoogleNet - 训练损失', marker='o', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, g_test_loss, label='GoogleNet - 测试损失', marker='s', color='#ff7f0e', linewidth=2)
    plt.plot(epochs_range, r_train_loss, label='ResNet - 训练损失', marker='^', color='#2ca02c', linewidth=2)
    plt.plot(epochs_range, r_test_loss, label='ResNet - 测试损失', marker='d', color='#d62728', linewidth=2)
    plt.xlabel('训练轮次（Epoch）', fontsize=12)
    plt.ylabel('损失值（Loss）', fontsize=12)
    plt.title('模型训练与测试损失对比', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs_range)

    # 子图2：准确率对比
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, g_test_acc, label='GoogleNet 测试准确率', marker='o', color='#1f77b4', linewidth=3)
    plt.plot(epochs_range, r_test_acc, label='ResNet 测试准确率', marker='s', color='#2ca02c', linewidth=3)
    plt.xlabel('训练轮次（Epoch）', fontsize=12)
    plt.ylabel('准确率（%）', fontsize=12)
    plt.title('模型测试准确率对比', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs_range)
    plt.ylim(95, 100)  # 限定y轴范围，更清晰展示差异

    # 调整布局并显示
    plt.tight_layout()
    plt.show()


# ===================== 9. 主执行代码（关键修复） =====================
if __name__ == '__main__':
    # 训练两个模型
    googlenet_model, g_train_loss, g_test_loss, g_test_acc = train_and_evaluate("googlenet")
    resnet_model, r_train_loss, r_test_loss, r_test_acc = train_and_evaluate("resnet")

    # 执行可视化
    plot_comparison(g_train_loss, g_test_loss, g_test_acc, r_train_loss, r_test_loss, r_test_acc)

    # 保存训练好的模型权重
    torch.save(googlenet_model.state_dict(), 'googlenet_mnist.pth')
    torch.save(resnet_model.state_dict(), 'resnet_mnist.pth')
    print("模型权重已保存为：googlenet_mnist.pth 和 resnet_mnist.pth")
