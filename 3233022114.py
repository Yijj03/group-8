"""
MNIST分类任务：GoogLeNet vs ResNet对比实验
作者：豆包编程助手
日期：2025年12月
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import matplotlib.style as style

# 设置matplotlib样式
style.use('seaborn-deep')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'SimHei'  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ========================= 配置类（统一管理超参数） =========================
class Config:
    """实验配置类"""
    BATCH_SIZE = 64
    TEST_BATCH_SIZE = 1000
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = './data'
    SAVE_DIR = './results'

    @classmethod
    def print_info(cls):
        """打印配置信息"""
        print(f"{'=' * 60}")
        print(f"实验配置信息")
        print(f"{'=' * 60}")
        print(f"设备: {cls.DEVICE}")
        print(f"训练批次大小: {cls.BATCH_SIZE}")
        print(f"测试批次大小: {cls.TEST_BATCH_SIZE}")
        print(f"训练轮数: {cls.EPOCHS}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print(f"{'=' * 60}\n")


# 创建保存目录
os.makedirs(Config.SAVE_DIR, exist_ok=True)


# ========================= 数据加载模块 =========================
def load_data():
    """加载MNIST数据集"""
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomHorizontalFlip(p=0.1),  # 轻微数据增强
    ])

    # 加载数据集
    train_dataset = datasets.MNIST(
        root=Config.DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=Config.DATA_DIR,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader


# ========================= 模型定义 =========================
class InceptionA(nn.Module):
    """Inception模块（多尺度特征融合）"""

    def __init__(self, in_channels):
        super().__init__()

        # 分支1：1x1卷积
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # 分支2：1x1卷积 → 5x5卷积
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, kernel_size=5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        # 分支3：1x1卷积 → 3x3卷积 × 2
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        # 分支4：池化 → 1x1卷积
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 24, kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        outputs = [
            self.branch1x1(x),
            self.branch5x5(x),
            self.branch3x3(x),
            self.branch_pool(x)
        ]
        return torch.cat(outputs, dim=1)


class GoogLeNet(nn.Module):
    """GoogLeNet（适配MNIST）"""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            InceptionA(10),
            nn.Conv2d(88, 20, kernel_size=5),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            InceptionA(20)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(1408, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        residual = self.residual(x)
        return F.relu(x + residual)


class ResNet(nn.Module):
    """ResNet（适配MNIST）"""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(16),
            ResidualBlock(16),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ========================= 训练工具函数 =========================
class Trainer:
    """训练器类"""

    def __init__(self, model, train_loader, test_loader):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=1e-5  # L2正则化
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=5,
            gamma=0.5
        )

        # 记录训练过程
        self.train_losses = []
        self.test_accuracies = []

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        # 使用tqdm显示进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch:2d}/{Config.EPOCHS}')

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })

        self.train_losses.append(total_loss / len(self.train_loader))
        self.scheduler.step()

        return self.train_losses[-1]

    @torch.no_grad()
    def test(self):
        """测试模型"""
        self.model.eval()
        correct = 0
        total = 0

        for data, target in self.test_loader:
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
            output = self.model(data)
            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        self.test_accuracies.append(accuracy)

        return accuracy

    def train(self):
        """完整训练过程"""
        print(f"\n开始训练 {self.model.__class__.__name__}...")

        for epoch in range(1, Config.EPOCHS + 1):
            avg_loss = self.train_epoch(epoch)
            accuracy = self.test()

            print(f'测试准确率: {accuracy:.2f}% | 平均损失: {avg_loss:.4f}')

        return self.test_accuracies


# ========================= 可视化工具 =========================
class Visualizer:
    """可视化工具类"""

    @staticmethod
    def plot_comparison(googlenet_acc, resnet_acc, googlenet_loss, resnet_loss):
        """绘制对比图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 准确率对比曲线
        epochs = range(1, Config.EPOCHS + 1)
        ax1.plot(epochs, googlenet_acc, 'b-o', linewidth=2, markersize=6, label='GoogLeNet', alpha=0.8)
        ax1.plot(epochs, resnet_acc, 'r-s', linewidth=2, markersize=6, label='ResNet', alpha=0.8)
        ax1.set_xlabel('训练轮数')
        ax1.set_ylabel('测试准确率 (%)')
        ax1.set_title('模型准确率对比', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(95, 100)

        # 2. 损失对比曲线
        ax2.plot(epochs, googlenet_loss, 'b-o', linewidth=2, markersize=6, label='GoogLeNet', alpha=0.8)
        ax2.plot(epochs, resnet_loss, 'r-s', linewidth=2, markersize=6, label='ResNet', alpha=0.8)
        ax2.set_xlabel('训练轮数')
        ax2.set_ylabel('训练损失')
        ax2.set_title('模型损失对比', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 最终准确率柱状图
        models = ['GoogLeNet', 'ResNet']
        final_accs = [googlenet_acc[-1], resnet_acc[-1]]
        colors = ['#3498db', '#e74c3c']
        bars = ax3.bar(models, final_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        # 在柱状图上添加数值标签
        for bar, acc in zip(bars, final_accs):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

        ax3.set_ylabel('最终准确率 (%)')
        ax3.set_title('模型最终准确率对比', fontweight='bold')
        ax3.set_ylim(95, 100)

        # 4. 准确率热力图
        acc_matrix = np.array([googlenet_acc, resnet_acc])
        im = ax4.imshow(acc_matrix, cmap='RdYlBu_r', aspect='auto', vmin=95, vmax=100)

        # 添加数值标注
        for i in range(2):
            for j in range(Config.EPOCHS):
                text = ax4.text(j, i, f'{acc_matrix[i, j]:.1f}',
                                ha="center", va="center", color="black", fontsize=8)

        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['GoogLeNet', 'ResNet'])
        ax4.set_xticks(range(Config.EPOCHS))
        ax4.set_xticklabels(range(1, Config.EPOCHS + 1))
        ax4.set_xlabel('训练轮数')
        ax4.set_title('准确率热力图', fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
        cbar.set_label('准确率 (%)')

        plt.tight_layout()
        plt.savefig(os.path.join(Config.SAVE_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_sample_predictions(model, test_loader):
        """可视化样本预测结果"""
        model.eval()

        # 获取一批测试数据
        data, target = next(iter(test_loader))
        data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)

        # 获取预测结果
        with torch.no_grad():
            output = model(data)
            _, predicted = torch.max(output, 1)

        # 绘制前16个样本
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        axes = axes.flatten()

        for i in range(16):
            img = data[i].cpu().squeeze().numpy()
            true_label = target[i].item()
            pred_label = predicted[i].item()

            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'真实: {true_label}\n预测: {pred_label}',
                              color='green' if true_label == pred_label else 'red')
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(Config.SAVE_DIR, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
        plt.show()


# ========================= 主程序 =========================
def main():
    """主函数"""
    # 打印配置信息
    Config.print_info()

    # 加载数据
    print("加载MNIST数据集...")
    train_loader, test_loader = load_data()

    # 训练GoogLeNet
    googlenet = GoogLeNet()
    googlenet_trainer = Trainer(googlenet, train_loader, test_loader)
    googlenet_acc = googlenet_trainer.train()
    googlenet_loss = googlenet_trainer.train_losses

    # 训练ResNet
    resnet = ResNet()
    resnet_trainer = Trainer(resnet, train_loader, test_loader)
    resnet_acc = resnet_trainer.train()
    resnet_loss = resnet_trainer.train_losses

    # 保存模型
    torch.save(googlenet.state_dict(), os.path.join(Config.SAVE_DIR, 'googlenet_mnist.pth'))
    torch.save(resnet.state_dict(), os.path.join(Config.SAVE_DIR, 'resnet_mnist.pth'))

    # 可视化结果
    visualizer = Visualizer()
    visualizer.plot_comparison(googlenet_acc, resnet_acc, googlenet_loss, resnet_loss)

    # 可视化样本预测
    print("\n生成样本预测可视化...")
    visualizer.plot_sample_predictions(googlenet, test_loader)
    visualizer.plot_sample_predictions(resnet, test_loader)

    # 打印最终结果
    print(f"\n{'=' * 60}")
    print(f"实验结果汇总")
    print(f"{'=' * 60}")
    print(f"GoogLeNet最终准确率: {googlenet_acc[-1]:.2f}%")
    print(f"ResNet最终准确率: {resnet_acc[-1]:.2f}%")
    print(f"最佳模型: {'ResNet' if resnet_acc[-1] > googlenet_acc[-1] else 'GoogLeNet'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()