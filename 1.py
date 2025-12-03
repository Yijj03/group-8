# 1. 导入模块（必须放在代码开头）
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# 2. 数据加载函数
def load_data():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,) * 3, (0.3081,) * 3)
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    batch_size = 256 if torch.cuda.is_available() else 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# 3. MiniInception模块
class MiniInception(nn.Module):
    def __init__(self, in_ch, ch1x1, ch3x3):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_ch, ch1x1, 1), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv2d(in_ch, ch3x3, 3, padding=1), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x)], 1)


# 4. MiniGoogleNet模型
class MiniGoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            MiniInception(16, 8, 16),
            MiniInception(24, 12, 24),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(36, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)


# 5. MiniResBlock模块
class MiniResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))


# 6. MiniResNet模型
class MiniResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            MiniResBlock(16, 16),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)


# 7. 训练函数
def train_model(model, model_name, train_loader, test_loader, epochs=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_losses, test_accs = [], []

    print(f"\n训练 {model_name} on {device}（{epochs}轮）")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            correct = sum(model(images.to(device)).argmax(dim=1).eq(labels.to(device)).sum().item()
                          for images, labels in test_loader)
        test_acc = 100 * correct / len(test_loader.dataset)
        test_accs.append(test_acc)
        print(f"Epoch {epoch + 1} | Loss: {train_losses[-1]:.4f} | Acc: {test_acc:.2f}%")

    return train_losses, test_accs


# 8. 可视化函数
def plot_results(gnet_loss, gnet_acc, resnet_loss, resnet_acc):
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 4), dpi=150)

    # 损失对比
    plt.subplot(1, 2, 1)
    plt.plot(gnet_loss, label='MiniGoogleNet', color='#ff7f0e', linewidth=2, marker='o', markersize=4)
    plt.plot(resnet_loss, label='MiniResNet', color='#2ca02c', linewidth=2, marker='s', markersize=4)
    for i, (g_val, r_val) in enumerate(zip(gnet_loss, resnet_loss)):
        plt.text(i, g_val + 0.05, f'{g_val:.2f}', ha='center', fontsize=7, color='#ff7f0e')
        plt.text(i, r_val - 0.1, f'{r_val:.2f}', ha='center', fontsize=7, color='#2ca02c')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.title('Training Loss Comparison', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.xticks(range(len(gnet_loss)), [f'Epoch {i + 1}' for i in range(len(gnet_loss))])

    # 准确率对比
    plt.subplot(1, 2, 2)
    plt.plot(gnet_acc, label='MiniGoogleNet', color='#ff7f0e', linewidth=2, marker='o', markersize=4)
    plt.plot(resnet_acc, label='MiniResNet', color='#2ca02c', linewidth=2, marker='s', markersize=4)
    for i, (g_val, r_val) in enumerate(zip(gnet_acc, resnet_acc)):
        plt.text(i, g_val + 0.3, f'{g_val:.1f}%', ha='center', fontsize=7, color='#ff7f0e')
        plt.text(i, r_val - 0.8, f'{r_val:.1f}%', ha='center', fontsize=7, color='#2ca02c')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.xticks(range(len(gnet_acc)), [f'Epoch {i + 1}' for i in range(len(gnet_acc))])
    plt.ylim(85, 100)

    # 保存图片
    plt.tight_layout(pad=1.0)
    save_path = 'fast_performance.png'
    try:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=150)
        if os.path.exists(save_path):
            print(f"\n✅ 可视化图保存至：{os.path.abspath(save_path)}")
    except Exception as e:
        temp_path = os.path.join(os.getenv('TEMP', '/tmp'), 'fast_performance.png')
        plt.savefig(temp_path, bbox_inches='tight', facecolor='white', dpi=150)
        print(f"\n✅ 可视化图保存至临时目录：{temp_path}")
    plt.close()


# 9. 主函数
def main():
    print("===== 加载数据 =====")
    train_loader, test_loader = load_data()

    print("\n===== 训练 MiniGoogleNet =====")
    gnet = MiniGoogleNet()
    gnet_loss, gnet_acc = train_model(gnet, 'MiniGoogleNet', train_loader, test_loader)

    print("\n===== 训练 MiniResNet =====")
    resnet = MiniResNet()
    resnet_loss, resnet_acc = train_model(resnet, 'MiniResNet', train_loader, test_loader)

    print("\n===== 生成可视化图 =====")
    plot_results(gnet_loss, gnet_acc, resnet_loss, resnet_acc)

    print(f"\n===== 运行完成！ =====")
    print(f"最终性能：MiniGoogleNet {gnet_acc[-1]:.2f}% | MiniResNet {resnet_acc[-1]:.2f}%")


if __name__ == "__main__":
    main()