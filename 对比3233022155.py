import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# ----------------------------
# 全局配置（保持不变）
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

batch_size = 64
epochs = 10
lr = 0.001
input_size = 64  # 保持64x64输入，兼顾速度和准确率

# ----------------------------
# 模型定义（保持不变）
# ----------------------------
class LightInception(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 8, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=1),
            nn.Conv2d(8, 16, kernel_size=3, padding=1)
        )
        self.branch_pool = nn.Conv2d(in_channels, 8, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        p1 = self.relu(self.branch1x1(x))
        p2 = self.relu(self.branch3x3(x))
        p3 = self.relu(self.branch_pool(nn.functional.avg_pool2d(x, 3, 1, 1)))
        return torch.cat([p1, p2, p3], dim=1)

class LightGoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            LightInception(16),
            nn.MaxPool2d(2, stride=2),
            LightInception(32),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(32, 10)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides) if use_1conv else None

    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLU(y + x)
        return y

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(Residual(64, 64), Residual(64, 64))
        self.b3 = nn.Sequential(Residual(64, 128, use_1conv=True, strides=2), Residual(128, 128))
        self.b4 = nn.Sequential(Residual(128, 256, use_1conv=True, strides=2), Residual(256, 256))
        self.b5 = nn.Sequential(Residual(256, 512, use_1conv=True, strides=2), Residual(512, 512))
        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10))

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x

# ----------------------------
# 训练/测试函数（保持不变）
# ----------------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    return total_loss / len(train_loader), correct / total

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return total_loss / len(test_loader), correct / total

def train_model(model, name):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'time': []}

    start_time = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['time'].append(time.time() - start_time)

        print(f"{name} Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    return history


if __name__ == '__main__':
    # 1. 数据准备（num_workers=0，关闭多进程）
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 修复点1：num_workers=0（Windows兼容模式）
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. 执行训练
    print(f"使用设备: {device}")
    print("\n开始训练轻量版GoogLeNet...")
    light_googlenet = LightGoogLeNet()
    googlenet_history = train_model(light_googlenet, "LightGoogLeNet")

    print("\n开始训练ResNet18...")
    resnet = ResNet18()
    resnet_history = train_model(resnet, "ResNet18")

    # 3. 可视化
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(googlenet_history['train_loss'], label='LightGoogLeNet Train')
    plt.plot(googlenet_history['test_loss'], label='LightGoogLeNet Test')
    plt.title('LightGoogLeNet Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(googlenet_history['train_acc'], label='LightGoogLeNet Train')
    plt.plot(googlenet_history['test_acc'], label='LightGoogLeNet Test')
    plt.title('LightGoogLeNet Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(resnet_history['train_loss'], label='ResNet18 Train')
    plt.plot(resnet_history['test_loss'], label='ResNet18 Test')
    plt.title('ResNet18 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(resnet_history['train_acc'], label='ResNet18 Train')
    plt.plot(resnet_history['test_acc'], label='ResNet18 Test')
    plt.title('ResNet18 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('mnist_light_model_comparison_fast.png')
    plt.show()

    # 4. 最终性能比较
    print("\n最终性能比较:")
    print(f"轻量版GoogLeNet 测试准确率: {googlenet_history['test_acc'][-1]:.4f}")
    print(f"ResNet18 测试准确率: {resnet_history['test_acc'][-1]:.4f}")
    print(f"轻量版GoogLeNet 总训练时间: {googlenet_history['time'][-1]:.2f}s")
    print(f"ResNet18 总训练时间: {resnet_history['time'][-1]:.2f}s")