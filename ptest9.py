import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=500, shuffle=False)


# Inception模块
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
        self.bn1 = nn.BatchNorm2d(ch1x1)
        self.bn2 = nn.BatchNorm2d(ch3x3)
        self.bn3 = nn.BatchNorm2d(ch5x5)
        self.bn4 = nn.BatchNorm2d(pool_proj)

    def forward(self, x):
        branch1 = self.bn1(self.branch1(x))
        branch2 = self.bn2(self.branch2(x))
        branch3 = self.bn3(self.branch3(x))
        branch4 = self.bn4(self.branch4(x))
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(192)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = F.relu(self.inception3a(x))
        x = F.relu(self.inception3b(x))
        x = self.maxpool3(x)
        x = F.relu(self.inception4a(x))
        x = F.relu(self.inception4b(x))
        x = F.relu(self.inception4c(x))
        x = F.relu(self.inception4d(x))
        x = F.relu(self.inception4e(x))
        x = self.maxpool4(x)
        x = F.relu(self.inception5a(x))
        x = F.relu(self.inception5b(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


#ResNet模块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


#训练函数
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10,
                       model_name='Model'):
    train_losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    all_preds = []
    all_labels = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        correct = 0
        total = 0
        epoch_preds = []
        epoch_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                epoch_preds.extend(predicted.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())

        all_preds.extend(epoch_preds)
        all_labels.extend(epoch_labels)

        accuracy = 100 * correct / total
        precision = precision_score(epoch_labels, epoch_preds, average='weighted', zero_division=0)
        recall = recall_score(epoch_labels, epoch_preds, average='weighted', zero_division=0)
        f1 = f1_score(epoch_labels, epoch_preds, average='weighted', zero_division=0)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(f'{model_name} - Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return {
        'model_name': model_name,
        'train_losses': train_losses,
        'accuracies': accuracies,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'final_accuracy': accuracies[-1],
        'final_precision': precisions[-1],
        'final_recall': recalls[-1],
        'final_f1': f1_scores[-1]
    }


num_epochs = 10
results_list = []

# 训练GoogleNet
print("开始训练 GoogleNet...")

googlenet = GoogLeNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(googlenet.parameters(), lr=0.001)

google_result = train_and_evaluate(
    googlenet, train_loader, test_loader,
    criterion, optimizer, device,
    num_epochs=num_epochs, model_name='GoogleNet'
)
results_list.append(google_result)

# 训练ResNet18
print("开始训练 ResNet18...")

resnet = ResNet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

resnet_result = train_and_evaluate(
    resnet, train_loader, test_loader,
    criterion, optimizer, device,
    num_epochs=num_epochs, model_name='ResNet18'
)
results_list.append(resnet_result)

#可视化比较
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)

plt.title('训练损失比较')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. 测试准确率比较折线图
plt.subplot(1, 3, 2)
plt.title('测试准确率比较')
plt.xlabel('Epoch')
plt.ylabel('准确率 (%)')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. 最终性能指标柱状图比较
plt.subplot(1, 3, 3)

metrics = ['精确率', '召回率', 'F1分数']
model_names = [result['model_name'] for result in results_list]
bar_width = 0.35
x_positions = np.arange(len(metrics))

for i, result in enumerate(results_list):
    values = [result['final_precision'], result['final_recall'], result['final_f1']]
    offset = i * bar_width - bar_width / 2
    bars = plt.bar(x_positions + offset, values, bar_width, label=result['model_name'], alpha=0.8)

    # 在柱子上添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=10)

plt.title('最终性能指标比较', fontsize=14, fontweight='bold')
plt.xlabel('指标')
plt.ylabel('分数')
plt.xticks(x_positions, metrics)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("模型性能对比总结")

for result in results_list:
    print(f"\n{result['model_name']}:")
    print(f"  最终测试准确率: {result['final_accuracy']:.2f}%")
    print(f"  最终精确率: {result['final_precision']:.4f}")
    print(f"  最终召回率: {result['final_recall']:.4f}")
    print(f"  最终F1分数: {result['final_f1']:.4f}")

# 保存模型
torch.save(googlenet.state_dict(), 'mnist_googlenet_model.pth')
torch.save(resnet.state_dict(), 'mnist_resnet18_model.pth')
print("\n模型已保存: mnist_googlenet_model.pth, mnist_resnet18_model.pth")