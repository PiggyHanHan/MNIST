import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import time
import os
import random

# 1. 设置设备（自动使用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 定义模型（一个简单的 CNN）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 7x7
        x = x.view(x.size(0), -1)                 # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. 数据预处理（训练集：增强；测试集：仅标准化）
transform_train = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 随机旋转、平移、缩放
    transforms.RandomApply([transforms.ElasticTransform(alpha=30.0)], p=0.3),      # 随机弹性变形
    transforms.RandomInvert(p=0.3),                                                # 随机颜色反转（黑底白字↔白底黑字）
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))                                      # MNIST 标准化参数
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 4. 加载 MNIST 数据集
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

# 5. 加载 EMNIST 数据集（只取数字部分，280,000 张手写数字）
emnist_train = datasets.EMNIST(root='./data', split='digits', train=True, download=True, transform=transform_train)
# 也可以将 EMNIST 测试集加入测试，但这里保留原 MNIST 测试集便于对比
# emnist_test = datasets.EMNIST(root='./data', split='digits', train=False, download=True, transform=transform_test)

# 6. 合并训练集（MNIST + EMNIST）
train_dataset = ConcatDataset([mnist_train, emnist_train])
print(f"训练集总样本数: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=False, num_workers=2)  # 仍用原 MNIST 测试集

# 7. 初始化模型、损失函数、优化器
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 99:  # 每 100 个批次打印一次
            print(f'Epoch {epoch}, Batch {batch_idx+1}, Loss: {running_loss/100:.4f}')
            running_loss = 0.0

# 9. 测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
    accuracy = 100. * correct / total
    print(f'测试准确率: {accuracy:.2f}%')
    return accuracy

# 10. 开始训练（增加轮次，因为数据量大了）
if __name__ == '__main__':
    epochs = 12  # 从 5 轮增加到 12 轮
    for epoch in range(1, epochs + 1):
        start = time.time()
        train(epoch)
        acc = test()
        end = time.time()
        print(f'Epoch {epoch} 用时: {end - start:.2f} 秒\n')

    # 11. 保存模型（新文件名）
    torch.save(model.state_dict(), 'mnist_cnn_enhanced.pth')
    print("模型已保存为 mnist_cnn_enhanced.pth")