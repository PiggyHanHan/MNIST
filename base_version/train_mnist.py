import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import random
import torchvision.transforms.functional as F

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

# 3. 数据预处理和加载
transform_train = transforms.Compose([
    transforms.ToTensor(),                     # 先转为张量（值范围0-1）
    transforms.Lambda(lambda x: 1 - x if random.random() < 0.5 else x),  # 随机反转（0-1之间反转：1-x）
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化
])

# 测试集不做反转，保持原样（或也做反转，但一般只对训练集做数据增强）
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 4. 初始化模型、损失函数、优化器
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练函数
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

# 6. 测试函数
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

# 7. 开始训练
if __name__ == '__main__':
    epochs = 5  # 训练 5 轮
    for epoch in range(1, epochs + 1):
        start = time.time()
        train(epoch)
        acc = test()
        end = time.time()
        print(f'Epoch {epoch} 用时: {end - start:.2f} 秒\n')

    # 8. 保存模型
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("模型已保存为 mnist_cnn.pth")

class RandomInvert(object):
    """以一定概率随机反转灰度图像（模拟颜色反转）"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # 反转灰度值：255 - 像素值
            return F.invert(img)
        return img