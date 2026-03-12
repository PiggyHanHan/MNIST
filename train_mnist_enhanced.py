import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import pickle

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 定义模型（输出改为47类）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 47)          # 47类：数字+字母
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 7x7
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. 数据预处理
transform_train = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomApply([transforms.ElasticTransform(alpha=30.0)], p=0.3),
    transforms.RandomInvert(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))        # 统一归一化到[-1,1]
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 4. 加载 EMNIST Balanced 数据集
train_dataset = datasets.EMNIST(
    root='./data', split='balanced', train=True,
    download=True, transform=transform_train
)
test_dataset = datasets.EMNIST(
    root='./data', split='balanced', train=False,
    download=True, transform=transform_test
)

print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

# 5. 保存类别映射（供预测使用）
classes = train_dataset.classes          # 长度为47的列表，每个元素为字符表示
with open('emnist_classes.pkl', 'wb') as f:
    pickle.dump(classes, f)
print("类别映射已保存至 emnist_classes.pkl")

# 6. 初始化模型
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. 训练函数
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
        if batch_idx % 100 == 99:
            print(f'Epoch {epoch}, Batch {batch_idx+1}, Loss: {running_loss/100:.4f}')
            running_loss = 0.0

# 8. 测试函数
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

# 9. 开始训练
if __name__ == '__main__':
    epochs = 35
    for epoch in range(1, epochs + 1):
        start = time.time()
        train(epoch)
        acc = test()
        end = time.time()
        print(f'Epoch {epoch} 用时: {end - start:.2f} 秒\n')

    # 10. 保存模型
    torch.save(model.state_dict(), 'emnist_cnn_balanced.pth')
    print("模型已保存为 emnist_cnn_balanced.pth")