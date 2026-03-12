import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import time
import pickle

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 定义改进后的模型（3层卷积 + BN + Dropout）
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=47):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 3. 数据预处理（与之前相同）
transform_train = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.RandomApply([transforms.ElasticTransform(alpha=30.0)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 4. 加载 EMNIST balanced 数据集
emnist_train = datasets.EMNIST(
    root='./data', split='balanced', train=True,
    download=True, transform=transform_train
)
emnist_test = datasets.EMNIST(
    root='./data', split='balanced', train=False,
    download=True, transform=transform_test
)

print(f"EMNIST训练集样本数: {len(emnist_train)}")
print(f"EMNIST测试集样本数: {len(emnist_test)}")

# 5. 加载 MNIST 数据集，并将标签映射到 EMNIST 的类别索引
# 注意：EMNIST balanced 的前10类是数字0-9，我们通过 emnist_train.classes 确认
# 假设 emnist_train.classes[0] = '0', emnist_train.classes[1] = '1', ..., emnist_train.classes[9] = '9'
# 因此MNIST的标签可以直接使用，不需要额外映射

mnist_train = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_train
)
mnist_test = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_test
)

print(f"MNIST训练集样本数: {len(mnist_train)}")
print(f"MNIST测试集样本数: {len(mnist_test)}")

# 6. 合并训练集和测试集
train_dataset = ConcatDataset([emnist_train, mnist_train])
test_dataset = ConcatDataset([emnist_test, mnist_test])

print(f"合并后训练集总样本数: {len(train_dataset)}")
print(f"合并后测试集总样本数: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

# 7. 保存类别映射（来自EMNIST，因为MNIST的标签已包含在内）
classes = emnist_train.classes
with open('emnist_classes.pkl', 'wb') as f:
    pickle.dump(classes, f)
print("类别映射已保存至 emnist_classes.pkl")

# 8. 初始化模型、损失函数、优化器、学习率调度器
model = ImprovedCNN(num_classes=47).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# 9. 训练函数
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

# 10. 测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    print(f'测试损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%')
    return avg_loss, accuracy

# 11. 开始训练
if __name__ == '__main__':
    epochs = 60
    best_loss = float('inf')
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        start = time.time()
        train(epoch)
        val_loss, val_acc = test()
        end = time.time()
        print(f'Epoch {epoch} 用时: {end - start:.2f} 秒')
        current_lr = optimizer.param_groups[0]['lr']
        print(f'当前学习率: {current_lr:.6f}')

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            torch.save(model.state_dict(), 'emnist_cnn_balanced.pth')
            print(f'*** 最佳模型已更新，验证损失: {best_loss:.4f}, 准确率: {best_acc:.2f}% ***\n')

    print(f'训练完成。最佳验证损失: {best_loss:.4f}, 最佳准确率: {best_acc:.2f}%')