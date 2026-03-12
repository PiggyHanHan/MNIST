import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pickle
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image
import time

# 定义模型（与之前完全相同）
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

# 定义二值化函数（与预测脚本一致）
def convert_to_mnist_style(image_pil):
    gray = image_pil.convert('L')
    img_np = np.array(gray)
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    corners = [binary[0, 0], binary[0, -1], binary[-1, 0], binary[-1, -1]]
    bg_color = np.mean(corners)
    if bg_color > 127:
        binary = 255 - binary
    return Image.fromarray(binary).convert('L')

# 自定义数据集类
class MyHandwritingDataset(Dataset):
    def __init__(self, csv_file, img_dir, class_to_idx, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label_str = str(self.df.iloc[idx, 1]).strip().upper()
        label = self.class_to_idx[label_str]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = convert_to_mnist_style(image)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载之前训练好的模型
    model = ImprovedCNN(num_classes=47).to(device)
    model.load_state_dict(torch.load('emnist_cnn_balanced.pth', map_location=device))
    print("原有模型加载成功！")

    # 加载类别映射
    with open('emnist_classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    class_to_idx = {ch: i for i, ch in enumerate(classes)}
    print(f"类别映射加载成功，共 {len(classes)} 类")

    # 数据预处理（减弱增强）
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载原有数据集（EMNIST+MNIST）
    print("加载原有数据集（EMNIST+MNIST）...")
    emnist_train = datasets.EMNIST(
        root='./data', split='balanced', train=True,
        download=False, transform=transform_train
    )
    mnist_train = datasets.MNIST(
        root='./data', train=True, download=False, transform=transform_train
    )
    original_dataset = ConcatDataset([emnist_train, mnist_train])
    print(f"原有数据集大小: {len(original_dataset)}")

    # 加载你自己的数据集（注意修改路径为你的文件夹）
    my_dataset = MyHandwritingDataset(
        csv_file='finetune_imgs/labels_my.csv',   # CSV文件路径
        img_dir='finetune_imgs',                   # 图片文件夹路径
        class_to_idx=class_to_idx,
        transform=transform_train
    )
    print(f"你自己的数据集大小: {len(my_dataset)}")

    # 合并数据集
    combined_dataset = ConcatDataset([original_dataset, my_dataset])
    print(f"合并后总数据集大小: {len(combined_dataset)}")

    # DataLoader 设置 num_workers=0 避免多进程问题
    train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True, num_workers=0)

    # 测试集（用EMNIST+MNIST的测试集）
    test_dataset = ConcatDataset([
        datasets.EMNIST(root='./data', split='balanced', train=False, download=False, transform=transform_test),
        datasets.MNIST(root='./data', train=False, download=False, transform=transform_test)
    ])
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)

    # 设置微调参数（小学习率）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 训练函数
    def train_epoch(epoch):
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
        return avg_loss

    # 开始微调
    print("开始微调...")
    epochs = 10
    best_loss = float('inf')

    for epoch in range(1, epochs+1):
        start = time.time()
        train_epoch(epoch)
        val_loss = test()
        scheduler.step(val_loss)
        end = time.time()
        print(f'Epoch {epoch} 用时: {end-start:.2f} 秒')
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'emnist_cnn_finetuned.pth')
            print(f"*** 最佳微调模型已保存，验证损失: {best_loss:.4f} ***")
        print()

    print("微调完成！")