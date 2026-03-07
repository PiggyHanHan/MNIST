import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# ----- 1. 重新定义模型结构（输入通道为3） -----
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 改为3通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----- 2. 加载训练好的彩色模型 -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('mnist_cnn_color.pth', map_location=device))
model.eval()
print("彩色模型加载成功！")

# ----- 3. 定义图片预处理（彩色，与训练测试集一致） -----
transform = transforms.Compose([
    transforms.Resize((28, 28)),                 # 缩放到 28x28
    transforms.ToTensor(),                        # 转为张量，值范围 [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5],    # 标准化到 [-1,1]
                         std=[0.5, 0.5, 0.5])
])

# ----- 4. 批量预测 -----
image_folder = 'test_imgs'
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
image_files = [f for f in os.listdir(image_folder)
               if f.lower().endswith(valid_extensions)]
image_files.sort()

print(f"找到 {len(image_files)} 张图片，开始识别...\n")

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    try:
        # 打开图片并确保为 RGB 模式
        img = Image.open(img_path).convert('RGB')
        # 预处理
        input_tensor = transform(img).unsqueeze(0).to(device)

        # 预测
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top2 = torch.topk(probs, 2)
            pred = output.argmax(dim=1).item()
            top2_indices = top2.indices[0].tolist()
            top2_values = top2.values[0].tolist()

        print(f"{img_file} -> 预测数字: {pred}  (置信度: {top2_values[0]:.4f})")
        print(f"    备选: {top2_indices[1]} 概率 {top2_values[1]:.4f}\n")

    except Exception as e:
        print(f"❌ {img_file} 处理失败: {e}\n")
        continue

print("识别完成！")