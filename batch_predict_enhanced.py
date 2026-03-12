import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
import pickle

# ----- 1. 定义模型结构（输出47类）-----
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 47)          # 47类
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----- 2. 加载模型和类别映射 -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('emnist_cnn_balanced.pth', map_location=device))
model.eval()
print("EMNIST Balanced 模型加载成功！")

with open('emnist_classes.pkl', 'rb') as f:
    classes = pickle.load(f)          # 列表，索引→字符
print(f"类别映射加载成功，共 {len(classes)} 类")

# ----- 3. 图片预处理（与训练一致）-----
base_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----- 4. 转换函数（与原代码相同）-----
def convert_to_mnist_style(image_pil):
    gray = image_pil.convert('L')
    img_np = np.array(gray)
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    corners = [binary[0, 0], binary[0, -1], binary[-1, 0], binary[-1, -1]]
    bg_color = np.mean(corners)
    if bg_color > 127:
        binary = 255 - binary
    result = Image.fromarray(binary).convert('L')
    return result

# ----- 5. 批量预测 -----
image_folder = 'test_imgs'
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
image_files.sort()

print(f"找到 {len(image_files)} 张图片，开始识别...\n")

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    try:
        img_color = Image.open(img_path).convert('RGB')
        img_gray = convert_to_mnist_style(img_color)
        input_tensor = base_transform(img_gray).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top2 = torch.topk(probs, 2)
            pred_idx = output.argmax(dim=1).item()
            pred_char = classes[pred_idx]               # 转换为字符
            top2_indices = top2.indices[0].tolist()
            top2_values = top2.values[0].tolist()

        print(f"{img_file} -> 预测字符: {pred_char}  (置信度: {top2_values[0]:.4f})")
        print(f"    备选: {classes[top2_indices[1]]} 概率 {top2_values[1]:.4f}\n")

    except Exception as e:
        print(f"❌ {img_file} 处理失败: {e}\n")

print("识别完成！")