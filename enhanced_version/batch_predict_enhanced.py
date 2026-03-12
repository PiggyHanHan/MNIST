import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np


# ----- 1. 重新定义模型结构（必须和训练时完全一样） -----
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
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ----- 2. 加载训练好的增强模型 -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('mnist_cnn_enhanced.pth', map_location=device))
model.eval()
print("增强模型加载成功！")

# ----- 3. 定义图片预处理（标准化部分）-----
# 注意：转换到黑底白字的步骤在下面的函数中完成，此处只做resize、转张量和标准化
base_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# ----- 4. 辅助函数：将彩色图片转换为黑底白字的灰度图 -----
def convert_to_mnist_style(image_pil):
    """
    输入：PIL Image（RGB彩色）
    输出：PIL Image（灰度图，黑底白字，适合灰度模型）
    """
    # 转为灰度
    gray = image_pil.convert('L')
    # 转为numpy
    img_np = np.array(gray)

    # OTSU二值化（自动阈值）
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 判断背景颜色（取四个角像素平均值）
    corners = [binary[0, 0], binary[0, -1], binary[-1, 0], binary[-1, -1]]
    bg_color = np.mean(corners)
    if bg_color > 127:  # 背景为白色，需要反转
        binary = 255 - binary  # 此时数字为白色（255），背景为黑色（0）

    # 转回PIL
    result = Image.fromarray(binary).convert('L')
    return result


# ----- 5. 批量预测 -----
image_folder = 'test_imgs'  # 修改为你的图片文件夹路径
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
image_files = [f for f in os.listdir(image_folder)
               if f.lower().endswith(valid_extensions)]
image_files.sort()

print(f"找到 {len(image_files)} 张图片，开始识别...\n")

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    try:
        # 打开彩色图片
        img_color = Image.open(img_path).convert('RGB')

        # 转换为黑底白字的灰度图
        img_gray = convert_to_mnist_style(img_color)

        # 预处理
        input_tensor = base_transform(img_gray).unsqueeze(0).to(device)

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