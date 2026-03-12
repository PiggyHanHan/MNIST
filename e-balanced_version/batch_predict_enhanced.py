import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
import pickle

# ----- 1. 定义模型结构（必须与训练时完全一致）-----
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

# ----- 2. 加载模型和类别映射 -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedCNN(num_classes=47).to(device)
model.load_state_dict(torch.load('emnist_cnn_finetuned.pth', map_location=device))
model.eval()
print("EMNIST Balanced 模型加载成功！")

with open('emnist_classes.pkl', 'rb') as f:
    classes = pickle.load(f)          # 列表，索引→字符
print(f"类别映射加载成功，共 {len(classes)} 类")

# ----- 3. 图片预处理函数（简化：不再自适应裁剪）-----
def convert_to_mnist_style(image_pil):
    """二值化并确保黑底白字"""
    gray = image_pil.convert('L')
    img_np = np.array(gray)
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 根据四个角判断背景颜色（假设背景占多数）
    corners = [binary[0, 0], binary[0, -1], binary[-1, 0], binary[-1, -1]]
    bg_color = np.mean(corners)
    if bg_color > 127:   # 背景为白色，则反转
        binary = 255 - binary
    return Image.fromarray(binary).convert('L')

def preprocess_for_model(image_pil):
    """完整预处理流水线：二值化 + 直接缩放至28x28"""
    img_binary = convert_to_mnist_style(image_pil)
    img_resized = img_binary.resize((28, 28), Image.Resampling.LANCZOS)
    return img_resized

# ----- 4. 最终变换（转为Tensor并归一化）-----
final_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----- 5. 批量预测 -----
image_folder = 'test_imgs'
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
image_files.sort()

print(f"找到 {len(image_files)} 张图片，开始识别...\n")

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    try:
        # 读取原图
        img_color = Image.open(img_path).convert('RGB')
        # 预处理
        img_processed = preprocess_for_model(img_color)
        input_tensor = final_transform(img_processed).unsqueeze(0).to(device)

        # （可选）保存预处理后的图像用于调试
        # img_processed.save(f"debug_{img_file}")

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top2 = torch.topk(probs, 2)
            pred_idx = output.argmax(dim=1).item()
            pred_char = classes[pred_idx]
            top2_indices = top2.indices[0].tolist()
            top2_values = top2.values[0].tolist()

        print(f"{img_file} -> 预测字符: {pred_char}  (置信度: {top2_values[0]:.4f})")
        print(f"    备选: {classes[top2_indices[1]]} 概率 {top2_values[1]:.4f}\n")

    except Exception as e:
        print(f"❌ {img_file} 处理失败: {e}\n")

print("识别完成！")