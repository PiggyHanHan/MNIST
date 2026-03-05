import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt   # 记得在文件开头导入

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

# ----- 2. 加载训练好的模型 -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
model.eval()
print("模型加载成功！")

# ----- 3. 定义图片预处理（必须和训练时一致） -----
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # 转成单通道灰度图
    transforms.Resize((28, 28)),                    # 缩放到 28x28
    transforms.ToTensor(),                           # 转成张量 (值范围 0~1)
    transforms.Normalize((0.1307,), (0.3081,))       # 标准化（MNIST的均值和标准差）
])

# ----- 4. 批量预测 -----
# 修改成你的图片文件夹路径
image_folder = 'test_imgs'   # 假设图片放在项目下的 test_imgs 文件夹里
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.png')]
# 按文件名排序，保证顺序（如果希望按1~9顺序）
image_files.sort(key=lambda x: int(x.split('.')[0]))  # 按数字文件名排序

print(f"找到 {len(image_files)} 张图片，开始识别...\n")

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    try:
        # 打开图片
        img = Image.open(img_path)
        # 预处理
        input_tensor = transform(img).unsqueeze(0).to(device)  # 增加 batch 维度
        # ===== 新增：可视化模型实际看到的图像 =====
        # 将tensor转回CPU并转为numpy数组（形状28x28）
        img_np = input_tensor.cpu().squeeze().numpy()
        plt.imshow(img_np, cmap='gray')
        plt.title(f'模型眼中的 {img_file} (28x28, 黑底白字)')
        plt.show()
        # =======================================
        # 预测
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)  # 转换成概率
            top2 = torch.topk(probs, 2)  # 概率最高的两个
            print(f"预测: {top2.indices[0].tolist()}, 概率: {top2.values[0].tolist()}")
            pred = output.argmax(dim=1).item()
        print(f"{img_file} -> 预测数字: {pred}")
    except Exception as e:
        print(f"{img_file} 处理失败: {e}")

print("\n识别完成！")