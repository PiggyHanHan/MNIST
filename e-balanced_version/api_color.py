import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import logging
import cv2
import numpy as np
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ----- 配置日志 -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- 1. 定义模型结构（与训练一致）-----
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
MODEL_PATH = 'emnist_cnn_balanced.pth'
CLASSES_PATH = 'emnist_classes.pkl'

try:
    model = ImprovedCNN(num_classes=47).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logger.info(f"EMNIST模型加载成功，设备：{device}")

    with open(CLASSES_PATH, 'rb') as f:
        classes = pickle.load(f)   # list, index -> char
    logger.info(f"类别映射加载成功，共 {len(classes)} 类")
except Exception as e:
    logger.error(f"加载失败：{e}")
    raise e

# ----- 3. 预处理（基础部分）-----
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----- 4. 辅助函数：将彩色图片转换为黑底白字灰度图 -----
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

# ----- 5. 改进的自适应裁剪（无额外边距，使字符填满图像）-----
def adaptive_crop(image_pil):
    """
    输入：二值化后的 PIL Image（黑底白字）
    输出：裁剪后紧贴字符的图像（不加边距），若找不到轮廓则返回原图
    """
    img_np = np.array(image_pil)
    contours, _ = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_pil  # 没有字符，返回原图
    # 取最大轮廓（假设是字符）
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    # 不加边距，直接裁剪
    cropped = image_pil.crop((x, y, x + w, y + h))
    return cropped

# ----- 6. 预测函数（可选自适应裁剪）-----
def predict_image(image_bytes, use_adaptive_crop=False):
    try:
        img_color = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_gray = convert_to_mnist_style(img_color)

        if use_adaptive_crop:
            img_gray = adaptive_crop(img_gray)
            logger.info("使用了自适应裁剪")

        # 统一缩放至28x28（自适应裁剪后的图像会被拉伸填满，符合训练数据分布）
        img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)

        input_tensor = base_transform(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top2 = torch.topk(probs, 2)
            pred_idx = output.argmax(dim=1).item()
            pred_char = classes[pred_idx]
            top2_indices = top2.indices[0].tolist()
            top2_values = top2.values[0].tolist()

        return {
            "prediction": pred_char,
            "confidence": top2_values[0],
            "top2": [
                {"class": classes[top2_indices[0]], "prob": top2_values[0]},
                {"class": classes[top2_indices[1]], "prob": top2_values[1]}
            ],
            "used_adaptive_crop": use_adaptive_crop
        }
    except Exception as e:
        logger.error(f"预测失败：{e}")
        raise HTTPException(status_code=400, detail=f"图片处理失败：{str(e)}")

# ----- 7. 创建 FastAPI 应用 -----
app = FastAPI(title="EMNIST 数字+字母识别API (47类，带自适应裁剪)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...), adaptive: bool = True):
    """
    上传图片，返回识别结果。
    adaptive=True 时启用自适应裁剪（推荐用于非居中的图片）。
    """
    contents = await file.read()
    result = predict_image(contents, use_adaptive_crop=adaptive)
    return JSONResponse(content=result)

@app.get("/")
async def root():
    return {"message": "服务已启动，请POST图片到 /predict 接口，adaptive参数控制是否裁剪"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)