import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import io
import logging
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pickle

# ----- 配置日志 -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- 1. 定义灰度模型结构（输出47类）-----
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
MODEL_PATH = 'emnist_cnn_balanced.pth'
CLASSES_PATH = 'emnist_classes.pkl'

try:
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logger.info(f"EMNIST模型加载成功，设备：{device}")

    with open(CLASSES_PATH, 'rb') as f:
        classes = pickle.load(f)
    logger.info(f"类别映射加载成功，共 {len(classes)} 类")
except Exception as e:
    logger.error(f"加载失败：{e}")
    raise e

# ----- 3. 定义预处理（与训练一致）-----
base_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----- 4. 辅助函数：将彩色图片转换为黑底白字灰度图（与原代码相同）-----
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

def adaptive_crop(image_pil):
    img_np = np.array(image_pil)
    contours, _ = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_pil
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img_np.shape[1] - x, w + 2 * pad)
    h = min(img_np.shape[0] - y, h + 2 * pad)
    cropped = image_pil.crop((x, y, x + w, y + h))
    return cropped

# ----- 5. 预测函数（返回字符）-----
def predict_image(image_bytes, use_adaptive_crop=False):
    try:
        img_color = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_gray = convert_to_mnist_style(img_color)

        if use_adaptive_crop:
            img_gray = adaptive_crop(img_gray)
            logger.info("使用了自适应裁剪")

        input_tensor = base_transform(img_gray).unsqueeze(0).to(device)

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

# ----- 6. 创建 FastAPI 应用 -----
app = FastAPI(title="EMNIST 数字+字母识别API (47类)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...), adaptive: bool = True):
    contents = await file.read()
    result = predict_image(contents, use_adaptive_crop=adaptive)
    return JSONResponse(content=result)

@app.get("/")
async def root():
    return {"message": "服务已启动，请POST图片到 /predict 接口，可选参数 adaptive=true/false 启用/关闭裁剪"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)