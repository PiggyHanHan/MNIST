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

# ----- 配置日志 -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----- 1. 定义模型结构 -----
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
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


# ----- 2. 加载模型 -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('mnist_cnn_color.pth', map_location=device))
    model.eval()
    logger.info(f"模型加载成功，设备：{device}")
except Exception as e:
    logger.error(f"模型加载失败：{e}")
    raise e

# ----- 3. 基础预处理（resize + 标准化）-----
base_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ----- 4. 辅助函数：自适应裁剪（用OpenCV找到最大数字区域）-----
def adaptive_crop(image_pil):
    """
    输入：PIL Image（RGB）
    输出：裁剪并放大的PIL Image（28x28），如果找不到区域则返回原图resize
    """
    # 转成OpenCV格式（灰度）
    img_cv = np.array(image_pil.convert('L'))  # 灰度图
    # 二值化（自适应阈值）
    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_pil  # 没找到轮廓，返回原图

    # 取最大轮廓（按面积）
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)

    # 扩大一点边界（比如加10像素）
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img_cv.shape[1] - x, w + 2 * pad)
    h = min(img_cv.shape[0] - y, h + 2 * pad)

    # 裁剪
    cropped_pil = image_pil.crop((x, y, x + w, y + h))
    return cropped_pil


# ----- 5. 预测函数（可选是否自适应裁剪）-----
def predict_image(image_bytes, use_adaptive_crop=False):
    try:
        # 打开图片并转为 RGB
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 如果需要自适应裁剪
        if use_adaptive_crop:
            img = adaptive_crop(img)
            logger.info("使用了自适应裁剪")

        # 预处理
        input_tensor = base_transform(img).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top2 = torch.topk(probs, 2)
            pred = output.argmax(dim=1).item()
            top2_indices = top2.indices[0].tolist()
            top2_values = top2.values[0].tolist()

        return {
            "prediction": pred,
            "confidence": top2_values[0],
            "top2": [
                {"class": top2_indices[0], "prob": top2_values[0]},
                {"class": top2_indices[1], "prob": top2_values[1]}
            ],
            "used_adaptive_crop": use_adaptive_crop
        }
    except Exception as e:
        logger.error(f"预测失败：{e}")
        raise HTTPException(status_code=400, detail=f"图片处理失败：{str(e)}")


# ----- 6. 创建 FastAPI 应用 -----
app = FastAPI(title="MNIST 彩色数字识别API（增强版）")

# 允许跨域（方便前端直接调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...), adaptive: bool = False):
    """
    上传图片，返回识别结果。
    adaptive 参数：是否启用自适应裁剪（默认False）
    """
    contents = await file.read()
    result = predict_image(contents, use_adaptive_crop=adaptive)
    return JSONResponse(content=result)


@app.get("/")
async def root():
    return {"message": "MNIST 彩色数字识别服务已启动，请POST图片到 /predict 接口，可选参数 adaptive=true 启用裁剪"}


# 启动（直接运行）
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)