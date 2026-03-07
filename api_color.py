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


# ----- 1. 定义灰度模型结构（输入通道为1）-----
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


# ----- 2. 加载灰度模型（请确保模型文件存在）-----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'mnist_cnn_enhanced.pth'  # 你的灰度增强模型文件名
try:
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logger.info(f"灰度模型加载成功，设备：{device}")
except Exception as e:
    logger.error(f"模型加载失败：{e}")
    raise e

# ----- 3. 定义灰度图预处理（与训练时测试集一致）-----
# 注意：灰度模型训练时用的是 Normalize((0.1307,), (0.3081,))
base_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# ----- 4. 辅助函数：将彩色图片转换为适合MNIST的黑底白字灰度图 -----
def convert_to_mnist_style(image_pil):
    """
    输入：PIL Image（RGB彩色）
    输出：PIL Image（灰度图，黑底白字，数字居中且大小适当）
    步骤：
      1. 转为灰度
      2. 二值化（OTSU）得到黑白图
      3. 判断极性：如果背景是白色（像素值高），则反转，确保数字为白色
      4. 可选：自适应裁剪（如果启用）
    """
    # 转为灰度
    gray = image_pil.convert('L')
    # 转为numpy
    img_np = np.array(gray)

    # OTSU二值化（自动阈值）
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 判断背景颜色（取四个角像素平均值，简单判断）
    corners = [binary[0, 0], binary[0, -1], binary[-1, 0], binary[-1, -1]]
    bg_color = np.mean(corners)  # 接近0或255
    if bg_color > 127:  # 背景为白色，需要反转
        binary = 255 - binary  # 此时数字为白色（255），背景为黑色（0）

    # 转回PIL
    result = Image.fromarray(binary).convert('L')
    return result


def adaptive_crop(image_pil):
    """
    输入：PIL Image（灰度图，黑底白字）
    输出：裁剪并放大的PIL Image（尽可能保留数字区域），如果找不到轮廓则返回原图
    """
    img_np = np.array(image_pil)
    # 二值图像已经是黑白，直接找轮廓
    contours, _ = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_pil
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    # 加一点边距
    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img_np.shape[1] - x, w + 2 * pad)
    h = min(img_np.shape[0] - y, h + 2 * pad)
    cropped = image_pil.crop((x, y, x + w, y + h))
    return cropped


# ----- 5. 预测函数 -----
def predict_image(image_bytes, use_adaptive_crop=False):
    try:
        # 打开彩色图片
        img_color = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 转换为MNIST风格灰度图（黑底白字）
        img_gray = convert_to_mnist_style(img_color)

        # 可选自适应裁剪
        if use_adaptive_crop:
            img_gray = adaptive_crop(img_gray)
            logger.info("使用了自适应裁剪")

        # 预处理（resize + 标准化）
        input_tensor = base_transform(img_gray).unsqueeze(0).to(device)

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
app = FastAPI(title="MNIST 数字识别API（灰度增强版，支持彩色转黑白）")

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
    return {"message": "服务已启动，请POST图片到 /predict 接口，可选参数 adaptive=true 启用裁剪"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)