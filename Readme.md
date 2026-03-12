# EMNIST 手写数字与字母识别项目

## 📌 项目概述
本项目基于 PyTorch 构建卷积神经网络（CNN），使用 **EMNIST Balanced** 数据集训练模型，可识别 **47 类**手写字符，包括数字 0-9、大写字母 A-Z 以及部分小写字母（由于合并易混淆字符，实际为 47 类）。项目涵盖训练、评估、批量预测和 API 部署，可作为通用 OCR 入门实践。

**核心特点**：
- 数据集：EMNIST Balanced（约 11.28 万张训练图，47 类）
- 模型：简单 CNN（两层卷积 + 两层全连接）
- 数据增强：随机仿射、弹性变形、颜色反转
- 输出：字符（如 `'3'`、`'A'`、`'d'` 等）
- 部署：FastAPI 提供 RESTful 服务，支持自适应裁剪

---

## 🚀 环境配置

### 1. 安装 Miniconda 与 PyCharm
- 下载 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（Python 3 版本）
- 安装 [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/)

### 2. 创建虚拟环境
在 PyCharm 中新建项目，选择 **Conda** 作为新环境，Python 版本选用 **3.12**（PyTorch 2.10 最高支持到 3.12）。

### 3. 安装依赖
在 PyCharm Terminal 中执行（以 CUDA 12.8 为例）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib pillow opencv-python fastapi uvicorn python-multipart
```

### 4. 验证 GPU
```python
import torch
print(torch.cuda.is_available())          # 应输出 True
```

---

## 📂 数据准备
- **EMNIST Balanced**：通过 `torchvision.datasets.EMNIST(split='balanced')` 自动下载，训练集约 11.28 万张，测试集约 1.88 万张，图片为 28×28 灰度图。
- 数据集默认保存在项目根目录下的 `data/` 文件夹中（训练脚本会自动创建）。
- 类别说明：共有 47 个类别，包含数字 0-9 和合并后的大小写字母（例如 `'C'` 与 `'c'` 合并为一类）。具体映射可通过训练脚本生成的 `emnist_classes.pkl` 查看。

---

## 🧠 模型定义
采用简单 CNN 结构（输入 1 通道，输出 47 类）：
```
Conv2d(1, 32, 3, padding=1) → ReLU → MaxPool2d(2)
Conv2d(32, 64, 3, padding=1) → ReLU → MaxPool2d(2)
Flatten → Linear(64*7*7, 128) → ReLU → Linear(128, 47)
```

---

## 🏋️ 训练过程

### 数据增强（训练集专用）
```python
transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9,1.1))
transforms.RandomApply([transforms.ElasticTransform(alpha=30.0)], p=0.3)
transforms.RandomInvert(p=0.3)           # 颜色反转增强鲁棒性
transforms.ToTensor()
transforms.Normalize((0.5,), (0.5,))      # 归一化至 [-1,1]
```

### 训练参数
- 优化器：Adam (lr=0.001)
- 损失函数：交叉熵
- 批量大小：64
- 训练轮次：35（可根据需要调整）
- 测试集准确率：约 **88% ~ 92%**（随随机种子略有波动）

训练完成后会生成两个文件：
- `emnist_cnn_balanced.pth`：模型权重
- `emnist_classes.pkl`：类别映射文件（索引 → 字符）

### 运行训练
```bash
python train_emnist_enhanced.py
```

---

## ✍️ 对手写图片进行预测

### 预处理流程
1. 读取彩色图片（RGB）
2. 转为灰度图，OTSU 二值化，极性归一化为 **黑底白字**
3. 可选自适应裁剪（根据轮廓定位字符）
4. 缩放至 28×28，转为张量，归一化至 [-1,1]

### 批量预测脚本 `batch_predict_enhanced.py`
- 自动遍历 `test_imgs/` 文件夹中的图片（支持 .png, .jpg, .jpeg, .bmp, .tiff）
- 输出预测字符及前两个候选概率
- 需要模型文件和类别映射文件在同一目录

示例输出：
```
1.png -> 预测字符: 3  (置信度: 0.9985)
    备选: 8 概率 0.0012

A.png -> 预测字符: A  (置信度: 0.9762)
    备选: 4 概率 0.0123
```

运行：
```bash
python batch_predict_enhanced.py
```

---

## 🗄️ API 服务器部署

### 启动 Python API（默认端口 8000）
```bash
nohup uvicorn api_color:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
```

### 启动 Go 后端（示例，根据实际需要调整）
```bash
nohup env BIND_ADDR=:8080 PY_API_URL=http://127.0.0.1:8000/predict?adaptive=true FRONTEND_DIR=/root/EMNIST/FrontEnd ./mnist-backend > backend.log 2>&1 &
```
- `adaptive=true` 启用自适应裁剪，可根据需要修改

### API 接口
- `GET /`：服务状态
- `POST /predict`：上传图片，返回 JSON 格式识别结果
  ```json
  {
    "prediction": "A",
    "confidence": 0.995,
    "top2": [
      {"class": "A", "prob": 0.995},
      {"class": "4", "prob": 0.003}
    ],
    "used_adaptive_crop": true
  }
  ```

---

## 📁 项目文件结构
```
项目根目录/
├── train_emnist_enhanced.py          # 训练脚本（生成模型和类别映射）
├── batch_predict_enhanced.py          # 批量预测脚本
├── api_color.py                        # FastAPI 服务
├── emnist_cnn_balanced.pth             # 训练好的模型权重
├── emnist_classes.pkl                   # 类别映射文件
├── test_imgs/                           # 存放待测图片
├── data/                                 # 数据集存放目录（自动下载）
└── README.md                             # 本文档
```

---

## ❓ 常见问题

### 1. 模型对某些字母识别不准怎么办？
- 收集自己手写的该类样本（几十张），使用 `ConcatDataset` 合并到训练集中微调即可。

### 2. 图片背景不是纯黑/白会影响结果吗？
- 预处理中的 OTSU 二值化和极性归一化会强制转换为黑底白字，因此背景颜色不影响。但若图片对比度极低，二值化可能失败，建议保证字符清晰。

### 3. 类别映射文件 (`emnist_classes.pkl`) 有什么作用？
- 训练时从数据集自动生成，保存了索引到实际字符的映射。预测时必须保证该文件存在且与模型匹配，否则输出将是数字索引而非字符。

### 4. 为什么输出只有 47 类而不是 62 类？
- EMNIST Balanced 将容易混淆的大小写字母（如 `'C'`/`'c'`）合并为一类，以降低分类难度，更适合实际应用。

### 5. 如何调整训练轮次或学习率？
- 直接修改 `train_emnist_enhanced.py` 中的 `epochs` 变量或 `optimizer` 的学习率参数。

### 6. 训练时数据集下载到哪里？
- 默认下载到项目根目录下的 `data/` 文件夹。如需更改路径，可修改 `root` 参数。

---

## 🎯 总结
本项目从 MNIST 数字识别扩展为 EMNIST 字母数字识别，展示了如何利用公开数据集和简单 CNN 实现多字符分类。通过数据增强和合理的预处理，模型对真实手写样本具有较好的泛化能力。后续可尝试更深的网络或迁移学习以进一步提升准确率。

**Happy Learning! 🚀**