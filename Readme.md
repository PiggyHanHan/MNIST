# MNIST 手写数字识别项目总结（彩色版）

## 📌 项目概述
本项目实现了基于 PyTorch 的卷积神经网络（CNN）来识别 MNIST 手写数字数据集，并进一步扩展为**支持任意前景/背景颜色的彩色数字识别**。通过引入 EMNIST 数据集、强数据增强以及颜色抖动，模型能够对红底蓝字、黄底绿字、黑白反转等任意颜色组合的数字进行准确分类。

项目包含完整的训练、评估和批量预测流程，适合深度学习入门到进阶实践。

## 🚀 环境配置

### 1. 安装 Miniconda 与 PyCharm
- 下载并安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（Python 3 版本）。
- 安装 [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/) 作为 IDE。

### 2. 创建虚拟环境
在 PyCharm 中新建项目，选择 **Conda** 作为新环境，Python 版本选用 **3.12**（PyTorch 2.10 最高支持到 3.12，避免使用 3.13 导致兼容性问题）。  
环境名称可设为 `MNIST`，位置可自定义。

### 3. 安装 PyTorch 及相关库
在 PyCharm 的 Terminal（确保环境激活）中执行以下命令（以 RTX 5060 显卡为例，需 CUDA 12.8 驱动 ≥570.00）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib pillow opencv-python
```
> **注意**：本项目使用了 `transforms.ElasticTransform` 和 `transforms.ColorJitter`，要求 **torchvision ≥ 0.8.0**（一般与 PyTorch 2.x 配套的版本均满足）。

### 4. 验证 GPU 可用性
```python
import torch
print(torch.cuda.is_available())          # 应输出 True
print(torch.cuda.get_device_name(0))      # 显示 GPU 型号
```

## 📂 数据准备
- **原始 MNIST**：通过 `torchvision.datasets.MNIST` 自动下载（6 万张训练图，1 万张测试图）。
- **增强数据集 EMNIST**（仅数字部分）：通过 `datasets.EMNIST(split='digits')` 自动下载，包含 **28 万张**额外的手写数字图片，风格更多样。
- 训练时使用 `ConcatDataset` 将两者合并，总训练样本数约 **34 万张**。

## 🧠 模型定义（彩色版）
为支持彩色输入，模型第一层卷积的输入通道改为 **3**（RGB），其余结构保持不变：
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    # forward 不变
```

## 🏋️ 训练过程（彩色增强版）

### 数据预处理与增强
- **训练集**：
  - 将灰度图转为 3 通道（`Grayscale(num_output_channels=3)`）
  - 随机仿射变换（旋转 ±15°，平移 ±10%，缩放 0.9~1.1）
  - 弹性变形（概率 0.3）
  - **颜色抖动**（`ColorJitter`）：随机调整亮度、对比度、饱和度、色调，模拟任意彩色数字
  - 转为张量，标准化到 `[-1, 1]`（`mean=0.5, std=0.5`）
- **测试集**：
  - 同样转为 3 通道，但只做标准化（无增强），保证评估稳定性

### 训练配置
- **损失函数**：交叉熵损失 `CrossEntropyLoss`
- **优化器**：Adam (`lr=0.001`)
- **批量大小**：64
- **训练轮次**：12（数据量大，充分收敛）
- **模型保存**：`mnist_cnn_color.pth`

训练过程中每 100 个 batch 打印一次 loss，每个 epoch 结束后在 MNIST 测试集上评估准确率（通常可达 99% 以上）。

## ✍️ 对手写数字图片进行预测

### 预处理关键点（彩色）
预测时，图片应直接以 RGB 模式读取，预处理与测试集一致：
```python
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```
- 无需手动转灰度，因为模型输入是 3 通道。
- 无需手动颜色反转，因为训练时已通过 `ColorJitter` 学习了对颜色的鲁棒性。

### 批量预测脚本
项目提供 `batch_predict_color.py`，功能：
- 自动遍历 `test_imgs/` 文件夹下的常见图片格式（.png, .jpg, .jpeg, .bmp, .tiff）
- 对每张图片进行预处理、模型推理，输出预测数字及前两个候选概率
- 异常处理：某张图片失败不影响后续处理

### 自适应裁剪（可选）
如果图片中数字很小或位置偏移，推荐使用 `batch_predict_adaptive.py`（需单独编写），该脚本会先通过轮廓检测自动定位数字区域，裁剪放大后再送入模型，可大幅提升对小数字的识别准确率。

## 📁 项目文件结构
```
MNIST/
├── train_mnist_color.py           # 彩色模型训练脚本（MNIST+EMNIST+颜色增强）
├── batch_predict_color.py          # 彩色模型批量预测脚本
├── mnist_cnn_color.pth             # 训练好的彩色模型参数
├── test_imgs/                       # 存放待预测的图片（任意颜色、格式）
├── requirements.txt                 # 依赖包列表（可选）
└── README.md                        # 本文件
```

## 🧩 常见问题与解决方法

### 1. 模型无法识别某些彩色数字
- **原因**：训练时颜色抖动范围可能不够覆盖某些极端颜色组合。
- **解决**：可调整 `ColorJitter` 参数（如增大 `hue` 范围）后重新训练，或收集特定颜色样本微调。

### 2. 图片中数字太小导致误判
- **现象**：6 被认成 0，9 被认成 4 等。
- **解决**：使用自适应裁剪预处理（先检测数字区域再缩放），或训练时增强尺度变化（将 `scale` 范围扩大到 `(0.5, 1.5)`）。

### 3. 模型加载时提示 `size mismatch`
- **原因**：模型结构与权重文件不匹配（如第一层通道数应为 3 但加载了旧版 1 通道权重）。
- **解决**：确保使用正确的训练脚本生成权重，且预测脚本中的模型定义与训练时完全一致。

### 4. PyCharm Terminal 激活环境报错
- **解决**：以管理员身份运行 PowerShell，执行 `Set-ExecutionPolicy RemoteSigned` 和 `conda init powershell`，重启 PyCharm。

### 5. 环境迁移
- 使用 `conda env export > environment.yml` 导出依赖，然后在新路径下用 `conda env create -f environment.yml` 重建。

## 🎯 总结
本项目从经典 MNIST 出发，逐步进阶到支持任意颜色的彩色数字识别，完整实践了深度学习项目的核心流程：
- 数据准备与增强
- 模型设计与修改（输入通道、颜色抖动）
- 训练、评估与调优
- 模型部署与批量预测

通过本项目，你已掌握：
- PyTorch 基础操作与 CNN 构建
- 数据增强（仿射、弹性变形、颜色抖动）
- 处理真实场景图片的预处理技巧
- 模型泛化能力分析与改进

下一步可挑战更复杂的任务，如多数字识别（CRNN+CTC）、实时视频流识别（目标检测+分类）、模型轻量化部署（ONNX/TensorRT）等。

---

**Happy Learning! 🚀**