# MNIST 手写数字识别项目总结

## 📌 项目概述
本项目实现了基于 PyTorch 的卷积神经网络（CNN）来识别 MNIST 手写数字数据集。通过完整的训练、评估和预测流程，展示了深度学习项目的典型步骤，包括环境配置、数据加载、模型定义、训练、测试、模型保存以及对手写图片的实时预测。

## 🚀 环境配置

### 1. 安装 Miniconda 与 PyCharm
- 下载并安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（Python 3 版本）。
- 安装 [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/) 作为 IDE。

### 2. 创建虚拟环境
在 PyCharm 中新建项目，选择 **Conda** 作为新环境，Python 版本选用 **3.12**（PyTorch 2.10 最高支持到 3.12，避免使用 3.13 导致兼容性问题）。  
环境名称可设为 `MNIST`，位置可自定义（若想放在 E 盘，需在创建时指定 `--prefix`，或在后续通过导出重建方式迁移）。

### 3. 安装 PyTorch 及相关库
在 PyCharm 的 Terminal（确保环境激活）中执行以下命令（以 RTX 5060 显卡为例，需 CUDA 12.8 驱动 ≥570.00）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib pillow
```

### 4. 验证 GPU 可用性
```python
import torch
print(torch.cuda.is_available())          # 应输出 True
print(torch.cuda.get_device_name(0))      # 显示 GPU 型号
```

## 📂 数据准备
MNIST 数据集可通过 `torchvision.datasets.MNIST` 自动下载，代码中设置 `download=True` 即可。数据预处理包括：
- 转为张量 (`ToTensor`)
- 标准化 (`Normalize`)：使用 MNIST 全局均值和标准差 (0.1307, 0.3081)

## 🧠 模型定义
采用简单的卷积神经网络 `SimpleCNN`：
- 卷积层：`Conv2d(1,32,3) + ReLU + MaxPool2d(2)` → `Conv2d(32,64,3) + ReLU + MaxPool2d(2)`
- 全连接层：`Linear(64*7*7, 128) + ReLU` → `Linear(128, 10)`

## 🏋️ 训练过程
- 损失函数：交叉熵损失 `CrossEntropyLoss`
- 优化器：Adam (lr=0.001)
- 批量大小：64
- 训练轮数：5 个 epoch
- 每 100 个 batch 打印一次损失，每个 epoch 结束后在测试集上评估准确率

训练完成后保存模型参数：
```python
torch.save(model.state_dict(), 'mnist_cnn.pth')
```

## 📊 测试与评估
训练结束后，模型在测试集上准确率达到 **99% 以上**，证明模型具有良好的泛化能力。

## ✍️ 对手写数字图片进行预测

### 图片预处理关键点
由于 MNIST 训练集是**黑底白字**（背景黑色，数字白色），而用户使用画图工具通常得到的是**白底黑字**图片，因此必须在预处理中加入颜色反转：
```python
transforms.Lambda(lambda x: 1 - x)   # 在 ToTensor 之后、Normalize 之前
```

完整的预处理流程：
```python
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),          # 颜色反转（若原图为白底黑字）
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### 批量预测脚本
脚本遍历指定文件夹中的所有 PNG 图片，输出每个文件的预测数字及前两个候选概率，便于分析模型的置信度。

## 🧩 常见问题与解决方法

### 1. PyCharm Terminal 激活环境报错
- 现象：打开 Terminal 显示红色错误 `Import-Module: 未能加载指定的模块...`
- 原因：PowerShell 执行策略限制或 Conda 初始化未完成
- 解决：以管理员身份运行 PowerShell，执行 `Set-ExecutionPolicy RemoteSigned` 和 `conda init powershell`，重启 PyCharm；或直接在 Terminal 中手动执行 `conda activate MNIST`

### 2. 颜色反转导致识别错误
- 如果手写图片是白底黑字但未反转，模型会完全认错（例如所有数字都变成 3、5、8 等）
- 解决方法：在预处理中加入 `transforms.Lambda(lambda x: 1 - x)`

### 3. 图片缩放与位置
- 模型期望输入为 28×28 像素，数字居中且大小适中。若原始图片比例不当或数字偏斜，可能影响识别。
- 建议直接在 28×28 画布上绘制，或确保数字位于中心且笔画较粗。

### 4. 模型对特定手写风格的误判
- 训练数据为 MNIST 标准手写数字（美式风格），若用户书写风格差异较大（如 9 的圆圈大、竖线短），模型可能误判为 4。
- 可通过数据增强或微调使模型适应更多风格，或接受此类误差作为数据集偏差的体现。

### 5. 环境迁移
- 若需将环境从 C 盘移至其他盘，推荐使用 `conda env export > environment.yml` 导出依赖，然后在新路径下用 `conda env create -f environment.yml` 重建。
- 注意导出前需激活正确的环境，否则 yml 文件可能只包含 base 环境的基础包。

## 📁 项目文件结构
```
MNIST/
├── train_mnist.py          # 训练脚本
├── batch_predict.py        # 批量预测脚本
├── mnist_cnn.pth           # 训练好的模型参数
├── test_imgs/              # 存放待预测的手写数字图片（命名如 1.png, 2.png ...）
├── requirements.txt        # 依赖包列表（可选）
└── README.md               # 本文件
```

## 🎯 总结
本项目完整实践了深度学习从环境搭建到模型部署的全流程，核心收获包括：
- 掌握 PyTorch 基本操作和 CNN 模型构建
- 学会处理真实手写图片的预处理（颜色反转、缩放、标准化）
- 理解数据分布对模型预测的影响
- 具备独立排查环境问题和迁移项目的能力

通过此项目，你已迈入深度学习实战的大门，后续可挑战更复杂的数据集（如 CIFAR-10、Fashion-MNIST）或尝试模型优化、部署等技术。

---

**Happy Learning! 🚀**