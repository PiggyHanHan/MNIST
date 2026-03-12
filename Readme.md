# MNIST 手写数字识别项目（扩展至 EMNIST 47类）

## 📌 项目概述
本项目基于 PyTorch 实现了卷积神经网络（CNN），最初用于识别 MNIST 手写数字（10类）。后续扩展至 **EMNIST Balanced 数据集（47类：数字+大小写字母）**，并通过融合 **MNIST** 和 **EMNIST** 数据训练出基础模型（测试准确率 93.5%）。为进一步提升对用户真实手写图片的识别能力，项目还利用 **用户自行收集的本地手写字母图片** 进行了微调，使模型在字母上的表现显著改善（部分字母准确率从几乎全错提升至 60%~90%）。下一步计划引入 **中国风格手写数据集**（如中科院自动化研究所手写数据集），以增强模型对中国用户手写习惯的泛化能力，最终部署为公开服务。

## 🚀 环境配置

### 1. 安装 Miniconda 与 PyCharm
- 下载并安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（Python 3 版本）。
- 安装 [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/) 作为 IDE。

### 2. 创建虚拟环境
在 PyCharm 中新建项目，选择 **Conda** 作为新环境，Python 版本选用 **3.12**（PyTorch 2.10 最高支持到 3.12，避免使用 3.13 导致兼容性问题）。  
环境名称可设为 `MNIST`。

### 3. 安装 PyTorch 及相关库
在 PyCharm 的 Terminal（确保环境激活）中执行以下命令（以 RTX 5060 显卡为例，需 CUDA 12.8 驱动 ≥570.00）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib pillow pandas opencv-python
```

### 4. 验证 GPU 可用性
```python
import torch
print(torch.cuda.is_available())          # 应输出 True
print(torch.cuda.get_device_name(0))      # 显示 GPU 型号
```

## 📂 数据准备与模型演进

### 第一阶段：仅 MNIST（基础数字识别）
- **数据**：MNIST 手写数字（6万张训练图，1万张测试图）。
- **模型**：简单 CNN（两层卷积 + 全连接），测试准确率 >99%。
- **脚本**：`train_mnist.py`、`batch_predict.py`。
- **问题**：无法识别字母，且对非常规手写风格泛化能力有限。

### 第二阶段：融合 EMNIST（47类数字+字母）
- **数据**：EMNIST Balanced（112,800 张训练图，18,800 张测试图）+ MNIST（额外增加数字样本）。
- **模型**：改进型 CNN（三层卷积 + BN + Dropout），测试准确率 **93.5%**。
- **脚本**：`train_emnist_enhanced.py`、`batch_predict_enhanced.py`、`api_color.py`（FastAPI 服务）。
- **效果**：数字识别几乎完美，但字母（尤其是手写体）识别率极低（多数错为数字）。

### 第三阶段：利用本地手写图片微调（当前状态）
- **数据**：在原有 EMNIST+MNIST 基础上，加入用户收集的 **133 张手写字母图片**（含 `handwritten_*` 和 `print_*` 系列），覆盖 A~F 字母（大写）。
- **操作**：用微调脚本 `finetune_emnist.py` 以低学习率（1e-4）训练 10 个 epoch，得到新模型 `emnist_cnn_finetuned.pth`。
- **效果**：字母识别率大幅提升（部分字母准确率从 0% 升至 60%~90%），但仍有混淆（如 A 常被误认为 4，C 与 0 混淆）。数字识别能力保持稳定。

### 第四阶段（计划）：引入中国风格手写数据集
- **目标**：替换或补充本地数据集，使模型真正适应中国大众的手写习惯，适合公开部署。
- **候选数据集**：
  - **中科院自动化研究所手写数据集**（约390万单字，含数字、英文字母），下载地址：http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html
  - **华南理工大学手写数据集(SCU-EPT Dataset)**（含真实试卷背景），需自行切分单字。
- **操作**：筛选出与 EMNIST 重合的 47 类，与原有数据混合训练，生成通用性更强的模型。
- **预期**：模型对中国用户的手写字母识别准确率显著提升，同时保持数字识别能力。

## 🧠 模型结构（当前）
采用改进型 CNN（与第二阶段相同）：
```
Conv2d(1,32) + BN + ReLU + MaxPool
Conv2d(32,64) + BN + ReLU + MaxPool
Conv2d(64,128) + BN + ReLU + MaxPool
Dropout(0.25) + Linear(128*3*3, 256) + ReLU
Linear(256, 47)
```
输入为 28×28 灰度图（黑底白字），输出 47 类（索引 0~46 对应 EMNIST 的字符列表）。

## 🔧 核心脚本说明

| 脚本文件 | 功能 | 备注 |
|----------|------|------|
| `train_emnist_enhanced.py` | 训练 EMNIST+MNIST 基础模型 | 输出 `emnist_cnn_balanced.pth` 和 `emnist_classes.pkl` |
| `batch_predict_enhanced.py` | 批量预测本地图片（支持二值化开关） | 用于测试模型效果 |
| `api_color.py` | FastAPI 服务（47类识别） | 部署到服务器，Go 后端调用 |
| `finetune_emnist.py` | 用本地手写图片微调模型 | 需准备 `finetune_imgs/` 文件夹及 `labels_my.csv` |
| `train_mnist.py` / `batch_predict.py` | 原数字版训练/预测脚本 | 保留作对比 |

## 🗄️ 服务器部署

### Python API 服务（uvicorn）
```bash
nohup env /root/MNIST/venv312/bin/uvicorn api_color:app --host 0.0.0.0 --port 8000 >uvicorn.log 2>&1 &
```

### Go 后端调用
```bash
# 不带 adaptive 参数（默认 True，但新版 API 内部已忽略裁剪）
nohup env BIND_ADDR=:8080 PY_API_URL=http://127.0.0.1:8000/predict FRONTEND_DIR=/root/MNIST/FrontEnd ./mnist-backend >backend.log 2>&1 &
```

## 📈 当前效果与后续计划

### 当前测试结果（基于微调后模型）
- **数字图片**：几乎全部正确，置信度普遍 >0.99。
- **字母图片**：部分正确，但仍有以下典型错误：
  - A ↔ 4 混淆（尤其手写 A）
  - C ↔ 0 混淆
  - 部分字母被误认为形状相似的数字
- **根本原因**：本地样本量少（仅133张），且用户个人手写风格不具广泛代表性。

### 下一步行动
1. **下载并处理中国风格手写数据集**（首选 CASIA-HWDB），筛选出与 EMNIST 相同的 47 类字符。
2. **重新训练/微调模型**：将新数据集与 EMNIST+MNIST 混合，使用更大 batch size 和适当学习率，训练更多轮次。
3. **评估与部署**：在新数据集测试集上验证，若效果理想则替换服务器模型；否则继续调整数据比例或增强策略。

## 📁 项目文件结构（最新）
```
MNIST/
├── train_emnist_enhanced.py          # EMNIST+MNIST 训练脚本（基础模型）
├── finetune_emnist.py                 # 本地数据微调脚本
├── batch_predict_enhanced.py          # 批量预测（47类）
├── api_color.py                        # FastAPI 服务（47类）
├── emnist_cnn_balanced.pth             # 基础模型（93.5%）
├── emnist_cnn_finetuned.pth            # 微调后模型（当前使用）
├── emnist_classes.pkl                   # 47类字符映射
├── finetune_imgs/                       # 本地手写图片（用于微调）
│   ├── handwritten_A_1.jpg
│   ├── ...
│   └── labels_my.csv                    # 标签文件（两列：filename,label）
├── test_imgs/                            # 通用测试图片
├── train_mnist.py                        # 原数字版训练脚本（保留）
├── batch_predict.py                       # 原数字版预测脚本
├── mnist_cnn.pth                          # 原数字版模型
└── README.md                              # 本文件
```

## 🧩 常见问题与解决

- **CSV 编码问题**：Windows 系统生成的 CSV 常为 GBK 编码，需在 pandas 读取时指定 `encoding='gbk'`，或另存为 UTF-8。
- **DataLoader 多进程错误**：Windows 下需设置 `num_workers=0` 并将主代码放入 `if __name__ == '__main__':`。
- **颜色反转**：预测时已固定为黑底白字（通过 Otsu 二值化 + 角点判断），训练时未使用 `RandomInvert` 增强，确保一致性。

## 🎯 结语
项目从简单的 MNIST 数字识别，逐步演进到能识别 47 类字符的通用手写识别系统，并通过微调初步适应了个人手写风格。下一步引入大规模中国风格数据集后，将真正成为一个适用于中国用户群体的手写数字字母识别服务。

欢迎继续关注项目进展！任何问题请提交 Issue 或联系维护者。

---

**Happy Learning! 🚀**