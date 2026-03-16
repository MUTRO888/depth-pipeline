# 浮雕深度估计管线 (Depth Pipeline)

将任意输入图片通过 AI 深度估计，自动生成适合大理石雕刻机使用的浮雕灰度图（BMP）。

## 运行环境要求

| 项目 | 配置 |
|------|------|
| OS | Windows 11 |
| GPU | NVIDIA GPU，显存 ≥ 4GB，支持 CUDA |
| Python | 3.10+ |

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/sonianmu/depth-pipeline.git
cd depth-pipeline
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. 安装 PyTorch（CUDA 版本）

根据你的 CUDA 版本，前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取安装命令。例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. 安装其他依赖

```bash
pip install -r requirements.txt
```

> **注意：** 首次运行时，程序会自动从 Hugging Face 下载 Marigold 深度估计模型（约 1.5GB），下载后缓存到本地，后续可离线使用。

## 运行方式

```bash
python app.py
```

启动后出现图形界面，操作步骤：

1. **上传图片** — 点击上传区域选择 JPG/PNG 图片（或拖拽文件到窗口）
2. **开始处理** — 点击"开始处理"按钮，等待深度估计完成
3. **保存结果** — 查看灰度图预览，确认导出路径后点击"保存文件"

输出文件为 8bit 灰度 BMP，可直接导入 JDPaint 使用。

## 目录结构

```
├── app.py                   # GUI 入口
├── pipeline.py              # 深度估计主流程
├── steps/
│   └── depth_estimation.py  # 深度估计模块（Marigold）
├── utils/
│   ├── image_io.py          # 图片读写
│   └── postprocess.py       # 后处理（归一化、反转、平滑）
├── config.yaml              # 参数配置
├── requirements.txt         # 依赖清单
└── README.md                # 本文件
```

## 配置说明

编辑 `config.yaml` 可调整模型参数：

```yaml
model:
  name: "prs-eth/marigold-depth-v1-1"  # 模型名称
  torch_dtype: "float16"               # 精度（4GB 显存必须用 fp16）
  device: "cuda"                        # 推理设备

inference:
  ensemble_size: 5                      # 集成次数（显存不足时自动降为 1）
  denoising_steps: 10                   # 去噪步数

postprocess:
  gaussian_radius: 2                    # 高斯平滑半径
```

若网络受限无法自动下载模型，可手动下载后将 `model.name` 改为本地目录路径。
