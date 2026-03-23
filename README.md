# 浮雕深度估计管线 (Depth Pipeline)

将任意输入图片通过 AI 深度估计，自动生成适合大理石雕刻机使用的浮雕灰度图（BMP）。

## 运行环境要求

| 项目 | 配置 |
|------|------|
| OS | Windows 11 |
| GPU | NVIDIA GPU，显存 ≥ 4GB，支持 CUDA |
| Python | 3.10+ |

## 快速开始（Windows，全效果版）

打开 PowerShell，依次执行：

```powershell
git lfs install
git clone https://github.com/MUTRO888/depth-pipeline.git
cd depth-pipeline
.\install_and_run_cn_full.bat
```

这个流程会：

1. 从 GitHub 拉取项目和仓库中的 LoRA 文件
2. 通过清华 PyPI 镜像安装依赖
3. 通过 `HF-Mirror` 下载全部 Hugging Face 模型
4. 直接按全效果配置启动程序

之后再次运行只需：

```powershell
cd depth-pipeline
.\install_and_run_cn_full.bat
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

## Windows 离线包（Mac 下载，Windows 直接运行）

如果你的 **Windows 电脑不能访问外网**，不要在 Mac 上尝试“安装 Windows 依赖”。正确做法是：

1. 在 Mac 上 **下载 Windows 专用 wheel 和模型文件**
2. 将它们和项目源码一起整理进 **一个可搬运文件夹**
3. 把这个文件夹拷到 Windows
4. 在 Windows 里执行 **一条命令** 完成本地安装并启动

### 1. 在 Mac 上生成离线包

在项目根目录执行：

```bash
python3 scripts/build_portable_folder.py --python-version 310
```

生成结果默认在：

```text
dist/depth-pipeline-offline-win/
```

这个目录里会包含：

- 项目源码
- `wheels/`：Windows `win_amd64` + Python 3.10 的离线依赖包
- `models/`：已经下载好的模型目录
- `config.offline.yaml`：离线专用配置
- `install_and_run_offline.bat`：Windows 一条命令启动脚本

> 这一步使用的是 `pip download --platform win_amd64 ...`，也就是在 Mac 上下载 **给 Windows 用** 的依赖，而不是把 Mac 当前环境直接复制给 Windows。

### 2. 拷到 Windows 后怎么运行

把整个 `dist/depth-pipeline-offline-win` 文件夹复制到 Windows，然后进入该目录，执行：

```powershell
.\install_and_run_offline.bat
```

这个脚本会：

1. 查找 **Python 3.10 x64**
2. 在当前目录创建 `.venv`
3. 从本地 `wheels/` 离线安装依赖
4. 强制启用离线模式
5. 使用本地 `models/` 直接启动 `app.py`

### 3. 注意事项

- 这个离线包默认目标是 **Windows x64 + Python 3.10**
- Windows 上仍然需要你本地已经有 **Python 3.10 x64**
- GPU 驱动 / CUDA 运行环境仍然是 Windows 本机负责，不会从 Mac 自动迁移
- 如果你想连可选的 SD 浮雕增强模型也一起打包，可在 Mac 上加：

```bash
python3 scripts/build_portable_folder.py --python-version 310 --include-sd-relief
```

## Windows 国内网络全效果方案

如果 Windows 可以访问 **GitHub + 国内互联网**，推荐直接从 GitHub 拉项目，然后固定走全量高效果链路。

### GitHub 里需要准备什么

仓库里需要包含：

- `config.full.yaml`
- `install_and_run_cn_full.bat`
- `scripts/download_full_models.py`
- `models/sd-relief/lora/relief_lora_438287.safetensors`

其中 LoRA 文件体积较大，建议使用 **Git LFS** 存进仓库：

```bash
git lfs track "models/sd-relief/lora/*.safetensors"
```

本项目已经在 `.gitattributes` 中为该路径配置了 Git LFS。

注意：

- 当前仓库使用的 LoRA 是 Civitai `Depth map Lora - SD1.5`
- 对应模型 / 版本是 `392921 @ 438287`
- 建议先从 Civitai 页面手动下载 `.safetensors`，再提交到 GitHub

### Windows 侧直接运行

第一次使用时：

```powershell
git lfs install
git clone https://github.com/MUTRO888/depth-pipeline.git
cd depth-pipeline
.\install_and_run_cn_full.bat
```

这个脚本会：

1. 创建 `.venv`
2. 通过清华 PyPI 镜像安装 Python 依赖
3. 通过 `HF-Mirror` 下载全效果需要的 Hugging Face 模型
4. 从仓库中的 LoRA 文件启用完整增强链路
5. 使用 `config.full.yaml` 启动程序

### 全效果模型清单

- 深度模型：`prs-eth/marigold-depth-v1-1`
- SD 基础模型：`runwayml/stable-diffusion-v1-5`
- ControlNet：`lllyasviel/control_v11f1p_sd15_depth`
- LoRA：Civitai `392921 @ 438287`

这个方案默认就是全量增强版，不再以“最小可运行”作为目标。
