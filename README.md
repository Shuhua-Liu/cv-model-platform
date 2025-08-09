# CV Model Platform 🔥

[![CI/CD](https://github.com/yourusername/cv-model-platform/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/yourusername/cv-model-platform/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Codecov](https://codecov.io/gh/yourusername/cv-model-platform/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/cv-model-platform)

**一个统一的计算机视觉模型管理和部署平台，支持检测、分割、分类、生成等多种CV任务。**

## ✨ 核心特性

🎯 **统一接口** - 一套API适配多种CV模型（YOLO、SAM、Stable Diffusion等）  
🚀 **开箱即用** - 一条命令完成环境设置和模型发现  
🔧 **智能配置** - 自动发现本地模型，无需手动配置  
🌍 **跨平台** - 完美支持Windows和Linux系统  
⚡ **高性能** - GPU加速，批处理，模型缓存优化  
🔌 **易扩展** - 插件化架构，轻松添加新模型支持  
📡 **多接口** - REST API、Python SDK、Web界面  
🐳 **容器化** - Docker支持，云原生部署  

## 🎬 快速演示

```python
from cv_platform.client import CVPlatformClient

# 创建客户端
client = CVPlatformClient()

# 目标检测
results = client.predict("yolov8n", "image.jpg")
print(f"检测到 {len(results)} 个对象")

# Image Segmentation
masks = client.predict("sam_vit_h", "image.jpg")

# 图像生成
generated = client.predict("stable_diffusion", 
                         prompt="a beautiful sunset over mountains")
```

## 🚀 Quick Start

### Method 1: One-Click Install (Recommended)

```bash
# 1. 克隆项目
git clone https://github.com/yourusername/cv-model-platform.git
cd cv-model-platform

# 2. 一条命令完成所有设置
python scripts/setup/setup_environment.py

# 3. 立即使用
python examples/basic_usage/detection_demo.py
```

### 方法二：手动安装

```bash
# 安装依赖
pip install -r requirements.txt
pip install -e .

# 发现本地模型
python scripts/models/detect_models.py

# 启动服务
python -m cv_platform.api.rest_api
```

📖 **详细教程**: [快速开始指南](QUICKSTART.md)

## 📁 模型文件组织

平台会自动搜索以下位置的模型文件：

```
~/cv_models/  # 推荐的模型存放位置
├── detection/
│   ├── yolo/
│   │   ├── yolov8n.pt
│   │   └── yolov8s.pt
│   └── detectron2/
├── segmentation/
│   ├── sam/
│   │   ├── sam_vit_h_4b8939.pth
│   │   └── sam_vit_b_01ec64.pth
│   └── mask2former/
├── generation/
│   ├── stable_diffusion/
│   │   └── v1-5-pruned-emaonly.safetensors
│   └── flux/
├── classification/
│   └── resnet/
└── multimodal/
    └── clip/
```

## 🎯 支持的模型类型

| 类别 | 支持的模型 | 状态 |
|------|------------|------|
| **检测** | YOLOv8/v9, Detectron2, RT-DETR | ✅ |
| **分割** | SAM, Mask2Former, DeepLabV3 | ✅ |
| **分类** | ResNet, EfficientNet, ViT | ✅ |
| **生成** | Stable Diffusion, FLUX, ControlNet | ✅ |
| **多模态** | CLIP, BLIP, LLaVA | 🚧 |

完整兼容性列表：[模型兼容性矩阵](docs/compatibility_matrix.md)

## 🌐 API使用

### REST API

```bash
# 启动API服务
python -m cv_platform.api.rest_api

# 使用API
curl -X POST "http://localhost:8000/predict/yolov8n" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@test_image.jpg"
```

### Python SDK

```python
from cv_platform.client import CVPlatformClient

# 同步客户端
client = CVPlatformClient("http://localhost:8000")
result = client.predict("yolov8n", "image.jpg")

# 异步客户端
import asyncio
from cv_platform.client import AsyncCVPlatformClient

async def main():
    async with AsyncCVPlatformClient() as client:
        result = await client.predict("sam_vit_h", "image.jpg")

# 批处理客户端
from cv_platform.client import BatchCVPlatformClient

batch_client = BatchCVPlatformClient()
results = batch_client.predict_batch("yolov8n", ["img1.jpg", "img2.jpg"])
```

### Web界面

```bash
# Streamlit界面
streamlit run src/cv_platform/web/streamlit_app.py

# Gradio界面
python src/cv_platform/web/gradio_app.py
```

## ⚙️ 高级配置

### 环境变量配置

```bash
export CV_MODELS_ROOT="/custom/models/path"
export CV_MODEL_YOLOV8_PATH="/special/yolo/path.pt"
export CV_GPU_DEVICES="0,1"  # 指定GPU设备
export CV_MAX_BATCH_SIZE=8   # 批处理大小
```

### 配置文件

```yaml
# config/models.yaml
models:
  yolov8n:
    path: "/path/to/yolov8n.pt"
    device: "cuda:0"
    batch_size: 4
    confidence: 0.5
    
platform:
  api:
    host: "0.0.0.0"
    port: 8000
  cache:
    enabled: true
    max_size: "2GB"
```

## 🐳 Docker部署

```bash
# 构建镜像
docker build -t cv-model-platform .

# 运行容器
docker run -p 8000:8000 -v /path/to/models:/models cv-model-platform

# 使用Docker Compose
docker-compose up -d
```

## 🔧 开发和扩展

### 添加新模型适配器

```python
from cv_platform.adapters.base import BaseModelAdapter

class MyCustomAdapter(BaseModelAdapter):
    def load_model(self, model_path):
        # 实现模型加载逻辑
        pass
    
    def predict(self, input_data):
        # 实现预测逻辑
        pass
    
    def preprocess(self, data):
        # 实现预处理逻辑
        pass
    
    def postprocess(self, results):
        # 实现后处理逻辑
        pass

# 注册适配器
from cv_platform.adapters.registry import register_adapter
register_adapter("my_model", MyCustomAdapter)
```

### 开发插件

```python
from cv_platform.plugins.base import BasePlugin

class MyPlugin(BasePlugin):
    def initialize(self):
        # 插件初始化逻辑
        pass
    
    def process(self, data):
        # 插件处理逻辑
        pass
```

## 📊 性能基准

| 模型 | 平台 | 推理时间 | 内存使用 | GPU显存 |
|------|------|----------|----------|---------|
| YOLOv8n | RTX 3080 | 12ms | 256MB | 1.2GB |
| SAM ViT-H | RTX 3080 | 1.8s | 2.1GB | 8.5GB |
| SD 1.5 | RTX 3080 | 3.2s | 1.8GB | 6.2GB |

完整基准测试：[性能报告](docs/benchmarks.md)

## 📚 Docs

- 📖 [用户指南](docs/user_guide/)
- 🔧 [开发者文档](docs/developer_guide/)  
- 📡 [API参考](docs/api_reference/)
- 🚀 [部署指南](docs/deployment/)
- ❓ [常见问题](docs/faq.md)
- 🐛 [故障排除](docs/troubleshooting.md)

## 🤝 贡献

我们欢迎任何形式的贡献！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request
