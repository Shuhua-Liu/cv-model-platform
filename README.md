# CV Model Platform 🔥

[![CI/CD](https://github.com/yourusername/cv-model-platform/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/yourusername/cv-model-platform/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Codecov](https://codecov.io/gh/yourusername/cv-model-platform/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/cv-model-platform)

**A unified computer vision model management and deployment platform supporting various CV tasks such as detection, segmentation, classificatgion, and generation.**

## ✨ Core Features

🎯 **Unified Interface** - A single API adapts to multiple CV models (YOLO, SAM, Stable Diffusion, etc.)  
🚀 **Out-of-the-box** - One command completes environment setup and model discovery  
🔧 **Intelligent Configuration** - Automatically discovers local models without manual configuration  
🌍 **Cross-platform** - Perfect support for Windows and Linux systems  
⚡ **High Performance** - GPU acceleration, batch processing, model cache optimization  
🔌 **Easy Extensibility** - Plugin architecture, easily add new model support  
📡 **Multiple Interfaces** - REST API、Python SDK、Web Interface  
🐳 **Containerization** - Docker support, cloud-native deployment  

## 🎬 Quick Demo

```python
from cv_platform.client import CVPlatformClient

# Create client
client = CVPlatformClient()

# Object detection
results = client.predict("yolov8n", "image.jpg")
print(f"Detected {len(results)} objects")

# Image segmentation
masks = client.predict("sam_vit_h", "image.jpg")

# Image generation
generated = client.predict("stable_diffusion",
                           prompt="a beautiful sunset over mountains")
```

## 🚀 Quick Start

### Method 1: One-Click Install (Recommended)

```bash
# 1. Clone the project
git clone https://github.com/Shuhua-Liu/cv-model-platform.git
cd cv-model-platform

# 2. Complete all setup with one command
python scripts/setup/setup_environment.py

# 3. Use immediately
python examples/basic_usage/detection_demo.py
```

### Method 2: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Discover local models
python scripts/models/detect_models.py

# Start service
python -m cv_platform.api.rest_api
```

📖 **Detailed tutorial**: [Quick Start Guide](QUICKSTART.md)

## 📁 Model File Organization

The platform automatically searches for model files in the following locations:

```
~/cv_models/  # Recommended model storage location
├── detection/
│   ├── yolo/
│   │   ├──v8/
│   │   │   ├── yolov8n.pt
│   │   │   └── yolov8s.pt
│   │   └──v11/
│   └── detectron2/
├── segmentation/
│   ├── sam/
│   │   ├── sam_vit_h_4b8939.pth
│   │   └── sam_vit_b_01ec64.pth
│   └── mask2former/
├── generation/
│   ├── stable_diffusion/
│   │   └── v1-5/
│   └── flux/
├── classification/
│   └── resnet/
└── multimodal/
    └── clip/
```

## 🎯 Supported Model Types

| Category | Supported Models | Status |
|------|------------|------|
| **Detection** | YOLOv8/v9, Detectron2, RT-DETR | ✅ |
| **Segmentation** | SAM, Mask2Former, DeepLabV3 | ✅ |
| **Classification** | ResNet, EfficientNet, ViT | ✅ |
| **Generation** | Stable Diffusion, FLUX, ControlNet | ✅ |
| **Multimodal** | CLIP, BLIP, LLaVA | 🚧 |

Full compatibility list: [Model Compatibility Matrix](docs/compatibility_matrix.md)

## 🌐 API Usage

### REST API

```bash
# Start API service
python -m cv_platform.api
python src/cv_platform/api/main.py
python scripts/start_api.py --host 0.0.0.0 --port 8000 workers 1

# Use API
curl -X POST "http://localhost:8000/predict/yolov8n" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@test_image.jpg"
```

### Python SDK

```python
from cv_platform.client import CVPlatformClient

# Synchronous client
client = CVPlatformClient("http://localhost:8000")
result = client.predict("yolov8n", "image.jpg")

# Asynchronous client
import asyncio
from cv_platform.client import AsyncCVPlatformClient

async def main():
    async with AsyncCVPlatformClient() as client:
        result = await client.predict("sam_vit_h", "image.jpg")

# Batch client
from cv_platform.client import BatchCVPlatformClient

batch_client = BatchCVPlatformClient()
results = batch_client.predict_batch("yolov8n", ["img1.jpg", "img2.jpg"])
```

### Web Interface

```bash
# Streamlit interface
streamlit run src/cv_platform/web/streamlit_app.py

# Gradio interface
python src/cv_platform/web/gradio_app.py
```

## ⚙️ Advanced Configuration

### Environment Variable Configuration

```bash
export CV_MODELS_ROOT="/custom/models/path"
export CV_MODEL_YOLOV8_PATH="/special/yolo/path.pt"
export CV_GPU_DEVICES="0,1"  # Specify GPU devices
export CV_MAX_BATCH_SIZE=8   # Batch size
```

### Configuration File

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

## 🐳 Docker Deployment

```bash
# Build image
docker build -t cv-model-platform .

# Run container
docker run -p 8000:8000 -v /path/to/models:/models cv-model-platform

# Use Docker Compose
docker-compose up -d
```

## 🔧 Development and Extension

### Add New Model Adaptor

```python
from cv_platform.adapters.base import BaseModelAdapter

class MyCustomAdapter(BaseModelAdapter):
    def load_model(self, model_path):
        # Implement model loading logic
        pass

    def predict(self, input_data):
        # Implement prediction logic
        pass

    def preprocess(self, data):
        # Implement preprocessing logic
        pass

    def postprocess(self, results):
        # Implement postprocessing logic
        pass

# Register adapter
from cv_platform.adapters.registry import register_adapter
register_adapter("my_model", MyCustomAdapter)
```

### Develop Plugin

```python
from cv_platform.plugins.base import BasePlugin

class MyPlugin(BasePlugin):
    def initialize(self):
        # Plugin initialization logic
        pass

    def process(self, data):
        # Plugin processing logic
        pass
```

## 📊 Performance Benchmark

| Model | Platform | Inference Time | Memory Usage | GPU VRAM |
|------|------|----------|----------|---------|
| YOLOv8n | RTX 6000 | 12ms | 256MB | 1.2GB |
| SAM ViT-H | RTX 6000 | 1.8s | 2.1GB | 8.5GB |
| SD 2.1 | RTX 6000 | 3.2s | 1.8GB | 6.2GB |

Full benchmark report: [Performance Report](docs/benchmarks.md)

## 📚 Docs

- 📖 [User Guide](docs/user_guide/)
- 🔧 [Developer Document](docs/developer_guide/)  
- 📡 [API Reference](docs/api_reference/)
- 🚀 [Deployment Guide](docs/deployment/)
- ❓ [FAQ](docs/faq.md)
- 🐛 [Troubleshooting](docs/troubleshooting.md)

## 🤝 Contribution

We welcome contributions of any kind!

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
