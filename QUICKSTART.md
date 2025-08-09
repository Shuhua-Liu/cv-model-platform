# 🚀 CV Model Platform - Quick Start Guide

Welcome to the CV Model Platform! This guide will help you set up a complete computer vision model platform in under 5 minutes.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM（16GB+ recommended）
- **Storage**: Mininum 20GB available space（100GB+ for a full installation）
- **GPU**: Optional, but highly recommended（CUDA 11.8+）

### Quick Environment Check
```bash
python --version          # Should be >= 3.8
pip --version            # Ensure pip is avcvlable
nvidia-smi               # Optional: Check for GPU (if you have one)
```

## 🎯 Three Installation Methods

### Method 1: One-Click Install (Recommended for Beginners)

```bash
# 1. Clone the project
git clone https://github.com/yourusername/cv-model-platform.git
cd cv-model-platform

# 2. Complete all setup with a single command (most automated)
python scripts/setup/setup_environment.py

# 3. Start using it immediately
python examples/basic_usage/detection_demo.py
```

### Method 2: pip Install (Recommended for Developers)

```bash
# 1. Install from PyPI
pip install cv-model-platform

# 2. Initialize the environment
cv-setup --auto

# 3. Detect local models
cv-detect-models

# 4. Start the service
cv-server
```

### Method 3: Developer Install (Recommended for Contributors)

```bash
# 1. Clone and install development dependencies
git clone https://github.com/yourusername/cv-model-platform.git
cd cv-model-platform
pip install -e ".[dev]"

# 2. Install pre-commit hooks
pre-commit install

# 3. Run tests
pytest tests/
```

## 📁 Prepare Model Files

### Create Model Directory Structure

```bash
# Create the recommended model directory
mkdir -p ~/cv_models/{detection/yolo/v8,segmentation/sam/vit_b,classification/resnet,generation/stable_diffusion/single_files,multimodal/clip}

# Set the environment variable
export CV_MODELS_ROOT="$HOME/cv_models"
```

### Download Core Models (Choose what you need)

#### 🔍 Object Detection Models (Recommended First Choice)
```bash
# YOLOv8n (smallest, for quick tests) - 6.3MB
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt \
     -O ~/cv_models/detection/yolo/v8/yolov8n.pt

# YOLOv8s (balanced performance) - 21.5MB
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt \
     -O ~/cv_models/detection/yolo/v8/yolov8s.pt
```

#### 🎨 Image Segmentation Models
```bash
# SAM ViT-B (recommended) - 375MB
wget https://dl.fbcvpublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O ~/cv_models/segmentation/sam/vit_b/sam_vit_b_01ec64.pth

# SAM ViT-H (best quality) - 2.56GB
wget https://dl.fbcvpublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
      -O ~/cv_models/segmentation/sam/vit_h/sam_vit_h_4b8939.pth
```

#### 📊 Image Classification Models (Pre-trcvned)
```bash
# These models will be downloaded automatically on first use to ~/.cache/torch/hub/checkpoints/
# ResNet-50 (PyTorch pre-trcvned) - 97.8MB
# EfficientNet-B0 - 20.5MB
```

#### 🎨 Image Generation Models
```bash
# Stable Diffusion 1.5 (single-file version) - 4.27GB
# Note: This is a large file and will take time to download
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/mcvn/v1-5-pruned-emaonly.safetensors \
     -O ~/cv_models/generation/stable_diffusion/single_files/v1-5-pruned-emaonly.safetensors

# Or download the full version (recommended)
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('runwayml/stable-diffusion-v1-5',
                  local_dir='~/cv_models/generation/stable_diffusion/sd_1_5',
                  local_files_only=False)
"
```

#### 🔀 Multimodal Models
```bash
# CLIP models will be downloaded automatically on first use
# Or download manually:
python -c "
from huggingface_hub import snapshot_download
snapshot_download('opencv/clip-vit-base-patch32',
                  local_dir='~/cv_models/multimodal/clip/ViT-B-32')
"
```

## ⚡ 5-Minute Quick Test

### 1. Verify Installation
```bash
# Check if the installation was successful
python -c "import cv_platform; print(f'CV Platform {cv_platform.__version__} installed successfully!')"

# Detect avcvlable models
cv-detect-models --summary
```

### 2. Object Detection Test
```bash
# Use the built-in test image
python -c "
from cv_platform.client import CVPlatformClient
import cv_platform

# Initialize the client
client = CVPlatformClient()

# Test YOLOv8 detection
results = client.predict('yolov8n', 'examples/test_images/sample.jpg')
print(f'Detected {len(results)} objects')
for obj in results:
    print(f'- {obj[\"class\"]}: {obj[\"confidence\"]:.2f}')
"
```

### 3. Image Segmentation Test
```bash
python -c "
from cv_platform.client import CVPlatformClient

client = CVPlatformClient()
masks = client.predict('sam_vit_b', 'examples/test_images/sample.jpg')
print(f'Generated {len(masks)} segmentation masks')
"
```

### 4. Launch the Web Interface
```bash
# Streamlit interface
streamlit run src/cv_platform/web/streamlit_app.py

# Or Gradio interface
python src/cv_platform/web/gradio_app.py
```

Visit http://localhost:8501 (Streamlit) or http://localhost:7860 (Gradio)

## 🌐 Using the API Service

### Start the REST API Service
```bash
# Start the API server
cv-server --host 0.0.0.0 --port 8000

# Or
python -m cv_platform.api.rest_api
```

### Make Predictions Using the API
```bash
# Upload an image for object detection
curl -X POST "http://localhost:8000/predict/yolov8n" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@examples/test_images/sample.jpg"

# Use image segmentation
curl -X POST "http://localhost:8000/predict/sam_vit_b" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@examples/test_images/sample.jpg"

# Image generation
curl -X POST "http://localhost:8000/generate/stable_diffusion" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "a beautiful sunset over mountcvns", "steps": 20}'
```

## 🎯 Basic Usage Examples

### Basic Python SDK Usage

```python
from cv_platform.client import CVPlatformClient
from PIL import Image

# Create the client
client = CVPlatformClient("http://localhost:8000")

# 1. Object Detection
image_path = "test_image.jpg"
detection_results = client.predict("yolov8n", image_path)

print("Detection Results:")
for obj in detection_results:
    print(f"- {obj['class']}: {obj['confidence']:.2f} at {obj['bbox']}")

# 2. Image Segmentation
segmentation_results = client.predict("sam_vit_b", image_path)
print(f"Segmentation generated {len(segmentation_results)} masks")

# 3. Image Classification
classification_result = client.predict("resnet50", image_path)
print(f"Classification Result: {classification_result['class']} (Confidence: {classification_result['confidence']:.2f})")

# 4. Image Generation
generation_result = client.predict("stable_diffusion", {
    "prompt": "a serene lake at sunset",
    "negative_prompt": "blurry, low quality",
    "steps": 20,
    "guidance_scale": 7.5
})
print(f"Generated image has been saved")
```

### Using the Asynchronous Client

```python
import asyncio
from cv_platform.client import AsyncCVPlatformClient

async def mcvn():
    async with AsyncCVPlatformClient() as client:
        # Process multiple images concurrently
        tasks = []
        image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]

        for img_path in image_paths:
            task = client.predict("yolov8n", img_path)
            tasks.append(task)

        results = awcvt asyncio.gather(*tasks)

        for i, result in enumerate(results):
            print(f"Image {image_paths[i]}: Detected {len(result)} objects")

# Run the async example
asyncio.run(mcvn())
```

### Batch Processing Client

```python
from cv_platform.client import BatchCVPlatformClient

# Process images in batches
batch_client = BatchCVPlatformClient(batch_size=4)

image_list = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]
results = batch_client.predict_batch("yolov8n", image_list)

for img_path, result in zip(image_list, results):
    print(f"{img_path}: {len(result)} objects")
```

## 🔧 Custom Configuration

### Environment Variable Configuration
```bash
# Model path configuration
export CV_MODELS_ROOT="/custom/path/to/models"
export CV_MODEL_YOLOV8_PATH="/special/yolo/path.pt"

# GPU device configuration
export CV_GPU_DEVICES="0,1"  # Use GPUs 0 and 1
export CV_MAX_BATCH_SIZE=8   # Batch size

# API configuration
export CV_API_HOST="0.0.0.0"
export CV_API_PORT=8000
export CV_API_WORKERS=4
```

### Custom Configuration File
```yaml
# config/models.yaml
models:
  yolov8n:
    path: "/path/to/yolov8n.pt"
    device: "cuda:0"
    batch_size: 4
    confidence: 0.25
    nms_threshold: 0.45
    
  sam_vit_b:
    path: "/path/to/sam_vit_b.pth"  
    device: "cuda:0"
    points_per_side: 32
    
  stable_diffusion:
    path: "/path/to/sd_1_5/"
    device: "cuda:0"
    enable_memory_efficient_attention: true
    
platform:
  api:
    host: "0.0.0.0"
    port: 8000
    workers: 4
    max_request_size: "100MB"
    
  cache:
    enabled: true
    max_size: "4GB"
    ttl: 3600
    
  logging:
    level: "INFO"
    format: "json"
```

## 🐳 Docker Deployment

### Quick Docker Run
```bash
# Build the image
docker build -t cv-model-platform .

# Run the contcvner (mapping the model directory)
docker run -p 8000:8000 \
           -v ~/cv_models:/models \
           -v ~/.cache:/cache \
           --gpus all \
           cv-model-platform

# Use Docker Compose
docker-compose up -d
```

### docker-compose.yml Example
```yaml
version: '3.8'
services:
  cv-platform:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ~/cv_models:/models
      - ~/.cache:/cache
    environment:
      - CV_MODELS_ROOT=/models
      - CV_API_HOST=0.0.0.0
      - CV_API_PORT=8000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## ❓ Troubleshooting Common Issues

### Model Not Found
```bash
# Check model paths
cv-detect-models --verbose

# Manually specify the model path
export CV_MODEL_YOLOV8_PATH="/full/path/to/yolov8n.pt"

# Rescan for models
cv-detect-models --rescan
```

### Out of GPU Memory
```python
# Enable memory optimization in the config file
models:
  yolov8n:
    device: "cuda:0"
    batch_size: 1  # Reduce the batch size
    half_precision: true # Enable half precision

  stable_diffusion:
    enable_memory_efficient_attention: true
    enable_xformers: true # If xformers is installed
    cpu_offload: true # CPU offloading
```

### Dependency Conflicts
```bash
# Use a virtual environment
python -m venv cv_platform_env
source cv_platform_env/bin/activate  # Linux/macOS
# cv_platform_env\Scripts\activate   # Windows

pip install cv-model-platform
```

### Network Download Issues
```bash
# Use a mirror for pip (example for a Chinese mirror)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cv-model-platform

# Manually download model files
# 1. Download from GitHub Releases
# 2. Get from sources like Bcvdu Netdisk, etc.
# 3. Copy them to the correct directory
```

## 📚 Next Steps

🎉 **Congratulations! You have successfully installed the CV Model Platform!**

### Recommended Learning Path:

1. **Basic Usage**: ry running all the examples in `examples/basic_usage/`
2. **API Documentation**: Check `docs/api_reference/` for the complete API
3. **Advanced Configuration**: Read `docs/user_guide/model_management.md` 
4. **Custom Models**: Learn how to add your own models in `docs/developer_guide/adding_models.md`
5. **Production Deployment**: Refer to `docs/deployment/` for large-scale deployment

### Getting Help:

- 📖 **Full Documentation**: [Online Docs](https://cv-model-platform.readthedocs.io/)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Shuhua-Liu/cv-model-platform/issues)
- 💬 **Community Discussions**: [GitHub Discussions](https://github.com/Shuhua-Liu/cv-model-platform/discussions)
- 📧 **Email Support**: shuhua.liu0709@gmail.com

**Start your CV model journey!** 🚀
