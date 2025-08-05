# cv-model-platform
A unified platform for managing and serving computer vision models
## Features
- **Multi-task Support**: Detection, Segmentation, Generation, Classification
- **Smart GPU Management**: Intelligent resource allocation across multiple GPUs
- **Easy Integration**: Plugin-based architecture for adding new models
- **Multiple Interfaces**: REST API, Python SDK, Web UI, Juypter extensions
- **Real-time Monitoring**: GPU usage, model performance, and system health
- **Docker Ready**: Easy deployment with Docker and Kubernetes
## Quick Start
```bash
# Clone the repository
git clone https://github.com/Shuhua-Liu/cv-model-platform.git
cd cv-model-platform
# Setup environment
bash scripts/setup_env.sh
# Download example models
python scripts/install_models.py --basic
# Start the platform
python -m cv_platform.api.rest_api
