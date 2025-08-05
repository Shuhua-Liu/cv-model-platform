from setuptools import setup, find_packages
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Python版本特定的依赖
python_version = sys.version_info
install_requires = [
    "torch>=2.0.0",  # 更新到支持Python 3.12的版本
    "torchvision>=0.15.0",
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "diffusers>=0.24.0",
    "transformers>=4.35.0",
    "ultralytics>=8.0.200",
    "segment-anything>=1.0",
    "open-clip-torch>=2.20.0",
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
    "pydantic>=2.0.0",  # Pydantic v2 for better Python 3.11+ support
    "typing-extensions>=4.5.0",  # 向后兼容
]

# Python 3.11+ 特定优化
if python_version >= (3, 11):
    install_requires.extend([
        "accelerate>=0.24.0",  # 更好的Python 3.11+支持
        "safetensors>=0.4.0",  # 更快的模型加载
    ])

# Python 3.12 特定依赖（如果需要）
if python_version >= (3, 12):
    # Python 3.12可能需要的特定版本
    pass

setup(
    name="cv-model-platform",
    version="0.1.0",
    author="Shuhua Liu",
    author_email="shuhua.liu0709@gmail.com",
    description="A unified platform for managing computer vision models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shuhua-Liu/cv-model-platform",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-xdist>=3.0",  # 并行测试
            "pytest-asyncio>=0.21.0",  # 异步测试支持
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.5.0",
            "ruff>=0.1.0",  # 更快的linting工具
            "pre-commit>=3.0.0",
        ],
        "web": [
            "streamlit>=1.28.0",
            "gradio>=4.0.0",
            "plotly>=5.17.0",
        ],
        "monitoring": [
            "prometheus-client>=0.18.0",
            "grafana-client>=3.5.0",
            "psutil>=5.9.0",
        ],
        "optimization": [
            "xformers>=0.0.22",  # 内存优化
            "triton>=2.1.0",     # GPU加速
            "flash-attn>=2.3.0", # 注意力优化
        ],
        "cloud": [
            "boto3>=1.34.0",     # AWS支持
            "google-cloud-storage>=2.10.0",  # GCP支持
            "azure-storage-blob>=12.19.0",   # Azure支持
        ],
    },
    entry_points={
        "console_scripts": [
            "cv-platform=cv_platform.cli:main",
            "cv-platform-server=cv_platform.api.rest_api:main",
            "cv-platform-web=cv_platform.web.app:main",
        ],
    },
    # Python 3.11+ 特定配置
    zip_safe=False,
    include_package_data=True,
    package_data={
        "cv_platform": [
            "config/*.yaml",
            "web/static/**/*",
            "web/templates/*.html",
        ],
    },
)