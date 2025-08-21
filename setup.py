#!/usr/bin/env python
"""Setup script for CV Model Platform."""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure Python version compatibility
if sys.version_info < (3, 8):
    raise RuntimeError("cv-model-platform requires Python 3.8 or higher")

# Get the project root directory
HERE = Path(__file__).parent.absolute()

# Read version from __init__.py
def get_version():
    """Extract version from __init__.py"""
    version_file = HERE / "src" / "cv_platform" / "__init__.py"
    version_line = [line for line in version_file.read_text().splitlines() 
                   if line.startswith("__version__")]
    if version_line:
        return version_line[0].split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read the README file
def get_long_description():
    """Get long description from README.md"""
    readme_file = HERE / "README.md"
    if readme_file.exists():
        return readme_file.read_text(encoding="utf-8")
    return ""

# Read requirements
def get_requirements(filename="requirements.txt"):
    """Parse requirements file"""
    requirements_file = HERE / filename
    if not requirements_file.exists():
        return []
    
    requirements = []
    with open(requirements_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-"):
                requirements.append(line)
    return requirements

# Core dependencies
INSTALL_REQUIRES = [
    "torch>=1.13.0",
    "torchvision>=0.14.0",
    "numpy>=1.21.0",
    "opencv-python>=4.6.0",
    "pillow>=9.0.0",
    "pyyaml>=6.0",
    "pydantic>=1.10.0",
    "fastapi>=0.95.0",
    "uvicorn>=0.20.0",
    "httpx>=0.24.0",
    "aiofiles>=23.0.0",
    "python-multipart>=0.0.6",
    "rich>=13.0.0",
    "typer>=0.7.0",
    "loguru>=0.7.0",
    "psutil>=5.9.0",
    "tqdm>=4.64.0",
    "packaging>=21.0",
]

# Optional dependencies for different categories
EXTRAS_REQUIRE = {
    # Detection models
    "detection": [
        "ultralytics>=8.0.0",
        "yolov5>=7.0.0",
    ],
    
    # Segmentation models  
    "segmentation": [
        "segment-anything>=1.0",
        "detectron2 @ git+https://github.com/facebookresearch/detectron2.git",
    ],
    
    # Generation models
    "generation": [
        "diffusers>=0.21.0",
        "transformers>=4.25.0",
        "accelerate>=0.20.0",
        "xformers>=0.0.20",
    ],
    
    # Classification models
    "classification": [
        "timm>=0.9.0",
        "torchmetrics>=0.11.0",
    ],
    
    # Multimodal models
    "multimodal": [
        "open-clip-torch>=2.20.0",
        "transformers>=4.25.0",
    ],
    
    # Backend support
    "onnx": [
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "onnxruntime-gpu>=1.15.0",
    ],
    
    "tensorrt": [
        "tensorrt>=8.6.0",
        "pycuda>=2022.2",
    ],
    
    "openvino": [
        "openvino>=2023.0.0",
    ],
    
    # Web interfaces
    "web": [
        "streamlit>=1.25.0",
        "gradio>=3.40.0",
        "plotly>=5.15.0",
        "bokeh>=3.2.0",
    ],
    
    # Development tools
    "dev": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.0",
        "black>=23.7.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "pre-commit>=3.3.0",
        "bandit>=1.7.0",
    ],
    
    # Documentation
    "docs": [
        "mkdocs>=1.5.0",
        "mkdocs-material>=9.2.0",
        "mkdocstrings[python]>=0.22.0",
        "mkdocs-jupyter>=0.24.0",
    ],
    
    # Testing with real models
    "test": [
        "pytest-benchmark>=4.0.0",
        "memory-profiler>=0.60.0",
        "line-profiler>=4.0.0",
    ],
}

# Convenience extras
EXTRAS_REQUIRE["all"] = list(set().union(*[
    deps for key, deps in EXTRAS_REQUIRE.items() 
    if key not in ["dev", "docs", "test"]
]))

EXTRAS_REQUIRE["full"] = list(set().union(*EXTRAS_REQUIRE.values()))

# Console scripts entry points
CONSOLE_SCRIPTS = [
    "cv-platform=cv_platform.cli.main:app",
    "cv-detect-models=cv_platform.scripts.detect_models:main",
    "cv-setup=cv_platform.scripts.setup_environment:main",
    "cv-server=cv_platform.api:main",
    "cv-benchmark=cv_platform.scripts.benchmark:main",
]

# Project classifiers
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
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
    "Topic :: Scientific/Engineering :: Image Processing",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Multimedia :: Graphics",
]

# Project keywords
KEYWORDS = [
    "computer-vision", "deep-learning", "machine-learning", "pytorch",
    "yolo", "sam", "stable-diffusion", "detectron2", "cv", "ai",
    "image-processing", "object-detection", "segmentation", "classification",
    "image-generation", "model-serving", "api", "platform"
]

setup(
    name="cv-model-platform",
    version=get_version(),
    description="A unified platform for computer vision model management and deployment",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    # Author information
    author="Shuhua Liu",
    author_email="shuhua.liu0709@gmail.com",
    
    # Project URLs
    url="https://github.com/Shuhua-Liu/cv-model-platform",
    project_urls={
        "Documentation": "https://cv-model-platform.readthedocs.io/",
        "Source": "https://github.com/Shuhua-Liu/cv-model-platform",
        "Tracker": "https://github.com/Shuhua-Liu/cv-model-platform/issues",
        "Changelog": "https://github.com/Shuhua-Liu/cv-model-platform/blob/main/CHANGELOG.md",
    },
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "cv_platform": [
            "config/*.yaml",
            "config/examples/*.yaml", 
            "config/schemas/*.json",
            "web/static/*",
            "web/templates/*",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Console scripts
    entry_points={
        "console_scripts": CONSOLE_SCRIPTS,
    },
    
    # Metadata
    classifiers=CLASSIFIERS,
    keywords=", ".join(KEYWORDS),
    license="MIT",
    
    # Options
    zip_safe=False,
    
    # Test configuration
    test_suite="tests",
    
    # Platform compatibility
    platforms=["any"],
    
    # Development status
    obsoletes_dist=[],
    provides_dist=[],
)
