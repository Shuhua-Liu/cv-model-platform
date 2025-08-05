from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "fastapi>=0.70.0",
        "uvicorn>=0.15.0",
        "diffusers>=0.20.0",
        "transformers>=4.20.0",
        "ultralytics>=8.0.0",
        "segment-anything>=0.1.0",
        "open-clip-torch>=2.20.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "pydantic>=1.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=22.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "isort>=5.9",
        ],
        "web": [
            "streamlit>=1.20.0",
            "gradio>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cv-model-platform=cv_model_platform.cli:main",
            "cv-model-platform-web=cv_model_platform.web.app:main",
            "cv-model-platform-server=cv_model_platform.api.rest_api:main",
        ],
    },
)
