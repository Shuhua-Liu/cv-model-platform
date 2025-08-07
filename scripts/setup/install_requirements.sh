#!/bin/bash
# scripts/setup_env_multi_python.sh
# Environment setup script for Python 3.8-3.12

set -e

# Color Definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 CV Model Platform Multi-Python Environment Setup${NC}"

# Detect operating system
OS=$(uname -s)
echo "Operating system detected: $OS"

# Get user's choice of Python version
echo "Please select the Python version to install:"
echo "1) Python 3.8"
echo "2) Python 3.9" 
echo "3) Python 3.10"
echo "4) Python 3.11 (Recommended)"
echo "5) Python 3.12 (Latest)"
echo "6) Install all versions for testing"

read -p "Enter your choice (1-6): " choice

case $choice in
    1) PYTHON_VERSIONS=("3.8") ;;
    2) PYTHON_VERSIONS=("3.9") ;;
    3) PYTHON_VERSIONS=("3.10") ;;
    4) PYTHON_VERSIONS=("3.11") ;;
    5) PYTHON_VERSIONS=("3.12") ;;
    6) PYTHON_VERSIONS=("3.8" "3.9" "3.10" "3.11" "3.12") ;;
    *) echo -e "${RED}Invalid choice${NC}"; exit 1 ;;
esac

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Miniconda or Anaconda first.${NC}"
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create an environment for each Python version
for py_version in "${PYTHON_VERSIONS[@]}"; do
    env_name="cv_platform_py${py_version//./}"
    
    echo -e "${YELLOW}Creating environment for Python ${py_version}: ${env_name}${NC}"
    
    # Create conda environment
    conda create -n "$env_name" python="$py_version" -y
    
    # Activate environment and install dependencies
    eval "$(conda shell.bash hook)"
    conda activate "$env_name"
    
    echo -e "${YELLOW}Installing base dependencies...${NC}"
    
    # Install PyTorch (select a suitable version based on the Python version)
    if [[ "$py_version" == "3.12" ]]; then
        # Python 3.12 requires the latest version of PyTorch
        pip install torch>=2.1.0 torchvision>=0.16.0 torchaudio>=2.1.0 --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$py_version" == "3.11" ]]; then
        # Recommended version for Python 3.11
        pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
    else
        # Python 3.8-3.10
        pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
    fi
    
    # Install project dependencies
    echo -e "${YELLOW}Installing project dependencies...${NC}"
    pip install -e ".[dev]"
    
    # Python version-specific optimizations
    if [[ "$py_version" == "3.11" ]] || [[ "$py_version" == "3.12" ]]; then
        echo -e "${YELLOW}Installing performance optimization packages for Python ${py_version}...${NC}"
        # Install updated optimization packages
        pip install -e ".[optimization]" || echo -e "${YELLOW}Some optimization packages may not be compatible, skipping.${NC}"
    fi
    
    # Verify installation
    echo -e "${YELLOW}Verifying installation...${NC}"
    python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'PyTorch import failed: {e}')

try:
    import cv_platform
    print('CV Platform imported successfully')
except ImportError as e:
    print(f'CV Platform import failed: {e}')
"
    
    conda deactivate
    echo -e "${GREEN}✅ Environment ${env_name} setup complete.${NC}"
    echo
done

echo -e "${GREEN}🎉 All environments set up successfully!${NC}"
echo
echo "Usage:"
for py_version in "${PYTHON_VERSIONS[@]}"; do
    env_name="cv_platform_py${py_version//./}"
    echo "  conda activate $env_name  # Activate Python ${py_version} environment"
done

echo
echo "To test all environments:"
echo "  bash scripts/test_all_versions.sh"

# Create a convenient script for switching environments
cat > scripts/switch_python.sh << 'EOF'
#!/bin/bash
# Convenient Python version switching script

echo "Available CV Platform environments:"
conda env list | grep cv_platform

echo
echo "Please select an environment to activate:"
echo "1) cv_platform_py38  (Python 3.8)"
echo "2) cv_platform_py39  (Python 3.9)" 
echo "3) cv_platform_py310 (Python 3.10)"
echo "4) cv_platform_py311 (Python 3.11)"
echo "5) cv_platform_py312 (Python 3.12)"

read -p "Enter your choice (1-5): " choice

case $choice in
    1) conda activate cv_platform_py38 ;;
    2) conda activate cv_platform_py39 ;;
    3) conda activate cv_platform_py310 ;;
    4) conda activate cv_platform_py311 ;;
    5) conda activate cv_platform_py312 ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

echo "Activated environment: $CONDA_DEFAULT_ENV"
python --version
EOF

chmod +x scripts/switch_python.sh

#
