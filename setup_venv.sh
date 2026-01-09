#!/bin/bash
# Setup script using virtualenv instead of conda
# Use this if conda is not available on your HPC

set -e  # Exit on error

echo "=========================================="
echo "PointTransformerV3 Virtual Environment Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.8" && "$PYTHON_VERSION" != "3.9" && "$PYTHON_VERSION" != "3.10" ]]; then
    echo "WARNING: Recommended Python version is 3.8-3.10, you have $PYTHON_VERSION"
fi

# Check CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "WARNING: nvcc not found. Make sure CUDA is loaded (module load cuda)"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv ptv3_env

# Activate virtual environment
echo "Activating virtual environment..."
source ptv3_env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (adjust for your CUDA version)
echo ""
echo "Installing PyTorch..."
# For CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 11.6, use:
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu116

# Install PyTorch Geometric dependencies
echo ""
echo "Installing PyTorch Geometric..."
pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install remaining requirements
echo ""
echo "Installing remaining dependencies..."
pip install -r requirements.txt

# Build pointops library
echo ""
echo "=========================================="
echo "Building pointops library..."
echo "=========================================="
cd libs/pointops
python setup.py install
cd ../..

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source ptv3_env/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python3 prep_data.py --help"
echo ""
echo "NOTES:"
echo "- Adjust CUDA version in the script if needed (edit setup_venv.sh)"
echo "- If flash-attn fails, comment it out in requirements.txt"
echo ""
