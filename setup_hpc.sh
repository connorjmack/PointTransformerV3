#!/bin/bash
# Setup script for PointTransformerV3 on HPC
# Run this script to set up the environment

set -e  # Exit on error

echo "=========================================="
echo "PointTransformerV3 HPC Setup"
echo "=========================================="

# Check CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "WARNING: nvcc not found. Make sure CUDA is loaded (module load cuda)"
fi

# Option 1: Conda environment (recommended)
echo ""
echo "=========================================="
echo "Setting up Conda environment..."
echo "=========================================="

# Create conda environment
conda env create -f environment.yml

echo ""
echo "Activating environment..."
conda activate ptv3

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
echo "  conda activate ptv3"
echo ""
echo "To test the installation, run:"
echo "  python3 prep_data.py --help"
echo ""
echo "NOTES:"
echo "- If flash-attn fails to install, you can remove it and set enable_flash=false in configs"
echo "- Adjust spconv version in requirements.txt based on your CUDA version:"
echo "    CUDA 11.8: spconv-cu118"
echo "    CUDA 11.6: spconv-cu116"
echo "    CUDA 12.1: spconv-cu121"
echo ""
