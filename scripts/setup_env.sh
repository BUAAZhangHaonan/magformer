#!/bin/bash
# MAGFormer Environment Setup Script
# This script sets up the conda environment and installs dependencies

set -e

echo "========================================="
echo "MAGFormer Environment Setup"
echo "========================================="

# Environment name
ENV_NAME="magformer"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Create environment if it doesn't exist
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists"
else
    echo "Creating conda environment: ${ENV_NAME}"
    conda create -n ${ENV_NAME} python=3.10 -y
fi

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install PyTorch
echo ""
echo "Installing PyTorch..."
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo ""
echo "Installing magformer package..."
pip install -e .

echo ""
echo "========================================="
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To train the model, run:"
echo "  python tools/train.py --config-file configs/magformer.yaml --dataset-root /path/to/eccd"
echo "========================================="
