#!/bin/sh
# Build script for MultiScaleDeformableAttention CUDA extension

# Check if CUDA_HOME is set
if [ -z "$CUDA_HOME" ]; then
    echo "Warning: CUDA_HOME is not set. Trying to detect CUDA installation..."
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
        echo "Found CUDA at $CUDA_HOME"
    else
        echo "Error: CUDA not found. Please set CUDA_HOME environment variable."
        exit 1
    fi
fi

# Build and install
echo "Building MultiScaleDeformableAttention CUDA extension..."
python setup.py build install

echo "Done!"
