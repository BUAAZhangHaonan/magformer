"""
MAGFormer - Multi-modal Adaptive Gated MaskFormer
Setup script for installation
"""

from pathlib import Path
from setuptools import find_packages, setup

PACKAGE_INCLUDE = [
    "magformer",
    "magformer.*",
    "tools",
    "tools.*",
]

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="magformer",
    version="0.1.0",
    description="Multi-modal Adaptive Gated MaskFormer for RGB-D Instance Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MAGFormer Contributors",
    url="https://github.com/your-org/magformer",
    packages=find_packages(include=PACKAGE_INCLUDE),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "magformer-train=tools.train:main",
            "magformer-eval=tools.evaluate:main",
            "magformer-infer=tools.inference:main",
            "magformer-export=tools.export_results:main",
        ],
    },
    include_package_data=True,
    package_data={
        "magformer": [
            "configs/**/*.yaml",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache License 2.0",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
