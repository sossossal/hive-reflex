#!/usr/bin/env python3
"""
setup.py - IMC-22 Python SDK 安装脚本
"""

from setuptools import setup, find_packages

with open("../../README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="imc22-sdk",
    version="2.1.0",
    author="Hive-Reflex Team",
    author_email="hive-reflex@example.com",
    description="IMC-22 CIM 芯片 Python SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hive-reflex/imc22-sdk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "onnx": [
            "onnx>=1.12.0",
        ],
        "full": [
            "onnx>=1.12.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "imc22-infer=imc22.cli:main",
        ],
    },
)
