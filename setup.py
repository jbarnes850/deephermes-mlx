"""
Setup script for DeepHermes MLX.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="deephermes",
    version="0.1.0",
    author="Jarrod Barnes",
    author_email="jbarnes850@gmail.com",
    description="DeepHermes MLX Inference for Apple Silicon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jbarnes850/mlx-deephermes",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "deephermes=deephermes.cli:main",
            "deephermes-chat=deephermes.chat:main",
        ],
    },
)
