#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlorenceForge Setup Script

A comprehensive framework for fine-tuning Florence-2 models on multiple vision-language tasks.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure we're in the right directory
HERE = Path(__file__).parent.absolute()
os.chdir(HERE)

# Python version check
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher is required for FlorenceForge.")

# Read version from __init__.py
def get_version():
    """Extract version from florence_forge/__init__.py"""
    version_file = HERE / "florence_forge" / "__init__.py"
    if not version_file.exists():
        return "1.0.0"
    
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

# Read long description from README
def get_long_description():
    """Read long description from README.md"""
    readme_file = HERE / "README.md"
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return "A comprehensive framework for fine-tuning Florence-2 models."

# Read requirements from requirements.txt
def get_requirements():
    """Parse requirements from requirements.txt"""
    requirements_file = HERE / "requirements.txt"
    if not requirements_file.exists():
        return []
    
    requirements = []
    with open(requirements_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments, empty lines, and optional dependencies
            if (line and 
                not line.startswith("#") and 
                not line.startswith("-") and
                "# 可选" not in line and
                "# Optional" not in line.lower() and
                "sphinx" not in line.lower() and
                "mkdocs" not in line.lower() and
                "jupyter" not in line.lower() and
                "gradio" not in line.lower() and
                "streamlit" not in line.lower() and
                "boto3" not in line.lower() and
                "azure" not in line.lower() and
                "google-cloud" not in line.lower() and
                "cupy" not in line.lower() and
                "memory-profiler" not in line.lower() and
                "line-profiler" not in line.lower()):
                # Remove inline comments
                if "#" in line:
                    line = line.split("#")[0].strip()
                if line:
                    requirements.append(line)
    
    return requirements

# Define optional dependencies
optional_dependencies = {
    "dev": [
        "pytest>=7.0.0,<8.0.0",
        "pytest-asyncio>=0.21.0,<1.0.0",
        "pytest-cov>=4.0.0,<5.0.0",
        "black>=23.0.0,<24.0.0",
        "isort>=5.12.0,<6.0.0",
        "flake8>=6.0.0,<7.0.0",
        "mypy>=1.5.0,<2.0.0",
        "pre-commit>=3.3.0,<4.0.0",
    ],
    "docs": [
        "sphinx>=7.0.0,<8.0.0",
        "sphinx-rtd-theme>=1.3.0,<2.0.0",
        "mkdocs>=1.5.0,<2.0.0",
        "mkdocs-material>=9.0.0,<10.0.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0,<2.0.0",
        "ipywidgets>=8.0.0,<9.0.0",
        "notebook>=6.5.0,<8.0.0",
    ],
    "demo": [
        "gradio>=3.50.0,<5.0.0",
        "streamlit>=1.28.0,<2.0.0",
    ],
    "cloud": [
        "boto3>=1.28.0,<2.0.0",
        "azure-storage-blob>=12.17.0,<13.0.0",
        "google-cloud-storage>=2.10.0,<3.0.0",
    ],
    "performance": [
        "numba>=0.57.0,<1.0.0",
        "memory-profiler>=0.61.0,<1.0.0",
        "line-profiler>=4.0.0,<5.0.0",
    ],
}

# Add 'all' option that includes everything
optional_dependencies["all"] = [
    dep for deps in optional_dependencies.values() for dep in deps
]

# Classifiers
classifiers = [
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
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    "Topic :: Text Processing :: Linguistic",
]

# Keywords
keywords = [
    "florence-2",
    "computer-vision",
    "natural-language-processing",
    "multi-task-learning",
    "fine-tuning",
    "transformers",
    "pytorch",
    "machine-learning",
    "deep-learning",
    "vision-language",
    "image-captioning",
    "object-detection",
    "ocr",
    "visual-question-answering",
]

# Project URLs
project_urls = {
    "Documentation": "https://florence-forge.readthedocs.io/",
    "Source": "https://github.com/florenceforge/florence-forge",
    "Tracker": "https://github.com/florenceforge/florence-forge/issues",
    "Changelog": "https://github.com/florenceforge/florence-forge/blob/main/CHANGELOG.md",
}

# Entry points
entry_points = {
    "console_scripts": [
        "florence_forge_cli=florence_forge.cli.main:main",
        "florence-forge=florence_forge.cli.main:main",
        "florence-train=florence_forge.cli.train:main",
        "florence-evaluate=florence_forge.cli.evaluate:main",
        "florence-serve=florence_forge.cli.serve:main",
        "florence-convert=florence_forge.cli.convert:main",
        "florence-infer=florence_forge.cli.inference:main",
    ],
}

# Setup configuration
setup(
    name="florence-forge",
    version=get_version(),
    author="FlorenceForge Contributors",
    author_email="contact@florenceforge.ai",
    description="A comprehensive framework for fine-tuning Florence-2 models on multiple vision-language tasks",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/florenceforge/florence-forge",
    project_urls=project_urls,
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    include_package_data=True,
    package_data={
        "florence_forge": [
            "configs/*.yaml",
            "configs/**/*.yaml",
            "templates/*.yaml",
            "templates/**/*.yaml",
            "assets/*",
            "assets/**/*",
        ],
    },
    install_requires=get_requirements(),
    extras_require=optional_dependencies,
    python_requires=">=3.8",
    classifiers=classifiers,
    keywords=keywords,
    entry_points=entry_points,
    zip_safe=False,
    platforms=["any"],
    license="MIT",
    license_files=["LICENSE"],
)

# Post-installation message
if __name__ == "__main__":
    print("""
🎉 FlorenceForge installation completed!

Quick start:
  1. Check installation: florence-forge --version
  2. View examples: florence-forge examples
  3. Start training: florence-forge train --config config.yaml
  4. Launch demo: florence-forge serve --model path/to/model

Documentation: https://florence-forge.readthedocs.io/
Examples: https://github.com/florenceforge/florence-forge/tree/main/examples

Happy fine-tuning! 🚀
    """)