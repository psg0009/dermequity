"""Setup script for DermEquity package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dermequity",
    version="2.0.0",
    author="Parth Gosar",
    author_email="pgosar@usc.edu",
    description="Fairness Auditing & Bias Mitigation Toolkit for Dermatological AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pgosar/dermequity",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
    },
    keywords=[
        "fairness",
        "bias",
        "dermatology",
        "skin cancer",
        "medical AI",
        "healthcare",
        "machine learning",
        "deep learning",
        "fitzpatrick",
        "equity",
    ],
    project_urls={
        "Bug Reports": "https://github.com/pgosar/dermequity/issues",
        "Source": "https://github.com/pgosar/dermequity",
        "Paper": "https://sites.google.com/usc.edu/showcais2026",
    },
)
