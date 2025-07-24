"""Setup script for Belgian Document Deidentification System."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="belgian-deidentification",
    version="1.0.0",
    author="Manus AI",
    author_email="manus@example.com",
    description="A waterproof deidentification system for Belgian healthcare documents in Dutch",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/belgian-deidentification",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
        ],
        "api": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "python-multipart>=0.0.6",
        ],
        "gpu": [
            "torch>=2.1.0+cu118",
            "transformers[torch]>=4.35.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "belgian-deidentify=belgian_deidentification.cli:main",
            "belgian-deidentify-api=belgian_deidentification.api.main:main",
            "belgian-deidentify-batch=belgian_deidentification.cli:batch_main",
        ],
    },
    include_package_data=True,
    package_data={
        "belgian_deidentification": [
            "data/*.json",
            "data/*.yaml",
            "config/*.yaml",
        ],
    },
    zip_safe=False,
    keywords=[
        "deidentification",
        "privacy",
        "healthcare",
        "nlp",
        "dutch",
        "belgian",
        "gdpr",
        "phi",
        "clinical",
        "medical",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/belgian-deidentification/issues",
        "Source": "https://github.com/your-org/belgian-deidentification",
        "Documentation": "https://belgian-deidentification.readthedocs.io/",
    },
)

