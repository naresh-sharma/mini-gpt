#!/usr/bin/env python3
"""
Setup script for MiniGPT package
"""

import os

from setuptools import find_packages, setup


# Read the README file for long description
def read_readme():
    readme_file = os.path.join(os.path.dirname(__file__), "README.md")
    if not os.path.exists(readme_file):
        return "Build GPT from scratch to understand how LLMs work."
    with open(readme_file, encoding="utf-8") as fh:
        return fh.read()


# Read requirements
def read_requirements():
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.exists(req_file):
        return []
    with open(req_file, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]


setup(
    name="mini-gpt",
    version="0.1.0",
    author="Naresh Sharma",
    author_email="asyncthinking@gmail.com",
    description="Build GPT from scratch to understand how LLMs work. A hands-on series for software engineers.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/naresh-sharma/mini-gpt",
    project_urls={
        "Bug Reports": "https://github.com/naresh-sharma/mini-gpt/issues",
        "Source": "https://github.com/naresh-sharma/mini-gpt",
        "Documentation": "https://github.com/naresh-sharma/mini-gpt#readme",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    # Remove CLI entry point until we actually create cli.py
    # entry_points={
    #     "console_scripts": [
    #         "mini-gpt=mini_gpt.cli:main",
    #     ],
    # },
    include_package_data=True,
    zip_safe=False,
    keywords="gpt, llm, machine-learning, education, tutorial, transformer, attention",
)
