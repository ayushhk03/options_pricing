from setuptools import setup, find_packages

setup(
    name="options-pricing-nn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.62.0",
    ],
)