from setuptools import setup, find_packages

setup(
    name="markov_quantile_predictor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.45.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A hybrid predictor combining Markov chains and quantile regression",
    keywords="markov, quantile regression, time series, prediction",
    python_requires=">=3.7",
)