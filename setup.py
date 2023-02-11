from setuptools import setup, find_packages

setup(
    name="donni",
    version="0.0.1",
    description="Demography Optimization via Neural Network Inference",
    packages=["donni"],
    # packges=find_packages(),
    install_requires=[
        "dadi",
        "scikit-learn == 1.0.2",
        "mapie == 0.3.1",
        "pytest",
        "pylint",
        "flake8",
        "mypy",
        "pytest-pylint",
        "pytest-flake8",
        "pytest-mypy"
        ],
    entry_points={"console_scripts": ["donni=donni.__main__:main"]},
)