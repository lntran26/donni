from setuptools import setup, find_packages

setup(
    name="dadi-ml",
    version="0.0.1",
    description="Machine learning applications for dadi",
    author="Linh Tran, Connie Sun",
    packages=["dadinet"],
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
    entry_points={"console_scripts": ["dadi-ml=dadinet.__main__:main"]},
)