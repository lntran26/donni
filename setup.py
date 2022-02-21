from setuptools import setup, find_packages

setup(
    name="dadi-ml",
    version="0.0.1",
    description="Machine learning applications for dadi",
    author="Linh Tran",
    packages=["dadinet"],
    # packges=find_packages(),
    install_requires=[
        "dadi == 2.1.1",
        "scikit-learn == 1.0.1",
        "mapie == 0.3.1"
    ],
    entry_points={"console_scripts": ["dadi-ml=dadinet.__main__:main"]},
)