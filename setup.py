from setuptools import setup, find_packages

setup(
    name="dadi-ml",
    version="0.0.1",
    description="Machine learning applications for dadi",
    author="Linh Tran",
    packages=find_packages(),
    install_requires=[
        "dadi",
        "scikit-learn",
        "mapie"
    ],
    entry_points={"console_scripts": ["dadi-ml=mlpr.__main__:main"]},
)