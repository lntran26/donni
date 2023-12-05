from setuptools import setup

setup(
    name="donni",
    version="0.1.0",
    description="Demography Optimization via Neural Network Inference",
    packages=["donni"],

    install_requires=[
        "dadi",
        "matplotlib",
        "numpy",
        "scipy",
        "nlopt",
        "python-irodsclient == 1.1.6",
        "appdirs == 1.4.4",
        ],
    entry_points={"console_scripts": ["donni=donni.__main__:main"]},
)