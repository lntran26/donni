from setuptools import setup

setup(
    name="donni",
    version="0.9.0",
    description="Demography Optimization via Neural Network Inference",
    packages=["donni"],

    install_requires=[
        "dadi",
        "python-irodsclient == 1.1.6",
        "appdirs == 1.4.4",
        "keras-tuner==1.4.6",
        "tensorflow>=2.12",
        ],
    entry_points={"console_scripts": ["donni=donni.__main__:main"]},
)