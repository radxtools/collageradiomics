"""Compatibility package; the CLI ships with the core distribution."""

from setuptools import setup

setup(
    name="collageradiomicscli",
    version="1.1.0",
    description="Compatibility package for the CoLlAGe command-line interface",
    url="https://github.com/radxtools/collageradiomics",
    python_requires=">=3.10",
    license="BSD-3-Clause",
    install_requires=["collageradiomics[cli]>=0.4.0"],
)
