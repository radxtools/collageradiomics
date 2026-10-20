"""Compatibility package; the CLI ships with the core distribution."""

from pathlib import Path

from setuptools import setup

setup(
    name="collageradiomicscli",
    version="1.1.0",
    description="Compatibility package for the CoLlAGe command-line interface",
    long_description=Path(__file__).with_name("README.rst").read_text(encoding="utf-8"),
    long_description_content_type="text/x-rst",
    url="https://github.com/radxtools/collageradiomics",
    python_requires=">=3.10",
    license="BSD-3-Clause",
    install_requires=["collageradiomics[cli]>=0.4.0"],
)
