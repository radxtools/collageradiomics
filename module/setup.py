import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="collageradiomics",
    version="0.0.1a26",
    author="Toth Technology",
    author_email="toth-tech@hillyer.me",
    description="CoLliage Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Toth-Technology/rad-i",
    py_modules=["collageradiomics"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'matplotlib',
        'numpy',
        'scikit-learn',
        'scikit-build',
        'mahotas',
        'scipy'
    ],
    python_requires='>=3.7',
)