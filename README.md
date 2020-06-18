# Setup

## Pip
```
pip install collageradiomics
```

## Git

```
git clone https://github.com/Toth-Technology/collageradiomics && cd collageradiomics
```

## Docker Ubuntu 20.04 Jupyter Examples

Install docker, then run in root of git repository:
### Linux:
```
docker run -it -p 8888:8888 -v $PWD:/root nathanhillyer/collageradiomics-examples
```
### Windows/Powershell:
```
docker run -it -p 8888:8888 -v ${PWD}:/root nathanhillyer/collageradiomics-examples
```

Open in browser:
```
localhost:8888
```

Open `jupyter/examples` directory and explore the Jupyter notebooks.

## Docker Ubuntu 20.04 Pip Module

```
docker run -it -p 8888:8888 nathanhillyer/collageradiomics-pip
```
