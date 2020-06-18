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

1. [Install Docker.](https://www.docker.com/get-started)
2. Run one of these commands in the __root of git repository__:
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
This is a Docker image with the collageradiomics module pre-installed:
```
docker run -it -p 8888:8888 nathanhillyer/collageradiomics-pip
```
