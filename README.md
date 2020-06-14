# Setup

## Pip
```
pip install collageradiomics
```

## Git

```
git clone https://github.com/Toth-Technology/rad-i && cd rad-i
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

Open `Juptyer/CoLlage.ipynb` to begin working.

## Docker Ubuntu 20.04 Pip Module

### Linux:
```
docker run -it -p 8888:8888 nathanhillyer/collageradiomics-examples
```
### Windows/Powershell:
```
docker run -it -p 8888:8888 nathanhillyer/collageradiomics-pip
```
