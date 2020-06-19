# Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe)

CoLlAGe captures subtle anisotropic differences in disease pathologies by measuring entropy of co-occurrences of voxel-level gradient orientations on imaging computed within a local neighborhood. CoLlAGe is based on the hypothesis that disruption in tissue microarchitecture can be quantified on imaging by measuring the disorder in voxel-wise gradient orientations. CoLlAGe involves assigning every image voxel a ‘disorder value’ associated with the co-occurrence matrix of gradient orientations computed around every voxel. Details on extraction of CoLlAGe features are included in [1]. After feature extraction, the subsequent distribution or different statistics such as mean, median, variance etc can be computed and used in conjunction with a machine learning classifier to distinguish similar appearing pathologies. The feasibility of CoLlAGe in distinguishing cancer from treatment confounders/benign conditions and characterizing molecular subtypes of cancers has been demonstrated in the context of multiple challenging clinical problems.


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

# References and Citations

If you make use of this implementation, please cite the following paper:

[1] Prasanna, P., Tiwari, P., & Madabhushi, A. (2016). Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe): A new radiomics descriptor. Scientific Reports, 6:37241.
