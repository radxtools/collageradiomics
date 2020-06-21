# Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe)

CoLlAGe captures subtle anisotropic differences in disease pathologies by measuring entropy of co-occurrences of voxel-level gradient orientations on imaging computed within a local neighborhood. CoLlAGe is based on the hypothesis that disruption in tissue microarchitecture can be quantified on imaging by measuring the disorder in voxel-wise gradient orientations. CoLlAGe involves assigning every image voxel a ‘disorder value’ associated with the co-occurrence matrix of gradient orientations computed around every voxel. Details on extraction of CoLlAGe features are included in [1]. After feature extraction, the subsequent distribution or different statistics such as mean, median, variance etc can be computed and used in conjunction with a machine learning classifier to distinguish similar appearing pathologies. The feasibility of CoLlAGe in distinguishing cancer from treatment confounders/benign conditions and characterizing molecular subtypes of cancers has been demonstrated in the context of multiple challenging clinical problems.

## Feature Classes
Currently supports the following Haralick features:

- AngularSecondMoment
- Contrast
- Correlation
- SumOfSquareVariance
- SumAverage
- SumVariance
- SumEntropy
- Entropy
- DifferenceVariance
- DifferenceEntropy
- InformationMeasureOfCorrelation1 
  - Both interpretations.
- InformationMeasureOfCorrelation2 
- MaximalCorrelationCoefficient 

## Idempotence
collageradiomics includes parameter tuning information in the output. It contains the image(s) and mask(s), and the settings apllied upon them. This allows multiple fully reproducible runs without having to remember or find the original parameters.

## Documentation
The best source of documentation is found the examples-- instructions provided below.

# Installation

## Pip
```
pip install collageradiomics
```

## Git

```
git clone https://github.com/ccipd/collageradiomics && cd collageradiomics
```

## Docker Ubuntu 20.04 Jupyter Examples

1. [Install Docker.](https://www.docker.com/get-started)
2. Pull the latest image: 
```
docker pull radiomics/collageradiomics-examples:latest
```
3. Run one of these commands in the __root of git repository__:
#### Linux:
```
docker run -it -p 8888:8888 -v $PWD:/root ccipd/collageradiomics-examples
```
#### Windows/Powershell:
```
docker run -it -p 8888:8888 -v ${PWD}:/root ccipd/collageradiomics-examples
```

#### Open in browser:
```
localhost:8888
```

Open `jupyter/examples` directory and explore the Jupyter notebooks.

## Docker Ubuntu 20.04 Pip Module
This is a Docker image with the collageradiomics module pre-installed:
```
docker pull radiomics/collageradiomics-pip:latest
docker run -it -p 8888:8888 ccipd/collageradiomics-pip
```

## Usage
collageradiomics can be implemented in Python through the collageradiomics module. It has a intuitive interface-- simply create a Collage object or use one of the factory methods and run the `execute()` function.

## Dependencies:
We thank these generous developers that allowed us to build collageradiomics without reinventing the wheel:
- matplotlib
- numpy
- scikit-learn
- scikit-build
- mahotas
- scipy

# References and Citations

<a href="http://bric-lab.com"><img align="right" height=100 src="https://static.wixstatic.com/media/a0e8e5_809a649f13254ff293405c7476004e20~mv2.png/v1/fill/w_248,h_240,al_c,usm_0.66_1.00_0.01/a0e8e5_809a649f13254ff293405c7476004e20~mv2.png"></a>

If you make use of this implementation, please cite the following paper:

[1] Prasanna, P., Tiwari, P., & Madabhushi, A. (2016). Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe): A new radiomics descriptor. Scientific Reports, 6:37241.

Please report any issues or feature requests via the [Issues](https://github.com/ccipd/collageradiomics/issues) tab

Additional information can be found on the [BrIC Lab](http://bric-lab.com) website.
