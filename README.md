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
collageradiomics includes parameter tuning information in the output. It contains the image(s) and mask(s), and the settings applied upon them. This allows multiple fully reproducible runs without having to remember or find the original parameters.

## Documentation
The best source of documentation is found the examples - instructions provided below.

# Installation
These instructions will help set up the **CoLlAGe** core module and examples. They assume you are working out of a terminal such as **Powershell** on Windows or **Konsole** on Linux.

## Git
1. Install the ```git``` command based on your operating OS.
* _Ubuntu_: ```sudo apt -y install git```
* _Windows_: A quick online search will provide you with several packaged ```git``` executables.
2. In a shell (e.g. **Powershell** on Windows or **Konsole** on Linux) clone the repository:  
```git clone https://github.com/ccipd/collageradiomics.git```
3. Enter that directory and make sure you are in that directory for the examples below:  
```cd collageradiomics```

## Docker
### Overview
Docker is like a stand-alone operating system that comes pre-installed with all the dependencies already set up properly. It allows you to jump right into coding. 
We offer 2 docker images: a basic core image for you to start coding with the collage features, and an image that contains a running Jupyter notebook with collage pre-installed and examples ready to run.

### Setup
[Click here to install **Docker** for your operating system.](https://www.docker.com/get-started)

From here on, we will assuming you have a terminal open 

### Core

### Examples
From the cloned directory, we will start up a **Docker** image which will run a live web server and host a Jupyter notebook at the URL http://localhost:8888 which contains examples of using the code.

* _Linux_: (From a terminal) ```sudo docker pull ccipd/collageradiomics-examples:latest```
* _Windows

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

* How to setup docker on each OS
* Windows Powershell
* Linux sudo

### Core
* Pull the latest
* Windows powershell screenshot
* Windows-specific screenshots
* python import command

### Examples
* Docker run command
* Jupyter screenshot

## Pip
### Setup
* Set up pip on your appropriate operating system

```
pip3 install collageradiomics
```
*(Note: for some operating systems the command is simply ```pip```).*

## Git

```
git clone https://github.com/ccipd/collageradiomics && cd collageradiomics
```

## Docker Ubuntu 20.04 Jupyter Examples


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
