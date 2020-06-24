# Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe)

**CoLlAGe** captures subtle anisotropic differences in disease pathologies by measuring entropy of co-occurrences of voxel-level gradient orientations on imaging computed within a local neighborhood.

**CoLlAGe** is based on the hypothesis that disruption in tissue microarchitecture can be quantified on imaging by measuring the disorder in voxel-wise gradient orientations. CoLlAGe involves assigning every image voxel a ‘disorder value’ associated with the co-occurrence matrix of gradient orientations computed around every voxel.

Details on extraction of **CoLlAGe** features are included in [1]. After feature extraction, the subsequent distribution or different statistics such as mean, median, variance etc can be computed and used in conjunction with a machine learning classifier to distinguish similar appearing pathologies. The feasibility of CoLlAGe in distinguishing cancer from treatment confounders/benign conditions and characterizing molecular subtypes of cancers has been demonstrated in the context of multiple challenging clinical problems.

## Feature Classes
Currently supports the following Haralick [2] features:

1. `AngularSecondMoment`
2. `Contrast`
3. `Correlation`
4. `SumOfSquareVariance`
5. `SumAverage`
6. `SumVariance`
7. `SumEntropy`
8. `Entropy`
9. `DifferenceVariance`
10. `DifferenceEntropy`
11. `InformationMeasureOfCorrelation1`
12. `InformationMeasureOfCorrelation2`
13. `MaximalCorrelationCoefficient`

## Idempotence
Our **CoLlAGe** module includes parameter tuning information in the output. It contains the image(s) and mask(s), and the settings applied upon them. This allows multiple fully reproducible runs without having to remember or find the original parameters.

## Documentation
The best source of documentation is found the examples - instructions provided below.

# Installation
These instructions will help set up the **CoLlAGe** core module and examples. They assume you are working out of a terminal such as **Powershell** on Windows or **Konsole** on Linux.

## Git
1. Install the **git** software based on your OS:
* _Ubuntu_: `sudo apt -y install git`
* _Windows_: A quick online search will provide you with several packaged executables.
2. In a shell (e.g. **Powershell** on Windows or **Konsole** on Linux) clone the repository:  
`git clone https://github.com/ccipd/collageradiomics.git`
3. Enter that directory and make sure you are in that directory for the examples below:  
`cd collageradiomics`

## Docker
### Overview
**Docker** is like a stand-alone operating system container that comes pre-installed with all the dependencies already set up properly. It allows you to jump right into coding with **CoLlAGe**.

We offer 2 **Docker** images: a basic core image for you to start coding with the **CoLlAGe** features (called `ccipd/collageradiomics-pip`) and an image that contains a running **Jupyter** notebook with **CoLlAGe** pre-installed and examples ready to run (called `ccipd/collageradiomics-examples`).

### Setup
1. [Click here](https://www.docker.com/get-started) to install **Docker** for your operating system.
2. For _Windows_ users, on the **Docker** graphical user interface program, go to _Resources_ :arrow_right: _Shared Folders_ and add your cloned repository to the list of folders that **Docker** will share with the container.

### Examples
From the cloned repository directory, we will start up a **Docker** image which will run a live web server and host a **Jupyter** notebook at the URL http://localhost:8888 which contains examples of using the code.

1. Pull the latest **Docker** image:
* _Linux_:
```console
user@machine:~/code/collageradiomics$ sudo docker pull ccipd/collageradiomics-examples:latest
latest: Pulling from ccipd/collageradiomics-pip
Digest: sha256:8fc7d61dbe6ad64eeff9c69cfaa788d90c61861bff8aaf8865ed1318c5666250
Status: Image is up to date for ccipd/collageradiomics-pip:latest
docker.io/ccipd/collageradiomics-pip:latest
```
* _Windows_: `docker pull ccipd/collageradiomics-examples:latest`
2. Run the **Docker** image:
* _Linux_:
```console
user@machine:~/code/collageradiomics$ sudo docker run -it -p 8888:8888 -v $PWD:/root ccipd/collageradiomics-examples
```
* _Windows_: `docker run -it -p 8888:8888 -v ${PWD}:/root ccipd/collageradiomics-examples`
3. Open up a web browser to http://localhost:8888
4. Navigate to the _Jupyter_ :arrow_right: _Examples_ directory.
5. Click on one of the example `*.ipynb` files.
6. Run all cells.
7. Feel free to add your own cells and run them to get familiar with the **CoLlAGe** code.

_TODO: Jupyter screenshot.__

### Core
1. Pull the latest **Docker** image:
* _Linux_: `sudo docker pull ccipd/collageradiomics-pip:latest`
* _Windows_: `docker pull ccipd/collageradiomics-pip:latest`
2. Run the **Docker** image:
* _Linux_:
```console
user@machine:~/code/collageradiomics$ sudo docker run -it -v $PWD:/root ccipd/collageradiomics-pip
root@12b12d2bff59:/# 
```
* _Windows_: `docker run -it -p 8888:8888 -v ${PWD}:/root ccipd/collageradiomics-pip`
3. If your terminal prompt changes to `root@[random_string]:/#` then you are working inside the **Docker** container environment.
* _Windows_: There is a button in the docker graphical user interface to load a terminal from a given downloaded
4. Test the python module by making sure the following command outputs `True` to the terminal:  
```console
root@12b12d2bff59:/# python -c 'import numpy as np; import collageradiomics; print(not not len(collageradiomics.__name__) and not not len(collageradiomics.Collage.from_rectangle(np.random.rand(20,20,3), 10, 10, 10, 10).execute()));'
True
root@12b12d2bff59:/# 
```
* _Windows_: _(TODO)_
5. Code in **Python**:
* `python`
* `>>> import collageradiomics`

This should output (something).

## Pip
To use this module in your existing **Python** development environment, you can install our **pip** module.

1. Make sure **pip** is set up and installed on your appropriate operating system. [See instructions here](https://pip.pypa.io/en/stable/installing/).
2. Install our **CoLlAGe** module:  
`pip3 install collageradiomics`

*(Note: for some operating systems the command is simply `pip` instead of `pip**3**`).*

## Usage
collageradiomics can be implemented in **Python** through the `collageradiomics` pip module. It has a intuitive interface - simply create a `Collage` object or use one of the factory methods and run the `execute()` function.

## Dependencies:
We thank these generous developers that allowed us to build collageradiomics without reinventing the wheel:
- `matplotlib`
- `numpy`
- `scikit-learn`
- `scikit-build`
- `mahotas`
- `scipy`

_(Note: We are using ```==``` for version numbers of our dependencies as a design choice.)_

# References and Citations

<a href="http://bric-lab.com"><img align="right" height=100 src="https://static.wixstatic.com/media/a0e8e5_809a649f13254ff293405c7476004e20~mv2.png/v1/fill/w_248,h_240,al_c,usm_0.66_1.00_0.01/a0e8e5_809a649f13254ff293405c7476004e20~mv2.png"></a>

If you make use of this implementation, please cite the following paper:

[1] Prasanna, P., Tiwari, P., & Madabhushi, A. (2016). "Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe): A new radiomics descriptor. Scientific Reports", 6:37241.

[2] R. M. Haralick, K. Shanmugam and I. Dinstein, "Textural Features for Image Classification," in IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-3, no. 6, pp. 610-621, Nov. 1973, [doi: 10.1109/TSMC.1973.4309314](https://doi.org/10.1109/TSMC.1973.4309314).

Please report any issues or feature requests via the [Issues](https://github.com/ccipd/collageradiomics/issues) tab

Additional information can be found on the [BrIC Lab](http://bric-lab.com) website.
