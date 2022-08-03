![Continuous Delivery](https://github.com/radxtools/collageradiomics/workflows/Continuous%20Delivery/badge.svg) [![Documentation Status](https://readthedocs.org/projects/collageradiomics/badge/?version=latest)](https://collageradiomics.readthedocs.io/en/latest/?badge=latest) [![doi](https://img.shields.io/badge/doi-10.1038/srep37241-brightgreen.svg)](https://doi.org/10.1038/srep37241)

# Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe)

# Table of Contents
- [Science](#science)
  - [Overview](#overview)
  - [Features](#features)
  - [References](#references)
- [Code](#code)
  - [Idempotence](#idempotence)
  - [Documentation](#documentation)
  - [Dependencies](#dependencies)
- [Installation & Usage](#installation--usage)
  - [Executive Summary for Experts](#executive-summary-for-experts)
  - [Docker](#docker)
    - [Docker Setup](#docker-setup)
    - [collageradiomics-examples Docker Image](#collageradiomics-examples-docker-image)
    - [collageradiomics-pip Docker Image](#collageradiomics-pip-docker-image)
  - [Pip](#pip)
  - [Python Usage](#python-usage)
    - [Basic Example](#basic-example)
    - [Real Data](#real-data)
- [Other Platforms](#other-platforms)
- [Contact](#contact)

# Science
## Overview
_[Back to **Table of Contents**](#table-of-contents)_

**CoLlAGe** captures subtle anisotropic differences in disease pathologies by measuring entropy of co-occurrences of voxel-level gradient orientations on imaging computed within a local neighborhood.

**CoLlAGe** is based on the hypothesis that disruption in tissue microarchitecture can be quantified on imaging by measuring the disorder in voxel-wise gradient orientations. CoLlAGe involves assigning every image voxel a ‘disorder value’ associated with the co-occurrence matrix of gradient orientations computed around every voxel.

Details on extraction of **CoLlAGe** features are included in [\[1\]](#references). After feature extraction, the subsequent distribution or different statistics such as mean, median, variance etc can be computed and used in conjunction with a machine learning classifier to distinguish similar appearing pathologies. The feasibility of CoLlAGe in distinguishing cancer from treatment confounders/benign conditions and characterizing molecular subtypes of cancers has been demonstrated in the context of multiple challenging clinical problems.

Each of the 13 **CoLlAGe** correlate to one of the 13 Haralick texture features[\[2\]](#references):
1. _AngularSecondMoment_
2. _Contrast_
3. _Correlation_
4. _SumOfSquareVariance_
5. _SumAverage_
6. _SumVariance_
7. _SumEntropy_
8. _Entropy_
9. _DifferenceVariance_
10. _DifferenceEntropy_
11. _InformationMeasureOfCorrelation1_
12. _InformationMeasureOfCorrelation2_
13. _MaximalCorrelationCoefficient_

## References
_[Back to **Table of Contents**](#table-of-contents)_

<a href="http://bric-lab.com"><img align="right" height=100 src="https://static.wixstatic.com/media/a0e8e5_809a649f13254ff293405c7476004e20~mv2.png/v1/fill/w_248,h_240,al_c,usm_0.66_1.00_0.01/a0e8e5_809a649f13254ff293405c7476004e20~mv2.png"></a>

If you make use of this implementation, please cite the following paper:

[1] Prasanna, P., Tiwari, P., & Madabhushi, A. (2016). "Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe): A new radiomics descriptor. Scientific Reports", 6:37241.

[2] R. M. Haralick, K. Shanmugam and I. Dinstein, "Textural Features for Image Classification," in IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-3, no. 6, pp. 610-621, Nov. 1973, [doi: 10.1109/TSMC.1973.4309314](https://doi.org/10.1109/TSMC.1973.4309314).

# Code
_[Back to **Table of Contents**](#table-of-contents)_

We made the collage object idempotent Our **CoLlAGe** module includes parameter tuning information in the output. It contains the image(s) and mask(s), and the settings applied upon them. This allows multiple fully reproducible runs without having to remember or find the original parameters.

Documentation can be found at
http://collageradiomics.rtfd.io/

We depend on the following libraries:
- `matplotlib`
- `numpy`
- `scikit-learn`
- `scikit-build`
- `sklearn`
- `mahotas`
- `scipy`

# Installation & Setup
_[Back to **Table of Contents**](#table-of-contents)_

## Pip Usage
```console
pip3 install --upgrade collageradiomics
```

## Local usage
```shell
# get the code
git clone https://github.com/radxtools/collageradiomics
cd collageradiomics

# set up virtual environment
python3 -m venv collageenv
source collageenv/bin/activate

# install requirements
sudo apt -y install build-essential gcc gfortran python-dev libopenblas-dev liblapack-dev cython libjpeg-dev zlib1g-dev
pip3 install -r requirements.txt

# run test script
python3 collageradiomics/modules/test_script.py
```

## Docker Notebooks

### Prepare Jupyter Notebooks
To load the example jupyter notebooks, run the following commands (with or without `sudo` depending on your environment):
```console
sudo docker pull radxtools/collageradiomics-pip:latest
sudo docker run -it radxtools/collageradiomics-pip

git clone https://github.com/radxtools/collageradiomics.git
sudo docker pull radxtools/collageradiomics-examples:latest
sudo docker run -it -p 8888:8888 -v $PWD:/root radxtools/collageradiomics-examples
```

### Exploring The Examples
_[Back to **Table of Contents**](#table-of-contents)_

1. Open up a web browser to http://localhost:8888  
![Jupyter Home](https://i.imgur.com/0XQ8OlT.png)
2. Navigate to the _Jupyter_ :arrow_right: _Examples_ directory.  
![Jupyter Examples](https://i.imgur.com/NjdMlOr.png)
3. Click on one of the example `*.ipynb` files.
4. Run _Cell_ :arrow_right: _Run all_.  
![Jupyter Run Cells](https://i.imgur.com/GaAaNAS.png)
![Jupyter Output](https://i.imgur.com/PapCcsg.png)
5. Feel free to add your own cells and run them to get familiar with the **CoLlAGe** code.
6. To stop the **Jupyter** notebook and exit the **Docker** image, press `Ctrl+C` twice:

## Docker Sandbox
_[Back to **Table of Contents**](#table-of-contents)_

This is the most straightforward way to start playing with the code. And it does not require the `git` commands that the **Jupyter** examples require. This is simply a pre-built container that lets you start trying out the module in **Python** immediately.

1. Pull the latest **Docker** image:
```console
sudo docker pull radxtools/collageradiomics-pip:latest
sudo docker run -it -v $PWD:/root radxtools/collageradiomics-pip
```
If your terminal prompt changes to `root@[random_string]:/#` then you are now working inside the standardized **Docker** sandbox container environment.

1. Test the python module by making sure the following command outputs `True` to the terminal:  
```shell
python -c 'import numpy as np; import collageradiomics; print(not not len(collageradiomics.__name__) and not not len(collageradiomics.Collage.from_rectangle(np.random.rand(20,20,3), 2, 2, 10, 10).execute()));'
```
This should display `True`.

To run python code with collage:
```console
root@12b12d2bff59:/# python
Python 3.8.2 (default, Apr 27 2020, 15:53:34) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import collageradiomics
>>> collageradiomics.__name__
'collageradiomics'
```

# Python Usage
```python

##################################################
# imports
import collageradiomics
import pydicom
import logging
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from skimage.exposure import equalize_hist
import numpy as np
from sklearn.preprocessing import minmax_scale
from random import randint
##################################################


##################################################
# logging
level = logging.INFO
logging.basicConfig(level=level)
logger = logging.getLogger()
logger.setLevel(level)
logger.info('Hello, world.')
##################################################


##################################################
# loading data
local_dcm_file = 'test.dcm'
instance = pydicom.dcmread(local_dcm_file)
slice_instance_uid = instance.SOPInstanceUID
logger.debug(f'slice_instance_uid  = {slice_instance_uid}')
##################################################


##################################################
# preprocessing
logger.info('Correcting image...')
np_array = instance.pixel_array
corrected = apply_modality_lut(np_array, instance)
corrected = apply_voi_lut(corrected, instance)
scaled_array = equalize_hist(corrected)
logger.debug(f'np.histogram(scaled_array) = {np.histogram(scaled_array)}')
logger.info('done.')
##################################################


##################################################
# rectangular selection
width = 50
height = 50
min_row = randint(30,300)
max_row = min_row + height
min_col = randint(30,300)
max_col = min_col + width

original_shape = np_array.shape
logger.debug(f'original_shape = {original_shape}')
logger.info('Calculating collage features...')
mask_array = np.zeros(original_shape, dtype='int')
mask_array[min_row:max_row, min_col:max_col] = 1
##################################################


##################################################
# run collage
textures = collageradiomics.Collage(scaled_array, mask_array).execute()

logger.debug(f'textures.shape = {textures.shape}')
logger.debug(f'textures.dtype = {textures.dtype}')
logger.debug(f'np.histogram(textures.flatten()) = {np.histogram(textures.flatten(), range=(np.nanmin(textures), np.nanmax(textures)))}')
##################################################

```
