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

## Features
_[Back to **Table of Contents**](#table-of-contents)_

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

## Idempotence
_[Back to **Table of Contents**](#table-of-contents)_

Our **CoLlAGe** module includes parameter tuning information in the output. It contains the image(s) and mask(s), and the settings applied upon them. This allows multiple fully reproducible runs without having to remember or find the original parameters.

## Documentation
_[Back to **Table of Contents**](#table-of-contents)_

http://collageradiomics.rtfd.io/

## Dependencies:
_[Back to **Table of Contents**](#table-of-contents)_
- `matplotlib`
- `numpy`
- `scikit-learn`
- `scikit-build`
- `mahotas`
- `scipy`

# Installation & Usage
_[Back to **Table of Contents**](#table-of-contents)_

## Pip Usage
```console
pip3 install collageradiomics
```

## Local usage
```console
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
```console
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

## Pip
_[Back to **Table of Contents**](#table-of-contents)_

To use this module in your existing **Python** development environment, you can install our **pip** module.

### Linux
1. Install **pip**:
```console
user@machine:~$ sudo apt -y install python3-pip
Reading package lists... Done
Building dependency tree       
Reading state information... Done
python3-pip is already the newest version (18.1-5).
0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.
user@machine:~$ 
```
2. Install our **CoLlAGe** module:  
```console
user@machine:~$ pip3 install collageradiomics --upgrade
Collecting collageradiomics
  Downloading https://files.pythonhosted.org/packages/58/46/73d6b5a6d0d2b952086a1c9c4ae339087e4678f421044847ab2ea8728adf/collageradiomics-0.0.1a39-py3-none-any.whl
...
(some output omitted for brevity)
...
Successfully installed collageradiomics-...
user@machine:~$ 
```
*(Note: For some operating systems, the command is simply `pip` instead of `pip3`.)*

### Windows
1. Install **Python** using [this link](https://www.python.org/downloads/windows/).
2. Test that **Python** is properly installed in the Powershell:
```console
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Try the new cross-platform PowerShell https://aka.ms/pscore6

PS C:\Users\user> python --version
Python 3.8.2
PS C:\Users\user> python
Python 3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 23:03:10) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> quit()
PS C:\Users\user>
```
3. Install pip by downloading this [get-pip.py file](https://bootstrap.pypa.io/get-pip.py) and running the following command:
```console
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Try the new cross-platform PowerShell https://aka.ms/pscore6

PS C:\Users\user> python get-pip.py
Collecting pip
  Downloading pip-20.1.1-py2.py3-none-any.whl (1.5 MB)
     |████████████████████████████████| 1.5 MB 3.2 MB/s
Collecting wheel
  Using cached wheel-0.34.2-py2.py3-none-any.whl (26 kB)
Installing collected packages: pip, wheel
  Attempting uninstall: pip
    Found existing installation: pip 20.0.2
    Uninstalling pip-20.0.2:
      Successfully uninstalled pip-20.0.2
Successfully installed pip-20.1.1 wheel-0.34.2
PS C:\Users\user> pip -V
pip 20.1.1 from c:\users\robto\appdata\local\programs\python\python38\lib\site-packages\pip (python 3.8)
PS C:\Users\user>
```
4. Install our module
```console
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Try the new cross-platform PowerShell https://aka.ms/pscore6

PS C:\Users\user> pip install collageradiomics --upgrade
```
5. Verify its installation in **Python**:
```console
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Try the new cross-platform PowerShell https://aka.ms/pscore6

PS C:\Users\user> python
>>> import collageradiomics
>>>
```
6. If you get an error like the one below, which can happen for some versions of python, call `import` again:
```console
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Try the new cross-platform PowerShell https://aka.ms/pscore6

PS C:\Users\user> python
>>> import collageradiomics
Could not import submodules (exact error was: DLL load failed while importing _bbox: The specified module could not be found.).

There are many reasons for this error the most common one is that you have
either not built the packages or have built (using `python setup.py build`) or
installed them (using `python setup.py install`) and then proceeded to test
mahotas **without changing the current directory**.

Try installing and then changing to another directory before importing mahotas.
>>> import collageradiomics
>>>
```

## Python Usage
_[Back to **Table of Contents**](#table-of-contents)_

collageradiomics can be implemented in **Python** through the `collageradiomics` pip module. It has a intuitive interface - simply create a `Collage` object and run the `execute()` function.

# Other Platforms
_[Back to **Table of Contents**](#table-of-contents)_

The RadxTools COLLAGE implementation is now available through the Cancer Phenomics Toolkit (CaPTk), developed by the Center for Biomedical Image Computing and Analytics (CBICA) at the University of Pennsylvania. For more information see https://github.com/CBICA/CaPTk


# Contact
_[Back to **Table of Contents**](#table-of-contents)_

Please report any issues or feature requests via the [Issues](https://github.com/radxtools/collageradiomics/issues) tab

Additional information can be found on the [BrIC Lab](http://bric-lab.com) website.
