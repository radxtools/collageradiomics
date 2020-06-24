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
This **Docker** image contains a running Jupyter notebook with the **CoLlAGe** module pre-installed. From the cloned repository directory, we will start up a **Docker** image which will run a live web server and host a **Jupyter** notebook at the URL http://localhost:8888 which contains examples of using the code.

_Note: Using this method requires you to pull the code from our repository. If you don't need the **Jupyter** examples and just want to start using **CoLlAGe** right away, you can skip this step and move onto the instructions for **Core** by [clicking here](#core)._

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
This is the most straightforward way to start playing with the code. And it does not require the `git` commands that the **Jupyter** examples require. This is simply a pre-built container

1. Pull the latest **Docker** image:
* _Linux_: `sudo docker pull ccipd/collageradiomics-pip:latest`
* _Windows_: `docker pull ccipd/collageradiomics-pip:latest`
2. Run the **Docker** image:
* _Linux_:
```console
user@machine:~/code/collageradiomics$ sudo docker run -it -v $PWD:/root ccipd/collageradiomics-pip
root@12b12d2bff59:/# 
```
* _Windows_:
```console
(powershell?)> docker run -it -p 8888:8888 -v ${PWD}:/root ccipd/collageradiomics-pip
```
3. If your terminal prompt changes to `root@[random_string]:/#` then you are working inside the **Docker** container environment.
* _Windows_: There is a button in the docker graphical user interface to load a terminal from a given downloaded
4. Test the python module by making sure the following command outputs `True` to the terminal:  
```console
root@12b12d2bff59:/# python -c 'import numpy as np; import collageradiomics; print(not not len(collageradiomics.__name__) and not not len(collageradiomics.Collage.from_rectangle(np.random.rand(20,20,3), 2, 2, 10, 10).execute()));'
True
root@12b12d2bff59:/# 
```
5. Starting coding with **CoLlAGe** in **Python**:
```console
root@12b12d2bff59:/# python
Python 3.8.2 (default, Apr 27 2020, 15:53:34) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import collageradiomics
>>> collageradiomics.__name__
'collageradiomics'
>>> 
```
6. Exit the **Docker** container:
```console
>>> quit()
root@12b12d2bff59:/# exit
exit
user@machine:~/code/collageradiomics$ 
```

## Pip
To use this module in your existing **Python** development environment, you can install our **pip** module.

1. Make sure **pip** is set up and installed on your appropriate operating system. [See instructions here](https://pip.pypa.io/en/stable/installing/).
* _Ubuntu_:
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

## Python Usage
collageradiomics can be implemented in **Python** through the `collageradiomics` pip module. It has a intuitive interface - simply create a `Collage` object or use one of the factory methods and run the `execute()` function.

### Basic Example
A simple example which executes it on a random array is as follows:
```console
user@machine:~$ python3
Python 3.7.5 (default, Apr 19 2020, 20:18:17) 
[GCC 9.2.1 20191008] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import collageradiomics
>>> import numpy as np
>>> image_w = 20; image_h = 20; image_d = 3;
>>> mask_min_x = 2; mask_min_y = 2; mask_w = 10; mask_h = 10;
>>> random_image = np.random.rand(image_h, image_w, image_d);
>>> collage = collageradiomics.Collage.from_rectangle(random_image, mask_min_x, mask_min_y, mask_w, mask_h)
>>> result = collage.execute();
>>> for f, slice in enumerate(result):
...   print(f'ColLiAGe Feature #{f+1} is a {type(slice)} of shape {slice.shape}');
... 
ColLiAGe Feature #1 is a <class 'numpy.ndarray'> of shape (20, 20)
ColLiAGe Feature #2 is a <class 'numpy.ndarray'> of shape (20, 20)
ColLiAGe Feature #3 is a <class 'numpy.ndarray'> of shape (20, 20)
ColLiAGe Feature #4 is a <class 'numpy.ndarray'> of shape (20, 20)
ColLiAGe Feature #5 is a <class 'numpy.ndarray'> of shape (20, 20)
ColLiAGe Feature #6 is a <class 'numpy.ndarray'> of shape (20, 20)
ColLiAGe Feature #7 is a <class 'numpy.ndarray'> of shape (20, 20)
ColLiAGe Feature #8 is a <class 'numpy.ndarray'> of shape (20, 20)
ColLiAGe Feature #9 is a <class 'numpy.ndarray'> of shape (20, 20)
ColLiAGe Feature #10 is a <class 'numpy.ndarray'> of shape (20, 20)
ColLiAGe Feature #11 is a <class 'numpy.ndarray'> of shape (20, 20)
ColLiAGe Feature #12 is a <class 'numpy.ndarray'> of shape (20, 20)
ColLiAGe Feature #13 is a <class 'numpy.ndarray'> of shape (20, 20)
>>> quit()
user@machine:~$
```

### Real Data
A simple example which uses a real-life sample image is as follows:
```console
user@machine:~$ git clone https://github.com/ccipd/collageradiomics
Cloning into 'collageradiomics'...
remote: Enumerating objects: 280, done.
remote: Counting objects: 100% (280/280), done.
remote: Compressing objects: 100% (157/157), done.
remote: Total 280 (delta 126), reused 231 (delta 86), pack-reused 0
Receiving objects: 100% (280/280), 1.48 MiB | 5.39 MiB/s, done.
Resolving deltas: 100% (126/126), done.
user@machine:~$ cd collageradiomics/sample_data
user@machine:~/collageradiomics/sample_data$ ll
total 472
drwxrwxr-x 2 user user   4096 Jun 24 02:18 ./
drwxrwxr-x 7 user user   4096 Jun 24 02:18 ../
-rw-rw-r-- 1 user user   1870 Jun 24 02:18 BrainSliceTumorMask.png
-rw-rw-r-- 1 user user  80680 Jun 24 02:18 BrainSliceTumor.png
-rw-rw-r-- 1 user user    375 Jun 24 02:18 ImageMask.png
-rw-rw-r-- 1 user user    604 Jun 24 02:18 ImageNonRectangularMask2.png
-rw-rw-r-- 1 user user    451 Jun 24 02:18 ImageNonRectangularMask.png
-rw-rw-r-- 1 user user 199662 Jun 24 02:18 ImageSlice2.png
-rw-rw-r-- 1 user user 172921 Jun 24 02:18 ImageSlice.png
user@machine:~/collageradiomics/sample_data$ python3
Python 3.7.5 (default, Apr 19 2020, 20:18:17) 
[GCC 9.2.1 20191008] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import collageradiomics
>>> 
```

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
