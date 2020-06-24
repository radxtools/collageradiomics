# Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe)

**CoLlAGe** captures subtle anisotropic differences in disease pathologies by measuring entropy of co-occurrences of voxel-level gradient orientations on imaging computed within a local neighborhood.

**CoLlAGe** is based on the hypothesis that disruption in tissue microarchitecture can be quantified on imaging by measuring the disorder in voxel-wise gradient orientations. CoLlAGe involves assigning every image voxel a ‘disorder value’ associated with the co-occurrence matrix of gradient orientations computed around every voxel.

Details on extraction of **CoLlAGe** features are included in [\[1\]](#references). After feature extraction, the subsequent distribution or different statistics such as mean, median, variance etc can be computed and used in conjunction with a machine learning classifier to distinguish similar appearing pathologies. The feasibility of CoLlAGe in distinguishing cancer from treatment confounders/benign conditions and characterizing molecular subtypes of cancers has been demonstrated in the context of multiple challenging clinical problems.

## Feature Classes
Currently supports the following Haralick [\[2\]](#references) features:

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

## Idempotence
Our **CoLlAGe** module includes parameter tuning information in the output. It contains the image(s) and mask(s), and the settings applied upon them. This allows multiple fully reproducible runs without having to remember or find the original parameters.

## Documentation
The best source of documentation is found the examples - instructions provided below.

# Installation & Use
These instructions will help set up the **CoLlAGe** core module and examples. They assume you are working out of a terminal such as **Powershell** on Windows or **Konsole** on Linux.

## Docker
**Docker** is like a stand-alone operating system container that comes pre-installed with all the dependencies already set up properly. It allows you to jump right into coding with **CoLlAGe**. We offer 2 **Docker** images: a basic core image for you to start coding with the **CoLlAGe** features (called `ccipd/collageradiomics-pip`) and an image that contains a running **Jupyter** notebook with **CoLlAGe** pre-installed and examples ready to run (called `ccipd/collageradiomics-examples`).

### Docker Setup
#### Linux
* [Click here](https://www.docker.com/get-started) and follow th einstructions to install **Docker**.
* There are Ubuntu-specific instructions here: https://docs.docker.com/engine/install/ubuntu/

#### Windows
1. [Click here](https://www.docker.com/get-started) and follow the instructions to install **Docker**.
2. Search for **Docker** in your start manu and run it:  
![Docker Search](https://i.imgur.com/QrhfUj9.png)
3. If it's running you should see an icon:  
![Docker Initializing](https://i.imgur.com/lylVdSc.png)  
![Docker Icon](https://i.imgur.com/NzGJQaO.png)

### collageradiomics-examples Docker Image
This **Docker** image contains a running Jupyter notebook with the **CoLlAGe** module pre-installed. From the cloned repository directory, we will start up a **Docker** image which will run a live web server and host a **Jupyter** notebook at the URL http://localhost:8888 which contains examples of using the code.

_Note: Using this method requires you to pull the code from our repository. If you don't need the **Jupyter** examples and just want to start using **CoLlAGe** right away, you can skip this step and jump to the instructions for **Core** by [clicking here](#collageradiomics-pip-docker-image)._

#### Linux
_Note: This was tested on Ubuntu 19.10 and 20.04_

1. Install git:
```console
user@machine:~$ sudo apt -y install git
Reading package lists... Done
Building dependency tree       
Reading state information... Done
git is already the newest version (1:2.20.1-2ubuntu1.19.10.3).
0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.
user@machine:~$ 
```
2. Clone the repository:
```console
user@machine:~$ git clone https://github.com/ccipd/collageradiomics.git
Cloning into 'collageradiomics'...
remote: Enumerating objects: 280, done.
remote: Total 280 (delta 0), reused 0 (delta 0), pack-reused 280user
Receiving objects: 100% (280/280), 1.48 MiB | 9.23 MiB/s, done.
Resolving deltas: 100% (125/125), done.
user@machine:~$ cd collageradiomics
user@machine:~/collageradiomics$ ls -l

```
3. Pull the latest **Docker** image:  
```console
user@machine:~/collageradiomics$ sudo docker pull ccipd/collageradiomics-examples:latest
latest: Pulling from ccipd/collageradiomics-examples
Digest: sha256:107a2804e76b156f40d571b8422f822a3712353645c86e5cc2275d2aea85c9be
Status: Image is up to date for ccipd/collageradiomics-examples:latest
docker.io/ccipd/collageradiomics-examples:latest
user@machine:~/collageradiomics$ 
```
4. Run the **Docker** image:  
```console
user@machine:~/collageradiomics$ sudo docker run -it -p 8888:8888 -v $PWD:/root ccipd/collageradiomics-examples
[I 06:35:13.806 NotebookApp] Writing notebook server cookie secret to /tmp/notebook_cookie_secret
[W 06:35:14.030 NotebookApp] All authentication is disabled.  Anyone who can connect to this server will be able to run code.
[I 06:35:14.033 NotebookApp] Serving notebooks from local directory: /root
[I 06:35:14.034 NotebookApp] The Jupyter Notebook is running at:
[I 06:35:14.034 NotebookApp] http://d41cc76f5035:8888/
[I 06:35:14.034 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

You can now skip over the _Windows_ installation instructions and jump straight to the [Exploring The Examples](#exploring-the-examples) section.

#### Windows
1. Install **git**. A quick online search for _"git for Windows"_ will provide you with several options for installing the ```git``` command. If it's correctly installed, the following should output your current version of git:  
```console
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Try the new cross-platform PowerShell https://aka.ms/pscore6

PS C:\Users\user> git --version
git version 2.26.2.windows.1
PS C:\Users\user>
```
2. Clone the repository:  
```console
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Try the new cross-platform PowerShell https://aka.ms/pscore6

PS C:\Users\user> git clone https://github.com/ccipd/collageradiomics.git
Cloning into 'collageradiomics'...
remote: Enumerating objects: 280, done.
Receiving objects:  97% (272/280), 1.24 MiB | 1.02 MiB/sused 280 eceiving objects:  91% (255/280), 1.24 MiB | 1.02 MiB/s
Receiving objects: 100% (280/280), 1.48 MiB | 1.09 MiB/s, done.
Resolving deltas: 100% (125/125), done.
PS C:\Users\user> cd collageradiomics
PS C:\Users\user\collageradiomics> dir


    Directory: C:\Users\user\collageradiomics


Mode                LastWriteTime         Length Name
----                -------------         ------ ----
d-----       2020-06-24      3:31                docker
d-----       2020-06-24      3:31                jupyter
d-----       2020-06-24      3:31                module
d-----       2020-06-24      3:31                sample_data
-a----       2020-06-24      3:31            215 .gitignore
-a----       2020-06-24      3:31          35823 LICENSE
-a----       2020-06-24      3:31           4045 README.md
-a----       2020-06-24      3:31            136 start.sh


PS C:\Users\user\collageradiomics>
```
3. Give **Docker** access to your cloned repository:
   1. Right click on the context menu near the clock:  
   ![Docker Context Menu](https://i.imgur.com/CSY0GzK.png)
   2. Select _Dashboard_:  
   ![Docker Dashboard](https://i.imgur.com/zIlGKvb.png)
   3. After you click on _Dashboard_, a window will pop up. Click on the gear icon for _Settings_ and move to _Resources_ :arrow_right: _File Sharing_.  
   ![Docker Filesharing](https://i.imgur.com/JLiVp72.png)
   4. Add your cloned repository folder:  
   ![Docker Add Repo](https://i.imgur.com/lb8RN1O.png)
4. Pull the latest **Docker** image:
```console
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Try the new cross-platform PowerShell https://aka.ms/pscore6

PS C:\Users\user\collageradiomics> docker pull ccipd/collageradiomics-examples:latest
latest: Pulling from ccipd/collageradiomics-examples
d51af753c3d3: Already exists
fc878cd0a91c: Already exists
6154df8ff988: Already exists
fee5db0ff82f: Already exists
a6501aa3ed52: Already exists
Digest: sha256:107a2804e76b156f40d571b8422f822a3712353645c86e5cc2275d2aea85c9be
Status: Downloaded newer image for ccipd/collageradiomics-examples:latest
docker.io/ccipd/collageradiomics-examples:latest
PS C:\Users\user\collageradiomics> docker pull ccipd/collageradiomics-examples:latest
latest: Pulling from ccipd/collageradiomics-examples
Digest: sha256:107a2804e76b156f40d571b8422f822a3712353645c86e5cc2275d2aea85c9be
Status: Image is up to date for ccipd/collageradiomics-examples:latest
docker.io/ccipd/collageradiomics-examples:latest
PS C:\Users\user\collageradiomics>
```
5. Run the **Docker** image:
```console
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Try the new cross-platform PowerShell https://aka.ms/pscore6

PS C:\Users\user> cd collageradiomics
PS C:\Users\user\collageradiomics> docker run -it -p 8888:8888 -v ${PWD}:/root ccipd/collageradiomics-examples
[I 08:28:36.091 NotebookApp] Writing notebook server cookie secret to /tmp/notebook_cookie_secret
[W 08:28:36.576 NotebookApp] All authentication is disabled.  Anyone who can connect to this server will be able to run code.
[I 08:28:36.585 NotebookApp] Serving notebooks from local directory: /root
[I 08:28:36.585 NotebookApp] The Jupyter Notebook is running at:
[I 08:28:36.585 NotebookApp] http://c5745f91dbee:8888/
[I 08:28:36.585 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

#### Exploring The Examples
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
```console
[I 07:05:36.271 NotebookApp] The Jupyter Notebook is running at:
[I 07:05:36.271 NotebookApp] http://4f033d68769d:8888/
[I 07:05:36.271 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
^C[I 07:05:37.628 NotebookApp] interrupted
Serving notebooks from local directory: /root
0 active kernels
The Jupyter Notebook is running at:
http://4f033d68769d:8888/
Shutdown this notebook server (y/[n])? ^C[C 07:05:38.744 NotebookApp] received signal 2, stopping
[I 07:05:38.745 NotebookApp] Shutting down 0 kernels
user@machine:~/collageradiomics$ 
```

### collageradiomics-pip Docker Image
This is the most straightforward way to start playing with the code. And it does not require the `git` commands that the **Jupyter** examples require. This is simply a pre-built container that lets you start trying out the module in **Python** immediately.

#### Linux
1. Pull the latest **Docker** image:
```console
user@machine:~$ sudo docker pull ccipd/collageradiomics-pip:latest
latest: Pulling from ccipd/collageradiomics-pip
Digest: sha256:8fc7d61dbe6ad64eeff9c69cfaa788d90c61861bff8aaf8865ed1318c5666250
Status: Image is up to date for ccipd/collageradiomics-pip:latest
docker.io/ccipd/collageradiomics-pip:latest
user@machine:~/collageradiomics$
```
2. Run the **Docker** image:
```console
user@machine:~/collageradiomics$ sudo docker run -it -v $PWD:/root ccipd/collageradiomics-pip
root@12b12d2bff59:/# 
```

#### Windows
1. Pull the latest **Docker** image:
```console
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Try the new cross-platform PowerShell https://aka.ms/pscore6

PS C:\Users\user> docker pull ccipd/collageradiomics-pip:latest
latest: Pulling from ccipd/collageradiomics-pip
d51af753c3d3: Already exists
fc878cd0a91c: Already exists
6154df8ff988: Already exists
fee5db0ff82f: Already exists
e4255cf4d4f9: Downloading [=================>                                 ]  62.34MB/178.6MB
14a983cf96b6: Downloading [===========================>                       ]  55.72MB/102.9MB      
14a983cf96b6: Pull complete
Digest: sha256:8fc7d61dbe6ad64eeff9c69cfaa788d90c61861bff8aaf8865ed1318c5666250
Status: Downloaded newer image for ccipd/collageradiomics-pip:latest
docker.io/ccipd/collageradiomics-pip:latest
PS C:\Users\user>
```
2. Run the **Docker** image:
```console
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Try the new cross-platform PowerShell https://aka.ms/pscore6

PS C:\Users\user> docker pull ccipd/collageradiomics-pip:latest
PS C:\Users\user> docker run -it ccipd/collageradiomics-pip
root@461c5017ce0e:/#
```

#### Inside The Container
If your terminal prompt changes to `root@[random_string]:/#` then you are now working inside the standardized **Docker** sandbox container environment.

1. Test the python module by making sure the following command outputs `True` to the terminal:  
```console
root@12b12d2bff59:/# python -c 'import numpy as np; import collageradiomics; print(not not len(collageradiomics.__name__) and not not len(collageradiomics.Collage.from_rectangle(np.random.rand(20,20,3), 2, 2, 10, 10).execute()));'
True
root@12b12d2bff59:/# 
```
2. Starting coding with **CoLlAGe** in **Python** [(click here to jump to code examples)](#python-usage):
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
3. Exit the **Docker** container:
```console
>>> quit()
root@12b12d2bff59:/# exit
exit
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

# References

<a href="http://bric-lab.com"><img align="right" height=100 src="https://static.wixstatic.com/media/a0e8e5_809a649f13254ff293405c7476004e20~mv2.png/v1/fill/w_248,h_240,al_c,usm_0.66_1.00_0.01/a0e8e5_809a649f13254ff293405c7476004e20~mv2.png"></a>

If you make use of this implementation, please cite the following paper:

[1] Prasanna, P., Tiwari, P., & Madabhushi, A. (2016). "Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe): A new radiomics descriptor. Scientific Reports", 6:37241.

[2] R. M. Haralick, K. Shanmugam and I. Dinstein, "Textural Features for Image Classification," in IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-3, no. 6, pp. 610-621, Nov. 1973, [doi: 10.1109/TSMC.1973.4309314](https://doi.org/10.1109/TSMC.1973.4309314).

Please report any issues or feature requests via the [Issues](https://github.com/ccipd/collageradiomics/issues) tab

Additional information can be found on the [BrIC Lab](http://bric-lab.com) website.
