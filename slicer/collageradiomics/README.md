[![doi](https://img.shields.io/badge/doi-10.1038/srep37241-brightgreen.svg)](https://doi.org/10.1038/srep37241)

# CollageRadiomics Slicer Extension

**CoLlAGe** captures subtle anisotropic differences in disease pathologies by measuring entropy of co-occurrences of voxel-level gradient orientations on imaging computed within a local neighborhood.

**CoLlAGe** is based on the hypothesis that disruption in tissue microarchitecture can be quantified on imaging by measuring the disorder in voxel-wise gradient orientations. CoLlAGe involves assigning every image voxel a ‘disorder value’ associated with the co-occurrence matrix of gradient orientations computed around every voxel.

Details on extraction of **CoLlAGe** features are included in [\[1\]](#references). After feature extraction, the subsequent distribution or different statistics such as mean, median, variance etc can be computed and used in conjunction with a machine learning classifier to distinguish similar appearing pathologies. The feasibility of CoLlAGe in distinguishing cancer from treatment confounders/benign conditions and characterizing molecular subtypes of cancers has been demonstrated in the context of multiple challenging clinical problems.

# Table of Contents
- [Slicer](#slicer)
  - [Overview](#overview)
- [Tutorials](#tutorials)
  - [End to End Demo](#end-to-end-demo)
  - [Multiple Angles & Textures](#multiple-angles-&-textures)
  - [Dependencies](#dependencies)
- [Contact](#contact)
- [References](#references)

# Slicer
_[Back to **Table of Contents**](#table-of-contents)_

The collageradiomics Slicer 3D extension allows a user to run the collage algorithm on 3D images and then visualize and save the results. This is done by providing an input volume and a mask via segmentation.

## Overview
_[Back to **Table of Contents**](#table-of-contents)_

The general operational flow for using the Slicer extension is to load a 3D image, segment a relavent portion, configure the collage parameters, and run the algorithm.

Once this is completed, the user can take the output volume(s) and visualize them within Slicer. Slicer also supports exporting in various formats, e.g. `.mha`.

# Tutorials
_[Back to **Table of Contents**](#table-of-contents)_

Here are some tutorials to get you started.

## End to End Demo

Here is a complete demonstration of loading sample data into Slicer, segmenting a tumor, visualizing the output, and saving it to a file.

[![Collage Demonstration](Tutorials/CollageFullDemo.png?raw=true)](https://youtu.be/9om8FMpY1vA "Collage Demonstration")

## Multiple Angles & Textures
_[Back to **Table of Contents**](#table-of-contents)_

Here is a demonstration of how to process and view multiple dominant directions and textures.

[![Collage Multiple Angles & Textures Demonstration](Tutorials/CollageMultipleDemo.png?raw=true)](https://youtu.be/9om8FMpY1vA "Collage Multiple Angles & Textures Demonstration")

# Contact
_[Back to **Table of Contents**](#table-of-contents)_

Please report any issues or feature requests via the [Issue Tracker](https://github.com/radxtools/collageradiomics/issues).

Additional information can be found on the [BrIC Lab](http://bric-lab.com) website.

# References
_[Back to **Table of Contents**](#table-of-contents)_

[![doi](https://img.shields.io/badge/doi-10.1038/srep37241-brightgreen.svg)](https://doi.org/10.1038/srep37241)

<a href="http://bric-lab.com"><img align="right" height=100 src="https://static.wixstatic.com/media/a0e8e5_809a649f13254ff293405c7476004e20~mv2.png/v1/fill/w_248,h_240,al_c,usm_0.66_1.00_0.01/a0e8e5_809a649f13254ff293405c7476004e20~mv2.png"></a>

If you make use of this implementation, please cite the following paper:

[1] Prasanna, P., Tiwari, P., & Madabhushi, A. (2016). "Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe): A new radiomics descriptor. Scientific Reports", 6:37241.

[2] R. M. Haralick, K. Shanmugam and I. Dinstein, "Textural Features for Image Classification," in IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-3, no. 6, pp. 610-621, Nov. 1973, [doi: 10.1109/TSMC.1973.4309314](https://doi.org/10.1109/TSMC.1973.4309314).
