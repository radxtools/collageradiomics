Welcome to collageradiomics's documentation!
============================================

CoLlAGe captures subtle anisotropic differences in disease pathologies by measuring entropy of co-occurrences of voxel-level gradient orientations on imaging computed within a local neighborhood.

CoLlAGe is based on the hypothesis that disruption in tissue microarchitecture can be quantified on imaging by measuring the disorder in voxel-wise gradient orientations. CoLlAGe involves assigning every image voxel a ‘disorder value’ associated with the co-occurrence matrix of gradient orientations computed around every voxel.

Details on extraction of CoLlAGe features are included in [1]. After feature extraction, the subsequent distribution or different statistics such as mean, median, variance etc can be computed and used in conjunction with a machine learning classifier to distinguish similar appearing pathologies. The feasibility of CoLlAGe in distinguishing cancer from treatment confounders/benign conditions and characterizing molecular subtypes of cancers has been demonstrated in the context of multiple challenging clinical problems.

Helpful Links
=============
Instructions: `README <https://github.com/ccipd/collageradiomics/blob/master/README.md>`_

BriC Lab Website: `BriC Lab <https://www.bric-lab.com/>`_

Original Paper: `Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe): A new radiomics descriptor <https://www.nature.com/articles/srep37241>`_

Code Documentation
==================

.. automodule:: collageradiomics
   :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
