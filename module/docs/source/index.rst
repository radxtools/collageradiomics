CoLlAGe radiomics
=================

Co-occurrence of Local Anisotropic Gradient Orientations describes local disorder
in image gradients. This package provides thirteen voxel-wise Haralick statistics,
a medical-image CLI, reproducible examples, and a 3D Slicer extension.

The scientific reference is Prasanna, Tiwari, and Madabhushi (2016),
`Scientific Reports 6:37241 <https://doi.org/10.1038/srep37241>`_.
The source and issue tracker are at
`radxtools/collageradiomics <https://github.com/radxtools/collageradiomics>`_.

.. toctree::
   :maxdepth: 2

   usage
   numerics
   contributing
   maintenance
   releases

Python API
----------

.. automodule:: collageradiomics
   :members: Collage, HaralickFeature, DifferenceVarianceInterpretation
