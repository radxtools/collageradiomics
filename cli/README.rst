CoLlAGe CLI compatibility package
=================================

This package installs ``collageradiomics[cli]>=0.4.0``. The command line interface
now ships with the core distribution. Both ``collageradiomics`` and the legacy
``collageradiomicscli.py`` commands remain available::

   python -m pip install collageradiomicscli
   collageradiomics --help
   collageradiomics -i image.nii.gz -m mask.nii.gz -o features.csv

See https://github.com/radxtools/collageradiomics for batch extraction,
installation from source, and numerical migration notes.
