Conda
=====

From the repository root, create a development environment with native
dependencies from conda-forge::

   conda env create -f environment.yml
   conda activate collageradiomics
   python -m pytest

The pip section installs the checkout in editable mode. Run the environment
creation command from the repository root so that ``./module`` resolves correctly.
Use ``conda env update -f environment.yml --prune`` after dependency changes.

Build and test a local Conda package::

   conda install --override-channels -c conda-forge conda-build
   conda build --override-channels -c conda-forge --python 3.12 packaging/conda
   conda install --use-local --override-channels -c conda-forge collageradiomics

The recipe builds the checkout, includes the CLI, and tests both 2D and 3D
extraction. CI independently tests the editable environment and built package.
Keep its explicit version aligned with ``module/VERSION`` when releasing.

This local recipe is not a published conda-forge feedstock. A conda-forge
submission must use the released source archive and its SHA256 in place of
``source.path``, and a maintainer must accept the feedstock invitation. The old
staged-recipes PR #14719 was closed without creating a feedstock. See issue #25
for the current submission status.
