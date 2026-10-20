[![Tests](https://github.com/radxtools/collageradiomics/actions/workflows/tests.yml/badge.svg)](https://github.com/radxtools/collageradiomics/actions/workflows/tests.yml)
[![Documentation](https://readthedocs.org/projects/collageradiomics/badge/?version=latest)](https://collageradiomics.readthedocs.io/)

# Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe)

CoLlAGe produces thirteen voxel-wise Haralick statistics from local gradient
orientations in 2D or 3D images. The project includes a medical-image CLI,
notebooks, Docker targets, and a 3D Slicer extension.

**The 0.4 development version changes numerical results.** Recompute cached
features from 0.3.x. See [numerical conventions](module/docs/source/numerics.rst).
The public import and thirteen-channel layout remain compatible.

## Install from this checkout

Use Python 3.10 or newer. Commands below run from the repository root:

```shell
python3 -m venv env_collage
source env_collage/bin/activate
python -m pip install -e './module[cli]'
```

After publication, install with `python -m pip install 'collageradiomics[cli]'`.
The core needs NumPy, SciPy, scikit-image, and Mahotas. Example dependencies are
optional: `python -m pip install -r requirements.txt` installs them from this checkout.

## Python usage

```python
import numpy as np
from collageradiomics import Collage, HaralickFeature

image = np.random.default_rng(42).random((32, 32))
mask = np.zeros(image.shape, dtype=bool)
mask[8:24, 8:24] = True
collage = Collage(image, mask, svd_radius=2, num_unique_angles=8)
features = collage.execute()  # (32, 32, 13)
entropy = collage.get_single_feature_output(HaralickFeature.Entropy)
print(entropy[mask].mean())
```

Arrays use `(height,width,depth)` in 3D and return `(H,W,D,13,2)`, with theta at
index 0 and phi at index 1. Values outside the mask are NaN. SVD and Haralick
windows use edge replication at image boundaries. Three-dimensional gradients
include z; texture neighborhoods remain slice-wise.

## Command line

```shell
collageradiomics -i image.nii.gz -m mask.nii.gz -o features.csv --extendstats
collageradiomics -i cases.csv -o features.csv --verbose
collageradiomics --help
```

Batch CSVs need `ID,Image,Mask` columns. Relative paths are resolved against the
CSV directory. Successful cases become output rows; failures are recorded in
`errors_features.csv` and cause a nonzero exit code. Image and mask size, spacing,
origin, and direction must match. `--dimensions 2` accepts 2D images or single-slice
volumes; it never discards slices. Use `--label` to select one mask label.

## Examples and Slicer

```shell
python -m pip install -e './module[examples]'
jupyter lab jupyter/examples/example.ipynb
python module/test_script.py --output-dir output_data
python examples/isbi.py IMAGE MASK --output-dir output_data
```

The notebook uses deterministic synthetic data and inline figures. The DICOM
example preserves numeric arrays in `.npy` and writes derived DICOM previews.
The [Slicer extension](slicer/collageradiomics/README.md) accepts labelmaps and
segmentations, supports both angles, and provides cancellable progress.

## Development and tests

For containerized extraction or notebooks, see [docker/README.rst](docker/README.rst).
Both targets build the checkout and include the CLI; obsolete personal base images
and the pinned 0.2.6 Docker hierarchy have been retired.

```shell
python -m pip install -r requirements-test.txt
python -m pip install --no-build-isolation -e './module[cli]'
python -m pytest
python -m mypy
python -m ruff check module/collageradiomics module/_collageradiomics tests cli
```

CI tests the minimum stack on Python 3.10, current dependencies on 3.11–3.13,
source/wheel rebuilding, and the extension in Slicer 5.12.4. Add
`-c constraints-test.txt` to install commands for the minimum stack, or use
`pipenv sync --dev` for the locked Python 3.11 environment.
Conda users can run `conda env create -f environment.yml` from the repository root;
see [the Conda guide](packaging/conda/README.rst) for local package builds.

See [CONTRIBUTING.rst](CONTRIBUTING.rst) for style, tests, and documentation builds.
The [backlog record](BACKLOG.rst) maps all 28 issues to fixes and external follow-ups.
[Full documentation](https://collageradiomics.readthedocs.io/).

# Science
## Overview

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

<a href="http://bric-lab.com"><img align="right" height=100 src="https://static.wixstatic.com/media/a0e8e5_809a649f13254ff293405c7476004e20~mv2.png/v1/fill/w_248,h_240,al_c,usm_0.66_1.00_0.01/a0e8e5_809a649f13254ff293405c7476004e20~mv2.png"></a>

If you make use of this implementation, please cite the following paper:

[1] Prasanna, P., Tiwari, P., & Madabhushi, A. (2016). "Co-occurrence of Local Anisotropic Gradient Orientations (CoLlAGe): A new radiomics descriptor. Scientific Reports", 6:37241.

[2] R. M. Haralick, K. Shanmugam and I. Dinstein, "Textural Features for Image Classification," in IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-3, no. 6, pp. 610-621, Nov. 1973, [doi: 10.1109/TSMC.1973.4309314](https://doi.org/10.1109/TSMC.1973.4309314).
