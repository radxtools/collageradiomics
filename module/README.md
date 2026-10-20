# CoLlAGe radiomics

Co-occurrence of Local Anisotropic Gradient Orientations captures local disorder
in image gradients. This Python package produces thirteen voxel-wise texture maps
for 2D and 3D images and includes optional medical-image CLI and notebook support.

```shell
python -m pip install 'collageradiomics[cli]'
collageradiomics -i image.nii.gz -m mask.nii.gz -o features.csv
```

Requires Python 3.10 or newer. The numerical core uses NumPy, SciPy, scikit-image,
and Mahotas. Install `collageradiomics[examples]` for notebooks and DICOM examples.

Version 0.4 corrects gradient orientation, feature mapping, and mask boundaries;
recompute cached results from earlier versions. The public import and
thirteen-feature output shape are preserved.

[Documentation](https://collageradiomics.readthedocs.io/)
| [Source and issues](https://github.com/radxtools/collageradiomics)
| [Scientific reference](https://doi.org/10.1038/srep37241)
