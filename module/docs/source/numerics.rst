Numerical conventions and migration
===================================

Version 0.4 changes numerical results. Recompute cached features when moving from
0.3.x, and record package versions and extraction parameters with experiments.
The Python import, feature names, and thirteen-channel layout remain compatible.

Corrected calculations
----------------------

* The principal gradient comes from the first row of SciPy's returned ``Vh``.
  Its largest-magnitude component is made positive to resolve SVD sign ambiguity.
* SVD and texture windows include their complete halo. Image boundaries use edge
  replication. Small ROIs, masks touching one edge, full-image masks, and every
  masked slice in a 3D image are calculated.
* 3D gradients include the z component; SVD neighborhoods and co-occurrence
  calculations remain slice-wise, following the existing 3D convention.
* Haralick windows are centered and have exactly the specified odd size.
* Angles are linearly binned separately for theta and phi over the ROI plus its
  texture halo; constant orientations use bin zero.
* CoLlAGe's documented feature list omits inverse difference moment and includes
  maximal correlation. Mahotas' first thirteen outputs have a different layout;
  the returned values are now explicitly mapped to the public feature names.
* The selected difference-variance interpretation is forwarded to Mahotas.
  ``XMinusYVariance`` computes the variance of the absolute bin difference;
  ``ProbabilityXMinusYVariance`` computes the variance of its probability vector.
* Maximal correlation is the second singular value of
  ``P(i,j)/sqrt(Px(i)*Py(j))``, after removing empty marginals. It is zero when
  either variable is constant. See Haralick's
  `Dependence, section 3 <https://haralick.org/journals/dependency_prl_1.pdf>`_.

Regression tests contain analytical checkerboard, independent, and constant GLCMs,
known principal-gradient directions, volume-end masks, and single-voxel corners.
These are small numerical references and do not establish clinical validation.

Summary conventions
-------------------

The CLI reports median, population skewness, Fisher excess kurtosis, and sample
variance (ddof=1). ``--extendstats`` adds mean and IQR. For constant or single-voxel
features, skewness and kurtosis are zero; one voxel has variance zero. Moments
are scaled before evaluation to keep nearly constant maps finite.
