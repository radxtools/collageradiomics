Changelog
=========

0.4.0 (unreleased)
------------------

Recompute cached features from 0.3.x. This release corrects principal-gradient
selection, SVD signs, centered texture windows, image/mask boundary handling,
3D end slices, the public Haralick feature mapping, maximal correlation, and
difference-variance selection. The thirteen feature names, public import, and
output axis order remain compatible. See the numerical migration guide.

* Require Python 3.10 or newer and use current numerical dependencies. Image I/O,
  plotting, DICOM, and notebook dependencies are optional extras.
* Validate arrays and parameters, including empty masks and nonfinite inputs.
  Small and edge-touching masks are supported. Expose throttled progress and use
  standard logging.
* Ship a working single/batch CLI with consistent summary statistics, mask label
  selection, physical-geometry checks, and per-case error reporting. Explicit 2D
  mode rejects multislice images. Constant-feature moments are finite.
* Repair the Slicer extension's labelmaps, segmentations, theta/phi outputs,
  parameter forwarding, cancellation, geometry, and temporary-node cleanup.
* Add numerical, CLI, distribution, example, and native Slicer tests; test the
  minimum Python 3.10 stack and current Python 3.11–3.13 stacks in CI.
* Replace external-viewer examples with a reproducible inline notebook. Fix
  derived DICOM serialization and the IBSI example's feature indexing.
* Add typed shared helpers, contributor/style guides, automatic documentation
  versions, CLI/notebook Docker builds, a Conda environment and package recipe,
  and validated release artifacts with PyPI trusted publishing.
