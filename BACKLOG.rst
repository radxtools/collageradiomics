Backlog resolution
===================

This record covers the original 28 issues and three dependency PRs. Implementation
and decisions are in merged
`PR #129 <https://github.com/radxtools/collageradiomics/pull/129>`_, with portable
Conda console entry points in
`PR #130 <https://github.com/radxtools/collageradiomics/pull/130>`_.
The code work is complete; four issues retain external completion steps below.
Version 0.4 changes numerical results: recompute cached 0.3.x features and read the
numerical migration guide. No clinical validation is implied by these tests.

Completed work
--------------

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Issue
     - Resolution and evidence
   * - #11
     - 110 Python tests cover validation, geometry, independent numerical values,
       CLI, examples, and sdist-to-wheel installation. CI tests Python 3.10–3.13;
       eight additional tests run in native Slicer 5.12.4.
   * - #24
     - README and CONTRIBUTING document virtual environments and checkout installs.
   * - #43
     - Both Docker targets include the CLI and SimpleITK. Container CI runs an
       actual image/mask extraction and checks 52 finite summary values.
   * - #52
     - The deterministic notebook uses inline Matplotlib and executes headlessly.
   * - #53
     - Typed helpers reject invalid array types, dimensions, shape, nonfinite
       values, empty masks, invalid enums, and malformed numerical parameters.
       Boundary and small-mask cases have regression tests.
   * - #54
     - CONTRIBUTING specifies Python syntax, formatting, logging, types, and
       numerical-test expectations. Ruff checks run in CI.
   * - #55
     - CONTRIBUTING documents setup, tests, Slicer testing, documentation, and PRs.
   * - #59
     - Core and I/O paths use TypeError, ValueError, RuntimeError, and I/O-specific
       failures. Tests check exception types and useful messages.
   * - #60
     - Public and internal packages are annotated, ship py.typed, and pass mypy
       across nine source files. CI checks types.
   * - #62
     - Validation, orientation, texture mapping, I/O, statistics, progress, and CLI
       live in the internal _collageradiomics package.
   * - #63
     - Keep major.minor.patch versioning. Numerical/API changes before 1.0 require
       a minor version and migration notes; dates do not convey compatibility.
       The maintenance guide records this decision.
   * - #68
     - Container CI builds and tests CLI/notebook targets. GHCR publication is an
       explicit manual input after successful builds.
   * - #69
     - Sphinx reads module/VERSION; Read the Docs installs the checkout. The docs
       build passes with warnings treated as errors.
   * - #77
     - The library uses the collageradiomics logger and leaves root configuration
       to callers. The legacy verbose_logging parameter is accepted and ignored.
   * - #80
     - The sdist includes VERSION, requirements, the DICOM sample, and its script.
       Distribution tests rebuild and run outside the checkout. The example
       locates its sample relative to its own file.
   * - #81
     - Keep shared helpers in this repository. There is no independent external
       consumer requiring a separate common-code repository. The maintenance
       guide records this decision.
   * - #84
     - NIfTI I/O validates geometry and converts axes consistently. Core geometry
       tests cover thin volumes and all masked slices; both angles remain finite.
   * - #85
     - Retain maintained numerical dependencies. Use NumPy rolling views, remove
       unused core dependencies, and keep GUI/I/O/example packages optional. The
       maintenance guide explains the decision against vendoring upstream code.
   * - #86
     - The CLI supports single/batch cases, label selection, all extraction
       parameters, consistent summaries, relative paths, and per-case error CSVs.
       Twenty CLI tests and container extraction cover its behavior.
   * - #90
     - Slicer creates separate theta (0) and phi (1) output volumes. Native tests
       select both angles and two features and verify all four outputs.
   * - #91
     - Throttled progress callbacks and INFO logging report stages; Slicer shows
       cancellable progress and removes temporary/incomplete nodes on cancellation.
   * - #97
     - Every masked 3D slice and both angles are calculated, including constant
       images and small/edge ROIs. Geometry/numerical tests require finite masked
       values. The original privately shared volume was unavailable for testing.
   * - #121
     - The original attached 512x512 image and mask now complete in about 30 seconds:
       all 92,918 masked pixels have finite thirteen-feature outputs, and background
       pixels remain NaN. Small boundary regressions preserve the shape fix.
   * - #124
     - Slicer accepts labelmaps directly and exports visible segmentations into
       the input geometry. Native tests cover both inputs and geometry/transform
       preservation.

External completion steps
-------------------------

These issues remain open so repository changes are not confused with third-party
acceptance or account ownership.

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Issue
     - Prepared work and remaining action
   * - #25
     - environment.yml and a native-tested local Conda recipe support checkout
       development and local packages. A separate conda-forge v1 recipe uses an
       official commit archive and verified SHA256, builds with rattler-build, and
       passes installed-package tests, including Linux and Windows builds.
       Explicit entry-point records regenerate the noarch console launchers on
       each platform. `Staged-recipes PR #35264
       <https://github.com/conda-forge/staged-recipes/pull/35264>`_ remains in draft
       for release-source substitution and acceptance. Feedstock maintenance is
       confirmed by mohsenhariri on the upstream PR.
   * - #73
     - `MRHub PR #119 <https://github.com/ISMRM/mrhub/pull/119>`_ adds a validated
       entry and original logo and is ready for review. MRHub maintainers must
       accept the listing.
   * - #78
     - The release guide specifies PyPI ownership, trusted publishers for both
       packages, and Read the Docs roles. A service owner must verify/add the
       maintainers and configure those accounts. Legacy GitHub secret names do
       not establish service ownership or working publication credentials.
       A draft 0.4.0 release with validated core/CLI artifacts and SHA256SUMS is
       prepared in `GitHub releases
       <https://github.com/radxtools/collageradiomics/releases>`_ for owner review;
       no package publication has been attempted.
   * - #83
     - `Slicer index PR #2424 <https://github.com/Slicer/ExtensionsIndex/pull/2424>`_
       supplies catalog metadata. The extension retries dependency imports after
       an explicit Install action and performs no package installation at startup.
       The source changes are merged and the complete remote-source catalog
       validator passes. Core 0.4 must reach PyPI, and Slicer maintainers must
       accept/build the catalog entry.

Dependency PR disposition
-------------------------

PRs #123 (SciPy 1.10.0), #125 (Pygments 2.15.0), and #126 (Tornado 6.3.3) were
closed as superseded after the coordinated refresh in #129 merged. The final
Python 3.11 lock uses
SciPy 1.17.1, Pygments 2.21.0, and Tornado 6.5.10. Applying the old lockfile diffs
would conflict and restore unrelated outdated versions. The refreshed lock
passes pip-audit with no known vulnerabilities as of
2026-10-20, including a fixed pytest version.
