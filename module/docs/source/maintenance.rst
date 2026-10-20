Maintenance decisions
=====================

Versioning
----------

Keep ``major.minor.patch`` versioning. Compatible fixes increment patch; before
1.0, numerical or API changes increment minor and require migration notes. Calendar
versions are accepted by Python packaging, but compatibility is more useful than
a date for this scientific library. This resolves the investigation in issue #63
without changing version schemes. See the
`Python packaging versioning guide <https://packaging.python.org/en/latest/discussions/versioning/>`_.

``module/VERSION`` is the single version source for package builds and Sphinx.
Read the Docs installs the checkout rather than a stale PyPI copy.

Shared code and dependencies
----------------------------

The internal package owns validation, SVD orientation, texture mapping, image I/O,
statistics, CLI, and progress. These helpers stay in this repository: there is no
second active consumer that justifies a separate common-code repository (#81).
The core depends on NumPy, SciPy, scikit-image, and Mahotas; GUI, I/O, and example
dependencies are optional. NumPy provides rolling views. Upstream co-occurrence
and texture implementations remain dependencies rather than vendored copies (#85).

External ownership and catalogs
-------------------------------

Repository code cannot grant PyPI or Read the Docs roles. A project owner must
verify organization ownership, add current maintainers, and configure the publisher.
Those checks remain the owner task in #78. Catalog and conda-forge submissions
also require acceptance by their maintainers; prepared contribution files and
submission links belong in the backlog record, not a claim of completed acceptance.
