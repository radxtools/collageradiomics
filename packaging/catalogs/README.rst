Catalog contributions
=====================

MRHub
-----

``mrhub.json`` is an entry for ``ismrm/mrhub``'s ``_data/projects.json``. Prepend
it to the existing list without changing other projects. Copy the unchanged
``slicer/collageradiomics/collageradiomics.png`` to
``images_packages/collageradiomics.png``. The entry uses MRHub's existing Image
processing category and leaves the citation count for its automatic updater.
Follow the upstream README and submit both files together. Acceptance is tracked
in issue #73.

3D Slicer
---------

``CollageRadiomics.json`` follows the current ExtensionsIndex schema and points
to the repository root. The root CMake entry point delegates to the extension
sources; ``build_subdirectory`` describes the build output, not a source subtree.
The extension can still be configured directly from ``slicer/collageradiomics``.

The extension retries its import after installing the required core and upgrades
an older core missing the progress API. Its native tests run in Slicer 5.12.4.
Submit the JSON file to ``Slicer/ExtensionsIndex`` after merging the source changes.
Keep the submission in draft until collageradiomics 0.4 is available from PyPI;
fresh Slicer installations need that dependency. Catalog acceptance is tracked
in issue #83.
