Releasing
=========

Version 0.4.0 changes numerical results. Read ``CHANGELOG.rst`` and the numerical
migration guide before updating an experiment. ``module/VERSION`` controls the
Python package and documentation version; update the Conda recipe's explicit
version at the same time. The compatibility CLI has its own version because it
is a small dependency shim around the core's ``[cli]`` extra.

Preparation
-----------

Run the documented tests, type/style checks, and documentation build. Require
green Python, native Slicer, example, container, Conda, and distribution workflows
on the final commit. Review the changelog, numerical differences, dependency lock,
and migration instructions. Merge the reviewed change before tagging it.

``sh module/setup.sh`` builds an sdist and a wheel from that sdist; it does not
upload anything. The Release packages workflow also builds the compatibility CLI,
runs the tests and ``twine check --strict``, and retains both artifact sets.
It builds on pull requests and supports manual validation without publishing.

Publisher setup (project owner)
--------------------------------

A PyPI owner must configure a trusted publisher for **each** project:
``collageradiomics`` and ``collageradiomicscli``. Use owner ``radxtools``, repository
``collageradiomics``, workflow ``python-publish.yml``, and environment ``pypi``.
Configure that GitHub environment's reviewers if the project requires a release
approval. See the official
`PyPI publisher setup <https://docs.pypi.org/trusted-publishers/adding-a-publisher/>`_.

The workflow uses short-lived OIDC credentials and does not read the legacy
``PYPI_USERNAME``/``PYPI_PASSWORD`` secrets. Their presence does not prove they
are current. A project owner should retire them after trusted publishing works.
PyPI and Read the Docs ownership and maintainer roles remain tracked in #78;
GitHub repository access does not grant either service's owner role.

Publication
-----------

Create a draft GitHub release with tag ``v`` followed by ``module/VERSION`` and
attach the validated artifacts for review. Publishing a non-prerelease GitHub
release triggers the PyPI job; draft releases and workflow-dispatch validation
do not upload packages. The workflow rejects tags that disagree with the package
version. Do not reuse a published version; PyPI files are immutable.

After publication, verify clean installs of ``collageradiomics[cli]`` and the
compatibility CLI, then update catalog submissions and the conda-forge recipe
to the released source URL and SHA256. External catalog/feedstock acceptance
is separate from producing tested contribution files in this repository.
