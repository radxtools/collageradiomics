Contributing
============

Work in a virtual environment from the repository root::

   python3 -m venv env_test
   source env_test/bin/activate
   python -m pip install -r requirements-test.txt
   python -m pip install --no-build-isolation -e './module[cli]'
   python -m pytest
   python -m mypy
   python -m ruff check module/collageradiomics module/_collageradiomics tests cli
   python -m ruff format --check module/collageradiomics module/_collageradiomics tests cli

Add ``-c constraints-test.txt`` to both install commands to reproduce the minimum
stack on Python 3.10. CI also tests current dependencies on 3.11, 3.12, and 3.13.
Pipenv users can run ``pipenv sync --dev`` using the checked-in Python 3.11 lock.

Code style
----------

Use Python 3.10 syntax, four spaces, descriptive names, and Ruff's 100-column
formatter. Keep the public import and feature order stable. Annotate the core and
shared helpers, then run mypy. Give errors specific types and useful parameter
names. Use the ``collageradiomics`` logger; library code does not configure root
logging or print progress. Keep shared calculations in ``_collageradiomics`` so
Python, CLI, examples, and Slicer use the same behavior. Retain upstream numerical
dependencies rather than copying their implementations into the repository.

Tests and changes
-----------------

Reproduce a reported problem with a small deterministic fixture. Numerical tests
should include independent expected values, shapes, mask placement, and NaN
behavior. Test batch and single-case CSV output together. A strict ``xfail`` is
appropriate only while a confirmed bug is being fixed; remove it with the fix.
Distribution tests must rebuild a wheel from its sdist and run outside the checkout.
Slicer changes must pass ``tests/run_slicer.py`` in the application, including GUI
parameter forwarding and temporary-node cleanup. Containers build in GitHub CI.

Use a short imperative commit subject. Explain changed numerical behavior and
validation in the PR. Prefer focused commits and include migration notes for
changes that affect cached features. Do not mix package publishing with tests.

Documentation
-------------

Install the documentation tools and build with warnings treated as errors::

   python -m pip install -r module/docs/requirements.txt
   python -m sphinx -W -b html module/docs/source /tmp/collage-docs

See the maintenance guide for versioning and release responsibilities.
