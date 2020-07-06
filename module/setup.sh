#!/bin/bash
rm -rf dist
python test_setup.py sdist bdist_wheel
twine upload dist/*