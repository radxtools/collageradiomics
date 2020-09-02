#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='Collageradiomics_CLI',
      version='1.0.0.',
      description='Get Collage features an image and a binary mask',
      url='https://github.com/radxtools/collageradiomics',
      python_requires='>=3.6',
    #   author='Sarthak Pati',
    #   author_email='software@cbica.upenn.edu',
      license='BSD-3-Clause',
      zip_safe=False,
      install_requires=[
      'collageradiomics',
      'setuptools>=47',
      'PyInstaller>=3.6',
      'SimpleITK==1.2.4'
      ],
      scripts=['collageradiomics_cli.py'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Operating System :: Unix'
      ]
      )