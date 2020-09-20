#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='collageradiomicscli',
      version='1.0.0',
      description='Get Collage features from an image and a binary mask',
      url='https://github.com/radxtools/collageradiomics',
      python_requires='>=3.6',
      author='Toth Technology',
      author_email='toth-tech@hillyer.me',
      license='BSD-3-Clause',
      zip_safe=False,
      install_requires=[
      'collageradiomics',
      'setuptools>=47',
      'SimpleITK==1.2.4',
      'click'
      ],
      scripts=['collageradiomicscli.py'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Operating System :: POSIX :: Linux',
          'Operating System :: Microsoft :: Windows :: Windows 10',
          'Operating System :: MacOS',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
      ]
      )
