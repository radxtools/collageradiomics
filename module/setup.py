from pathlib import Path

import setuptools

module_dir = Path(__file__).resolve().parent
requirements = (module_dir / 'requirements.txt').read_text(encoding='utf-8').splitlines()
long_description = (module_dir / 'README.md').read_text(encoding='utf-8')
version = (module_dir / 'VERSION').read_text(encoding='utf-8').strip()

setuptools.setup(
    name='collageradiomics',
    version=version,
    author='ThetaTech',
    author_email='robert.toth@thetatech.ai',
    description='Python implementation of COLLAGE texture features from BRIC & INVENTLAB.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/radxtools/collageradiomics',
    project_urls={
        'Docker Examples': 'https://github.com/radxtools/collageradiomics/tree/master/docker',
        'Documentation': 'https://collageradiomics.readthedocs.io/',
        'Github': 'https://github.com/radxtools/collageradiomics'
    },
    py_modules=['collageradiomics'],
    packages=setuptools.find_packages(where=str(module_dir), include=['_collageradiomics*']),
    entry_points={'console_scripts': [
        'collageradiomics=_collageradiomics.cli:run',
        'collageradiomicscli.py=_collageradiomics.cli:run',
    ]},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    install_requires=requirements,
    extras_require={
        'cli': ['click>=8.1,<9', 'SimpleITK>=2.3,<3'],
        'examples': [
            'matplotlib>=3.8,<4', 'pydicom>=2.4,<4', 'scikit-learn>=1.3,<2',
            'nibabel>=5,<6', 'jupyterlab>=4.3,<5', 'SimpleITK>=2.3,<3',
        ],
    },
    python_requires='>=3.10',
    keywords='radiomics cancerimaging medicalresearch computationalimaging',
)
