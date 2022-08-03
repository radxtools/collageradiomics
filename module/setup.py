import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('VERSION') as version_file:
    version = version_file.read().strip()

setuptools.setup(
    name='collageradiomics',
    version=version,
    author='ThetaTech',
    author_email='robert.toth@thetatech.ai',
    description='Python implementation of COLLAGE texture features from BRIC & INVENTLAB.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ccipd/collageradiomics',
    project_urls={
        'Docker Examples': 'https://hub.docker.com/repository/docker/ccipd/collageradiomics-examples',
        'Docker Module': 'https://hub.docker.com/repository/docker/ccipd/collageradiomics-pip',
        'Github': 'https://github.com/ccipd/collageradiomics'
    },
    py_modules=['collageradiomics'],
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
    install_requires=[
        'numpy',
        'scikit-build',
        'scikit-image',
        'mahotas',
        'scipy'
    ],
    python_requires='>=3.6',
    keywords='radiomics cancerimaging medicalresearch computationalimaging',
)
