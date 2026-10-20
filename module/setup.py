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
    install_requires=requirements,
    python_requires='>=3.6',
    keywords='radiomics cancerimaging medicalresearch computationalimaging',
)
