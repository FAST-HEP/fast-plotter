#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
from setuptools import setup, find_packages


with open('README.rst') as readme_file:
    readme = readme_file.read()


#with open('HISTORY.rst') as history_file:
#    history = history_file.read()


def get_version():
    _globals = {}
    with open(os.path.join("fast_plotter", "version.py")) as version_file:
        exec(version_file.read(), _globals)
    return _globals["__version__"]


requirements = ['matplotlib', 'pandas>=1.0.0', 'numpy>=1.16.5', 'scipy',
                'fast-curator', 'fast-flow']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', 'flake8', 'pytest-cov']

setup(
    author="Ben Krikler",
    author_email='fast-hep@cern.ch',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="F.A.S.T. plotter package",
    entry_points={
        'console_scripts': [
            'fast_plotter=fast_plotter.__main__:main',
            'fast_plotter_postproc=fast_plotter.postproc.__main__:main',
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme,
    include_package_data=True,
    keywords='fast_plotter',
    name='fast_plotter',
    packages=find_packages(include=['fast_plotter*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/fast-hep/fast-plotter',
    version=get_version(),
    zip_safe=False,
)
