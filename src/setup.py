#!/usr/bin/env python

from setuptools import find_packages, setup

install_requires = []

__version__ = '0.1.0'

d = setup(
    name='saiil_public',
    version=__version__,
    packages=find_packages(exclude=['tests*', 'docs*']),
    install_requires=install_requires,
    author='Guy Rosman',
    maintainer='Guy Rosman <rosman@csail.mit.edu>',
    url='https://github.com/SAIIL/SAIIL_public',
    keywords='laparoscopic,sdtm,model',
    classifiers=['Environment :: Console'],
    description="",
    long_description="",
    license='MIT',
    test_suite='nose.collector',
    tests_require=['nose'],
    scripts=[],
)
