from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

TESTS_REQUIRES = ['nose>=1.1.2']

setup(name='sklearn-sequence-classifiers',
      version='0.1',
      description='Sequence classifiers for scikit-learn',
      author='Antti Ajanki',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      tests_require=TESTS_REQUIRES,
      test_suite='nose.collector',
      author_email='antti.ajanki@iki.fi',
      license='BSD',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Software Development :: Libraries'
      ]
)
