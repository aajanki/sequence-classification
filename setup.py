from __future__ import print_function
import sys
import textwrap
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

TESTS_REQUIRES = ['nose>=1.1.2', 'theano']

setup(name='sklearn-sequence-classifiers',
      version='0.2',
      description='Sequence classifiers for scikit-learn',
      long_description=textwrap.dedent("""\
          Sequence classifiers for scikit-learn
          =====================================

          Convolutional neural network sequence classifier with a scikit-learn interface.

          Usage example
          -------------

          Predicting IMDB review sentiments.::

              from keras.datasets import imdb
              from keras.preprocessing import sequence
              from sequence_classifiers import CNNSequenceClassifier

              maxlen = 400
              (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
              x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
              x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

              clf = CNNSequenceClassifier(epochs=2)
              clf.fit(x_train, y_train)
              print(clf.score(x_test, y_test))
          """),
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
