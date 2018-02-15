# Sequence classifier with a scikit-learn interface

[![Build Status](https://travis-ci.org/aajanki/sequence-classification.svg?branch=master)](https://travis-ci.org/aajanki/sequence-classification)
[![PyPI version](https://badge.fury.io/py/sklearn-sequence-classifiers.svg)](https://badge.fury.io/py/sklearn-sequence-classifiers)
[![Sponsored](https://img.shields.io/badge/chilicorn-sponsored-brightgreen.svg)](http://spiceprogram.org/oss-sponsorship/)

Convolutional neural network sequence classifier in the spirit of [`[1]`](#references). Wraps a Keras implementation as a scikit-learn classifier.

## Software requirements

* Python (2.7 or >= 3.5)
* scikit-learn (tested on 0.19)
* Keras (tested on 2.0.5, with the Theano 0.9.0 backend)

## Installation

[Install Keras](https://keras.io/#installation). This has been tested on the Theano backend, should work on other backends, too.

```
pip3 install --user sklearn-sequence-classifiers
```

Installing from the source code:

```
git clone git@github.com:aajanki/sequence-classification.git
cd sequence-classification
python3 setup.py install --user
```

## Running tests

```
python3 setup.py test
```

## Usage example

Predicting IMDB review sentiments.

```python
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
```

## License

BSD

## References

[1] Yoon Kim: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
