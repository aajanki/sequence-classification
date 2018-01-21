# Sequence classifier with a scikit-learn interface

[![Build Status](https://travis-ci.org/aajanki/sequence-classification.svg?branch=master)](https://travis-ci.org/aajanki/sequence-classification)
[![Sponsored](https://img.shields.io/badge/chilicorn-sponsored-brightgreen.svg)](http://spiceprogram.org/oss-sponsorship/)

Convolutional neural network sequence classifier in the spirit of [`[1]`](#references). Wraps a Keras implementation as a scikit-learn classifier.

## Software requirements

* Python (2.7 or >= 3.5)
* scikit-learn (tested on 0.19)
* Keras (tested on 2.0.5, with the Theano 0.9.0 backend)

## Installation

```
python setup.py install
```

## Running tests

```
python setup.py test
```

## Usage example

Predict IMDB review sentiment.

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

`[1]` Yoon Kim: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
