# Sequence classifier with a scikit-learn interface

Convolutional neural network sequence classifier. Wraps a Keras implementation as a scikit-learn classifier.

Requires Keras (tested with the Theano backend).

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

clf = CNNSequenceClassifier(filter_size=3)
clf.fit(x_train, y_train, epochs=2)
print(clf.score(x_test, y_test))
```
