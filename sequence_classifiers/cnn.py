import os
import tempfile
import warnings
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array, assert_all_finite
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential, load_model
from keras.layers import Embedding, Dropout, Convolution1D, \
    GlobalMaxPooling1D, Dense, Activation
from sklearn.exceptions import DataConversionWarning


def serialize_net_(net):
    """Serialize a Keras net as HDF5 bytestring."""
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.file.close()
    net.save(tmp.name)

    with open(tmp.name, 'rb') as f:
        res = f.read()

    delete_ignore_errors(tmp.name)

    return res


def deserialize_net(bytes):
    """Deserialize a Keras net from an HDF5 byte representation."""
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.file.write(bytes)
    tmp.file.close()

    res = load_model(tmp.name)

    delete_ignore_errors(tmp.name)

    return res


def delete_ignore_errors(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


class CNNSequenceClassifier(BaseEstimator, ClassifierMixin):
    """Sequence classification using a convolutional neural network

    The input is a n_samples x sequence_length array of integers. Each row
    is one sequence of token indexes.

    The classifier fits a neural network consisting of the following layers to
    the data:

      word embedding -> 1-D convolution -> dense -> sigmoid

    The classifier is implemented using Keras.

    Parameters
    ----------
    embedding_dim : int, optional (default=50)
        The dimensionality of the learned embedding of the tokens.

    num_filters : int, optional (default=250)
        The number of convolutional filters to fit.

    filter_size : int, optional (default=3)
        The width of the convolutional filter, the number of the consecutive
         tokens covered by on filter.

    hidden_dim : int, optional (default=250)
        The dimensionality of the dense netwrok layer after the convolution.

    dropout_rates : tuple, optional (default=(0.2, 0.2))
        A two-tuple of dropout rates applied during the training as
        regularisation. The first element is the dropout between the embedding
        layer and the convolutional layer, the second is the dropout rate
        between the dense layer and the output layer.

    epochs : int, optional (default=4)
        Number of epochs, that is, full passes over the training data, to run
        on fit().

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    verbose : bool, optional (default=False)
        Enable verbose output during training.

    Attributes
    ----------
    net_ : Neural network object
        The fitted Keras neural network instance.

    References
    ----------
    "Convolutional Neural Networks for Sentence Classification" by Yoon Kim

    Examples
    --------
    >>> from keras.datasets import imdb
    >>> from keras.preprocessing import sequence
    >>> from sequence_classifiers import CNNSequenceClassifier

    >>> maxlen = 400
    >>> (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
    >>> x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    >>> x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    >>> clf = CNNSequenceClassifier(epochs=2)
    >>> clf.fit(x_train, y_train)
    >>> print(clf.score(x_test, y_test))
    ...                    # doctest: +SKIP
    ...
    0.87736
    """
    def __init__(self,
                 embedding_dim=50,
                 num_filters=250,
                 filter_size=3,
                 hidden_dim=250,
                 dropout_rates=(0.2, 0.2),
                 epochs=4,
                 batch_size=32,
                 verbose=False):
        super(CNNSequenceClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.hidden_dim = hidden_dim
        self.dropout_rates = dropout_rates
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        """Fit a convolutional neural network classifier according to X, y

        Parameters
        ----------
        X : array-like, shape (n_samples, sequence_length)
            Training sequences of integer tokens, where n_samples is the
            number of samples and sequence_length is the number of tokens in a
            sequence.

            Each sequence must be zero-padded to the same length
            (sequence_length) using, for example,
            keras.preprocessing.sequence.pad_sequences().

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        check_classification_targets(y)
        X, y = self._check_input(X, y)
        binarizer = LabelBinarizer()
        y = binarizer.fit_transform(y)
        self.classes_ = binarizer.classes_
        self.net_ = self._build_model(X.max() + 1,
                                      self.embedding_dim,
                                      X.shape[1],
                                      self.dropout_rates,
                                      self.num_filters,
                                      self.filter_size,
                                      self.hidden_dim,
                                      len(self.classes_))
        self.net_.fit(X, y,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      verbose=1 if self.verbose else 0)

        return self

    def predict_proba(self, X):
        """
        Return probability estimates for the test sequences X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, sequence_length]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        check_is_fitted(self, 'net_')
        X, _ = self._check_input(X)
        proba = self.net_.predict(X)
        if len(self.classes_) == 2:
            return np.hstack((1 - proba, proba))
        else:
            return proba

    def predict(self, X):
        """
        Perform classification on an array of test sequences X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, sequence_length]

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test sequences X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, sequence_length]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        return np.log(self.predict_proba(X))

    def _check_input(self, X, y=None):
        # We check for finiteness separately here, because the finiteness
        # check in check_X_y() does not work when X is converted to integer.
        # Moreover, assert_all_finite() is buggy on sparse input so we skip it
        # (check_X_y/check_array will anayway raise an exception on sparse
        # arrays).
        if not sparse.issparse(X):
            assert_all_finite(X)

        if y is None:
            X = check_array(X, dtype=[np.int32, np.int64],
                            warn_on_dtype=True,
                            ensure_min_features=self.filter_size)
        else:
            X, y = check_X_y(X, y, dtype=[np.int32, np.int64],
                             warn_on_dtype=True,
                             ensure_min_features=self.filter_size)

        if X.min() < 0:
            warnings.warn('Negative values in X cropped to zero.',
                          DataConversionWarning)
            X = np.maximum(X, 0)

        return X, y

    def _build_model(self,
                     vocabulary_size,
                     embedding_dim,
                     maxlen,
                     dropout_rates,
                     num_filters,
                     filter_size,
                     hidden_dim,
                     num_classes):
        model = Sequential()
        model.add(Embedding(vocabulary_size, embedding_dim,
                            input_length=maxlen))
        model.add(Dropout(dropout_rates[0]))

        model.add(Convolution1D(num_filters,
                                filter_size,
                                padding='valid',
                                activation='relu',
                                strides=1))
        model.add(GlobalMaxPooling1D())

        model.add(Dense(hidden_dim))
        model.add(Dropout(dropout_rates[1]))
        model.add(Activation('relu'))

        if num_classes <= 2:
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
        else:
            model.add(Dense(num_classes))
            model.add(Activation('softmax'))

        model.compile(loss='binary_crossentropy', optimizer='adam')

        return model

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'net_' in state:
            state['net_bytes_'] = serialize_net_(state['net_'])
            del state['net_']

        return state

    def __setstate__(self, state):
        if 'net_bytes_' in state:
            self.__dict__['net_'] = deserialize_net(state['net_bytes_'])
            del state['net_bytes_']

        self.__dict__.update(state)
