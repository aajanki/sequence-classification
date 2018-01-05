import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array, assert_all_finite
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Convolution1D, GlobalMaxPooling1D, Dense, Activation


class CNNSequenceClassifier(BaseEstimator, ClassifierMixin):
    """Sequence classification using a convolutional neural network

    The input is a n_samples x n_sequence_length array of integers. Each row
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

    dropout_rates : tuple, optional (default=(0.5, 0.2))
        A two-tuple of dropout rates applied during the training as
        regularisation. The first element is the dropout between the embedding
        layer and the convolutional layer, the second is the dropout rate
        between the dense layer and the output layer.

    verbose : bool, optional (default=False)
        Enable verbose output during training.

    Attributes
    ----------
    net_ : Neural network object
        The fitted Keras neural network instance.

    References
    ----------
    "Convolutional Neural Networks for Sentence Classification" by Yoon Kim
    """
    def __init__(self,
                 embedding_dim=50,
                 num_filters=250,
                 filter_size=3,
                 hidden_dim=250,
                 dropout_rates=(0.5, 0.2),
                 verbose=False):
        super(CNNSequenceClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.hidden_dim = hidden_dim
        self.dropout_rates = dropout_rates
        self.verbose = verbose

    def fit(self, X, y, batch_size=32, epochs=2):
        assert_all_finite(X)
        X, y = check_X_y(X, y, dtype=[np.int32, np.int64],
                         warn_on_dtype=True,
                         ensure_min_features=self.filter_size)
        check_classification_targets(y)

        self.net_ = self._build_model(X.max() + 1,
                                      self.embedding_dim,
                                      X.shape[1],
                                      self.dropout_rates,
                                      self.num_filters,
                                      self.filter_size,
                                      self.hidden_dim)
        self.net_.fit(X, y,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1 if self.verbose else 0)

        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'net_')
        assert_all_finite(X)
        X = check_array(X, dtype=[np.int32, np.int64],
                        ensure_min_features=self.filter_size)
        return self.net_.predict(X)[:, 0]

    def predict(self, X):
        proba = self.predict_proba(X)
        predictions = np.zeros(X.shape[0], dtype=np.int)
        predictions[proba > 0.5] = 1
        return predictions

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def _build_model(self,
                     vocabulary_size,
                     embedding_dim,
                     maxlen,
                     dropout_rates,
                     num_filters,
                     filter_size,
                     hidden_dim):
        model = Sequential()
        model.add(Embedding(vocabulary_size, embedding_dim, input_length=maxlen))
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

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam')

        return model
