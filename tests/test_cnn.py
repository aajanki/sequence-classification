import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sequence_classifiers import CNNSequenceClassifier
from sklearn.utils import check_random_state
from sklearn.utils.testing import assert_raises, assert_true, assert_greater
from sklearn.utils.testing import assert_equal, assert_array_equal, assert_allclose
from sklearn.metrics import accuracy_score


def test_check_cnn_classifier():
    return check_estimator(CNNSequenceClassifier)


def test_classifier_train():
    """sklearn's check_classifiers_train() with suitable training data"""
    generator = check_random_state(999)
    X_m = generator.randint(40, size=(200, 10))
    y_m = generator.randint(3, size=X_m.shape[0])
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]

    for (X, y) in [(X_m, y_m), (X_b, y_b)]:
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)

        classifier = CNNSequenceClassifier()

        # raises error on malformed input for fit
        assert_raises(ValueError, classifier.fit, X, y[:-1])

        classifier.fit(X, y)
        # with lists
        classifier.fit(X.tolist(), y.tolist(), epochs=50)

        assert_true(hasattr(classifier, "classes_"))
        y_pred = classifier.predict(X)
        assert_equal(y_pred.shape, (n_samples,))
        # training set performance
        assert_greater(accuracy_score(y, y_pred), 0.9)

        # predict_proba agrees with predict
        y_prob = classifier.predict_proba(X)
        assert_equal(y_prob.shape, (n_samples, n_classes))
        assert_array_equal(np.argmax(y_prob, axis=1), y_pred)
        # check that probas for all classes sum to one
        assert_allclose(np.sum(y_prob, axis=1), np.ones(n_samples))
        # raises error on malformed input
        assert_raises(ValueError, classifier.predict_proba, X.T)
        # raises error on malformed input for predict_proba
        assert_raises(ValueError, classifier.predict_proba, X.T)

        # predict_log_proba is a transformation of predict_proba
        y_log_prob = classifier.predict_log_proba(X)
        assert_allclose(y_log_prob, np.log(y_prob), 8, atol=1e-9)
        assert_array_equal(np.argsort(y_log_prob), np.argsort(y_prob))
