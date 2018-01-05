from sklearn.utils.estimator_checks import check_estimator
from sequence_classifiers import CNNSequenceClassifier


def test_check_cnn_classifier():
    return check_estimator(CNNSequenceClassifier)
