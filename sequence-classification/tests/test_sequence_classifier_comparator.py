from sequence_classifier_comparator import SequenceClassifierComparator
from classifiers.sequence_classifier import SequenceClassifier
from sequence_transformer import SequenceTransformer
from results_writer import ResultsWriter
from results_reader import ResultsReader

import numpy as np

NAME = 'name'
PARAMS = 'params'
CONF_TRAIN_MAT = 'conf_matrix_train'
CONF_TEST_MAT = 'conf_matrix_test'

TEST_PARAMS = {'a': 1}


class TestClassifier(SequenceClassifier):
    def __init__(self, name='TEST'):
        self.name = name

    def predict(self, X):
        return X

    def fit(self, X, y):
        pass

class TestTransformer(SequenceTransformer):
    def fit_transform(self, raw_data):
        return raw_data

    def transform(self, raw_data):
        return raw_data


class TestWriter(ResultsWriter):
    def __init__(self, base_dir="results", file_prefix=""):
        self.base_dir = base_dir
        self.file_prefix = file_prefix

    def write_results(self, name, params, conf_matrix_train, conf_matrix_test):
        self.results = {NAME: name, PARAMS: params, CONF_TRAIN_MAT: conf_matrix_train,
                        CONF_TEST_MAT: conf_matrix_test}


class TestReader(ResultsReader):
    def read_results(self, classifier_names):
        results = classifier_names
        return results


def test_should_add_classifier():
    # given
    seq_class_comparator = SequenceClassifierComparator(TestWriter(), TestReader())
    test_class = TestClassifier()

    # when
    seq_class_comparator.add_classifier(test_class)

    # then
    assert len(seq_class_comparator.classifier_triplets) == 1


def test_should_save_correctly_predicted_data():
    # given
    test_writer = TestWriter()
    seq_class_comparator = SequenceClassifierComparator(test_writer, TestReader())
    test_class = TestClassifier()
    seq_class_comparator.add_classifier(test_class)
    X = [1,2,3,4,5]
    y = [1,2,3,4,5]

    # when
    seq_class_comparator.fit_predict(X, y)

    # then
    assert test_writer.results[NAME] == "TEST"
    # assert test_writer.results[PARAMS] == PARAMS
    assert np.array_equal(test_writer.results[CONF_TRAIN_MAT], np.identity(len(test_writer.results[CONF_TRAIN_MAT])))
    assert np.array_equal(test_writer.results[CONF_TEST_MAT], np.identity(len(test_writer.results[CONF_TEST_MAT])))


def test_should_save_wrong_predicted_data():
    #given
    test_writer = TestWriter()
    seq_class_comparator = SequenceClassifierComparator(test_writer, TestReader())
    test_class = TestClassifier()
    seq_class_comparator.add_classifier(test_class)
    X = [1,2,3,4,5]
    y = [2,3,4,5,1]

    # when
    seq_class_comparator.fit_predict(X, y)

    # then
    assert np.add(test_writer.results[CONF_TRAIN_MAT], np.identity(len(test_writer.results[CONF_TRAIN_MAT]))).max() == 1
    assert np.add(test_writer.results[CONF_TEST_MAT], np.identity(len(test_writer.results[CONF_TEST_MAT]))).max() == 1
