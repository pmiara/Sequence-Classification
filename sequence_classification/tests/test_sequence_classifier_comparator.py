from sequence_classification.sequence_classifier_comparator import SequenceClassifierComparator
import numpy as np

from .mock import (MockClassifier, MockReader, MockWriter, PARAMS, DATASET,
    CLASSIFIER, DATASET_NAME, CONF_TRAIN_MAT, CONF_TEST_MAT)


def test_should_add_classifier():
    # given
    seq_class_comparator = SequenceClassifierComparator(MockWriter(), MockReader())
    test_class = MockClassifier()

    # when
    seq_class_comparator.add_classifier(test_class)

    # then
    assert len(seq_class_comparator.classifiers_with_params) == 1


def test_should_save_correctly_predicted_data():
    # given
    test_writer = MockWriter()
    seq_class_comparator = SequenceClassifierComparator(test_writer, MockReader())
    test_class = MockClassifier()
    seq_class_comparator.add_classifier(test_class)
    X = [1,2,3,4,5]
    y = [1,2,3,4,5]
    seq_class_comparator.add_custom_dataset(X, y, DATASET_NAME)

    # when
    seq_class_comparator.fit_predict_all()

    # then
    assert test_writer.results[DATASET] == DATASET_NAME
    assert test_writer.results[CLASSIFIER] == "TEST"
    assert test_writer.results[PARAMS] == {}
    assert np.array_equal(test_writer.results[CONF_TRAIN_MAT], np.identity(len(test_writer.results[CONF_TRAIN_MAT])))
    assert np.array_equal(test_writer.results[CONF_TEST_MAT], np.identity(len(test_writer.results[CONF_TEST_MAT])))


def test_should_save_wrong_predicted_data():
    #given
    test_writer = MockWriter()
    seq_class_comparator = SequenceClassifierComparator(test_writer, MockReader())
    test_class = MockClassifier()
    seq_class_comparator.add_classifier(test_class)
    X = [1,2,3,4,5]
    y = [2,3,4,5,1]
    seq_class_comparator.add_custom_dataset(X, y, DATASET_NAME)

    # when
    seq_class_comparator.fit_predict_all()

    # then
    assert np.diag(test_writer.results[CONF_TRAIN_MAT]).sum() == 0
    assert np.diag(test_writer.results[CONF_TEST_MAT]).sum() == 0


def test_should_change_name_and_params():
    # given
    test_writer = MockWriter()
    params = {"x":["a"]}
    seq_class_comparator = SequenceClassifierComparator(test_writer, MockReader())
    test_class = MockClassifier("NAME")
    seq_class_comparator.add_classifier(test_class, params=params)
    X = [1,2,3,4,5]
    y = [1,2,3,4,5]
    seq_class_comparator.add_custom_dataset(X, y, DATASET_NAME)

    # when
    seq_class_comparator.fit_predict_all()

    # then
    assert test_writer.results[CLASSIFIER] == "NAME"
    assert test_writer.results[PARAMS] == {'x': 'a'}
