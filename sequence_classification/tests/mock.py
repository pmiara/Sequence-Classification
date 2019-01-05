from sequence_classification.classifiers.sequence_classifier import SequenceClassifier
from sequence_classification.sequence_transformer import SequenceTransformer
from sequence_classification.results_writer import ResultsWriter
from sequence_classification.results_reader import ResultsReader


PARAMS = 'params'
DATASET = 'dataset'
CLASSIFIER = 'classifier'
DATASET_NAME = 'test_dataset'
CONF_TRAIN_MAT = 'conf_matrix_train'
CONF_TEST_MAT = 'conf_matrix_test'


class MockClassifier(SequenceClassifier):
    def __init__(self, name='TEST', x=''):
        self.name = name
        self.x = x

    def predict(self, X):
        return X

    def fit(self, X, y):
        pass

class MockTransformer(SequenceTransformer):
    def fit_transform(self, raw_data):
        return raw_data

    def transform(self, raw_data):
        return raw_data


class MockWriter():
    def __init__(self, base_dir="results", file_prefix=""):
        self.base_dir = base_dir
        self.file_prefix = file_prefix

    def write_results(self, dataset, classifier, params, conf_matrix_train, conf_matrix_test):
        self.results = {DATASET: dataset, CLASSIFIER: classifier, PARAMS: params, CONF_TRAIN_MAT: conf_matrix_train,
                        CONF_TEST_MAT: conf_matrix_test}


class MockReader():
    def read_results(self, results):
        return results
