from classifiers.sequence_classifier import SequenceClassifier
from sequence_transformer import SequenceTransformer
from results_writer import ResultsWriter
from results_reader import ResultsReader


NAME = 'name'
PARAMS = 'params'
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


class MockWriter(ResultsWriter):
    def __init__(self, base_dir="results", file_prefix=""):
        self.base_dir = base_dir
        self.file_prefix = file_prefix

    def write_results(self, name, params, conf_matrix_train, conf_matrix_test):
        self.results = {NAME: name, PARAMS: params, CONF_TRAIN_MAT: conf_matrix_train,
                        CONF_TEST_MAT: conf_matrix_test}


class MockReader(ResultsReader):
    def read_results(self, classifier_names):
        results = classifier_names
        return results
