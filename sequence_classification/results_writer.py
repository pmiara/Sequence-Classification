import json
import os
from collections import defaultdict


class ResultsWriter:
    def __init__(self, base_dir='results', file_prefix=''):
        self.file_prefix = file_prefix
        self.saved = defaultdict(set)
        self.base_dir = base_dir
        if not os.path.isdir(self.base_dir):
            os.mkdir(self.base_dir)

    def write_results(self, dataset_name, classifier_name, params, conf_matrix_train, conf_matrix_test):
        self.dataset_dir_exists(dataset_name)
        data = {'conf_matrix_train': conf_matrix_train.tolist(),
                'conf_matrix_test': conf_matrix_test.tolist(),
                'params': params}
        filename = self.file_prefix + classifier_name + '.txt'
        filename = os.path.join(self.base_dir, dataset_name, filename)
        if self.old_results_are_present(dataset_name, classifier_name, filename):
            os.remove(filename)
        with open(filename, 'a') as outfile:
            json.dump(data, outfile)
            outfile.write('\n')
        self.saved[dataset_name].add(classifier_name)

    def dataset_dir_exists(self, dataset_name):
        dataset_dir = os.path.join(self.base_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)

    def old_results_are_present(self, dataset_name, classifier_name, filename):
        return classifier_name not in self.saved[dataset_name] and os.path.exists(filename)
