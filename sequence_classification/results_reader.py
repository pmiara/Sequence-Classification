import json
from collections import defaultdict
from os import path
import numpy as np


class ResultsReader:
    def __init__(self, base_dir="results", file_prefix=""):
        self.file_prefix = file_prefix
        self.base_dir = base_dir
        if not path.isdir(self.base_dir):
            raise Exception("directory " + self.base_dir + " not found")

    def convert_data(self, data):
        result = {"conf_matrix_train": np.array(data["conf_matrix_train"]),
                  "conf_matrix_test": np.array(data["conf_matrix_test"]),
                  "params": data["params"]}
        return result

    def read_results(self, dataset_names, classifier_names):
        results = {}
        for dataset in dataset_names:
            results[dataset] = defaultdict(list)
            for classifier in classifier_names:
                filename = path.join(self.base_dir, dataset, self.file_prefix + classifier + ".txt")
                with open(filename) as f:
                    for line in f.readlines():
                        data = json.loads(line)
                        results[dataset][classifier].append(self.convert_data(data))
        return results
