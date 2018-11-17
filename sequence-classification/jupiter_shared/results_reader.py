import json
import os.path
from glob import glob
import numpy as np


class ResultsReader:
    def __init__(self, base_dir="results", file_prefix=""):
        self.file_prefix = file_prefix
        self.base_dir = base_dir
        if not os.path.isdir(self.base_dir):
            raise Exception("directory " + self.base_dir + " not found")

    def convert_data(self, data):
        result = {"conf_matrix_train": np.array(data["conf_matrix_train"]),
                  "conf_matrix_test": np.array(data["conf_matrix_test"]),
                  "params": data["params"]}
        return result

    def read_results(self, classifier_names):
        results = []
        for name in classifier_names:
            pathname = os.path.join(self.base_dir, self.file_prefix + name + "*.csv")
            for file_name in glob(pathname):
                with open(file_name) as file:
                    data = json.load(file)
                results.append((name, self.convert_data(data)))
        return results
