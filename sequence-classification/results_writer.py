import json
import os


class ResultsWriter:
    def __init__(self, base_dir="results", file_prefix=""):
        self.file_prefix = file_prefix
        self.series_numbers = {}
        self.base_dir = base_dir
        if not os.path.isdir(self.base_dir):
            os.mkdir(self.base_dir)

    def write_results(self, name, params, conf_matrix_train, conf_matrix_test):
        if name not in self.series_numbers.keys():
            self.series_numbers[name] = 0
        data = {"conf_matrix_train": conf_matrix_train.tolist(),
                "conf_matrix_test": conf_matrix_test.tolist(),
                "params": params}
        filename = self.file_prefix + name + str(self.series_numbers[name]) + ".csv"
        with open(os.path.join(self.base_dir, filename), 'w') as outfile:
            json.dump(data, outfile)
        self.series_numbers[name] = self.series_numbers[name] + 1
