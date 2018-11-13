import json
import os.path


class ResultsReader:
    def __init__(self, base_dir="results", file_prefix=""):
        self.file_prefix = file_prefix
        self.base_dir = base_dir
        if not os.path.isdir(self.base_dir):
            raise Exception("directory " + self.base_dir + " not found")

    def read_results(self, classifiers):
        matrices = []
        for c in classifiers:
            series_number = 0
            classifier_name = c[0].name
            file_name = os.path.join(self.base_dir, self.file_prefix + classifier_name + str(series_number) + ".csv")
            while os.path.isfile(file_name):
                with open(file_name) as file:
                    data = json.load(file)
                matrices.append((classifier_name, data))
                series_number += 1
                file_name = os.path.join(self.base_dir, self.file_prefix + classifier_name + str(series_number) + ".csv")
        return matrices
