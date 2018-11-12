import numpy
import os.path


class ResultsReader:
    def __init__(self, file_prefix=""):
        self.file_prefix = file_prefix

    def read_confusion_matrices(self, classifiers):
        matrices = []
        for c in classifiers:
            series_number = 0
            classifier_name = c[0].name
            file_name = "results/confusion_matrix/" + self.file_prefix + classifier_name + str(series_number) + ".csv"
            while os.path.isfile(file_name):
                matrices.append((classifier_name, numpy.genfromtxt(file_name, delimiter=',').astype(int)))
                series_number += 1
                file_name = "results/confusion_matrix/" + self.file_prefix + c[0].name + str(series_number) + ".csv"
        return matrices

    def read_params(self, classifiers):
        parameters = []
        for c in classifiers:
            classifier_name = c[0].name
            f = open("results/parameters/" + self.file_prefix + classifier_name + ".csv", "r")
            data = f.read()
            f.close()
            parameters.append((classifier_name, eval(data)))
        return parameters
