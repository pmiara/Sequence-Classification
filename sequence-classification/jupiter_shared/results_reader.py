import numpy
import os.path


class ResultsReader:
    def __init__(self, file_name=""):
        self.file_name = file_name

    def read_confusion_matrices(self, classifiers):
        matrices = []
        for c in classifiers:
            series_number = 0
            classifier_name = c[0].name
            file_name = "confusion_matrix/" + self.file_name + classifier_name + str(series_number) + ".csv"
            while os.path.isfile(file_name):
                matrices.append((classifier_name,numpy.genfromtxt(file_name, delimiter=',').astype(int)))
                series_number += 1
                file_name = "confusion_matrix/" + self.file_name + c[0].name + str(series_number) + ".csv"
        return matrices
