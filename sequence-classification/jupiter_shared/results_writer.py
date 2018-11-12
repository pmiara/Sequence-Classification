import numpy


class ResultsWriter:
    def __init__(self, file_name=""):
        self.file_name = file_name
        self.series_numbers = {}

    def write_confusion_matrix(self, name, matrix):
        if name not in self.series_numbers.keys():
            self.series_numbers[name] = 0
        numpy.savetxt("confusion_matrix/" + self.file_name + name + str(self.series_numbers[name]) + ".csv", matrix,
                      delimiter=",")
        self.series_numbers[name] = self.series_numbers[name] + 1
