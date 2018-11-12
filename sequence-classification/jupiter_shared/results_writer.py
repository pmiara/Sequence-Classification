import numpy


class ResultsWriter:
    def __init__(self, file_prefix=""):
        self.file_prefix = file_prefix
        self.series_numbers = {}

    def write_confusion_matrix(self, name, matrix):
        if name not in self.series_numbers.keys():
            self.series_numbers[name] = 0
        numpy.savetxt("results/confusion_matrix/" + self.file_prefix + name + str(self.series_numbers[name]) + ".csv", matrix,
                      delimiter=",")
        self.series_numbers[name] = self.series_numbers[name] + 1

    def write_params(self, name, param_dict):
        f = open("results/parameters/" + self.file_prefix + name + ".csv", 'w')
        f.write(str(param_dict))
        f.close()