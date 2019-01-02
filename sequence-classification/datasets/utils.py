from os import path


class DatasetLoader:

    @staticmethod
    def load_data(name):
        raise NotImplementedError()

    @staticmethod
    def get_dataset_file(file_path):
        return path.join(path.dirname(__file__), *file_path)


class Dataset:
    def __init__(self, X, y, name):
        self.X = X
        self.y = y
        self.name = name
