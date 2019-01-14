import numpy
from keras.datasets import imdb

from .utils import DatasetLoader, Dataset


class IMDbDataset(DatasetLoader, ):

    @staticmethod
    def load_data(name='IMDb'):
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
        return Dataset(numpy.concatenate([X_train, X_test]), numpy.concatenate([y_train, y_test]), name)
