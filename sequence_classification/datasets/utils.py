import re
import numpy as np
from os import path
from sklearn.feature_extraction.text import CountVectorizer


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


def get_X_y(data):
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 token_pattern=r'\b\w+\b')
    docs = [x[0] for x in data]
    vectorizer.fit(docs)
    integers_from_strings = [[vectorizer.vocabulary_.get(y.lower()) for y in re.sub(r'[.!,;?]', ' ', x).split() if
                              vectorizer.vocabulary_.get(y.lower()) is not None] for x in docs]
    return np.array(integers_from_strings), np.array([x[1] for x in data])
