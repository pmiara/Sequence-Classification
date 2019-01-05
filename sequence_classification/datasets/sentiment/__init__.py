import numpy as np

from ..utils import DatasetLoader, Dataset, get_X_y


class CommentSentimentDataset(DatasetLoader):

    @staticmethod
    def load_data(name='Comment Sentiment'):
        data = np.genfromtxt(CommentSentimentDataset.get_dataset_file(['sentiment', 'amazon_cells_labelled.txt']),
                                      delimiter='\t', encoding="utf-8", dtype=None)
        X, y = get_X_y(data)
        return Dataset(X, y, name)
