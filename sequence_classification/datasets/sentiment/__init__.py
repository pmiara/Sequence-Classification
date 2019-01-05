import numpy as np

from ..utils import DatasetLoader, Dataset, get_X_y


class CommentSentimentDataset(DatasetLoader):

    @staticmethod
    def load_data(name='Comment Sentiment'):
        data = []
        for file in ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']:
            data.append(np.genfromtxt(CommentSentimentDataset.get_dataset_file(['sentiment', file]),
                                      delimiter='\t', encoding="utf-8", dtype=None))
        sentiment_data = np.concatenate(tuple(data), axis=0)
        X, y = get_X_y(sentiment_data)
        return Dataset(X, y, name)
