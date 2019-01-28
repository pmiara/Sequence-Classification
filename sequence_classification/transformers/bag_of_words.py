from .sequence_transformer import SequenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


class BagOfWordsTransformer(SequenceTransformer):
    def __init__(self, token_pattern=r'\b\w+\b', **kwargs):
        self.vectorizer = CountVectorizer(token_pattern=token_pattern, **kwargs)

    def fit_transform(self, raw_data):
        strings_from_list = [' '.join([str(x) for x in data]) for data in raw_data]
        return self.vectorizer.fit_transform(strings_from_list).toarray()

    def transform(self, raw_data):
        strings_from_list = [' '.join([str(x) for x in data]) for data in raw_data]
        return self.vectorizer.transform(strings_from_list).toarray()
