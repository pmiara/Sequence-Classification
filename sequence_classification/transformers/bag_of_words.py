from ..sequence_transformer import SequenceTransformer
from sklearn.feature_extraction.text import CountVectorizer



class BagOfWordsTransformer(SequenceTransformer):
    def __init__(self):
        self.vectorizer = CountVectorizer(analyzer="word",
                                          tokenizer=None,
                                          preprocessor=None,
                                          stop_words=None,
                                          token_pattern=r'\b\w+\b')

    def fit_transform(self, raw_data):
        strings_from_list = [' '.join([str(x) for x in data]) for data in raw_data]
        self.vectorizer.fit_transform(strings_from_list)
        return self.vectorizer.fit_transform(strings_from_list).toarray()

    def transform(self, raw_data):
        strings_from_list = [' '.join([str(x) for x in data]) for data in raw_data]
        return self.vectorizer.transform(strings_from_list).toarray()



