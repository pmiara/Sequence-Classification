from .transformers.sequence_transformer import SequenceTransformer
from .classifiers.sequence_classifier import SequenceClassifier


class CustomTransformer(SequenceTransformer):
    def __init__(self):
        # initialize transformer
        self.vectorizer = CustomVectorizer()

    def fit_transform(self, raw_data):
        # fit transformer to given data and transform given data
        self.vectorizer.fit(raw_data)
        return self.vectorizer.transform(raw_data)

    def transform(self, raw_data):
        # transform given data
        return self.vectorizer.transform(raw_data)


class CustomClassifier(SequenceClassifier):
    def __init__(self, name='CustomName', transformer=None):
        super(CustomClassifier, self).__init__(name, transformer)
        # set up model
        self.model = CustomModel()

    def _fit(self, X, y):
        # train model
        self.model.fit(X, y)
        return self

    def _predict(self, X):
        # predict classes for new data
        return self.model.predict(X)
