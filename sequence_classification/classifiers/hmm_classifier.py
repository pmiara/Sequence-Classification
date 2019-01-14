from seqlearn.hmm import MultinomialHMM
from sklearn.feature_extraction.text import CountVectorizer

from .sequence_classifier import SequenceClassifier
from ..sequence_transformer import SequenceTransformer


class HMMClassifier(SequenceClassifier):
    def __init__(self, name='HMM', transformer=None):
        super(HMMClassifier, self).__init__(name, transformer)
        self.model = MultinomialHMM()

    def _fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, len(X_train))

    def _predict(self, X):
        return self.model.predict(X)
