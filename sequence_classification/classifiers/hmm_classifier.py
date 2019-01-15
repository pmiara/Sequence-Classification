from seqlearn.hmm import MultinomialHMM

from .sequence_classifier import SequenceClassifier


class HMMClassifier(SequenceClassifier):
    def __init__(self, name='HMM', transformer=None):
        super(HMMClassifier, self).__init__(name, transformer)
        self.model = MultinomialHMM()

    def _fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, len(X_train))

    def _predict(self, X):
        return self.model.predict(X)
